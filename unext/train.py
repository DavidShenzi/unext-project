"""Training entry point.

Changes from upstream:
  * Defaults now match the *paper* (lr 1e-4, batch 8, 400 epochs), not the repo's
    generic CLI scaffolding (lr 1e-3, batch 16, 100 epochs).
  * `--seed` seeds everything and is also used as the train/val split seed, so
    baseline and modified runs see identical splits. `--split_seed` overrides it
    if you want to sweep the 3 splits the paper averages over.
  * `num_workers` defaults to 0 on Windows: the spawn-based worker start method
    makes >0 slow to start and prone to hanging here.
  * Device is resolved once (`cuda` if available, else `cpu`) instead of hard
    `.cuda()` calls, so the code runs on CPU for smoke tests.
  * AMP (`--amp`, on by default) — roughly halves memory and speeds up training
    materially on your RTX 5060.
  * Logs train loss/iou AND val loss/iou/dice per epoch to log.csv, plus the
    per-image metrics, and records total wall-clock training time.
  * albumentations imports use the modern top-level API (`A.Flip` etc. moved out
    of `albumentations.augmentations.transforms` in v1.x → v2.x).
"""

import argparse
import os
import time
from collections import OrderedDict
from glob import glob

import albumentations as A
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score, iou_dice_per_image
from utils import AverageMeter, count_params, seed_everything, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__ + ['BCEWithLogitsLoss']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='run name (default: <dataset>_<arch>)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs (paper: 400)')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N',
                        help='mini-batch size (paper: 8)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext',
                        choices=ARCH_NAMES, help='|'.join(ARCH_NAMES))
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--se_reduction', default=16, type=int,
                        help='SE bottleneck ratio (UNext_SE only)')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)

    # dataset
    parser.add_argument('--dataset', default='busi', help='folder name under inputs/')
    parser.add_argument('--img_ext', default='.png')
    parser.add_argument('--mask_ext', default='.png')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (paper: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau',
                                 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N')

    # runtime
    parser.add_argument('--num_workers', default=0, type=int,
                        help='0 is the safe default on Windows')
    parser.add_argument('--seed', default=41, type=int)
    parser.add_argument('--split_seed', default=None, type=int,
                        help='train/val split seed; defaults to --seed')
    parser.add_argument('--amp', default=True, type=str2bool,
                        help='mixed precision training')
    parser.add_argument('--stop_after', default=None, type=int, metavar='N',
                        help='halt after N epochs WITHOUT changing the cosine schedule '
                             '(T_max stays --epochs). Use this to run a partial job that '
                             'can later be resumed to the full --epochs; passing a smaller '
                             '--epochs instead would compress the whole LR curve.')
    parser.add_argument('--resume', default=False, type=str2bool,
                        help='resume from models/<name>/last.pth if it exists')
    parser.add_argument('--init_from', default=None, type=str, metavar='PATH',
                        help='graft: load these weights (strict=False) before training, '
                             'e.g. models/busi_split43_aug/model.pth. Layers absent from '
                             'the checkpoint keep their fresh init.')
    parser.add_argument('--freeze_except', default=None, type=str, metavar='PREFIXES',
                        help='comma-separated parameter-name prefixes to keep trainable; '
                             'everything else is frozen. e.g. "se,dse" trains only the '
                             'SE blocks. Requires --init_from to be meaningful.')
    parser.add_argument('--skip_identity_init', default=False, type=str2bool,
                        help='initialise SkipFusion gates at 0.5 so the fusion reduces to '
                             'the baseline `out + t` at insertion (for grafting)')
    parser.add_argument('--se_identity_init', default=False, type=str2bool,
                        help='initialise SE gates at ~1.0 so the block is an identity map '
                             'at insertion (required when grafting onto a trained net; '
                             'default init would halve every activation)')
    parser.add_argument('--aug', default='paper', choices=['paper', 'strong'],
                        help="'paper' = rotate90 + flips only (the paper's setup); "
                             "'strong' adds shift/scale/rotate, brightness/contrast, "
                             "gamma and elastic deformation to fight overfitting")

    return parser.parse_args()


def build_train_transform(config):
    """Training augmentation pipeline.

    'paper' reproduces the UNeXt paper exactly: RandomRotate90 + flips, nothing else.

    'strong' targets the overfitting measured on BUSI (train IoU 0.91 vs val 0.58 at
    epoch 400, best val at epoch 80). With only 518 training images, geometric and
    photometric variation is the cheapest way to enlarge the effective dataset.
    Elastic deformation is standard in medical segmentation (U-Net used it precisely
    because annotated medical data is scarce) and is well suited to ultrasound, where
    tissue genuinely deforms between scans.
    """
    resize_and_norm = [A.Resize(config['input_h'], config['input_w']), A.Normalize()]

    if config['aug'] == 'paper':
        return A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ] + resize_and_norm)

    return A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.8, 1.2),
                 rotate=(-30, 30), border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, border_mode=0, p=0.3),
    ] + resize_and_norm)


def _freeze_bn_stats(model):
    """Put BatchNorm layers whose params are frozen into eval mode.

    `requires_grad = False` stops gradient updates but NOT the running-mean/var buffers,
    which keep tracking batch statistics in train() mode. When fine-tuning a grafted
    model with a frozen backbone that silently changes the "frozen" network, so the
    measured effect would not be attributable to the new layers alone.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()
                n += 1
    return n


def train_one_epoch(config, train_loader, model, criterion, optimizer, device, scaler):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    if config.get('freeze_except'):
        _freeze_bn_stats(model)

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            output = model(input)
            loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        iou, _ = iou_score(output.detach().float(), target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        pbar.set_postfix(OrderedDict([('loss', avg_meters['loss'].avg),
                                      ('iou', avg_meters['iou'].avg)]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(),
                  'iou_pi': AverageMeter(), 'dice_pi': AverageMeter()}
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            iou, dice = iou_score(output, target)
            iou_pi, dice_pi = iou_dice_per_image(output, target)

            n = input.size(0)
            avg_meters['loss'].update(loss.item(), n)
            avg_meters['iou'].update(iou, n)
            avg_meters['dice'].update(dice, n)
            avg_meters['iou_pi'].update(iou_pi, n)
            avg_meters['dice_pi'].update(dice_pi, n)

            pbar.set_postfix(OrderedDict([('loss', avg_meters['loss'].avg),
                                          ('iou', avg_meters['iou'].avg),
                                          ('dice', avg_meters['dice'].avg)]))
            pbar.update(1)
        pbar.close()

    return OrderedDict([(k, m.avg) for k, m in avg_meters.items()])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = '%s_%s' % (config['dataset'], config['arch'])
    if config['split_seed'] is None:
        config['split_seed'] = config['seed']

    seed_everything(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.join('models', config['name'])
    os.makedirs(out_dir, exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('device: %s' % device)
    print('-' * 20)

    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    cudnn.benchmark = True

    model_kwargs = {}
    if config['arch'] == 'UNext_SE':
        model_kwargs['se_reduction'] = config['se_reduction']
        model_kwargs['se_identity_init'] = config['se_identity_init']
    elif config['arch'] == 'UNext_SkipFusion':
        model_kwargs['skip_identity_init'] = config['skip_identity_init']
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'],
                                           **model_kwargs)

    # ---- graft: load a pretrained backbone, leaving new layers at their fresh init ----
    if config['init_from']:
        src = torch.load(config['init_from'], map_location='cpu', weights_only=False)
        if isinstance(src, dict) and 'model' in src:
            src = src['model']
        r = model.load_state_dict(src, strict=False)
        print(f"=> grafted from {config['init_from']}: "
              f"{len(src)} tensors loaded, {len(r.missing_keys)} new, "
              f"{len(r.unexpected_keys)} unused")
        if r.unexpected_keys:
            print(f"   WARNING unused keys: {r.unexpected_keys[:5]}")

    # ---- freeze everything except the named prefixes ----
    if config['freeze_except']:
        keep = tuple(p.strip() for p in config['freeze_except'].split(',') if p.strip())
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(keep)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        trainable_mods = sorted({n.split('.')[0] for n, p in model.named_parameters()
                                 if p.requires_grad})
        print(f'=> frozen all but {keep}: {n_train:,}/{n_all:,} trainable '
              f'({100 * n_train / n_all:.2f}%) in modules {trainable_mods}')
        if n_train == 0:
            raise SystemExit(f'--freeze_except {keep!r} matched no parameters')

    model = model.to(device)

    config['n_params'] = count_params(model)
    print('trainable params: %s' % f"{config['n_params']:,}")

    with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'],
                              weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config['factor'], patience=config['patience'],
            min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
            gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # ---- data ----
    img_dir = os.path.join('inputs', config['dataset'], 'images')
    mask_dir = os.path.join('inputs', config['dataset'], 'masks')
    img_ids = glob(os.path.join(img_dir, '*' + config['img_ext']))
    img_ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in img_ids)
    if not img_ids:
        raise SystemExit(
            f"No images found in {img_dir} matching *{config['img_ext']}.\n"
            f"Run the preprocessing script first (see README.md)."
        )
    print('found %d images' % len(img_ids))

    train_img_ids, val_img_ids = train_test_split(
        img_ids, test_size=0.2, random_state=config['split_seed'])

    train_transform = build_train_transform(config)
    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    common = dict(img_dir=img_dir, mask_dir=mask_dir, img_ext=config['img_ext'],
                  mask_ext=config['mask_ext'], num_classes=config['num_classes'])
    train_dataset = Dataset(img_ids=train_img_ids, transform=train_transform, **common)
    val_dataset = Dataset(img_ids=val_img_ids, transform=val_transform, **common)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], drop_last=True,
        pin_memory=(device.type == 'cuda'))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], drop_last=False,
        pin_memory=(device.type == 'cuda'))

    scaler = torch.amp.GradScaler(device.type) if (config['amp'] and device.type == 'cuda') else None

    log = OrderedDict([(k, []) for k in
                       ['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou',
                        'val_dice', 'val_iou_per_image', 'val_dice_per_image']])

    best_iou = 0
    best_epoch = -1
    trigger = 0
    start_epoch = 0
    t_start = time.time()

    # ---- resume ----
    ckpt_path = os.path.join(out_dir, 'last.pth')
    if config['resume'] and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        if ck.get('scheduler') is not None and scheduler is not None:
            scheduler.load_state_dict(ck['scheduler'])
        if ck.get('scaler') is not None and scaler is not None:
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        best_iou = float(ck['best_iou'])
        best_epoch = ck['best_epoch']
        # keep the epochs already written to log.csv so curves stay continuous
        prev_log = os.path.join(out_dir, 'log.csv')
        if os.path.exists(prev_log):
            done = pd.read_csv(prev_log)
            done = done[done['epoch'] < start_epoch]
            for k in log:
                log[k] = done[k].tolist()
        print(f'=> resumed from epoch {ck["epoch"]} '
              f'(best {best_iou:.4f} @ep{best_epoch}), continuing at {start_epoch}')

    # --stop_after halts early WITHOUT touching the scheduler: T_max is still
    # config['epochs'], so the LR curve is identical to an uninterrupted full run
    # and `--resume` picks the trajectory back up exactly where it left off.
    last_epoch = config['epochs']
    if config['stop_after'] is not None:
        last_epoch = min(config['epochs'], start_epoch + config['stop_after'])
        print(f'=> will stop after epoch {last_epoch - 1} '
              f'(cosine T_max remains {config["epochs"]})')

    for epoch in range(start_epoch, last_epoch):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        current_lr = optimizer.param_groups[0]['lr']

        train_log = train_one_epoch(config, train_loader, model, criterion,
                                    optimizer, device, scaler)
        val_log = validate(config, val_loader, model, criterion, device)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        elif config['scheduler'] == 'MultiStepLR':
            scheduler.step()

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'],
                 val_log['iou'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(current_lr)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_iou_per_image'].append(val_log['iou_pi'])
        log['val_dice_per_image'].append(val_log['dice_pi'])
        pd.DataFrame(log).to_csv(os.path.join(out_dir, 'log.csv'), index=False)

        trigger += 1
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
            best_iou = val_log['iou']
            best_epoch = epoch
            print("=> saved best model")
            trigger = 0

        # Always save the latest full state as well. `model.pth` holds the BEST epoch's
        # weights only, so it cannot be resumed from -- the first 400-epoch runs ended
        # with no optimizer/scheduler state on disk and could not be continued.
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'best_iou': best_iou,
                    'best_epoch': best_epoch},
                   os.path.join(out_dir, 'last.pth'))

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    completed = len(log['epoch'])
    summary = {'best_iou': float(best_iou), 'best_epoch': best_epoch,
               'epochs_run': completed, 'train_seconds': round(elapsed, 1),
               'n_params': config['n_params'],
               'target_epochs': config['epochs'],
               'partial': completed < config['epochs'],
               'resumable_from': completed - 1}
    with open(os.path.join(out_dir, 'summary.yml'), 'w') as f:
        yaml.dump(summary, f)
    print('\nbest val IoU %.4f @ epoch %d | %.1f min total'
          % (best_iou, best_epoch, elapsed / 60))


if __name__ == '__main__':
    main()
