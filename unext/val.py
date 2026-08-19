"""Evaluation: reload a trained run, score the held-out split, dump predicted masks.

Changes from upstream:
  * Same split seed as training is read back from config.yml, so the val set here
    is exactly the one held out during that run (upstream hardcoded 41 and would
    silently evaluate on training images if you had trained with another seed).
  * Reports both the upstream batch-aggregate metric and the per-image mean.
  * Measures GPU and CPU per-image inference latency (the paper's table reports
    CPU inference time), plus parameter count and GFLOPs when ptflops is installed.
  * `--save_images false` skips writing PNGs when you only want the numbers.
"""

import argparse
import os
import time
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score, iou_dice_per_image
from utils import AverageMeter, count_params, seed_everything, str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='run name under models/')
    parser.add_argument('--save_images', default=True, type=str2bool)
    parser.add_argument('--bench_cpu', default=True, type=str2bool,
                        help='measure single-image CPU latency (paper reports this)')
    parser.add_argument('--bench_runs', default=50, type=int)
    return parser.parse_args()


def build_model(config, device):
    model_kwargs = {}
    if config['arch'] == 'UNext_SE':
        model_kwargs['se_reduction'] = config.get('se_reduction', 16)
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'],
                                           **model_kwargs)
    ckpt = os.path.join('models', config['name'], 'model.pth')
    if not os.path.exists(ckpt):
        # Checkpoints are not committed (see .gitignore) -- only the training
        # logs are. Say so plainly instead of raising a bare FileNotFoundError.
        raise SystemExit(
            f"no checkpoint at {ckpt}\n"
            "Model weights are not distributed with this repository; "
            "train the run first, e.g.\n"
            f"  python train.py --dataset busi --arch {config['arch']} "
            f"--name {config['name']}")
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def benchmark_latency(model, device, h, w, runs, warmup=10):
    """Mean single-image forward latency in ms."""
    x = torch.randn(1, 3, h, w, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def main():
    args = parse_args()

    with open(os.path.join('models', args.name, 'config.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    seed_everything(config.get('seed', 41))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    model = build_model(config, device)

    img_dir = os.path.join('inputs', config['dataset'], 'images')
    mask_dir = os.path.join('inputs', config['dataset'], 'masks')
    img_ids = glob(os.path.join(img_dir, '*' + config['img_ext']))
    img_ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in img_ids)

    split_seed = config.get('split_seed', config.get('seed', 41))
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=split_seed)
    print('evaluating on %d held-out images' % len(val_img_ids))

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])
    val_dataset = Dataset(
        img_ids=val_img_ids, img_dir=img_dir, mask_dir=mask_dir,
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config.get('num_workers', 0), drop_last=False)

    meters = {k: AverageMeter() for k in ['iou', 'dice', 'iou_pi', 'dice_pi']}

    if args.save_images:
        for c in range(config['num_classes']):
            os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            iou, dice = iou_score(output, target)
            iou_pi, dice_pi = iou_dice_per_image(output, target)
            n = input.size(0)
            meters['iou'].update(iou, n)
            meters['dice'].update(dice, n)
            meters['iou_pi'].update(iou_pi, n)
            meters['dice_pi'].update(dice_pi, n)

            if args.save_images:
                probs = torch.sigmoid(output).cpu().numpy()
                probs = (probs >= 0.5).astype('uint8')
                for i in range(len(probs)):
                    for c in range(config['num_classes']):
                        cv2.imwrite(
                            os.path.join('outputs', config['name'], str(c),
                                         meta['img_id'][i] + '.png'),
                            (probs[i, c] * 255).astype('uint8'))

    results = {
        'iou_batch_aggregate': round(float(meters['iou'].avg), 4),
        'dice_from_iou': round(float(meters['dice'].avg), 4),
        'iou_per_image': round(float(meters['iou_pi'].avg), 4),
        'dice_per_image': round(float(meters['dice_pi'].avg), 4),
        'n_params': count_params(model),
    }

    h, w = config['input_h'], config['input_w']
    if device.type == 'cuda':
        results['gpu_ms_per_image'] = round(
            benchmark_latency(model, device, h, w, args.bench_runs), 2)
    if args.bench_cpu:
        cpu_model = build_model(config, torch.device('cpu'))
        results['cpu_ms_per_image'] = round(
            benchmark_latency(cpu_model, torch.device('cpu'), h, w,
                              max(5, args.bench_runs // 5)), 2)

    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            build_model(config, torch.device('cpu')), (3, h, w),
            as_strings=False, print_per_layer_stat=False, verbose=False)
        # ptflops reports MACs; the paper's "GFLOPs" convention counts 1 MAC = 1 FLOP
        results['gflops'] = round(macs / 1e9, 3)
    except ImportError:
        print('(ptflops not installed - skipping GFLOPs; pip install ptflops)')

    print('\n' + '=' * 40)
    for k, v in results.items():
        print(f'{k:24s} {v}')
    print('=' * 40)

    with open(os.path.join('models', config['name'], 'eval.yml'), 'w') as f:
        yaml.dump(results, f)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
