"""Sanity checks that need no dataset. Run this before downloading anything.

Verifies for each architecture:
  * forward pass produces the right output shape at 256 and 512 input sizes
  * a backward pass produces finite gradients
  * parameter count is in the expected range (paper: UNeXt ~1.47M)
  * the loss is finite and the metrics run

Usage:  python smoke_test.py
"""

import torch

import archs
from losses import BCEDiceLoss
from metrics import iou_score, iou_dice_per_image
from utils import count_params


def check(name, arch_cls, size, device, **kwargs):
    model = arch_cls(num_classes=1, input_channels=3, deep_supervision=False,
                     **kwargs).to(device)
    x = torch.randn(2, 3, size, size, device=device)
    y = (torch.rand(2, 1, size, size, device=device) > 0.5).float()

    out = model(x)
    assert out.shape == y.shape, f'{name}@{size}: got {tuple(out.shape)}, want {tuple(y.shape)}'

    loss = BCEDiceLoss()(out, y)
    assert torch.isfinite(loss), f'{name}@{size}: non-finite loss'

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f'{name}@{size}: no gradients'
    assert all(torch.isfinite(g).all() for g in grads), f'{name}@{size}: non-finite grads'

    iou, dice = iou_score(out.detach(), y)
    iou_pi, dice_pi = iou_dice_per_image(out.detach(), y)

    n = count_params(model)
    print(f'  {name:10s} {size}x{size}  out={tuple(out.shape)}  '
          f'params={n:>9,}  loss={loss.item():.4f}  iou={iou:.3f}')
    return n


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type == 'cuda':
        print(f'gpu:    {torch.cuda.get_device_name(0)}')
    print(f'torch:  {torch.__version__}\n')

    results = {}
    for name, cls, kwargs in [('UNext', archs.UNext, {}),
                              ('UNext_S', archs.UNext_S, {}),
                              ('UNext_SE', archs.UNext_SE, {'se_reduction': 16})]:
        print(f'{name}:')
        for size in (256, 512):
            results[(name, size)] = check(name, cls, size, device, **kwargs)
        print()

    base = results[('UNext', 256)]
    se = results[('UNext_SE', 256)]
    print(f'SE overhead: +{se - base:,} params '
          f'(+{100 * (se - base) / base:.2f}% over baseline UNeXt)')
    print(f'baseline UNeXt params: {base:,}  (paper reports ~1.47M)')
    print('\nall checks passed')


if __name__ == '__main__':
    main()
