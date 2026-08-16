"""Convert ISIC 2018 Task 1 (lesion boundary segmentation) into the loader layout.

Raw ISIC 2018 Task 1 ships as two folders:

    ISIC2018_Task1-2_Training_Input/     ISIC_0000000.jpg ...
    ISIC2018_Task1_Training_GroundTruth/ ISIC_0000000_segmentation.png ...

Produces:

    inputs/isic/
    ├── images/ISIC_0000000.png
    └── masks/0/ISIC_0000000.png

Both are resized on disk (default 512x512) because the raw ISIC images vary in
size and some are very large (up to ~6700x4400); decoding those every epoch makes
the dataloader, not the GPU, the bottleneck. Training resizes again to --input_h/w,
so keep this >= the training resolution.
"""

import argparse
import os
from glob import glob

import cv2
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True,
                   help='path to ISIC2018_Task1-2_Training_Input')
    p.add_argument('--masks', required=True,
                   help='path to ISIC2018_Task1_Training_GroundTruth')
    p.add_argument('--out', default=os.path.join('inputs', 'isic'))
    p.add_argument('--size', default=512, type=int,
                   help='resize longest edge to this square size on disk')
    return p.parse_args()


def main():
    args = parse_args()

    img_out = os.path.join(args.out, 'images')
    mask_out = os.path.join(args.out, 'masks', '0')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    img_paths = sorted(glob(os.path.join(args.images, '*.jpg')))
    if not img_paths:
        raise SystemExit(f'no .jpg files found in {args.images}')

    n = 0
    missing = 0
    for img_path in tqdm(img_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(args.masks, f'{stem}_segmentation.png')
        if not os.path.exists(mask_path):
            missing += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            missing += 1
            continue

        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        mask = ((mask > 127) * 255).astype('uint8')

        cv2.imwrite(os.path.join(img_out, stem + '.png'), img)
        cv2.imwrite(os.path.join(mask_out, stem + '.png'), mask)
        n += 1

    print(f'\nwrote {n} pairs to {args.out}' +
          (f' ({missing} skipped for missing/unreadable masks)' if missing else ''))


if __name__ == '__main__':
    main()
