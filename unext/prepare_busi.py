"""Convert the raw BUSI dataset into the layout the loader expects.

Raw BUSI (Dataset_BUSI_with_GT) ships as:

    <raw>/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png
    │   ├── benign (1)_mask_1.png     <- some cases have multiple mask files
    │   └── ...
    ├── malignant/
    └── normal/                       <- all-zero masks, no lesion

Produces:

    inputs/busi/
    ├── images/<class>_<n>.png
    └── masks/0/<class>_<n>.png

Notes
  * Multiple `_mask_k` files for one image are merged with a pixel-wise OR, which
    is the standard handling -- they are separate lesions in the same scan.
  * The `normal` class is EXCLUDED by default. Those cases have empty masks, and
    including them changes what the benchmark measures; the UNeXt paper reports
    647 images, which is the benign (437) + malignant (210) count. Pass
    --include_normal to add the 133 normal cases (780 total) if you want that
    ablation, but say so explicitly in the report.
  * Masks are binarised at >127 and written as 0/255.
"""

import argparse
import os
import re
from glob import glob

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True,
                   help='path to Dataset_BUSI_with_GT (the folder containing '
                        'benign/ malignant/ normal/)')
    p.add_argument('--out', default=os.path.join('inputs', 'busi'))
    p.add_argument('--include_normal', action='store_true',
                   help='include the 133 normal (empty-mask) cases')
    return p.parse_args()


def main():
    args = parse_args()

    classes = ['benign', 'malignant']
    if args.include_normal:
        classes.append('normal')

    img_out = os.path.join(args.out, 'images')
    mask_out = os.path.join(args.out, 'masks', '0')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    total = 0
    for cls in classes:
        cls_dir = os.path.join(args.raw, cls)
        if not os.path.isdir(cls_dir):
            raise SystemExit(f'missing class folder: {cls_dir}')

        # images are the files WITHOUT "_mask" in the name
        img_paths = sorted(p for p in glob(os.path.join(cls_dir, '*.png'))
                           if '_mask' not in os.path.basename(p))

        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            # "benign (12)" -> 12
            m = re.search(r'\((\d+)\)', stem)
            idx = m.group(1) if m else stem.replace(' ', '_')
            out_id = f'{cls}_{idx}'

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f'  skip unreadable image: {img_path}')
                continue

            # collect "<stem>_mask.png" and any "<stem>_mask_1.png", "_mask_2.png"...
            mask_paths = sorted(glob(os.path.join(cls_dir, f'{stem}_mask*.png')))
            if not mask_paths:
                print(f'  skip (no mask found): {img_path}')
                continue

            merged = None
            for mp in mask_paths:
                m_img = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m_img is None:
                    continue
                if merged is None:
                    merged = np.zeros_like(m_img)
                if m_img.shape != merged.shape:
                    m_img = cv2.resize(m_img, (merged.shape[1], merged.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                merged = np.maximum(merged, m_img)

            if merged is None:
                print(f'  skip (unreadable masks): {img_path}')
                continue

            merged = ((merged > 127) * 255).astype('uint8')

            if merged.shape[:2] != img.shape[:2]:
                merged = cv2.resize(merged, (img.shape[1], img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(os.path.join(img_out, out_id + '.png'), img)
            cv2.imwrite(os.path.join(mask_out, out_id + '.png'), merged)
            total += 1

        print(f'{cls}: done')

    print(f'\nwrote {total} image/mask pairs to {args.out}')
    print(f'  images: {img_out}')
    print(f'  masks:  {mask_out}')


if __name__ == '__main__':
    main()
