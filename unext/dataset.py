"""Dataset loader.

Expected layout (same as upstream):

    inputs/<dataset_name>/
    ├── images/
    │   ├── 0a7e06.png
    │   └── ...
    └── masks/
        └── 0/                 <- one subdir per class; binary tasks use only "0"
            ├── 0a7e06.png
            └── ...

Change from upstream: `cv2.imread` returning None (missing/corrupt file) previously
produced an opaque `TypeError: 'NoneType' object is not subscriptable` deep in the
loader. It now raises a FileNotFoundError naming the offending path.
"""

import os

import cv2
import numpy as np
import torch.utils.data


def _imread(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes,
                 transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = _imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            m = _imread(os.path.join(self.mask_dir, str(i), img_id + self.mask_ext),
                        cv2.IMREAD_GRAYSCALE)
            mask.append(m[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            # A.Normalize already scaled the image by mean/std -> roughly [-2.5, 2.5].
            # Upstream divided by 255 again here, squashing inputs into ~[-0.01, 0.01]
            # and throwing away nearly all the signal. Only un-normalized images
            # (no transform) need the /255.
            img = img.astype('float32')
        else:
            img = img.astype('float32') / 255

        img = img.transpose(2, 0, 1)
        # Masks pass through A.Normalize untouched, so they are still 0/255 here.
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
