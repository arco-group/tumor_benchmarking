import os
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiDataset(Dataset):
    def __init__(
        self,
        data_dir,
        tumor,
        mode="Training",
        img_size=256,
        num_classes=None,
        transform=None,
        mask_transform=None,
        augment=False,
        cache_size=0,
        test_masks_dir=None,
    ):
        tumor_path_labels = {
            "lung": "Dataset002_Lung1",
            "breast": "Dataset008_ISPY1",
            "liver": "Dataset009_Liver2",
            "kidney": "Dataset012_Kidney",
            "brain": "Dataset004_BraTS",
        }

        self.mode = mode
        self.tumor = tumor
        self.task = tumor_path_labels[tumor]
        self.dataset = self.task.split("_")[-1]
        self.data_path = os.path.join(data_dir, self.task)
        if mode == "Training":
            self.img_path = os.path.join(self.data_path, "imagesTr")
            self.mask_path = os.path.join(self.data_path, "labelsTr")
            samples_path = os.path.join(self.data_path, f"{tumor}_train.txt")
        else:
            if test_masks_dir is None:
                raise ValueError("test_masks_dir is required when mode is not 'Training'.")
            self.img_path = os.path.join(self.data_path, "imagesTs")
            self.mask_path = os.path.join(test_masks_dir, f"{self.dataset}_labelsTs")
            samples_path = os.path.join(self.data_path, f"{tumor}_test.txt")
        self.img_size = int(img_size)
        self.num_classes = int(num_classes if num_classes is not None else (4 if tumor == "brain" else 2))
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.cache_size = max(0, int(cache_size))
        self._image_cache = OrderedDict()
        self._mask_cache = OrderedDict()

        with open(samples_path, "r") as f:
            self.samples_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples_list)

    def _load_volume(self, path, cache):
        if self.cache_size > 0 and path in cache:
            cache.move_to_end(path)
            return cache[path]

        volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
        if self.cache_size > 0:
            cache[path] = volume
            cache.move_to_end(path)
            while len(cache) > self.cache_size:
                cache.popitem(last=False)
        return volume

    @staticmethod
    def _extract_slice(volume, slice_idx):
        if volume.shape[0] == volume.shape[1]:
            return volume[:, :, slice_idx]
        if volume.shape[1] == volume.shape[2]:
            return volume[slice_idx, :, :]
        axis = int(np.argmin(volume.shape))
        return np.take(volume, slice_idx, axis=axis)

    @staticmethod
    def _normalize(image):
        image = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(image)
        if not finite.any():
            return np.zeros_like(image, dtype=np.float32)

        image = np.where(finite, image, 0.0)
        values = image[finite]
        lo, hi = np.percentile(values, [1.0, 99.0])
        image = np.clip(image, lo, hi)
        mean = float(image[finite].mean())
        std = float(image[finite].std())
        if std < 1e-6:
            std = 1.0
        return (image - mean) / std

    def _apply_augment(self, image, mask):
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1).item() < 0.2:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                image = torch.rot90(image, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])
        return image, mask

    def __getitem__(self, idx):
        sample = self.samples_list[idx]
        volume_name, slice_idx = sample.split("/")
        slice_idx = int(slice_idx)

        img_path = os.path.join(self.img_path, volume_name)
        mask_path = os.path.join(self.mask_path, volume_name.split("_0000.nii")[0] + ".nii.gz")

        image_volume = self._load_volume(img_path, self._image_cache)
        mask_volume = self._load_volume(mask_path, self._mask_cache)

        image = self._extract_slice(image_volume, slice_idx)
        mask = self._extract_slice(mask_volume, slice_idx)

        image = torch.from_numpy(self._normalize(image).copy()).float().unsqueeze(0)
        if self.tumor == "lung":
            mask = (mask == 3).astype(np.float32)
        elif self.num_classes > 2:
            mask = np.clip(np.rint(mask), 0, self.num_classes - 1).astype(np.float32)
        else:
            mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if image.shape[-2:] != (self.img_size, self.img_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="nearest",
            ).squeeze(0)

        if self.augment and self.mode == "Training":
            image, mask = self._apply_augment(image, mask)

        return {
            "image": image.contiguous(),
            "mask": mask.contiguous(),
            "sample_id": sample,
            "volume_id": volume_name,
            "slice_idx": slice_idx,
        }
