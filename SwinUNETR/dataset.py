import os
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiDataset(Dataset):
    def __init__(self, data_dir, tumor, img_size=256, mode="Training", transform=None, mask_transform=None, augment=False, test_masks_dir=None):
        
        tumor_path_labels = {
            "lung": "Dataset002_Lung1",
            "breast": "Dataset008_ISPY1",
            "liver": "Dataset009_Liver2",
            "kidney": "Dataset012_Kidney",
            "brain": "Dataset004_BraTS",
        }

        self.tumor = tumor
        self.task = tumor_path_labels[tumor]
        self.data_path = os.path.join(data_dir, self.task)
        self.img_path = os.path.join(self.data_path, "imagesTr")
        self.mask_path = os.path.join(self.data_path, "labelsTr")
        self.img_size = int(img_size)
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.mode = mode
        self.test_masks_dir = test_masks_dir

        if not self.mode == 'Training':
            self.mask_path = os.path.join(f'{self.test_masks_dir}/{self.dataset}_labelsTs')  
            self.img_path = os.path.join(self.data_path, "imagesTs") 
            samples_path = os.path.join(self.data_path, f'{tumor}_test.txt')
        else:
            self.mask_path = os.path.join(self.data_path, "labelsTr")  
            self.img_path = os.path.join(self.data_path, "imagesTr") 
            samples_path = os.path.join(self.data_path, f'{tumor}_train.txt')

        with open(samples_path, 'r') as f:
            self.samples_list = [x[:-1] for x in f]


    def __len__(self):
        return len(self.samples_list)

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

        image_volume = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
        mask_volume = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)

        if image_volume.shape[0] == image_volume.shape[1]:
            image_volume[:, :, slice_idx]
        if mask_volume.shape[1] == mask_volume.shape[2]:
            mask_volume[slice_idx, :, :]

        image = torch.from_numpy(self._normalize(image).copy()).float().unsqueeze(0)
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

        if self.augment:
            image, mask = self._apply_augment(image, mask)

        return {
            "image": image.contiguous(),
            "mask": mask.contiguous(),
            "sample_id": sample,
            "volume_id": volume_name,
            "slice_idx": slice_idx,
        }
