from torch.utils.data import Dataset
import torch
import os
import nibabel as nib
import torch.nn.functional as F


class MultiDataset(Dataset):
    def __init__(self, data_dir, tumor, mode="Training", transform=None, mask_transform=None, num_classes=2, test_masks_dir=None):
        """ Lung dataset class

        Args:
            data_root (str): The root directory of the dataset.
            semantic (bool): Whether to use semantic segmentation masks.
            transform (torchvision.transforms.Compose): The transforms to apply to the data.
        """

        tumor_path_labels = {'lung': 'Dataset002_Lung1',
                        'breast': 'Dataset008_ISPY1',
                        'liver' : 'Dataset009_Liver2',
                        'kidney' : 'Dataset012_Kidney',
                        'brain': 'Dataset004_BraTS'          
        }  # Set here the names of your own datasets

        self.mode = mode
        self.tumor = tumor
        self.task = tumor_path_labels[tumor]
        self.dataset = self.task.split('_')[-1]
        self.data_path = data_dir + '/' + self.task
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

        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.samples_list[idx].split("/")[0])
        mask_path = os.path.join(self.mask_path, self.samples_list[idx].split("/")[0].split("_0000.nii")[0] + ".nii.gz")

        image = nib.load(img_path).get_fdata()
        if image.shape[0] == image.shape[1]:
            image = image[:, :, int(self.samples_list[idx].split("/")[-1])]

        elif image.shape[1] == image.shape[2]:
            image = image[int(self.samples_list[idx].split("/")[-1]), :, :]

        image = torch.from_numpy(image.copy()).float().unsqueeze(0)
        
        mask = nib.load(mask_path).get_fdata()

        if mask.shape[0] == mask.shape[1]:
            mask = mask[:, :, int(self.samples_list[idx].split("/")[-1])]

        elif mask.shape[1] == mask.shape[2]:
            mask = mask[int(self.samples_list[idx].split("/")[-1]), :, :]

        mask = torch.from_numpy(mask.copy()).float()

        if self.tumor == 'lung':
            mask[mask!=3] = 0
            mask[mask==3] = 1
            
        if self.transform:
            image = self.transform(image)
                
        target_processed = F.one_hot(mask.long(), num_classes=self.num_classes).permute(2, 0, 1).float()

        if self.mask_transform:
            target_processed = self.mask_transform(target_processed)

        if self.mode != "Training":
            return image, target_processed.long(), self.samples_list[idx]
        else:
            return image, target_processed.long()

