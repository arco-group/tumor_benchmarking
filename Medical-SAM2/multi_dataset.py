import os
import numpy as np
import torch
import yaml
from PIL import Image
import nibabel as nib
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox


class MultiDataset(Dataset):
    def __init__(self, args, data_dir, transform=None, transform_msk=None, mode='Training', prompt='bbox', seed=None,
                 variation=0):
        
        tumor_path_labels = {'lung': 'Dataset002_Lung1',
                             'breast': 'Dataset008_ISPY1',
                             'liver' : 'Dataset009_Liver2',
                             'kidney' : 'Dataset012_Kidney',
                             'brain': 'Dataset004_BraTS'          
        }  # Set here the names of your own datasets

        self.task = tumor_path_labels[args.tumor]
        self.tumor = args.tumor
        self.dataset = self.task.split('_')[-1]
        self.data_path = data_dir + '/' + self.task

        # Set the data list for training and test
        self.mode = mode
        if self.mode == 'Training':
            self.name_list = sorted(os.listdir(os.path.join(self.data_path, 'imagesTr')))
        else:
            self.name_list = sorted(os.listdir(os.path.join(self.data_path, 'imagesTs')))

        # Set the basic information of the dataset
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        if self.mode == 'Training':
            img_path = os.path.join(self.data_path, 'imagesTr', name)
            mask_path = os.path.join(self.data_path, 'labelsTr', name.replace('_0000.nii', '.nii'))
        else:
            img_path = os.path.join(self.data_path, 'imagesTs', name)
            mask_path = os.path.join('/nnUNetv2/Data/LabelsTs', f'{self.dataset}_labelsTs', name.replace('_0000.nii', '.nii'))
        
        try:
            data_seg_3d = (nib.load(mask_path)).get_fdata()
            if self.tumor == 'lung':
                data_seg_3d = (data_seg_3d == 3).astype(np.uint8)
        except Exception:
            print(f"Skipping {name} because mask path {mask_path} does not exist.")

        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i
        for j in reversed(range(data_seg_3d.shape[-1])):
                if np.sum(data_seg_3d[..., j]) > 0:
                    data_seg_3d = data_seg_3d[..., :j + 1]
                    break
        num_frame = data_seg_3d.shape[-1]

        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length

        if num_frame > video_length and self.mode == 'Training':
            frames = [a for a in range(0, data_seg_3d.shape[2]) if np.any(data_seg_3d[:, :, a] > 0)]

            if self.tumor == 'brain':
                valid_starts = []
                max_start = data_seg_3d.shape[2] - video_length

                for s in range(max_start + 1):
                    window = data_seg_3d[:, :, s:s + video_length]
                    found_all_labels = False

                    for k in range(window.shape[2]):
                        labels_in_slice = np.unique(window[:, :, k])
                        if 1 in labels_in_slice and 2 in labels_in_slice and 3 in labels_in_slice:
                            found_all_labels = True
                            break

                    if found_all_labels:
                        valid_starts.append(s)

                if len(valid_starts) > 0:
                    starting_frame = np.random.choice(valid_starts)
                else:
                    try:
                        starting_frame = np.random.randint(frames[0], frames[-1] - video_length + 1)
                    except ValueError:
                        starting_frame = frames[0] if len(frames) > 0 else 0

            else:
                try:
                    starting_frame = np.random.randint(frames[0], frames[-1] - video_length + 1)
                except ValueError:
                    starting_frame = frames[0] if len(frames) > 0 else 0

        else:
            frames = [a for a in range(0, data_seg_3d.shape[2]) if np.any(data_seg_3d[:, :, a] > 0)]

            if self.tumor == 'brain':
                valid_starts = []
                max_start = max(0, data_seg_3d.shape[2] - video_length)

                for s in range(max_start + 1):
                    window = data_seg_3d[:, :, s:s + video_length]
                    found_all_labels = False

                    for k in range(window.shape[2]):
                        labels_in_slice = np.unique(window[:, :, k])
                        if 1 in labels_in_slice and 2 in labels_in_slice and 3 in labels_in_slice:
                            found_all_labels = True
                            break

                    if found_all_labels:
                        valid_starts.append(s)

                if len(valid_starts) > 0:
                    starting_frame = valid_starts[0]
                else:
                    starting_frame = frames[0] if len(frames) > 0 else 0

            else:
                starting_frame = frames[0] if len(frames) > 0 else 0

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        
        data_img_3d = (nib.load(img_path)).get_fdata()
        data_img_3d = data_img_3d[..., starting_frame_nonzero:starting_frame_nonzero + num_frame]

        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            try:
                data_slice = data_img_3d[..., frame_index]
                data_slice = np.repeat(data_slice[:, :, np.newaxis], 3, axis=2) 
                data_slice = np.nan_to_num(data_slice, nan=0.0)
                vmin, vmax = data_slice.min(), data_slice.max()
                if vmax == vmin:
                    data_slice = np.zeros_like(data_slice)
                else:
                    data_slice = (255 * (data_slice - vmin) / (vmax - vmin)).astype(np.uint8)
                img = Image.fromarray(data_slice, 'RGB')
            except IndexError:
                continue
            try:
                mask = data_seg_3d[..., frame_index]
            except IndexError:
                continue
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)

            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'prompted_range': (starting_frame_nonzero + starting_frame, starting_frame_nonzero + starting_frame + video_length -1),
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'prompted_range': (starting_frame_nonzero + starting_frame, starting_frame_nonzero + starting_frame + video_length -1),
            }
