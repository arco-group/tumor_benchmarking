import numpy as np
import os

join = os.path.join
import torch
from torch.utils.data import Dataset
import random
from skimage.transform import resize
import nibabel as nib


def resize_bbox(bbox, scale, image_shape):
    """
    bbox: tensor/list/array [x_min, y_min, x_max, y_max]
    scale: float. 1.05 = +5%, 0.95 = -5%
    image_shape: (H, W)

    return: rescaled bbox and cropped to the image, as torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
    """
    H, W = image_shape

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().float().tolist()

    x_min, y_min, x_max, y_max = bbox

    # centro
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # ancho y alto originales
    bw = x_max - x_min
    bh = y_max - y_min

    # nuevo ancho y alto
    new_bw = bw * scale
    new_bh = bh * scale

    # reconstrucción
    new_x_min = cx - new_bw / 2.0
    new_x_max = cx + new_bw / 2.0
    new_y_min = cy - new_bh / 2.0
    new_y_max = cy + new_bh / 2.0

    # clamp a la imagen
    new_x_min = max(0, min(new_x_min, W - 1))
    new_x_max = max(0, min(new_x_max, W - 1))
    new_y_min = max(0, min(new_y_min, H - 1))
    new_y_max = max(0, min(new_y_max, H - 1))

    return torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max], dtype=torch.float32)


def shift_bbox(bbox, shift_percent, image_shape, direction=None, return_direction=False):
    """
    Desplaza una bbox sin cambiar su tamaño.

    Args:
        bbox: tensor/list/array [x_min, y_min, x_max, y_max]
        shift_percent: float, por ejemplo 0.05, 0.10, 0.20
        image_shape: (H, W)
        direction: None o una de:
            ['left', 'right', 'up', 'down',
             'up_left', 'up_right', 'down_left', 'down_right']
        return_direction: bool

    Returns:
        bbox desplazada como torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        opcionalmente también la dirección usada
    """
    H, W = image_shape

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().float().view(-1).tolist()
    else:
        bbox = np.array(bbox, dtype=np.float32).reshape(-1).tolist()

    x_min, y_min, x_max, y_max = bbox

    bw = x_max - x_min
    bh = y_max - y_min

    directions = {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
        "up_left": (-1, -1),
        "up_right": (1, -1),
        "down_left": (-1, 1),
        "down_right": (1, 1),
    }

    if direction is None:
        direction = random.choice(list(directions.keys()))

    if direction not in directions:
        raise ValueError(f"Dirección no válida: {direction}. Opciones: {list(directions.keys())}")

    sx, sy = directions[direction]

    # Desplazamiento en píxeles relativo al tamaño de la bbox
    dx = sx * bw * shift_percent
    dy = sy * bh * shift_percent

    # Limitar el shift para que la bbox no salga de la imagen
    dx = max(-x_min, min(dx, (W - 1) - x_max))
    dy = max(-y_min, min(dy, (H - 1) - y_max))

    new_x_min = x_min + dx
    new_x_max = x_max + dx
    new_y_min = y_min + dy
    new_y_max = y_max + dy

    result = torch.tensor([
        int(round(new_x_min)),
        int(round(new_y_min)),
        int(round(new_x_max)),
        int(round(new_y_max)),
    ], dtype=torch.int64)

    if return_direction:
        return result, direction
    return result


class MultiDataset(Dataset):
    def __init__(self, data_dir, tumor, mode="Training", bbox_shift=0, label=None, rescale_bbox=None, shift_percent=None):

        tumor_path_labels = {'lung': 'Dataset002_Lung1',
                        'breast': 'Dataset008_ISPY1',
                        'liver' : 'Dataset009_Liver2',
                        'kidney' : 'Dataset012_Kidney',
                        'brain': 'Dataset004_BraTS'          
        }

        self.mode = mode
        self.tumor = tumor
        self.task = tumor_path_labels[tumor]
        self.dataset = self.task.split('_')[-1]
        self.data_path = data_dir + '/' + self.task

        if self.tumor == "brain":
            self.label = label

        if not self.mode == 'Training':
            self.mask_path = os.path.join("/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/nnUNetv2/Data/LabelsTs", f'{self.dataset}_labelsTs')  
            self.img_path = os.path.join(self.data_path, "imagesTs") 
            if self.tumor == "brain":
                file_path = os.path.join(self.data_path, f'{self.tumor}_{self.label}_test_pos.txt')
            else:
                file_path = os.path.join(self.data_path, f'{self.tumor}_test_pos.txt')

        else:
            self.mask_path = os.path.join(self.data_path, "labelsTr")  
            self.img_path = os.path.join(self.data_path, "imagesTr") 
            file_path = os.path.join(self.data_path, f'{self.tumor}_train.txt')


        with open(file_path, 'r') as f:
            self.samples_list = [x[:-1] for x in f]

        self.bbox_shift = bbox_shift

        self.rescale_bbox = rescale_bbox
        self.shift_percent = shift_percent

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_path, self.samples_list[idx].split("/")[0])
            mask_path = os.path.join(self.mask_path, self.samples_list[idx].split("/")[0].split("_0000.nii")[0] + ".nii.gz")

            image = nib.load(img_path).get_fdata()
            if image.shape[0] == image.shape[1]:
                image = image[:, :, int(self.samples_list[idx].split("/")[-1])]

            elif image.shape[1] == image.shape[2]:
                image = image[int(self.samples_list[idx].split("/")[-1]), :, :]

            img = np.stack([image] * 3, axis=2)
            img_1024 = resize(img, (1024, 1024))
            # convert the shape to (3, H, W)
            img_1024 = np.transpose(img_1024, (2, 0, 1))

            img_min = img_1024.min()
            img_max = img_1024.max()
            den = img_max - img_min

            normalized_array = (img_1024 - img_min) / den

            mask = nib.load(mask_path).get_fdata()
            if mask.shape[0] == mask.shape[1]:
                mask = mask[:, :, int(self.samples_list[idx].split("/")[-1])]

            elif mask.shape[1] == mask.shape[2]:
                mask = mask[int(self.samples_list[idx].split("/")[-1]), :, :]

            gt = mask.copy()
            gt[gt > 1] = 1
            label_ids = np.unique(gt)[1:]
            gt = resize(gt, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False)
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            )  # only one label, (256, 256)
            assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])

            if self.rescale_bbox is not None:
                bbox = resize_bbox(
                    bbox,
                    scale=self.rescale_bbox, 
                    image_shape=(1024, 1024)
                )

            if self.shift_percent is not None:
                bbox = shift_bbox(
                    bbox,
                    shift_percent=self.shift_percent,
                    image_shape=(1024, 1024)
                )

            return (
                torch.tensor(normalized_array).float(),
                torch.tensor(gt2D[None, :, :]).long(),
                torch.tensor(bboxes).float(),
                self.samples_list[idx],
            )
        
        except Exception as e:
            print(f"[WARNING] Error in sample {self.samples_list[idx]}: {e}")
            return None