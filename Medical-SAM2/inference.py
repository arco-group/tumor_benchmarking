import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import numpy as np
import random
import pandas as pd
import argparse

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from func_3d.utils import eval_seg


parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='sam2', help='net type')
parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
parser.add_argument('-exp_name', type=str, help='experiment name')
parser.add_argument('-vis', type=bool, default=False, help='Generate visualisation during validation')
parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
parser.add_argument('-prompt', type=str, default='bbox', help='type of prompt, bbox or click')
parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
parser.add_argument('-pretrain', type=str, default="/checkpoints/MedSAM2_pretrain.pth", help='path of pretrain weights')
parser.add_argument('-val_freq', type=int, default=5, help='interval between each validation')
parser.add_argument('-gpu', type=bool, default=False, help='use gpu or not')
parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
parser.add_argument('-image_size', type=int, default=1024, help='image_size')
parser.add_argument('-out_size', type=int, default=1024, help='output_size')
parser.add_argument('-distributed', default='none', type=str, help='multi GPU ids to use')
parser.add_argument('-dataset', default='multi_dataset', type=str, help='dataset name')
parser.add_argument('-tumor', type=str, help='tumor type')
parser.add_argument('-dataset_yaml', default=False, type=str, help='yaml file with samples of training set')
parser.add_argument('-sam_ckpt', type=str, default="/checkpoints/sam2_hiera_small.pt", help='sam checkpoint address')
parser.add_argument('-sam_config', type=str, default="sam2_hiera_s", help='sam checkpoint address')
parser.add_argument('-video_length', type=int, default=None, help='sam checkpoint address')
parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-weights', type=str, default=0, help='the weights file you want to test')
parser.add_argument('-model', type=str, help='the name of the model you are training')
parser.add_argument('-multimask_output', type=int, default=1, help='the number of masks output for multi-class segmentation')
parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')
parser.add_argument('-data_path', type=str, default='./data/btcv', help='The path of segmentation data')
parser.add_argument('-rescale_bbox', type=float, default=None, help='Rescale of bounding box, i.e. 0.95 would give a bounding box a 95percent smaller')
parser.add_argument('-shift_percent', type=float, default=None, help='Percent of bounding box shifting, i.e. 0.1 would move the bounding box a 10percent in a random direction')
args = parser.parse_args()


def overlay_segmentation(image, segmentation, alpha=0.5):
    """Overlays a segmentation mask on a grayscale image."""
    color_map = np.array([
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [0, 255, 0],    # Green
        [255, 0, 0],    # Red
        [0, 0, 255],    # Blue
    ], dtype=np.uint8)

    seg_colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    unique_classes = np.unique(segmentation)

    for i, cls in enumerate(unique_classes):
        if cls > 0:  # Ignore background
            seg_colored[segmentation == cls] = color_map[i % len(color_map)]

    image_rgb = np.stack([image] * 3, axis=-1)
    blended = (1 - alpha) * image_rgb + alpha * seg_colored
    return blended.astype(np.uint8)


GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(
    args,
    args.net,
    use_gpu=args.gpu,
    gpu_device=GPUdevice,
    distribution=args.distributed
)
net.to(dtype=torch.float16)

if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights, strict=False)

torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

nice_train_loader, nice_test_loader = get_dataloader(args)

net.eval()
threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
prompt_freq = args.prompt_freq
prompt = args.prompt

# Métricas globales reales
global_iou_sum = 0.0
global_dice_sum = 0.0
global_count = 0

# Métricas por clase dinámicas
# class_results[label] = {"iou_sum": float, "dice_sum": float, "count": int}
class_results = {}

results = []

# Casos válidos reales
valid_cases = 0

for pack in tqdm(nice_test_loader, total=len(nice_test_loader), desc='Validation round', unit='batch', leave=False):
    imgs_tensor = pack['image']
    mask_dict = pack['label']

    if prompt == 'click':
        pt_dict = pack['pt']
        point_labels_dict = pack['p_label']
    elif prompt == 'bbox':
        bbox_dict = pack['bbox']

    if len(imgs_tensor.size()) == 5:
        imgs_tensor = imgs_tensor.squeeze(0)

    frame_id = list(range(imgs_tensor.size(0)))

    train_state = net.val_init_state(imgs_tensor=imgs_tensor)
    prompt_frame_id = list(range(0, len(frame_id), prompt_freq))

    obj_list = []
    for id in frame_id:
        try:
            obj_list += list(mask_dict[id].keys())
        except KeyError:
            continue

    obj_list = list(set(obj_list))

    if len(obj_list) == 0:
        continue

    valid_cases += 1

    name = pack['image_meta_dict']['filename_or_obj']

    with torch.no_grad():
        for id in prompt_frame_id:
            for ann_obj_id in obj_list:
                try:
                    if prompt == 'click':
                        points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                        labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                        _, _, _ = net.train_add_new_points(
                            inference_state=train_state,
                            frame_idx=id,
                            obj_id=ann_obj_id,
                            points=points,
                            labels=labels,
                            clear_old_points=False,
                        )

                    elif prompt == 'bbox':
                        bbox = bbox_dict[id][ann_obj_id]

                        if args.rescale_bbox is not None:
                            bbox = resize_bbox(
                                bbox,
                                scale=args.rescale_bbox,
                                image_shape=(imgs_tensor.shape[2], imgs_tensor.shape[3])
                            )

                        if args.shift_percent is not None:
                            bbox = shift_bbox(
                                bbox,
                                shift_percent=args.shift_percent,
                                image_shape=(imgs_tensor.shape[2], imgs_tensor.shape[3])
                            )

                        _, _, _ = net.train_add_new_bbox(
                            inference_state=train_state,
                            frame_idx=id,
                            obj_id=ann_obj_id,
                            bbox=bbox.to(device=GPUdevice),
                            clear_old_points=False,
                        )

                except KeyError:
                    _, _, _ = net.train_add_new_mask(
                        inference_state=train_state,
                        frame_idx=id,
                        obj_id=ann_obj_id,
                        mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                    )

        video_segments = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(
            train_state,
            start_frame_idx=0
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i]
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        for id in frame_id:
            preds = []
            masks = []

            for ann_obj_id in obj_list:
                # Predicción
                try:
                    pred = video_segments[id][ann_obj_id]
                except KeyError:
                    continue

                pred = pred.unsqueeze(0)

                # GT
                try:
                    mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                except KeyError:
                    mask = torch.zeros_like(pred).to(device=GPUdevice)

                # Eval
                temp = eval_seg(pred, mask, threshold)
                iou_val = temp[0]
                dice_val = temp[1]

                results.append({
                    "name": name[0],
                    "frame": id,
                    "label": int(ann_obj_id),
                    "iou": float(iou_val),
                    "dice": float(dice_val),
                })

                global_iou_sum += iou_val
                global_dice_sum += dice_val
                global_count += 1

                if ann_obj_id not in class_results:
                    class_results[ann_obj_id] = {
                        "iou_sum": 0.0,
                        "dice_sum": 0.0,
                        "count": 0
                    }

                class_results[ann_obj_id]["iou_sum"] += iou_val
                class_results[ann_obj_id]["dice_sum"] += dice_val
                class_results[ann_obj_id]["count"] += 1

                preds.append((pred[0, 0, :, :] > 0.5) * ann_obj_id)
                masks.append(mask[0, 0, :, :] * ann_obj_id)

            if len(preds) == 0:
                continue

            all_preds = sum(preds)
            all_masks = sum(masks)

            # Para visualización: clamp al máximo label presente
            max_label = max([int(x) for x in obj_list]) if len(obj_list) > 0 else 0
            all_preds[all_preds > max_label] = max_label

            if args.vis:
                os.makedirs(f'./temp/val_1/{args.model}/{name[0]}', exist_ok=True)

                overlayed_pred = overlay_segmentation(
                    imgs_tensor[id, 0, :, :].cpu().numpy().astype(np.uint8),
                    all_preds.cpu().numpy(),
                    alpha=0.3
                )

                plt.imshow(np.asarray(overlayed_pred))
                plt.savefig(
                    f'./temp/val_1/{args.model}/{name[0]}/{id}.png',
                    bbox_inches='tight',
                    pad_inches=0
                )
                plt.close()

# Impresión final
if global_count > 0:
    eiou = global_iou_sum / global_count
    edice = global_dice_sum / global_count
else:
    eiou = np.nan
    edice = np.nan

# Ordenar labels para imprimir bonito
sorted_labels = sorted(class_results.keys(), key=lambda x: float(x))

for cls in sorted_labels:
    stats = class_results[cls]
    mean_iou = stats["iou_sum"] / stats["count"] if stats["count"] > 0 else np.nan
    mean_dice = stats["dice_sum"] / stats["count"] if stats["count"] > 0 else np.nan
    print(f'IOU for label {cls}: {mean_iou}, DICE: {mean_dice}.')

print(f'Global: {eiou}, DICE: {edice}.')
print(f'Valid cases: {valid_cases}')
print(f'Total class evaluations: {global_count}')

df = pd.DataFrame(results)
output_path = f'metrics_{args.tumor}_{args.prompt}_zero_shot.xlsx'
df.to_excel(output_path, index=False)

print(f'Metrics saved to {output_path}')

