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

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from func_3d.utils import eval_seg


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


def resize_bbox(bbox, scale, image_shape):
    """
    bbox: tensor/list/array [x_min, y_min, x_max, y_max]
    scale: float. 1.05 = +5%, 0.95 = -5%
    image_shape: (H, W)

    return: bbox escalada y recortada a la imagen
    """
    H, W = image_shape

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().float().view(-1).tolist()
    else:
        bbox = np.array(bbox).reshape(-1).tolist()

    x_min, y_min, x_max, y_max = bbox

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    bw = x_max - x_min
    bh = y_max - y_min

    new_bw = bw * scale
    new_bh = bh * scale

    new_x_min = cx - new_bw / 2.0
    new_x_max = cx + new_bw / 2.0
    new_y_min = cy - new_bh / 2.0
    new_y_max = cy + new_bh / 2.0

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


def main():
    args = cfg.parse_args()
    args.vis = False
    args.net = 'sam2'
    args.sam_ckpt = '/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/checkpoints/sam2_hiera_small.pt' 
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Breast_click_2026_03_25_08_17_25/Model/latest_epoch.pth"  # breast click
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Breast_new_2026_03_25_08_11_49/Model/latest_epoch.pth"  # breast bbox
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Liver_click_2026_03_25_08_20_02/Model/latest_epoch.pth"  # liver click
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Liver_new_2026_03_25_08_06_12/Model/latest_epoch.pth"  # liver bbox
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Lung_click_2026_03_25_08_18_22/Model/latest_epoch.pth"  # lung click
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Lung_new_2026_03_25_08_10_55/Model/latest_epoch.pth"  # lung bbox
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Kidney_click_2026_03_25_08_19_28/Model/latest_epoch.pth"  # kidney click
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Kidney_new_2026_03_25_08_05_14/Model/latest_epoch.pth"  # kidney bbox
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Brain_click_2026_03_25_10_27_57/Model/latest_epoch.pth"  # brain click
    # args.sam_ckpt = "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/logs/Brain_new_2026_03_25_10_28_37/Model/latest_epoch.pth"  # brain bbox    
    args.sam_config = 'sam2_hiera_s'
    args.image_size = 1024
    args.val_freq = 5
    args.prompt = 'click'
    args.prompt_freq = 2
    args.dataset = 'multi_dataset'
    args.tumor = 'liver'
    args.data_path = '/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/nnUNetv2/Data/nnUNet_raw'
    args.model = 'test'
    args.pretrain = '/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/medsam2/automatic-medsam2/checkpoints/MedSAM2_pretrain.pth'
    args.video_length = None
    args.rescale_bbox = None
    args.shift_percent = None

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


if __name__ == '__main__':
    main()