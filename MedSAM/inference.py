'''
This file uses the code implementation from bowang-lab:
https://github.com/bowang-lab/MedSAM
The inference code is adapted to fit our needs. The original code can be found here: https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py
'''


import numpy as np
import os

join = os.path.join
import torch
from model import MedSAM
from dataset import MultiDataset
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from torch.utils.data import DataLoader
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/nnUNetv2/Data/nnUNet_raw", help="Path to training files")
parser.add_argument("--test_masks_dir", type=str, default="/nnUNetv2/Data/LabelsTs", help="Path to test files")
parser.add_argument("--tumor", type=str)
parser.add_argument("-checkpoint", type=str, default="sam_vit_b_01ec64.pth")
parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument("-rescale_bbox", type=float, default=None)
parser.add_argument("-shift_percent", type=float, default=None)
parser.add_argument("-zero_shot", type=bool, default=False)
args = parser.parse_args()


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


results = []

device = args.device

sam_model = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
medsam_model = MedSAM(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)

ckpt = torch.load(args.checkpoint, map_location=device)
if args.zero_shot == False:
    medsam_model.load_state_dict(ckpt["model"], strict=True)
elif args.zero_shot == True:
    medsam_model.load_state_dict(ckpt, strict=True)
medsam_model.eval()

miou = MeanIoU(num_classes=args.num_classes, per_class=True).to(device)
gen_dice = GeneralizedDiceScore(num_classes=args.num_classes, per_class=True).to(device)

metric_miou = []
metric_dice = []

ts_dataset = MultiDataset(data_dir=args.data_dir, tumor=args.tumor, mode="Test", test_masks_dir=args.test_masks_dir, rescale_bbox=args.rescale_bbox, shift_percent=args.shift_percent)
ts_dataloader = DataLoader(ts_dataset, batch_size=1, shuffle=False)

for step, (image, gt2D, boxes, name) in enumerate(tqdm(ts_dataloader, desc='Processing samples', total=len(ts_dataloader))):
    patient, slice = name[0].split('/')
    slice = slice.split('.')[0]
    boxes_np = boxes.detach().cpu().numpy()
    image, gt2D = image.to(device), gt2D.to(device)

    image_embedding = medsam_model.image_encoder(image)  # (1, 256, 64, 64)

    _, _, H, W = image.shape
    medsam_seg = torch.tensor(medsam_inference(medsam_model, image_embedding, boxes_np, H, W), dtype=torch.long)

    gt2D = gt2D.squeeze(1)
    gt =  F.one_hot(gt2D, num_classes=2)
    gt = gt.permute(0, 3, 1, 2).bool()

    medsam_seg = medsam_seg.unsqueeze(0)
    medsam_seg = F.one_hot(medsam_seg, num_classes=2)
    medsam_seg = medsam_seg.permute(0, 3, 1, 2).bool()

    iou_value = miou(medsam_seg.to(device), gt.to(device))
    dice_value = gen_dice(medsam_seg.to(device), gt.to(device))

    metric_miou.append(iou_value)
    metric_dice.append(dice_value)

    results.append({
        "name": name[0],
        "iou": iou_value[1].item(),
        "dice": dice_value[1].item()
    })

df = pd.DataFrame(results)

output_path = f"MedSAM_{args.tumor}_metrics_per_sample.xlsx"
df.to_excel(output_path, index=False)

mean_miou = torch.stack(metric_miou).mean(dim=0).to(device)
mean_dice = torch.stack(metric_dice).mean(dim=0).to(device)

print(f'IoU: {mean_miou}, \n Dice: {mean_dice}')
