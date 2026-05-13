from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import MultiDataset
from model import LitSwinUNETR2D


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", "--data_root", dest="data_dir", type=str, default="nnUNetv2/Data/nnUNet_raw")
parser.add_argument("--test_masks_dir", type=str, default="nnUNetv2/Data/LabelsTs")
parser.add_argument("--tumor", type=str, required=True)
parser.add_argument("-checkpoint_path", "--checkpoint_path", "--checkpoint", type=str, required=True)
parser.add_argument("-num_classes", "--num_classes", type=int, default=None)

parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("-batch_size", "--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--cache_size", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--precision", type=str, default="16-mixed")

args = parser.parse_args()


checkpoint_path = Path(args.checkpoint_path)
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

test_dataset = MultiDataset(data_dir=args.data_dir, tumor=args.tumor, mode="Test", img_size=args.img_size, test_masks_dir=args.test_masks_dir)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

model = LitSwinUNETR2D.load_from_checkpoint(str(checkpoint_path))
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=args.devices,
    precision=args.precision,
    logger=False,
)

trainer.test(model=model, dataloaders=test_loader)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
output_xlsx = output_dir / f"SwinUNETR_{args.tumor}.xlsx"

df = pd.DataFrame(model.test_records)
with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="per_slice", index=False)

print(f"[OK] Test metrics saved to: {output_xlsx}")
print(f"[OK] Test samples: {len(df)}")
if len(df) > 0:
    print(f"[OK] Mean Dice: {df['dice'].mean():.4f}")
    print(f"[OK] Mean IoU: {df['iou'].mean():.4f}")



