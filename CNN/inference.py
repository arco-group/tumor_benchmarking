#!/usr/bin/env python3
import os
import json
import yaml
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tumor_benchmarking.CNN import dataset
from tumor_benchmarking.CNN import pl_model


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--tumor", type=str, choices=["lung", "liver", "kidney", "brain", "breast"])
parser.add_argument("--model", type=str, choices=["deeplabv3_resnet101", "unet"])
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_classes", type=int, default=2, help="Number of classes depends on the labels of the mask. It is 2 for all tumors, except for brain, that has 4 labels. Background is included")
args = parser.parse_args()


def dice_iou_per_class_from_labels(pred_labels, target_labels, num_classes, eps=1e-7):
    pred_one_hot = F.one_hot(pred_labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target_labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    pred_fg = pred_one_hot[:, 1:]
    target_fg = target_one_hot[:, 1:]

    intersection = (pred_fg * target_fg).sum(dim=(0, 2, 3))
    pred_sum = pred_fg.sum(dim=(0, 2, 3))
    target_sum = target_fg.sum(dim=(0, 2, 3))
    union = pred_sum + target_sum - intersection

    dice_per_class = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou_per_class = (intersection + eps) / (union + eps)

    valid = (pred_sum + target_sum) > 0
    return dice_per_class, iou_per_class, valid, intersection, pred_sum, target_sum, union


@torch.no_grad()
def evaluate_semantic(model, dataloader, device, num_classes):
    model.eval()

    total_intersection = torch.zeros(num_classes - 1, dtype=torch.float64, device=device)
    total_pred_sum = torch.zeros(num_classes - 1, dtype=torch.float64, device=device)
    total_target_sum = torch.zeros(num_classes - 1, dtype=torch.float64, device=device)
    total_union = torch.zeros(num_classes - 1, dtype=torch.float64, device=device)

    per_case_rows = []

    for batch in tqdm(dataloader):
        images, targets, paths = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, dict) and "out" in outputs:
            logits = outputs["out"]
        else:
            logits = outputs

        pred_labels = torch.argmax(logits, dim=1)
        target_labels = torch.argmax(targets, dim=1)

        _, _, _, inter, pred_sum, target_sum, union = dice_iou_per_class_from_labels(
            pred_labels=pred_labels,
            target_labels=target_labels,
            num_classes=num_classes,
        )

        total_intersection += inter.double()
        total_pred_sum += pred_sum.double()
        total_target_sum += target_sum.double()
        total_union += union.double()

        for i in range(images.shape[0]):
            d_case, j_case, valid_case, _, _, _, _ = dice_iou_per_class_from_labels(
                pred_labels=pred_labels[i:i+1],
                target_labels=target_labels[i:i+1],
                num_classes=num_classes,
            )

            row = {
                "case": paths[i],
                "dice_macro_fg": d_case[valid_case].mean().item() if valid_case.any() else 0.0,
                "iou_macro_fg": j_case[valid_case].mean().item() if valid_case.any() else 0.0,
            }

            for cls_idx in range(1, num_classes):
                row[f"dice_class_{cls_idx}"] = d_case[cls_idx - 1].item()
                row[f"iou_class_{cls_idx}"] = j_case[cls_idx - 1].item()
                row[f"valid_class_{cls_idx}"] = bool(valid_case[cls_idx - 1].item())

            per_case_rows.append(row)

    eps = 1e-7
    global_dice_per_class = (2.0 * total_intersection + eps) / (total_pred_sum + total_target_sum + eps)
    global_iou_per_class = (total_intersection + eps) / (total_union + eps)
    global_valid = (total_pred_sum + total_target_sum) > 0

    metrics = {
        "num_test_samples": len(dataloader.dataset),
        "dice_macro_fg": global_dice_per_class[global_valid].mean().item() if global_valid.any() else 0.0,
        "iou_macro_fg": global_iou_per_class[global_valid].mean().item() if global_valid.any() else 0.0,
    }

    for cls_idx in range(1, num_classes):
        metrics[f"dice_class_{cls_idx}"] = global_dice_per_class[cls_idx - 1].item()
        metrics[f"iou_class_{cls_idx}"] = global_iou_per_class[cls_idx - 1].item()
        metrics[f"valid_class_{cls_idx}"] = bool(global_valid[cls_idx - 1].item())

    return metrics, pd.DataFrame(per_case_rows)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

transform = transforms.Compose([
    dataset.GrayToRGB(),
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
])
mask_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
])

# Create Dataset and Dataloader
test_data = dataset.MultiDataset(data_dir=args.data_dir, tumor=args.tumor, mode="Test", transform=transform, mask_transform=mask_transform, num_classes=args.num_classes)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=config["training"].get("num_workers", 2), pin_memory=False)

# Instantiate the LightningModule
if args.model == "unet":
    model = pl_model.UnetModel(num_classes=args.num_classes)
elif args.model == "deeplabv3_resnet101":
    model = pl_model.TorchvisionModel(model_name=args.model, num_classes=args.num_classes)  # Any model reported on https://docs.pytorch.org/vision/0.9/models.html#semantic-segmentation is supported
else:
    print(f"{args.model} model is not supported")

# Use the checkpoint to load the model
if args.model == "unet":
    model = pl_model.UnetModel.load_from_checkpoint(args.checkpoint_path, num_classes=args.num_classes).model
elif args.model == "deeplabv3_resnet101":    
    model = pl_model.DetectionModelV2.load_from_checkpoint(checkpoint_path=args.checkpoint_path, model_name=args.model, num_classes=args.num_classes)

model.eval()
model = model.to(device)

metrics, per_case_df = evaluate_semantic(
    model=model,
    dataloader=test_loader,
    device=device,
    num_classes=config["training"]["num_classes"],
)

out_dir = os.path.join("results", f"{args.model}_{args.tumor}")
os.makedirs(out_dir, exist_ok=True)

metrics_path = os.path.join(out_dir, "test_metrics.json")
per_case_path = os.path.join(out_dir, "test_metrics_per_case.csv")

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

    per_case_df.to_csv(per_case_path, index=False)

print("Test results:")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"Overall metrics saved at: {metrics_path}")
print(f"Cases metrics saved at: {per_case_path}")