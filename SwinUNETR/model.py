from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR


def segmentation_stats_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = 2 if logits.shape[1] == 1 else logits.shape[1]

    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits[:, 0]) >= threshold).long()
    else:
        pred = torch.argmax(logits, dim=1)

    target = target.long().squeeze(1).clamp(0, num_classes - 1)
    pred_oh = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    pred_fg = pred_oh[:, 1:]
    target_fg = target_oh[:, 1:]
    intersection = (pred_fg * target_fg).sum(dim=(1, 2, 3))
    pred_sum = pred_fg.sum(dim=(1, 2, 3))
    target_sum = target_fg.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection

    eps = 1e-7
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou


class LitSwinUNETR2D(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        feature_size: int = 48,
        swin_depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        swin_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.is_multiclass = num_classes > 2
        out_channels = num_classes if self.is_multiclass else 1

        self.model = SwinUNETR(
            in_channels=1,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=swin_depths,
            num_heads=swin_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=drop_path_rate,
            spatial_dims=2,
            use_checkpoint=use_checkpoint,
            use_v2=False,
        )

        if self.is_multiclass:
            self.seg_loss = nn.CrossEntropyLoss()
            self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        else:
            self.seg_loss = nn.BCEWithLogitsLoss()
            self.dice_loss = DiceLoss(sigmoid=True)

        self._test_records: list[dict[str, Any]] = []
        self.test_records: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.is_multiclass:
            ce = self.seg_loss(logits, target.long().squeeze(1))
            dice = self.dice_loss(logits, target.long())
        else:
            target = target.float()
            ce = self.seg_loss(logits, target)
            dice = self.dice_loss(logits, target)
        return ce + dice

    def _shared_step(self, batch: dict[str, Any], stage: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]
        masks = batch["mask"]
        logits = self(images)
        loss = self._loss(logits, masks)

        dice, iou = segmentation_stats_from_logits(
            logits.detach(),
            masks,
            threshold=float(self.hparams.threshold),
        )

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=stage == "train", on_epoch=True, batch_size=images.size(0))
        self.log(f"{stage}/dice", dice.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log(f"{stage}/iou", iou.mean(), prog_bar=False, on_step=False, on_epoch=True, batch_size=images.size(0))
        return loss, dice.detach(), iou.detach()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_test_epoch_start(self) -> None:
        self._test_records = []
        self.test_records = []

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, dice, iou = self._shared_step(batch, "test")

        sample_ids = batch.get("sample_id", [""] * len(dice))
        volume_ids = batch.get("volume_id", [""] * len(dice))
        slice_indices = batch.get("slice_idx", [0] * len(dice))

        for idx in range(len(dice)):
            slice_idx = slice_indices[idx]
            if torch.is_tensor(slice_idx):
                slice_idx = slice_idx.item()
            self._test_records.append(
                {
                    "sample_id": str(sample_ids[idx]),
                    "volume_id": str(volume_ids[idx]),
                    "slice_idx": int(slice_idx),
                    "dice": float(dice[idx].cpu()),
                    "iou": float(iou[idx].cpu()),
                }
            )

    def on_test_epoch_end(self) -> None:
        self.test_records = sorted(self._test_records, key=lambda row: (row["volume_id"], row["slice_idx"]))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
