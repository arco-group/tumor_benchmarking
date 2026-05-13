from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset import MultiDataset
from model import LitSwinUNETR2D, _count_trainable_params


TUMOR_NUM_CLASSES = {
    "lung": 2,
    "breast": 2,
    "liver": 2,
    "kidney": 2,
    "brain": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simple 2D MONAI SwinUNETR model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", "--data_root", dest="data_dir", type=str, default="nnUNetv2/Data/nnUNet_raw")
    parser.add_argument("--tumor", type=str, required=True, choices=list(TUMOR_NUM_CLASSES))
    parser.add_argument("-num_classes", "--num_classes", type=int, default=None)
    parser.add_argument("-checkpoint_path", "--checkpoint_path", type=str, default="results/checkpoints")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_size", type=int, default=0)
    parser.add_argument("--no_aug", action="store_true")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--swin_depths", type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument("--swin_heads", type=int, nargs=4, default=[3, 6, 12, 24])
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_checkpoint", action="store_true")

    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="benchmarking")
    parser.add_argument("--wandb_offline", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    num_classes = args.num_classes or TUMOR_NUM_CLASSES[args.tumor]
    checkpoint_dir = Path(args.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    dataset = MultiDataset(
        data_dir=args.data_dir,
        tumor=args.tumor,
        mode="Training",
        img_size=args.img_size,
        num_classes=num_classes,
        augment=not args.no_aug,
        cache_size=args.cache_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = LitSwinUNETR2D(
        img_size=args.img_size,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        feature_size=args.feature_size,
        swin_depths=tuple(args.swin_depths),
        swin_heads=tuple(args.swin_heads),
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
    )

    print(f"[INFO] tumor={args.tumor}")
    print(f"[INFO] data_dir={args.data_dir}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] trainable_params={_count_trainable_params(model) / 1e6:.2f}M")
    print(f"[INFO] checkpoints={checkpoint_dir}")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="train/dice",
            mode="max",
            save_top_k=args.save_top_k,
            save_last=True,
            filename=f"SwinUNETR_{args.tumor}" + "-{epoch:03d}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = None
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"SwinUNETR_{args.tumor}",
            mode="offline" if args.wandb_offline else "online",
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        num_nodes=args.nodes,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=25,
        benchmark=True,
    )

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=args.resume_ckpt)
    trainer.save_checkpoint(str(checkpoint_dir / f"SwinUNETR_{args.tumor}.ckpt"))


if __name__ == "__main__":
    main()
