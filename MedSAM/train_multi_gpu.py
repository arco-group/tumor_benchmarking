'''
This file uses the code implementation from bowang-lab:
https://github.com/bowang-lab/MedSAM
The training code is adapted to fit our needs. The original code can be found here: https://github.com/bowang-lab/MedSAM/blob/main/train_one_gpu.py
'''

import argparse
import os
import random
import shutil
from datetime import datetime

join = os.path.join

import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from dataset import MultiDataset
from model import MedSAM
from segment_anything import sam_model_registry


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return distributed, rank, local_rank, world_size

    rank = 0
    local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return distributed, rank, local_rank, world_size


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def build_zero_sync_loss(trainable_params, device):
    # Keep DDP synchronization consistent when a whole batch is filtered out by collate_skip_none.
    zero_loss = torch.zeros((), device=device)
    for p in trainable_params:
        zero_loss = zero_loss + p.view(-1)[0] * 0.0
    return zero_loss


def load_model_state(model, checkpoint_state):
    if any(k.startswith("module.") for k in checkpoint_state.keys()):
        checkpoint_state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in checkpoint_state.items()
        }
    model.load_state_dict(checkpoint_state)


os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")

distributed, rank, local_rank, world_size = setup_distributed()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2023 + rank)
np.random.seed(2023 + rank)
random.seed(2023 + rank)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/nnUNetv2/Data/nnUNet_raw", help="Path to training files")
parser.add_argument("-tumor", type=str)
parser.add_argument("-checkpoint", type=str, default="sam_vit_b_01ec64.pth")
# train
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01)
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-use_wandb", type=bool, default=False)
parser.add_argument("-use_amp", action="store_true", default=False)
parser.add_argument("-resume", type=str, default=None)
parser.add_argument("-ddp_bucket_cap_mb", type=int, default=25)
parser.add_argument("-find_unused_parameters", type=bool, default=True)
args = parser.parse_args()

wandb_run = None
if args.use_wandb and is_main_process():
    import wandb

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    else:
        wandb.login()

    wandb_run = wandb.init(
        project="benchmarking",
        name=f'MedSAM_{args.tumor}',
        config={
            "lr": args.lr,
            "batch_size_per_gpu": args.batch_size,
            "world_size": world_size,
            "effective_batch_size": args.batch_size * world_size,
            "data_path": args.data_dir,
        },
    )


run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
if distributed:
    run_id_list = [run_id] if is_main_process() else [None]
    dist.broadcast_object_list(run_id_list, src=0)
    run_id = run_id_list[0]

model_save_path = f'MedSAM_{args.tumor}-{run_id}'

if is_main_process():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
if distributed:
    dist.barrier()

sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = MedSAM(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)

if distributed:
    medsam_model = DDP(
        medsam_model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        find_unused_parameters=args.find_unused_parameters,
        bucket_cap_mb=args.ddp_bucket_cap_mb,
    )

model_for_optim = medsam_model.module if distributed else medsam_model
img_mask_encdec_params = list(model_for_optim.image_encoder.parameters()) + list(
    model_for_optim.mask_decoder.parameters()
)

optimizer = torch.optim.AdamW(
    img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
)

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

train_dataset  = MultiDataset(data_dir=args.data_dir, tumor=args.tumor, mode="Training")

train_sampler = (
    DistributedSampler(train_dataset, shuffle=True) if distributed else None
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=collate_skip_none,
)

start_epoch = 0
if args.resume is not None and os.path.isfile(args.resume):
    checkpoint = torch.load(args.resume, map_location=device)
    start_epoch = checkpoint.get("epoch", -1) + 1
    load_model_state(model_for_optim, checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

if distributed:
    dist.barrier()

scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

best_loss = 1e10
losses = []

try:
    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        medsam_model.train()
        epoch_loss_sum_local = 0.0
        valid_steps_local = 0

        progress = tqdm(
            train_dataloader,
            disable=not is_main_process(),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
        )

        for batch in progress:
            local_has_data = 0 if batch is None else 1
            if distributed:
                has_data_tensor = torch.tensor(
                    [local_has_data], device=device, dtype=torch.int32
                )
                dist.all_reduce(has_data_tensor, op=dist.ReduceOp.SUM)
                global_has_data = has_data_tensor.item() > 0
            else:
                global_has_data = local_has_data > 0

            if not global_has_data:
                continue

            optimizer.zero_grad(set_to_none=True)

            if batch is None:
                zero_loss = build_zero_sync_loss(img_mask_encdec_params, device)
                if args.use_amp:
                    scaler.scale(zero_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    zero_loss.backward()
                    optimizer.step()
                continue

            image, gt2d, boxes, _ = batch
            boxes_np = boxes.detach().cpu().numpy()
            image = image.to(device, non_blocking=True)
            gt2d = gt2d.to(device, non_blocking=True)

            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch.float16,
                enabled=args.use_amp,
            ):
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2d) + ce_loss(
                    medsam_pred, gt2d.float()
                )

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_value = loss.detach().item()
            epoch_loss_sum_local += loss_value
            valid_steps_local += 1

            if is_main_process():
                progress.set_postfix(loss=f"{loss_value:.4f}")

        if distributed:
            stats = torch.tensor(
                [epoch_loss_sum_local, float(valid_steps_local)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_loss_sum = stats[0].item()
            valid_steps = int(stats[1].item())
        else:
            epoch_loss_sum = epoch_loss_sum_local
            valid_steps = valid_steps_local

        epoch_loss = epoch_loss_sum / max(valid_steps, 1)

        if is_main_process():
            losses.append(epoch_loss)

            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, "epoch_loss": epoch_loss})

            checkpoint = {
                "model": model_for_optim.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

            plt.figure()
            plt.plot(losses)
            plt.title("Dice + Cross Entropy Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
            plt.close()

        if distributed:
            dist.barrier()

finally:
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()