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

from segment_anything import sam_model_registry

join = os.path.join


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


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.contour(mask_image[:, :, 0])


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, data_dir, tumor, mode="Training", bbox_shift=20, label=None):
        tumor_path_labels = {
            "lung": "Dataset002_Lung1",
            "breast": "Dataset008_ISPY1",
            "liver": "Dataset009_Liver2",
            "kidney": "Dataset012_Kidney",
            "brain": "Dataset004_BraTS",
        }

        self.mode = mode
        self.tumor = tumor
        self.task = tumor_path_labels[tumor]
        self.dataset = self.task.split("_")[-1]
        self.data_path = data_dir + "/" + self.task

        if self.tumor == "brain":
            self.label = label

        if self.mode != "Training":
            self.mask_path = os.path.join(
                "/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/nnUNetv2/Data/LabelsTs",
                f"{self.dataset}_labelsTs",
            )
            self.img_path = os.path.join(self.data_path, "imagesTs")
            if self.tumor == "brain":
                file_path = os.path.join(
                    self.data_path, f"{self.tumor}_{self.label}_test_pos.txt"
                )
            else:
                file_path = os.path.join(self.data_path, f"{self.tumor}_test_pos.txt")
        else:
            self.mask_path = os.path.join(self.data_path, "labelsTr")
            self.img_path = os.path.join(self.data_path, "imagesTr")
            if self.tumor == "brain":
                file_path = os.path.join(
                    self.data_path, f"{self.tumor}_{self.label}_train_pos.txt"
                )
            else:
                file_path = os.path.join(self.data_path, f"{self.tumor}_train_pos.txt")

        with open(file_path, "r") as f:
            self.samples_list = [x[:-1] for x in f]

        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_path, self.samples_list[idx].split("/")[0])
            mask_path = os.path.join(
                self.mask_path,
                self.samples_list[idx].split("/")[0].split("_0000.nii")[0] + ".nii.gz",
            )

            image = nib.load(img_path).get_fdata(dtype=np.float32)
            slice_idx = int(self.samples_list[idx].split("/")[-1])

            if image.shape[0] == image.shape[1]:
                image = image[:, :, slice_idx]
            elif image.shape[1] == image.shape[2]:
                image = image[slice_idx, :, :]

            img = np.stack([image] * 3, axis=2)
            img_1024 = resize(img, (1024, 1024))
            img_1024 = np.transpose(img_1024, (2, 0, 1))

            img_min = img_1024.min()
            img_max = img_1024.max()
            den = img_max - img_min

            if (not np.isfinite(img_1024).all()) or den < 1e-8:
                print(f"[WARNING] Imagen invalida o constante: {self.samples_list[idx]}")
                return None

            normalized_array = (img_1024 - img_min) / den

            mask = nib.load(mask_path).get_fdata(dtype=np.float32)
            if mask.shape[0] == mask.shape[1]:
                mask = mask[:, :, slice_idx]
            elif mask.shape[1] == mask.shape[2]:
                mask = mask[slice_idx, :, :]

            if self.tumor == "lung":
                mask[mask != 3] = 0
                mask[mask == 3] = 1
            elif self.tumor == "brain":
                mask[mask != self.label] = 0
                mask[mask == self.label] = 1

            gt = mask.copy()
            gt[gt > 1] = 1
            gt = resize(gt, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False)
            gt2D = np.uint8(gt == 1)

            assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"

            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            h, w = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(w, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(h, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])

            return (
                torch.tensor(normalized_array).float(),
                torch.tensor(gt2D[None, :, :]).long(),
                torch.tensor(bboxes).float(),
                self.samples_list[idx],
            )
        except Exception as e:
            print(f"[WARNING] Error en sample {self.samples_list[idx]}: {e}. Mask {mask.shape}, label {self.label}, values ")
            return None


class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--tr_npy_path",
        type=str,
        default="/mimer/NOBACKUP/groups/naiss2023-6-336/emulero/nnUNetv2/Data/nnUNet_raw",
        help="path to training files",
    )
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
    parser.add_argument("-tumor", type=str)
    parser.add_argument("-label", type=int, default=None)
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument(
        "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("--load_pretrain", type=bool, default=True)
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")

    parser.add_argument("-num_epochs", type=int, default=1000)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-num_workers", type=int, default=0)

    parser.add_argument("-weight_decay", type=float, default=0.01)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-use_wandb", type=bool, default=False)
    parser.add_argument("-wandb_project", type=str, default="benchmarking")
    parser.add_argument("-wandb_api_key", type=str, default="")

    parser.add_argument("-use_amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--ddp_bucket_cap_mb", type=int, default=25)
    parser.add_argument("--find_unused_parameters", type=bool, default=True)

    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")

    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d-%H%M%S")
    if distributed:
        run_id_list = [run_id] if is_main_process() else [None]
        dist.broadcast_object_list(run_id_list, src=0)
        run_id = run_id_list[0]

    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)

    if is_main_process():
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    if distributed:
        dist.barrier()

    wandb_run = None
    if args.use_wandb and is_main_process():
        import wandb

        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key)
        else:
            wandb.login()

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.task_name,
            config={
                "lr": args.lr,
                "batch_size_per_gpu": args.batch_size,
                "world_size": world_size,
                "effective_batch_size": args.batch_size * world_size,
                "data_path": args.tr_npy_path,
                "model_type": args.model_type,
                "tumor": args.tumor,
                "label": args.label,
            },
        )

    # REMOVEEEEEE
    args.tumor = "brain"
    args.label = 1
    #####


    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
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

    if is_main_process():
        print(
            "Number of total parameters:",
            sum(p.numel() for p in medsam_model.parameters()),
        )
        print(
            "Number of trainable parameters:",
            sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
        )

    model_for_optim = medsam_model.module if distributed else medsam_model
    img_mask_encdec_params = list(model_for_optim.image_encoder.parameters()) + list(
        model_for_optim.mask_decoder.parameters()
    )

    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )

    if is_main_process():
        print(
            "Number of image encoder and mask decoder parameters:",
            sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
        )

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    train_dataset = NpyDataset(
        data_dir=args.tr_npy_path,
        tumor=args.tumor,
        mode="Training",
        label=args.label,
    )

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

    if is_main_process():
        print("Number of training samples:", len(train_dataset))

    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint.get("epoch", -1) + 1
        load_model_state(model_for_optim, checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

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

                print(
                    f"Time: {datetime.now().strftime('%Y%m%d-%H%M')}, "
                    f"Epoch: {epoch}, Loss: {epoch_loss}"
                )

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


if __name__ == "__main__":
    main()