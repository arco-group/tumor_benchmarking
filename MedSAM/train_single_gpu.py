'''
This file uses the code implementation from bowang-lab:
https://github.com/bowang-lab/MedSAM
The training code is adapted to fit our needs. The original code can be found here: https://github.com/bowang-lab/MedSAM/blob/main/train_one_gpu.py
'''

"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
from segment_anything import sam_model_registry
import argparse
from datetime import datetime
import shutil
from torch.utils.data._utils.collate import default_collate
from dataset import MultiDataset


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


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

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/nnUNetv2/Data/nnUNet_raw", help="Path to training files")
parser.add_argument("--test_masks_dir", type=str, default="/nnUNetv2/Data/LabelsTs", help="Path to test files")
parser.add_argument("--tumor", type=str)
parser.add_argument("-checkpoint", type=str, default="sam_vit_b_01ec64.pth")
parser.add_argument('-device', type=str, default='cuda:0')
# train
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("-use_wandb", type=bool, default=False, help="use wandb to monitor training")
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("-resume", type=str, default=None, help="Resuming training from checkpoint")
parser.add_argument("-device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login(key='your_key')
    wandb.init(
        project="benchmarking",
        name=f'MedSAM_{args.tumor}',
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.data_dir,
        },
    )

# set up model for training
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = f'MedSAM_{args.tumor}-{run_id}'
os.makedirs(model_save_path, exist_ok=True)
shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

device = torch.device(args.device)

# Set up model
sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = MedSAM(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)
medsam_model.train()

img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
    medsam_model.mask_decoder.parameters()
)
optimizer = torch.optim.AdamW(
    img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
)

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

# Cross entropy loss
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

# train
iter_num = 0
losses = []
best_loss = 1e10
train_dataset  = MultiDataset(data_dir=args.data_dir, tumor=args.tumor, mode="Training")


print("Number of training samples: ", len(train_dataset))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=collate_skip_none,
)

start_epoch = 0
if args.resume is not None:
    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

for epoch in range(start_epoch, args.num_epochs):
    epoch_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        image, gt2D, boxes, _ = batch
        optimizer.zero_grad()
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        if args.use_amp:
            ## AMP
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            medsam_pred = medsam_model(image, boxes_np)
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        iter_num += 1

    epoch_loss /= step
    losses.append(epoch_loss)
    if args.use_wandb:
        wandb.log({"epoch_loss": epoch_loss})
    # save the latest model
    checkpoint = {
        "model": medsam_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

    # plot loss
    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(model_save_path, args.tumor + "train_loss.png"))
    plt.close()
