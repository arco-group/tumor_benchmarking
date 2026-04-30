# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import argparse
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader


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
args = parser.parse_args()


GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
net.to(dtype=torch.bfloat16)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

sam_layers = (
                []
            #   + list(net.image_encoder.parameters())
            #   + list(net.sam_prompt_encoder.parameters())
                + list(net.sam_mask_decoder.parameters())
                )
mem_layers = (
                []
                + list(net.obj_ptr_proj.parameters())
                + list(net.memory_encoder.parameters())
                + list(net.memory_attention.parameters())
                + list(net.mask_downsample.parameters())
                )
if len(sam_layers) == 0:
    optimizer1 = None
else:
    optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
if len(mem_layers) == 0:
    optimizer2 = None
else:
    optimizer2 = optim.Adam(mem_layers, lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

nice_train_loader, nice_test_loader = get_dataloader(args)

'''checkpoint path and tensorboard'''
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.model)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.model))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
best_dice = 0.0

for epoch in range(settings.EPOCH):
    net.train()
    time_start = time.time()
    loss, prompt_loss, non_prompt_loss = function.train_sam(args, net, optimizer1, optimizer2, nice_train_loader, epoch)
    logger.info(f'Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))

writer.close()