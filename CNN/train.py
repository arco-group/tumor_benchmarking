from torchvision import transforms
from tumor_benchmarking.CNN import dataset
from torch.utils.data import DataLoader
from tumor_benchmarking.CNN import pl_model
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--tumor", type=str, choices=["lung", "liver", "kidney", "brain", "breast"])
parser.add_argument("--model", type=str, choices=["deeplabv3_resnet101", "unet"])
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-num_epoch", type=int, default=100)
parser.add_argument("-num_classes", type=int, default=2, help="Number of classes depends on the labels of the mask. It is 2 for all tumors, except for brain, that has 4 labels. Background is included")
args = parser.parse_args()

transform = transforms.Compose([
    dataset.GrayToRGB(),
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
])

# Create Dataset
training_data = dataset.MultiDataset(data_dir=args.data_dir, tumor=args.tumor, mode="Training", transform=transform, mask_transform=mask_transform, num_classes=args.num_classes)
test_data = dataset.MultiDataset(data_dir=args.data_dir, tumor=args.tumor, mode="Test", transform=transform, mask_transform=mask_transform, num_classes=args.num_classes)

# Create PyTorch DataLoader
train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)


# Instantiate the LightningModule
if args.model == "unet":
    model = pl_model.UnetModel(num_classes=args.num_classes)
elif args.model == "deeplabv3_resnet101":
    model = pl_model.TorchvisionModel(model_name=args.model, num_classes=args.num_classes)  # Any model reported on https://docs.pytorch.org/vision/0.9/models.html#semantic-segmentation is supported
else:
    print(f"{args.model} model is not supported")


wandb_logger = WandbLogger(
    project="benchmarking",
    log_model=False,
    name=f"{args.model}_{args.tumor}"
)

lr_monitor = LearningRateMonitor(logging_interval="step")

monitor_metric = "val/dice"
monitor_mode = "max"

checkpoint_callback = ModelCheckpoint(
    monitor=monitor_metric,
    mode=monitor_mode,
    save_top_k=1,
    save_last=True,
    filename="{epoch:02d}-{" + monitor_metric.replace("/", "_") + ":.4f}",
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices="auto",
    strategy="ddp",
    num_nodes=2,
    max_epochs=args.num_epoch,
    callbacks=[lr_monitor, checkpoint_callback],
    logger=wandb_logger,
)

# Train the model
trainer.fit(model, train_dataloader, test_dataloader)


