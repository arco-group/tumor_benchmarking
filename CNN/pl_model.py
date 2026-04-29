import lightning as pl
from torch.optim import SGD, Adam
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F


def segmentation_stats_from_logits(logits, targets, eps=1e-7):
    """
    logits:  [B, C, H, W]
    targets: [B, C, H, W] one-hot
    """
    pred_labels = torch.argmax(logits, dim=1)
    target_labels = torch.argmax(targets, dim=1)

    num_classes = logits.shape[1]

    pred_one_hot = F.one_hot(pred_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()

    pred_fg = pred_one_hot[:, 1:]
    target_fg = target_one_hot[:, 1:]

    intersection = (pred_fg * target_fg).sum(dim=(0, 2, 3))
    pred_sum = pred_fg.sum(dim=(0, 2, 3))
    target_sum = target_fg.sum(dim=(0, 2, 3))
    union = pred_sum + target_sum - intersection

    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()

    return dice, iou


class UNet(nn.Module):
    """
    U-Net model for image segmentation.

    The U-Net is a convolutional network architecture designed for biomedical image segmentation.
    It consists of an encoder (downsampling path) to capture context and a decoder (upsampling path)
    to enable precise localization.

    Attributes
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    out_channels : int
        Number of output channels (e.g., 1 for binary segmentation or multiple for multi-class segmentation).

    As implemented in: Primakov, Sergey P., et al. "Automated detection and segmentation of non-small cell lung cancer
    computed tomography images." Nature communications 13.1 (2022): 3423.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the U-Net model.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout=False):
        """
        Creates a convolutional block with two sequential convolutional layers.

        Parameters
        ----------
        in_channels : int
            Number of input channels for the convolutional block.
        out_channels : int
            Number of output channels for the convolutional block.
        dropout : float
            Dropout rate.

        Returns
        -------
        nn.Sequential
            A sequential container with two convolutional layers, each followed by a ReLU activation and
            batch normalization.
        """
        if dropout:
            return nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True)
            )

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W) where:
            - N is the batch size,
            - C is the number of channels,
            - H and W are the height and width of the input image.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W).
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.output_conv(dec1)


class UnetModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(UnetModel, self).__init__()
        self.model = UNet(in_channels=3, out_channels=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = images.shape[0]
        logits = self(images)

        loss = self.criterion(logits, targets.argmax(dim=1))
        dice, iou = segmentation_stats_from_logits(logits, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train/dice", dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train/iou", iou, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        batch_size = images.shape[0]
        logits = self(images)

        loss = self.criterion(logits, targets.argmax(dim=1))
        dice, iou = segmentation_stats_from_logits(logits, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val/dice", dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return {
            "val_loss": loss.detach(),
            "val_dice": dice.detach(),
            "val_iou": iou.detach(),
        }

    def test_step(self, batch, batch_idx):
        images, targets, _ = batch
        batch_size = images.shape[0]
        logits = self(images)

        loss = self.criterion(logits, targets.argmax(dim=1))
        dice, iou = segmentation_stats_from_logits(logits, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test/dice", dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test/iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class TorchvisionModel(pl.LightningModule):
    def __init__(self, model_name, num_classes):
        super(TorchvisionModel, self).__init__()
        self.num_classes = num_classes

        self.model = torchvision.models.get_model(model_name, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batch_size = images.shape[0]
        logits = self(images)["out"]

        loss = self.criterion(logits, targets.argmax(dim=1))
        dice, iou = segmentation_stats_from_logits(logits, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train/dice", dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train/iou", iou, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, paths = batch
        batch_size = images.shape[0]
        logits = self(images)["out"]

        loss = self.criterion(logits, targets.argmax(dim=1))
        dice, iou = segmentation_stats_from_logits(logits, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val/dice", dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return {
            "val_loss": loss.detach(),
            "val_dice": dice.detach(),
            "val_iou": iou.detach(),
        }

    def test_step(self, batch, batch_idx):
        images, targets, _ = batch
        batch_size = images.shape[0]
        logits = self(images)["out"]

        loss = self.criterion(logits, targets.argmax(dim=1))
        dice, iou = segmentation_stats_from_logits(logits, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test/dice", dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test/iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer


