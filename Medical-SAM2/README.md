# Medical SAM 2

This repository contains the code used to train and evaluate Medical SAM 2 on custom medical imaging datasets.

## Installation

Install Medical SAM 2 and download the required weights by following the instructions in the [official repository](https://github.com/ImprintLab/Medical-SAM2).

## Using a Custom Dataset

If you want to train or evaluate on your own dataset, copy `multi_dataset.py` into `func_3d/dataset/` and update `func_3d/dataset/__init__.py` to include the custom dataset branch.

```python
elif args.dataset == 'multi_dataset':
    multi_train_dataset = MultiDataset(args, args.data_path, transform=None, transform_msk=None, mode='Training', prompt=args.prompt)
    multi_test_dataset = MultiDataset(args, args.data_path, transform=None, transform_msk=None, mode='Test', prompt=args.prompt)

    nice_train_loader = DataLoader(multi_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(multi_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
```

### Dataset Configuration

In the `multi_dataset.py` file, the `MultiDataset` class contains a dictionary called `tumor_path_labels` that maps tumor types to their corresponding dataset folder names in the nnUNet_raw directory. This dictionary looks like:

```python
tumor_path_labels = {
    'lung': 'Dataset002_Lung1',
    'breast': 'Dataset008_ISPY1',
    'liver': 'Dataset009_Liver2',
    'kidney': 'Dataset012_Kidney',
    'brain': 'Dataset004_BraTS'
}
```

**Important**: You must modify this dictionary to match your own dataset folder names and tumor types. Each key should correspond to the tumor type you use as the `-tumor` argument, and each value should be the exact folder name in your `nnUNet_raw` directory.

## Training

From the project root, you can launch training with:

```bash
srun --export=ALL python -u train.py \
    -data_path /nnUNetv2/Data/nnUNet_raw \
    -dataset multi_dataset \
    -tumor tumor \
    -prompt bbox \
    -exp_name model_name \
    -model model_name \
    -pretrain /checkpoints/MedSAM2_pretrain.pth
```

Use `-prompt click` instead of `-prompt bbox` if you want click-based prompting.

## Inference

From the project root, you can launch inference with:

```bash
srun --export=ALL python -u inference.py \
    -data_path /nnUNetv2/Data/nnUNet_raw \
    -dataset multi_dataset \
    -tumor tumor \
    -prompt bbox \
    -exp_name model_name \
    -model model_name \
    -sam_ckpt checkpoint_path \
    -rescale_bbox None \
    -shift_percent None
```

For zero-shot inference, set `-sam_ckpt /checkpoints/sam2_hiera_small.pt`.

`-prompt` can be set to `bbox` or `click`. The `-rescale_bbox` and `-shift_percent` arguments can be used to tighten, expand, or shift bounding boxes during inference.

