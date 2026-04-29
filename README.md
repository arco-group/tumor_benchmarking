# Tumor Benchmarking

## Overview

This repository contains a comprehensive benchmarking framework for tumor segmentation across multiple deep learning architectures. It includes all models, datasets, and evaluation code used in our benchmarking paper (reference pending), along with reproducible training and inference pipelines.

## Repository Contents

The repository is organized into X main model implementations:

- **nnUNetv2**: State-of-the-art automated medical image segmentation framework
- **SAM-based**: Segment Anything Model variants adapted for medical imaging (forse faccio una per modello)
- **CNN**: Convolutional Neural Networks for tumor segmentation
- **Swin UNETR**: Transformer-based architecture for medical image segmentation

## Dataset Structure

To ensure compatibility across all models, please organize your datasets following the structure below. This structure is consistent with nnUNet conventions for easy integration.

### Directory Layout

```
tumor_benchmarking/
├── README.md
├── nnUNetv2/
│   ├── README.md
│   ├── Data/
│   │   ├── LabelsTs/
│   │   │   └── {DatasetName}_labelsTs/
│   │   │       ├── tumor_002.nii.gz
│   │   │       ├── tumor_004.nii.gz
│   │   │       └── ...
│   │   ├── nnUNet_raw/
│   │   │   └── Dataset000_{DatasetName}/
│   │   │       ├── imagesTr/
│   │   │       │   ├── tumor_000_0000.nii.gz
│   │   │       │   ├── tumor_001_0000.nii.gz
│   │   │       │   ├── tumor_003_0000.nii.gz
│   │   │       │   └── ...
│   │   │       ├── imagesTs/
│   │   │       │   ├── tumor_002_0000.nii.gz
│   │   │       │   ├── tumor_004_0000.nii.gz
│   │   │       │   └── ...
│   │   │       ├── labelsTr/
│   │   │       │   ├── tumor_000.nii.gz
│   │   │       │   ├── tumor_001.nii.gz
│   │   │       │   ├── tumor_003.nii.gz
│   │   │       │   └── ...
│   │   │       ├── dataset.json
│   │   │       ├── tumor_train.txt
│   │   │       └── tumor_test.txt
│   │   ├── nnUNet_preprocessed/
│   │   └── nnUNet_results/
│   └── ...
├── SAM-based/
├── CNN/
└── Swin UNETR/
```

### Train/Test Split Files

The `tumor_train.txt` and `tumor_test.txt` files define which samples belong to the training and testing sets. Each file should contain entries with the following structure, listing all available slices for each sample:

```
tumor_000_0000.nii.gz/0
tumor_000_0000.nii.gz/1
tumor_000_0000.nii.gz/2
tumor_000_0000.nii.gz/3
tumor_001_0000.nii.gz/0
tumor_001_0000.nii.gz/1
tumor_001_0000.nii.gz/2
tumor_001_0000.nii.gz/3
...
```

Each line represents a specific slice of a sample, allowing 2D image handling for some models.

## Model-Specific Instructions

Each model has its own implementation with detailed setup and training instructions:

- **[nnUNetv2](nnUNetv2/README.md)**: Automated medical image segmentation
- **[SAM-based Models](SAM-based/README.md)**: Prompt-guided segmentation
- **[CNN](CNN/README.md)**: Convolutional neural network implementation
- **[Swin UNETR](Swin%20UNETR/README.md)**: Transformer-based segmentation

Please refer to each model's README for specific installation requirements, hyperparameters, and training procedures.

## Using Your Own Dataset

All models and training scripts are ready to work with datasets organized according to the structure above. To train a model with your custom dataset:

1. Organize your data following the directory structure provided above
2. Configure the `dataset.json` file with appropriate metadata
3. Create `tumor_train.txt` and `tumor_test.txt` files with the required format
4. Navigate to the desired model directory and follow the model-specific training instructions in its README

## Dataset Format Requirements

Images and labels must be stored in `.nii.gz` format to ensure compatibility with our code. Case naming follows the nnUNet conventions described in the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#what-do-training-cases-look-like). 

Note: we have also added an additional folder `LabelsTs/` containing ground-truth labels for test cases. This folder is not part of the original nnUNet dataset format but is included here to enable evaluation of our test sets using the ground-truth labels alongside the provided test images.
