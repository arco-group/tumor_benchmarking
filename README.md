# Tumor benchmarking

qua dire che come medsam e medicalsam 2 si basando su segment anything, abbiamo fatto un environment per tutti e due

spiegazione di come istallare sam_venv

spiegazione della struttura dei datasets nella cartella per fare match con il codice

```
tumor_benchmarking/
├── README.md
├── nnUNetv2/
│   ├── README.md
│   ├── Data
│   │   ├── LabelsTs/
│   │   │   └── Name_labelsTs
│   │   │       ├── tumor_002.nii.gz
│   │   │       ├── tumor_004.nii.gz
│   │   │       └── ...
│   │   ├── nnUNet_raw/
│   │   │   └── Dataset000_Name/
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

spiegare che train e test.txt files devono avere questa struttura
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
