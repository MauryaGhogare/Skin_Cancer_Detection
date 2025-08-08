# SLICE-3D 2024 Challenge Dataset

## Overview
The SLICE-3D dataset is designed for skin lesion classification using images extracted from 3D Total Body Photography (TBP). The primary goal is to differentiate between benign and malignant cases using diagnostically labeled images with associated metadata.

### Dataset Includes:
- **Images**: JPEG lesion images.
- **Binary Classification Target**: 
  - 0 for benign
  - 1 for malignant

## Dataset Description

### Files Included:
- `train-image/` - Folder containing JPEG training images.
- `train-image.hdf5` - HDF5 file with training image data, indexed by `isic_id`.
- `train-metadata.csv` - Metadata for training images.
- `test-image.hdf5` - HDF5 file with test image data.
- `test-metadata.csv` - Metadata for test images.
- `sample_submission.csv` - Example submission format.

### Metadata Fields
The metadata includes the following fields:

- **Patient Details**:
  - `age_approx`: Approximate age of the patient.
  - `sex`: Gender of the patient.
  - `patient_id`: Unique identifier for the patient.

- **Lesion Information**:
  - `lesion_id`: Unique identifier for the lesion.
  - `anatom_site_general`: General anatomical site of the lesion.
  - `clin_size_long_diam_mm`: Clinical size of the lesion (long diameter in mm).
  - `tbp_lv_areaMM2`: Area of the lesion in mm².

- **Image-Based Features**:
  - `tbp_lv_H`: Height of the lesion in the TBP image.
  - `tbp_lv_L`: Length of the lesion in the TBP image.
  - `tbp_lv_norm_border`: Normalized border of the lesion.
  - `tbp_lv_radial_color_std_max`: Maximum radial color standard deviation.

- **Classification Labels** (only in `train-metadata.csv`):
  - `target`: 0 for benign, 1 for malignant.

## Data Imbalance and Preprocessing
The dataset is highly imbalanced, with approximately 99% benign and 1% malignant cases. To address this issue:

- **Undersampling and Oversampling Techniques**: Applied to balance the dataset.
- **Augmentations**: Techniques such as flipping, rotation, and color shifts are used to enhance model robustness.

### Model Architecture
- **EfficientNetV2Backbone**: Utilized for feature extraction from lesion images.
  
