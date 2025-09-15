
# DepthSS
# Project: Remote Sensing Image Semantic Segmentation based on HRNet with our depthSS


## Project Structure

```
your-project-name/
├── core/                  # Contains all custom modules (models, data transforms, etc.)
├── data/                  # Directory for datasets
├── work_dirs/             # Directory to save training logs and model weights
├── config.py              # The main project configuration file
├── train.py               # Script to start the training process
├── predict.py             # Script for inference on a single image
├── requirements.txt       # A list of dependencies
└── README.md              # This README file
```

## 1. Environment Setup

**a. Create a Conda Environment (Recommended)**
```bash
conda create -n hrnet-fusion python=3.8 -y
conda activate hrnet-fusion
```

**b. Install PyTorch**
Please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to install the version that corresponds to your CUDA environment. For example, for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

**c. Install MMLab Dependencies**
We highly recommend using the official OpenMMLab `mim` tool for installation, as it automatically resolves compatibility issues between `mmcv` and your PyTorch/CUDA versions.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmsegmentation>=1.0.0"
```

**d. Install Other Dependencies**
```bash
pip install -r requirements.txt
```

## 2. Dataset Preparation

This project requires images, their corresponding segmentation annotations, and pre-generated normal vector maps. Please organize your dataset according to the following directory structure:

```
your-project-name/
└── data/
    ├── vaihingen/
    │   ├── img_dir/
    │   │   ├── train/
    │   │   │   ├── image1.png
    │   │   │   └── ...
    │   │   └── val/
    │   │       ├── image100.png
    │   │       └── ...
    │   └── ann_dir/
    │       ├── train/
    │       │   ├── image1.png  (Segmentation Annotation)
    │       │   └── ...
    │       └── val/
    │           ├── image100.png (Segmentation Annotation)
    │           └── ...
    │
    └── output_results/
        ├── train/
        │   └── normal_maps/
        │       ├── norm_image1.png (Normal Map)
        │       └── ...
        └── val/
            └── normal_maps/
                ├── norm_image100.png (Normal Map)
                └── ...
```
-   The `data/vaihingen` directory contains the original images (`img_dir`) and the ground truth segmentation maps (`ann_dir`).
-   The `data/output_results` directory contains the normal maps generated from the original images. The filename of a normal map must be prefixed with `norm_` followed by the original image's filename.

## 3. Training the Model

**a. (Optional) Download Pre-trained Weights**
For improved performance and faster convergence, you can download the official pre-trained weights for HRNet (e.g., `hrnetv2_w18_imagenet_pretrained.pth`). In `config.py`, set the `load_from` variable to the path of the downloaded weights file.

**b. Start Training**
Execute the following command to begin training. Logs and model checkpoints will be saved in the `work_dirs/config/` directory.
```bash
python train.py config.py
```

**c. Resume Training**
If training is interrupted, you can resume from the latest checkpoint in the working directory:
```bash
python train.py config.py --resume
```

## 4. Inference and Prediction

Use a trained model to perform inference on a single image.

```bash
# Syntax: python predict.py <config_file> <checkpoint_file> <input_image> <output_path>
python predict.py config.py \
                  work_dirs/config/iter_80000.pth \
                  data/vaihingen/img_dir/val/example.png \
                  work_dirs/predictions/result.png
```
-   Ensure that the input image also has a corresponding normal map, as it is required by the model during inference.

## 5. Validation Visualization

During the training process, validation is performed every `val_interval` iterations (set to 2000 in `config.py`). The visualization results for the first 5 images of each validation run—which include the original image, ground truth, prediction, and normal map—are automatically saved to the `work_dirs/val_visualizations/` directory. This allows for convenient and periodic monitoring of the model's performance.

