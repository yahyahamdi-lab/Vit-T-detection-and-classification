## Vit-T-detection-and-classification
### Overview
This project implements a Vision Transformer (ViT) model for image classification tasks using PyTorch and the timm library. The model is trained on a custom dataset, evaluated using performance metrics (accuracy, ROC curve, confusion matrix), and saved for future deployment.
### Requirement:
python==3.8.10  # Tested versions

torch==2.0.1+cu118  # With CUDA 11.8

torchvision==0.15.2+cu118

timm==0.9.2
scikit-learn==1.2.2
matplotlib==3.7.1
albumentations==1.3.0  # For advanced augmentations
wandb==0.15.0  # For experiment tracking
tensorboard==2.12.0  # For visualization
### Key Features
Data Loading & Preprocessing:

Resizes images to 224x224 resolution.

### Data augmentation (random horizontal flipping for training).

Pixel value normalization.

#### ViT Model:

Uses a ViT-Base (patch16_224) model via timm.

Replaces the classifier head to match the number of target classes.

### Training:

Optimized with Adam (learning rate = 0.0001).

Loss function: CrossEntropyLoss.


### Evaluation:

Computes test accuracy.

Generates a ROC curve (one-vs-rest) with AUC scores.

Displays a confusion matrix.
