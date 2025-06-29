## Vit-T-detection-and-classification
### Overview
This project implements a Vision Transformer (ViT) model for image classification tasks using PyTorch and the timm library. The model is trained on a custom dataset, evaluated using performance metrics (accuracy, ROC curve, confusion matrix), and saved for future deployment.
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

Saves the best model based on validation accuracy.

### Evaluation:

Computes test accuracy.

Generates a ROC curve (one-vs-rest) with AUC scores.

Displays a confusion matrix.
