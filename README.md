## Vit-T-detection-and-classification
### Overview
This project implements a Vision Transformer (ViT) model for Aquatic Floating Objects classification and detection tasks using PyTorch and the timm library. The proposed model combines GAN network for image generation and Vit for detection and classification tasks. The model is trained on a custom dataset, evaluated using performance metrics (accuracy, ROC curve, confusion matrix), and saved for future deployment.
![Architecture](https://github.com/user-attachments/assets/363034e6-ea8f-4be0-a894-f71b907f0c2c)

### Requirement:
- python==3.8.10
- torch==2.0.1+cu118  # With CUDA 11.8
- torchvision==0.15.2+cu118s
- timm==0.9.2
- scikit-learn==1.2.2
- matplotlib==3.7.1
- albumentations==1.3.0  
- wandb==0.15.0  
- tensorboard==2.12.0  # For visualization

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

#### Dataset:
The dataset used was a combination of two datasets exported via Roboflow, the first one (Ocean Plastics Waste Detection - Float
Plastics Dataset) [1] includes 4987 images, and the second one (Ocean Pollution dataset.v1-ocean-debris) [2] includes 626 images. 

![image](https://github.com/user-attachments/assets/9101153f-4c68-4ea1-b608-18071a42d76a)

### References
[1] Roboflow, “Float Plastics Dataset.” [Online]. Available:
https://universe.roboflow.com/ocean-plastics-waste-detection/float-plastics-dataset
[2] Roboflow, “Ocean Debris Dataset (v1),” [Online]. Available:
https://universe.roboflow.com/ocean-pollution/dataset.v1-ocean-debris
