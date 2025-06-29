import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
}

# Load datasets
data_dir = "/kaggle/input/databalance/splitDataSetVersion3"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test', 'valid']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'test', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'valid']}
class_names = image_datasets['train'].classes
print(f"Classes: {class_names}")

# Load a pretrained ViT model using timm
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(class_names))

# Modify the classifier head (if necessary)
model.head = nn.Linear(model.head.in_features, len(class_names))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Définir le device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Déplacer le modèle sur le device
model = model.to(device)

# Initialisation des variables
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Listes pour stocker les résultats
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print('-' * 10)

    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Enregistrer les résultats
        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
        else:
            valid_losses.append(epoch_loss)
            valid_accuracies.append(epoch_acc.item())

        # Sauvegarder le meilleur modèle
        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model updated!")
# Charger les poids du meilleur modèle
model.load_state_dict(best_model_wts)

# Sauvegarder le meilleur modèle dans un fichier
torch.save(model.state_dict(), 'best_model.pth')
print("Training complete! Best model saved as 'best_model.pth'")


# Définir le chemin du dossier
save_dir = '/kaggle/working/model/'
os.makedirs(save_dir, exist_ok=True)  

# Définir le chemin complet pour sauvegarder le modèle
model_path = os.path.join(save_dir, 'best_model.pth')

# Sauvegarder le modèle
torch.save(model.state_dict(), model_path)
print(f"Training complete! Best model saved at '{model_path}'")

plt.show()

model.eval()
test_loss = 0.0
test_corrects = 0

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)

test_acc = test_corrects.double() / dataset_sizes['test']
print(f"Test Accuracy: {test_acc:.4f}")

num_classes = len(class_names)

# Collect true labels and predicted probabilities
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
# Concatenate all results
all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

# Binarize labels for ROC calculation (one-vs-rest approach)
all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

# Plot ROC Curve
plt.figure(figsize=(20, 20))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"Class {i}").plot()

plt.title("ROC Curve")
plt.show()

# Get predictions and true labels
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        
# Concatenate all results
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()