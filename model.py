import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import pandas as pd
import numpy as np
from collections import Counter
import random
import cv2


def crop_eye_region(image, cascade_path="haarcascade_eye.xml"):
    img_array = np.array(image)[:, :, ::-1]  # RGB to BGR
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        padding = int(max(w, h) * 0.2)
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(img_array.shape[1] - x, w + 2 * padding), min(img_array.shape[0] - y, h + 2 * padding)
        cropped = img_array[y:y+h, x:x+w]
        return Image.fromarray(cropped[:, :, ::-1])
    return image

class EyeDiseaseDataset(Dataset):
    def __init__(self, tiff_dir=None, png_dir=None, csv_file=None, transform=None, balance_data=True):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        keratitis_images = []
        epiphora_images = []
        normal_images = []

        
        if tiff_dir and os.path.exists(tiff_dir):
            tiff_files = [f for f in os.listdir(tiff_dir) if f.lower().endswith('.tif')]
            keratitis_images = [(os.path.join(tiff_dir, f), 2) for f in tiff_files]
            print(f"Found {len(tiff_files)} TIFF files in {tiff_dir}")

    
        if csv_file and png_dir and os.path.exists(png_dir) and os.path.exists(csv_file):
            data = pd.read_csv(csv_file).dropna()
            data.iloc[:, 0] = data.iloc[:, 0].astype(str)
            label_mapping = {"normal": 0, "mild": 1, "moderate": 1, "severe": 1}
            png_files = [(os.path.join(png_dir, f), label_mapping[l]) 
                         for f, l in zip(data.iloc[:, 0], data.iloc[:, 1]) 
                         if os.path.exists(os.path.join(png_dir, f))]
            
            for path, label in png_files:
                if label == 0:
                    normal_images.append((path, label))
                elif label == 1:
                    epiphora_images.append((path, label))
            print(f"Found {len(png_files)} valid PNG files from CSV")

     
        if balance_data:
            num_keratitis = len(keratitis_images)  
            normal_images = random.sample(normal_images, min(num_keratitis, len(normal_images)))
            epiphora_images = random.sample(epiphora_images, min(num_keratitis, len(epiphora_images)))

       
        self.image_paths, self.labels = zip(*(keratitis_images + epiphora_images + normal_images))

    
        print(f"Balanced Dataset - Normal: {len(normal_images)}, Epiphora: {len(epiphora_images)}, Keratitis: {len(keratitis_images)}")

        if not self.image_paths:
            raise ValueError("Dataset is empty after initialization. Check file paths and data availability.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
        
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = crop_eye_region(image)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


try:
    dataset = EyeDiseaseDataset(
        tiff_dir=r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\Blue_Light",
        png_dir=r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\26172919\train\train",
        csv_file=r"C:\Users\Ludovic\Pytorch\ExpoSciences\Datasets\26172919\SLID_E_information.csv",
        transform=train_transform,
        balance_data=True
    )
except ValueError as e:
    print(e)
    exit(1)


print(f"Dataset size: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError("Dataset is empty. .")


train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
print(f"Train indices: {len(train_indices)}, Val indices: {len(val_indices)}")

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
val_dataset.dataset.transform = val_transform


label_counts = Counter(dataset.labels)
class_weights = {label: 1.0 / count for label, count in label_counts.items()}
sample_weights = [class_weights[label] for label in [dataset.labels[i] for i in train_indices]]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=3):
        super(EyeDiseaseModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeDiseaseModel(num_classes=3).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


MODEL_PATH = "newer_model.pth"

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    if os.path.exists(MODEL_PATH):
        choice = input("Modèle existant trouvé. Entrez 'train' pour réentraîner ou 'load' pour charger : ").strip().lower()
        if choice == "load":
            print("Chargement du modèle existant.")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            return
        elif choice != "train":
            print("Entrée invalide. Chargement par défaut.")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            return
    
    print("Entraînement démarré...")
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for i, batch in enumerate(train_loader):
            if batch is None:
                print(f"Batch {i} is None, skipping...")
                continue
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        train_accuracy = 100 * correct_preds / total_preds if total_preds > 0 else 0
        print(f"Époque [{epoch+1}/{num_epochs}], Perte: {running_loss/len(train_loader):.4f}, Précision entraînement: {train_accuracy:.2f}%")

      
        model.eval()
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        val_accuracy = 100 * correct_preds / total_preds if total_preds > 0 else 0
        print(f"Précision validation: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Nouveau meilleur modèle sauvegardé: {val_accuracy:.2f}%")

    print("Entraînement terminé")


train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)
print("Chargement du meilleur modèle")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
