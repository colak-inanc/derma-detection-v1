import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import numpy as np
import os

# class weights ile loss fonksiyonu dengelendi

def get_model(num_classes):
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_train_loader(data_dir="data-sets/Dataset/s-train", batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(260, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
    ])
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1
    class_weights = 1. / np.array(class_counts)
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    return loader, dataset.classes, class_counts

def train_model(model, train_loader, device, epochs=15, class_weights=None, patience=3):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    best_loss = float('inf')
    patience_counter = 0
    print(f"Kullanılan cihaz: {device}", flush=True)
    print("Eğitim başlıyor...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\n{epoch+1}. epoch başlıyor...", flush=True)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)
        # Early stopping ve en iyi modeli kaydetme
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/efficientnet_b2_skin_disease_best.pth")
            print("Yeni en iyi model kaydedildi.", flush=True)
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print("Early stopping ile eğitim durduruldu.", flush=True)
                break
    print("\n✅ Eğitim tamamlandı. En iyi model 'models/efficientnet_b2_skin_disease_best.pth' olarak kaydedildi.", flush=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, classes, class_counts = get_train_loader()
    class_weights = torch.FloatTensor(1. / np.array(class_counts)).to(device)
    model = get_model(num_classes=len(classes))
    train_model(model, train_loader, device, epochs=15, class_weights=class_weights, patience=3)
