import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.model import get_model 

def validate_model(model_path="models/efficientnet_b2_skin_disease.pth", data_dir="data-sets/Dataset/s-test"):
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    val_dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    class_names = val_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(class_names)
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\n✅ Validation/Test Accuracy: {accuracy:.2f}%")

    print("\n📊 Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=1 
    ))


if __name__ == "__main__":
    validate_model()
