import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import get_model  

def validate_model(model_path="models/efficientnet_b2_skin_disease.pth", data_dir="data-sets/Dataset/s-test"):
    transform = transforms.Compose([
        transforms.Resize((260, 260)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EK: Modeli sınıf sayısına göre oluştur (yüklenmeden önce)
    num_classes = len(val_dataset.classes)
    model = get_model(num_classes=num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\n✅ Validation/Test Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    validate_model()
