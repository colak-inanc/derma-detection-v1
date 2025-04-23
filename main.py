from src.dataloader import get_train_loader
from src.model import get_model
from src.train import train_model
import torch

train_loader, classes = get_train_loader()
model = get_model(num_classes=len(classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model, train_loader, device)

# Eğitim bittiğinde modeli kaydet
torch.save(model.state_dict(), "models/efficientnet_b2_skin_disease.pth")
