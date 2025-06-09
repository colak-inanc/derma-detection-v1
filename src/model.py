import torch 
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

def get_model(num_classes):
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model
