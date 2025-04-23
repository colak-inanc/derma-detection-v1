from torchvision.models import efficientnet_b2
import torch.nn as nn

def get_model(num_classes):
    model = efficientnet_b2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
