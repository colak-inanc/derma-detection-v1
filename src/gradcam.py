import torch
from torchvision import transforms
from torchvision.models import efficientnet_b2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.model_targets import SoftmaxOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.image import deprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.model_targets import SoftmaxOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.find_layers import find_layer_types

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def run_gradcam(model, image_path, class_names, target_layer="features.7"):
    model.eval()

    # Görseli yükle ve normalizasyon
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

    # GradCAM hazırla
    target_layers = [getattr(model, target_layer.split('.')[0])[int(target_layer.split('.')[1])]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(0)]  # sınıf belirtmezsen en yüksek skorlu sınıfa bakar

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    img_rgb = img.resize((260, 260))
    img_np = (transforms.ToTensor()(img_rgb).permute(1, 2, 0).numpy())

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title("Grad-CAM Visualization")
    plt.axis("off")
    plt.show()
