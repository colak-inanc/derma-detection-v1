import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from datetime import datetime

# === MODEL YÜKLE ===
def load_trained_model(model_path, num_classes=6, device='cuda' if torch.cuda.is_available() else 'cpu'):
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

# === GÖRSEL DÖNÜŞÜMÜ ===
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === GÖRSEL YÜKLE ===
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    return img, input_tensor

# === TAHMİN YAP ===
def predict_topk(model, input_tensor, class_names, device, topk=3):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, topk)

    return top_indices[0], top_probs[0]

# === GRAD-CAM ÜRET ===
def generate_gradcam(model, input_tensor, target_class, device):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    target_layer = model.features[-1][0]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor.to(device))
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    forward_handle.remove()
    backward_handle.remove()

    gradients_ = gradients[0]
    activations_ = activations[0]
    weights = torch.mean(gradients_, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations_, dim=1).squeeze()
    cam = np.maximum(cam.cpu().numpy(), 0)
    cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    return cam

# === CAM GÖSTER ===
def show_cam_on_image(original_img, cam, alpha=0.5):
    img = np.array(original_img.resize((380, 380))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    result = heatmap * alpha + img * (1 - alpha)

    plt.figure(figsize=(8, 8))
    plt.imshow(result)
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

def save_gradcam_image(original_img, cam, class_name):
    """Grad-CAM görselini kaydeder."""
    original_img_np = np.array(original_img)
    original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
    
    cam = cv2.resize(cam, (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(original_img_np) / 255
    cam_img = cam_img / np.max(cam_img)
    
    os.makedirs(GRADCAM_SAVE_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{class_name}_gradcam.jpg"
    save_path = os.path.join(GRADCAM_SAVE_DIR, filename)
    
    cv2.imwrite(save_path, np.uint8(255 * cam_img))
    return save_path

# === ANA FONKSİYON ===
if __name__ == "__main__":
    from main import MODEL_PATH, IMAGE_PATH, CLASS_NAMES, DEVICE

    model = load_trained_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=DEVICE)
    original_img, input_tensor = load_image(IMAGE_PATH)

    top_indices, top_probs = predict_topk(model, input_tensor, CLASS_NAMES, DEVICE)
    print("\nGrad-CAM Üretiliyor...")
    cam = generate_gradcam(model, input_tensor, top_indices[0].item(), DEVICE)
    show_cam_on_image(original_img, cam)
