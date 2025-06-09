import os
import traceback
import torch
import cv2
import numpy as np
from datetime import datetime

from analyze_image import (
    load_trained_model,
    load_image,
    predict_topk,
    generate_gradcam,
    show_cam_on_image
)

from analyze_report import analyze_medical_report

# === Dosya Yolları ve Ayarlar ===
MODEL_PATH = "models/efficientnet_b4_skin_best.pth"
IMAGE_PATH = "src/deneme.jpg"
PDF_PATH = "src/analysis-pdf/tahlil.pdf"
GRADCAM_SAVE_DIR = "src/gradcam-detect"

CLASS_NAMES = [
    'acne_inflammatory',
    'autoimmune_chronic',
    'benign_lesions',
    'fungal_infectious',
    'non_melanoma_pigmented',
    'skin_cancer_precancerous'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_gradcam_image(original_img, cam, class_name):
    """
    Grad-CAM görselini kaydeder.
    """
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
    print(f"\nGrad-CAM görseli kaydedildi: {save_path}")


def run_image_analysis():
    """
    Görüntü tabanlı tıbbi analiz sürecini başlatır.
    """
    if not os.path.exists(IMAGE_PATH):
        print("Bilgi: Görsel dosyası bulunamadı. Görüntü analizi atlanıyor.")
        return

    print("\n--- Görsel Analizi Başlatıldı ---")
    try:
        model = load_trained_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=DEVICE)
        original_img, input_tensor = load_image(IMAGE_PATH)
        top_indices, top_probs = predict_topk(model, input_tensor, CLASS_NAMES, DEVICE)

        print("\nGrad-CAM hesaplanıyor. Lütfen bekleyin...")
        cam = generate_gradcam(model, input_tensor, top_indices[0].item(), DEVICE)
        
        # En yüksek olasılıklı sınıfı al
        predicted_class = CLASS_NAMES[top_indices[0].item()]
        
        # Grad-CAM görselini kaydet
        save_gradcam_image(original_img, cam, predicted_class)

        print("\nBilgi: Görsel analizi başarıyla tamamlandı.")

    except Exception as error:
        print("Hata: Görsel analizi sırasında beklenmeyen bir sorun oluştu.")
        traceback.print_exception(type(error), error, error.__traceback__)


def run_pdf_analysis():
    """
    PDF tabanlı tıbbi rapor analiz sürecini başlatır.
    """
    if not os.path.exists(PDF_PATH):
        print("Bilgi: PDF dosyası bulunamadı. Rapor analizi atlanıyor.")
        return

    print("\n--- PDF Rapor Analizi Başlatıldı ---")
    try:
        result = analyze_medical_report(PDF_PATH)

        if result["status"] == "success":
            print("\n--- Rapor Özeti ---\n")
            print(result["analysis"])
            print("\nBilgi: PDF raporu başarıyla analiz edildi.")
        else:
            print(f"Hata: Rapor analizi başarısız. Detay: {result['error']}")

    except Exception as error:
        print("Hata: PDF analizi sırasında beklenmeyen bir sorun oluştu.")
        traceback.print_exception(type(error), error, error.__traceback__)


def main():
    """
    Ana yürütücü fonksiyon. Tıbbi görsel ve PDF analizlerini sırasıyla çalıştırır.
    Eksik dosyalar göz önünde bulundurularak süreç dinamik yürütülür.
    """
    print("=" * 60)
    print("Tıbbi Görsel ve Rapor Analizi Başlatılıyor")
    print("=" * 60)

    run_image_analysis()
    run_pdf_analysis()

    print("\n" + "=" * 60)
    print("Bilgi: Tüm analiz süreçleri tamamlandı.")
    print("=" * 60)


if __name__ == "__main__":
    main()
