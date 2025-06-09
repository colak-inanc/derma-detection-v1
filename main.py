import os
import traceback
import torch
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO

from analyze_image import (
    load_trained_model,
    load_image,
    predict_topk,
    generate_gradcam
)

from analyze_report import analyze_medical_report
from api.gemini_api import ask_gemini

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

def image_to_base64(image):
    """PIL Image'ı base64 formatına dönüştürür."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def save_gradcam_image(original_img, cam, class_name):
    """Grad-CAM görselini kaydeder ve base64 formatında döndürür."""
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
    
    # Grad-CAM görselini PIL Image'a dönüştür
    cam_pil = Image.fromarray(cv2.cvtColor(np.uint8(255 * cam_img), cv2.COLOR_BGR2RGB))
    return save_path, image_to_base64(cam_pil)

def run_image_analysis():
    """Görüntü tabanlı tıbbi analiz sürecini başlatır."""
    if not os.path.exists(IMAGE_PATH):
        return None

    try:
        model = load_trained_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=DEVICE)
        original_img, input_tensor = load_image(IMAGE_PATH)
        top_indices, top_probs = predict_topk(model, input_tensor, CLASS_NAMES, DEVICE)
        cam = generate_gradcam(model, input_tensor, top_indices[0].item(), DEVICE)
        
        predicted_class = CLASS_NAMES[top_indices[0].item()]
        gradcam_path, gradcam_base64 = save_gradcam_image(original_img, cam, predicted_class)
        original_base64 = image_to_base64(original_img)
        
        return {
            "status": "success",
            "predictions": [
                {"class": CLASS_NAMES[idx], "probability": float(prob)} 
                for idx, prob in zip(top_indices, top_probs)
            ],
            "original_image": original_base64,
            "gradcam_image": gradcam_base64,
            "gradcam_path": gradcam_path
        }

    except Exception as error:
        return None

def generate_medical_report(image_analysis=None, pdf_analysis=None):
    """Görüntü ve PDF analizlerini birleştirerek kapsamlı bir tıbbi rapor oluşturur."""
    if not (image_analysis and image_analysis["status"] == "success" and 
            pdf_analysis and pdf_analysis["status"] == "success"):
        return "Hata: Analiz için tüm veriler eksiksiz olmalıdır."

    prompt = f"""
    Aşağıda bir hastaya ait görsel tabanlı teşhis analizi ve laboratuvar raporu özetlenmiştir. Lütfen bu iki veri kümesini dikkatle inceleyiniz ve aralarındaki olası ilişkileri değerlendirerek kapsamlı, açıklayıcı ve klinik açıdan anlamlı bir tıbbi rapor oluşturunuz.

    GÖRÜNTÜ ANALİZİ SONUÇLARI:
    - En Olası Teşhis: {image_analysis['predictions'][0]['class']} (%{image_analysis['predictions'][0]['probability']:.2f})
    - Diğer Olası Teşhisler:
    {chr(10).join([f'  * {pred["class"]}: %{pred["probability"]:.2f}' for pred in image_analysis['predictions'][1:]])}

    LABORATUVAR RAPORU ANALİZİ:
    {pdf_analysis['analysis']}

    Beklentiler:
    - Görüntü analizi ile laboratuvar bulguları birbirini destekliyor mu, çelişiyor mu, yoksa birbirini tamamlayıcı nitelikte mi?
    - Her bir bulgu, klinik açıdan neden-sonuç ilişkisi içerisinde değerlendirilmelidir.
    - Görsel ve biyokimyasal verilerin birleştirilerek bütünsel bir değerlendirme sunulması beklenmektedir.

    Aşağıdaki başlıklar doğrultusunda yapılandırılmış, hasta odaklı bir tıbbi rapor hazırlayınız:

    1. GENEL DEĞERLENDİRME
       - Görsel analiz ve laboratuvar bulgularının genel özeti
       - Hastanın mevcut durumu hakkında bütüncül değerlendirme

    2. TEŞHİSLER VE KLİNİK UYUMLULUK
       - Görüntü analizinden elde edilen teşhis ve olasılıkların değerlendirilmesi
       - Laboratuvar bulgularının bu teşhislerle uyumu, çelişkisi veya tamamlayıcılığı
       - Klinik olarak anlamlı senaryoların oluşturulması

    3. DETAYLI KLİNİK BULGULAR
       - Görsel verilerdeki öne çıkan bulguların açıklanması
       - Laboratuvar sonuçlarında dikkat çeken değerlerin yorumlanması
       - İki veri kümesi arasındaki klinik bağlantıların vurgulanması

    4. TEDAVİ YAKLAŞIMLARI VE ÖNERİLER
       - Uygun görülen tedavi seçenekleri
       - İzlem ve kontrol önerileri
       - Gerekli görülen yaşam tarzı düzenlemeleri

    5. RİSKLER, UYARILAR VE TAKİP PLANI
       - Dikkat edilmesi gereken klinik durumlar
       - Göz ardı edilmemesi gereken belirtiler
       - Hekime başvuru gerektiren senaryolar ve önerilen kontrol zamanlaması

    Yazım kuralları:
    - Rapor, hasta için anlaşılır ve sade bir dille yazılmalıdır.
    - Teknik terimler sadeleştirilerek açıklanmalı, gerekirse örneklerle desteklenmelidir.
    - Görsel ve laboratuvar bulguları arasındaki ilişkiler açık ve mantıklı bir şekilde kurulmalıdır.
    - Bulgular bilimsel doğruluk çerçevesinde tutarlı şekilde yorumlanmalıdır.
    """

    return ask_gemini(prompt)


def main():
    """Ana yürütücü fonksiyon."""
    # Görüntü analizi
    image_analysis = None
    if os.path.exists(IMAGE_PATH):
        image_analysis = run_image_analysis()
    
    # PDF analizi
    pdf_analysis = None
    if os.path.exists(PDF_PATH):
        pdf_analysis = analyze_medical_report(PDF_PATH)
    
    # Her iki analiz de başarılı ise rapor oluştur
    if (image_analysis and image_analysis["status"] == "success") and \
       (pdf_analysis and pdf_analysis["status"] == "success"):
        final_report = generate_medical_report(image_analysis, pdf_analysis)
        print("\nLABORATUVAR RAPORU ANALİZİ:")
        print("=" * 60)
        print(final_report)
        print("=" * 60)
    else:
        print("\nHata: Analiz için tüm veriler (görüntü ve laboratuvar raporu) eksiksiz olmalıdır.")

if __name__ == "__main__":
    main() 