from blur import process_pdf
from gemini_api import ask_gemini
import os
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import re

# Tesseract OCR yolunu ayarla
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def format_analysis_output(analysis_text):
    """
    Gemini'den gelen analiz metnini sade, başlıkları büyük harfli ve aralıklı bir formata çevirir. Başlıktan hemen sonra bir alt satıra geçer.
    """
    # Yıldızları ve madde işaretlerini temizle
    text = re.sub(r'\*\*(.*?)\*\*:?\s*', r'\1', analysis_text)
    text = re.sub(r'\*\s*(.*?)\s*\*', r'• \1', text)
    
    lines = text.split('\n')
    formatted = []
    for line in lines:
        line = line.strip()
        if re.match(r'^(\d+\.|ÖNEMLİ NOT)', line, re.IGNORECASE):
            title = re.sub(r'^(\d+\.|ÖNEMLİ NOT)(.*)', lambda m: m.group(0).upper(), line)
            formatted.append(f"\n{title}")
            formatted.append("\n")
        elif line.startswith('•'):
            formatted.append(f"{line}")
        elif line:
            formatted.append(f"{line}\n")
    return '\n'.join([l for l in formatted if l.strip() or l == ''])

def extract_text_from_blurred_pdf(pdf_path):
    """
    Blurlanmış PDF'den OCR kullanarak metin çıkarır
    """
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page_num, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # RGBA ise RGB'ye çevir
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        text = pytesseract.image_to_string(gray, lang='tur')
        all_text.append(text)
        
    doc.close()
    return "\n".join(all_text)

def analyze_medical_report(pdf_path: str) -> dict:
    """
    Tıbbi rapor analiz süreci:
    1. PDF'deki kişisel bilgileri blurlar
    2. Blurlanmış PDF'i Gemini'ye gönderir ve analiz ettirir
    
    Args:
        pdf_path (str): Tıbbi rapor PDF'inin yolu
    
    Returns:
        dict: Analiz sonuçları
    """
    try:
        print("Veri Güvenliği Sağlanıyor...")
        if not os.path.exists(pdf_path):
            raise Exception(f"PDF dosyası bulunamadı: {pdf_path}")
            
        file_name = os.path.basename(pdf_path)
        name, ext = os.path.splitext(file_name)
        blurred_pdf = os.path.join(os.path.dirname(pdf_path), f"{name}_blurred{ext}")
        
        process_pdf(pdf_path, blurred_pdf)
        
        if not os.path.exists(blurred_pdf):
            raise Exception("Blurlama işlemi başarısız oldu!")
    
        print("\nVeriler Yorumlanmaya Hazırlanıyor...")
        try:
            pdf_text = extract_text_from_blurred_pdf(blurred_pdf)
            
            if not pdf_text.strip():
                print("Uyarı: PDF'den çıkarılan metin boş!")
                raise Exception("PDF'den metin çıkarılamadı!")
                
        except Exception as e:
            print(f"PDF metin çıkarma hatası: {str(e)}")
            raise Exception(f"PDF'den metin çıkarılırken hata oluştu: {str(e)}")
            
        print("\nRapor Analiz Ediliyor...")
        prompt = f"""
        Bu bir tıbbi rapordur. Lütfen aşağıdaki içeriği analiz et ve özetle:

        {pdf_text}

        Lütfen şu başlıklar altında özetle:
        1. Genel Değerlendirme
        2. Önemli Bulgular
        3. Risk Faktörleri
        4. Öneriler

        Not: Lütfen sadece tıbbi içeriğe odaklanın ve kişisel bilgileri göz ardı edin.
        """
        
        analysis = ask_gemini(prompt)
        formatted_analysis = format_analysis_output(analysis)
        
        return {
            "status": "success",
            "blurred_pdf_path": blurred_pdf,
            "analysis": formatted_analysis
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    pdf_path = "../../src/tahlil.pdf"
    print("\nTıbbi Rapor Analiz Süreci Başlatılıyor...")
    print("=" * 50)
    
    result = analyze_medical_report(pdf_path)
    
    print("\n" + "=" * 50)
    if result["status"] == "success":
        print("\nAnaliz Sonucu:")
        print(result["analysis"])
    else:
        print(f"\nHata: {result['error']}")

if __name__ == "__main__":
    main()