import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.blur import process_pdf
from api.gemini_api import ask_gemini

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
        if not os.path.exists(pdf_path):
            raise Exception(f"PDF dosyası bulunamadı: {pdf_path}")
            
        file_name = os.path.basename(pdf_path)
        name, ext = os.path.splitext(file_name)
        blurred_pdf = os.path.join(os.path.dirname(pdf_path), f"{name}_blurred{ext}")
        
        process_pdf(pdf_path, blurred_pdf)
        
        if not os.path.exists(blurred_pdf):
            raise Exception("Blurlama işlemi başarısız oldu!")
    
        try:
            pdf_text = extract_text_from_blurred_pdf(blurred_pdf)
            
            if not pdf_text.strip():
                raise Exception("PDF'den metin çıkarılamadı!")
                
        except Exception as e:
            raise Exception(f"PDF'den metin çıkarılırken hata oluştu: {str(e)}")
            
        prompt = f"""
        Bu bir tıbbi rapordur. Lütfen aşağıdaki içeriği analiz et ve özetle:

        {pdf_text}
        Lütfen aşağıdaki başlıklar altında yalnızca tıbbi değerlendirmeye odaklanarak bir özet hazırlayınız. Kişisel bilgiler bu analiz dışında tutulmalıdır.

        1. Genel Değerlendirme  
        2. Tespit Edilen Bulgular  
        3. Olası Risk Faktörleri  
        4. Klinik Öneriler
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
    from main import PDF_PATH
    result = analyze_medical_report(PDF_PATH)
    if result["status"] == "success":
        print(result["analysis"])
    else:
        print(f"Hata: {result['error']}")

if __name__ == "__main__":
    main()