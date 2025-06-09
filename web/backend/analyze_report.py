import sys
import os

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.blur import process_pdf
from api.gemini_api import ask_gemini

import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import re

def analyze_report(pdf_path):
    # PDF'i işle
    processed_pdf = process_pdf(pdf_path)
    
    # PDF'ten metin çıkar
    doc = fitz.open(processed_pdf)
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Metni Gemini API'ye gönder
    analysis = ask_gemini(text)
    
    return analysis 