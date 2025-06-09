import cv2
import os
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import re

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

BLUR_KERNEL = (35, 35)
BLUR_SIGMA = 50

# Anahtar kelimeler için blur genişliği/yüksekliği
BLUR_EXTRA_W = 500
BLUR_EXTRA_H = 30

# Tarih için özel blur genişliği/yüksekliği
DATE_EXTRA_W = 30
DATE_EXTRA_H = 30

# Saat için özel blur genişliği/yüksekliği
TIME_EXTRA_W = 30
TIME_EXTRA_H = 30

KEYWORDS = ["Adı", "Soyadı", "Tarih"]
def blur_region(img, x, y, w, h):
    roi = img[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, BLUR_KERNEL, BLUR_SIGMA)
    img[y:y+h, x:x+w] = blurred
    return img

def find_and_blur_text(img, keywords=KEYWORDS):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang="tur")

    date_pattern = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")
    time_pattern = re.compile(r"\d{1,2}:\d{2}:\d{2}")

    for i, word in enumerate(data['text']):
        word_str = str(word)
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        if any(kw.lower() in word_str.lower() for kw in keywords):
            img = blur_region(
                img,
                max(x - 10, 0),
                max(y - 10, 0),
                w + BLUR_EXTRA_W,
                h + BLUR_EXTRA_H
            )
        elif date_pattern.fullmatch(word_str):
            img = blur_region(
                img,
                max(x - 10, 0),
                max(y - 10, 0),
                w + DATE_EXTRA_W,
                h + DATE_EXTRA_H
            )
        elif time_pattern.fullmatch(word_str):
            img = blur_region(
                img,
                max(x - 10, 0),
                max(y - 10, 0),
                w + TIME_EXTRA_W,
                h + TIME_EXTRA_H
            )
    return img

def find_and_blur_qr(img):
    qr = cv2.QRCodeDetector()
    data, points, _ = qr.detectAndDecode(img)
    if points is not None:
        points = points[0].astype(int)
        x, y, w, h = cv2.boundingRect(points)
        img = blur_region(img, x, y, w, h)
    return img

def pdf_page_to_image(page, dpi=300):
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).copy().reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def save_images_as_pdf(image_paths, output_pdf):
    new_doc = fitz.open()
    for path in image_paths:
        img_doc = fitz.open(path)
        pdf_bytes = fitz.open("pdf", img_doc.convert_to_pdf())
        new_doc.insert_pdf(pdf_bytes)
        img_doc.close()
        os.remove(path)
    new_doc.save(output_pdf)
    new_doc.close()

def process_pdf(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)
    output_images = []

    for i, page in enumerate(doc):
        img = pdf_page_to_image(page)
        # Tüm sayfalarda anahtar kelime, tarih ve saat blur
        img = find_and_blur_text(img, keywords=KEYWORDS)
        if i == len(doc) - 1:
            img = find_and_blur_qr(img)
        temp_path = f"_temp_page_{i}.png"
        cv2.imwrite(temp_path, img)
        output_images.append(temp_path)

    save_images_as_pdf(output_images, output_pdf)
    #print(f"İşlem tamamlandı. Çıktı: {output_pdf}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        input_pdf = sys.argv[1]
        output_pdf = sys.argv[2]
        process_pdf(input_pdf, output_pdf)
    else:
        print("Kullanım: python blur.py <girdi.pdf> <çıktı.pdf>")
