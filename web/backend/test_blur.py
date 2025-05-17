import os
from blur import process_pdf

def test_blur_process():
    """
    Blur işlemini test eden fonksiyon
    """
    # Test dosyası yolu
    test_pdf = "src/tahlil.pdf"
    
    print("Blur Testi Başlatılıyor...")
    print("=" * 50)
    
    try:
        # 1. PDF dosyasının varlığını kontrol et
        print("1. PDF dosyası kontrol ediliyor...")
        if not os.path.exists(test_pdf):
            raise Exception(f"Test PDF dosyası bulunamadı: {test_pdf}")
        print("✓ PDF dosyası mevcut")
        
        # 2. Blurlanmış PDF için yeni dosya adı oluştur
        file_name = os.path.basename(test_pdf)
        name, ext = os.path.splitext(file_name)
        blurred_pdf = os.path.join(os.path.dirname(test_pdf), f"{name}_blurred{ext}")
        
        # 3. Blur işlemini uygula
        print("\n2. Blur işlemi uygulanıyor...")
        process_pdf(test_pdf, blurred_pdf)
        
        # 4. Blurlanmış PDF'in oluşturulduğunu kontrol et
        print("\n3. Sonuçlar kontrol ediliyor...")
        if not os.path.exists(blurred_pdf):
            raise Exception("Blurlanmış PDF oluşturulamadı!")
            
        # 5. Dosya boyutlarını karşılaştır
        original_size = os.path.getsize(test_pdf)
        blurred_size = os.path.getsize(blurred_pdf)
        
        print(f"\nTest Sonuçları:")
        print(f"✓ Orijinal PDF boyutu: {original_size / 1024:.2f} KB")
        print(f"✓ Blurlanmış PDF boyutu: {blurred_size / 1024:.2f} KB")
        print(f"✓ Blurlanmış PDF kaydedildi: {blurred_pdf}")
        
        print("\nTest başarıyla tamamlandı! ✓")
        
    except Exception as e:
        print(f"\nTest başarısız! ❌")
        print(f"Hata: {str(e)}")

if __name__ == "__main__":
    test_blur_process() 