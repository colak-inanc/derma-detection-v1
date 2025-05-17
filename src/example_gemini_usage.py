from gemini_helper import GeminiHelper

def main():
    # API anahtarınızı buraya girin veya GOOGLE_API_KEY çevre değişkenini ayarlayın
    api_key = "YOUR_API_KEY_HERE"
    
    # Gemini yardımcı sınıfını başlat
    gemini = GeminiHelper(api_key)
    
    # Metin üretme örneği
    text_prompt = "Deri hastalıkları hakkında kısa bir bilgi verir misin?"
    response = gemini.generate_text(text_prompt, temperature=0.7, max_tokens=1024)
    print("Metin Yanıtı:", response)
    
    # Sohbet örneği
    chat_messages = [
        {"role": "user", "content": "Merhaba, deri hastalıkları konusunda uzman mısın?"},
        {"role": "assistant", "content": "Evet, deri hastalıkları konusunda size yardımcı olabilirim."},
        {"role": "user", "content": "Egzama hakkında bilgi verir misin?"}
    ]
    
    chat_response = gemini.chat(chat_messages)
    print("\nSohbet Yanıtı:", chat_response)
    
    # Görüntü analizi örneği
    image_path = "path/to/your/image.jpg"
    image_prompt = "Bu deri lezyonunu analiz et ve olası durumları açıkla."
    try:
        image_analysis = gemini.analyze_image(image_path, image_prompt)
        print("\nGörüntü Analizi:", image_analysis)
    except Exception as e:
        print(f"Görüntü analizi sırasında hata oluştu: {e}")

if __name__ == "__main__":
    main() 