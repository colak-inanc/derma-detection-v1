import requests
import os
from typing import Optional, List, Dict
import time

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY


def ask_gemini(prompt: str, temperature: float = 0.7, max_tokens: int = 2048, max_retries: int = 3) -> str:
    """
    Gemini 1.5 Flash modeline metin tabanlı soru sorar
    
    Args:
        prompt (str): Sorulacak soru veya istek
        temperature (float, optional): Yaratıcılık seviyesi (0.0 - 1.0). Varsayılan 0.7.
        max_tokens (int, optional): Maksimum token sayısı. Varsayılan 2048.
        max_retries (int, optional): Maksimum yeniden deneme sayısı. Varsayılan 3.
    
    Returns:
        str: Model yanıtı
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                json=data,
                timeout=30  # 30 saniye timeout
            )
            
            # HTTP durum kodunu kontrol et
            if response.status_code == 429:  # Rate limit
                wait_time = (attempt + 1) * 2
                print(f"Rate limit aşıldı. {wait_time} saniye bekleniyor...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "API'den geçerli bir yanıt alınamadı."
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Zaman aşımı. Yeniden deneniyor... (Deneme {attempt + 1}/{max_retries})")
                continue
            return "Sunucu yanıt vermedi. Lütfen daha sonra tekrar deneyin."
            
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"Bağlantı hatası. Yeniden deneniyor... (Deneme {attempt + 1}/{max_retries})")
                time.sleep(2)  # Bağlantı hatası durumunda 2 saniye bekle
                continue
            return "Sunucuya bağlanılamadı. Lütfen internet bağlantınızı kontrol edin."
            
        except requests.exceptions.RequestException as e:
            return f"API İsteği Hatası: {str(e)}"
            
        except Exception as e:
            return f"Beklenmeyen Hata: {str(e)}"
    
    return "Maksimum deneme sayısına ulaşıldı. Lütfen daha sonra tekrar deneyin."

def analyze_image(image_path: str, prompt: str) -> str:
    """
    Gemini 1.5 Flash Vision modeli ile görüntü analizi yapar
    
    Args:
        image_path (str): Analiz edilecek görüntünün dosya yolu
        prompt (str): Görüntü için analiz isteği
    
    Returns:
        str: Analiz sonucu
    """
    vision_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-vision:generateContent?key=" + GEMINI_API_KEY
    
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data.hex()
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(vision_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except FileNotFoundError:
        return "Görüntü dosyası bulunamadı."
    except requests.exceptions.RequestException as e:
        return f"API İsteği Hatası: {str(e)}"
    except Exception as e:
        return f"Beklenmeyen Hata: {str(e)}"

def chat(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """
    Gemini 1.5 Flash modeli ile sohbet benzeri etkileşim
    
    Args:
        messages (List[Dict[str, str]]): Mesaj geçmişi. Her mesaj 'role' ve 'content' içermelidir.
        temperature (float, optional): Yaratıcılık seviyesi. Varsayılan 0.7.
    
    Returns:
        str: Model yanıtı
    """
    headers = {"Content-Type": "application/json"}
    
    # Mesaj geçmişini API formatına dönüştür
    contents = []
    for message in messages:
        contents.append({
            "role": message["role"],
            "parts": [{"text": message["content"]}]
        })
    
    data = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"API İsteği Hatası: {str(e)}"
    except Exception as e:
        return f"Beklenmeyen Hata: {str(e)}"