from PIL import Image
from PIL.Image import Resampling
import os

target_size = (260, 260)
input_base = "data-sets/Dataset"
output_base = "data-sets/Dataset"

# Hem train hem test klasörlerini işlemek için
subsets = ["train", "test"]

for subset in subsets:
    input_root = os.path.join(input_base, subset)
    output_root = os.path.join(output_base, f"s-{subset}")
    os.makedirs(output_root, exist_ok=True)

    for class_folder in os.listdir(input_root):
        input_class_path = os.path.join(input_root, class_folder)
        output_class_path = os.path.join(output_root, class_folder)

        if os.path.isdir(input_class_path):
            os.makedirs(output_class_path, exist_ok=True)

            for img_name in os.listdir(input_class_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        img_path = os.path.join(input_class_path, img_name)
                        img = Image.open(img_path)
                        img = img.resize(target_size, Resampling.LANCZOS)
                        img.save(os.path.join(output_class_path, img_name))
                    except Exception as e:
                        print(f"Hata: {subset}/{class_folder}/{img_name} dosyası işlenemedi. {e}")
