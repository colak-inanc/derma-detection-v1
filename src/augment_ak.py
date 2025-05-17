import os
from PIL import Image, ImageOps, ImageEnhance
import random

# Kendi veri yolunu buraya yaz
CLASS_DIR = "data-sets/Dataset/s-train/actinic_keratosis_basal_cell_carcinoma_and_other_malignant_lesions"
TARGET_MULTIPLIER = 4  # Veriyi yaklaşık 4 katına çıkar

def augment_image(img):
    # PIL ile uyumlu augmentasyonlar
    ops = [
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(random.randint(-30, 30)),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.7, 1.3)),
    ]
    op = random.choice(ops)
    return op(img)

def main():
    files = [f for f in os.listdir(CLASS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    n_files = len(files)
    print(f"Orijinal görsel sayısı: {n_files}")

    # Kaç augmentasyon yapılacak?
    total_needed = n_files * (TARGET_MULTIPLIER - 1)
    per_image = (total_needed // n_files) + 1

    new_files = 0
    for idx, file in enumerate(files):
        img_path = os.path.join(CLASS_DIR, file)
        img = Image.open(img_path).convert("RGB")
        base_name = os.path.splitext(file)[0]
        for aug_idx in range(per_image):
            aug_img = augment_image(img)
            new_name = f"{base_name}_aug{aug_idx+1}_{idx}.jpg"
            save_path = os.path.join(CLASS_DIR, new_name)
            if not os.path.exists(save_path):
                aug_img.save(save_path)
                new_files += 1

    print(f"Augmentasyon tamamlandı! Yeni eklenen görsel sayısı: {new_files}")

if __name__ == "__main__":
    main()
