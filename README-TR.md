# Derma-Detection
Görsel ve Tıbbi Verilere Dayalı Dermatolojik Hastalık Tespiti

>[Scrum Takip Tahtamız](https://miro.com/app/board/uXjVIMWBJzw=/)

## Proje Detayları
1. <b>Modüller</b>
    1. <b>Laboratuvar Test Verilerine Dayalı Teşhis</b>
    - Kan veya idrar değerleri gibi test sonuçlarını makine öğrenmesi modelleriyle analiz eder.
    - Karar ağaçları, rastgele ormanlar ve XGBoost gibi algoritmalar kullanılır.
    - Özellik önemini görüntülemek ve şeffaflık sağlamak için SHAP uygulanır.<br><br>

    2. <b>Dermatolojik Görüntü Analizi</b>
    - Cilt koşullarını tespit etmek için CNN tabanlı modeller (ResNet50, EfficientNet vb.) kullanılır.
    - Grad-CAM, görüntünün önemli bölgelerini vurgulamak için uygulanır.
    - Model eğitimi için HAM10000 ve ISIC gibi kamuya açık veri kümesi kullanılır.<br><br>

    3. <b>Hibrit Teşhis Sistemi</b>
    - Kullanıcıdan hem görüntü hem de ilgili laboratuvar sonuçları (PDF veya tablo formatında) alır.
    - Görüntü ve yapılandırılmış verileri ayrı ayrı işler ve son hastalık tahmini için birleştirir.
    - Gerçek dünya senaryolarını taklit ederek hem görsel hem de klinik verileri kullanarak teşhis yapar.<br><br><br>

2. <b>Teknolojiler ve Araçlar</b>

| Alan                       | Kütüphane / Araç                   |
|----------------------------|------------------------------------|
| Model Geliştirme           | scikit-learn, XGBoost, PyTorch     |
| Görsel Analiz              | CNN, ResNet, Grad-CAM              |
| Tablosal Veri İşleme       | pandas, SHAP, LightGBM             |
| PDF Ayrıştırm              | pdfplumber, PyMuPDF                |
| Açıklanabilirlik           | SHAP, Captum, Gemini               |
| Arayüz(Demo)               | Streamlit                          |


<br><br>3. <b>Kullanılacak Veri Kümesi</b>
> Dermatolojik Görüntüler için kullanılacak veri setleri
1. [Skin Cancer Malignant vs Benign Dataset](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
2. [Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
3. [Skin Disease Dataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset/data)
4. [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
5. [ISIC (International Skin Imaging Collaboration) Archive](https://api.isic-archive.com/collections/?pinned=true)

> Dermatolojik verilerin yetersiz kalması durumunda veya test için kullanılacak veri setleri 
1. [Skin Disease Text Classification](https://www.kaggle.com/datasets/rafsunahmad/skin-disease-text-classification)
2. [20 Skin Diseases Dataset](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset)
3. [Skin Disease Classification Dataset](https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes)

> Tıbbi veriler için kullanılacak veri setleri
1. [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. [Health Test By Blood Dataset](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset)
3. [Disease Diagnosis Dataset](https://www.kaggle.com/datasets/s3programmer/disease-diagnosis-dataset)
4. [Medical Information Mart for Intensive Care](https://mimic.mit.edu)
5. [Complete Blood Count (CBC) Test](https://www.kaggle.com/datasets/ahmedelsayedtaha/complete-blood-count-cbc-test)

> ⚠️ **Not:** KVKK ve HIPAA gibi veri gizliliği düzenlemeleri nedeniyle, ayrıntılı kan testi sonuçlarını ve diğer biyokimyasal değerleri içeren halka açık veri kümeleri sınırlıdır. Mevcut veri setleri genellikle tek bir hastalığa (örneğin diyabet) odaklanmıştır veya uygun etiketleme ve yeterli biyokimyasal derinlikten yoksundur.  
>
> Bu sınırlamanın üstesinden gelmek için, **Gemini** gibi bir büyük dil modeli (LLM) doğal dil yorumlama katmanı olarak sisteme entegre edilecektir. Bu bileşen, tıbbi veri modelinin tahminlerini insan dostu terimlerle açıklamaya yardımcı olacak ve tıbbi olarak bilgilendirilmiş akıl yürütme ve öneriler sunarak potansiyel veri boşluklarını telafi edecektir.

## <br><br>Örnek Kullanım Senaryosu
1. Hasta, kolundaki bir yara görüntüsünü ve kan testine ait bir PDF belgesini yükler.
2. Sistem, PDF'den glikoz seviyesi, CRP, WBC gibi verileri çıkarır.
3. ResNet ile görüntü analiz edilir, XGBoost ile tablo verisi ayrı ayrı değerlendirilir.
4. Sonuçlar birleştirilir: "Bu yara yüksek glikoz ile ilişkili olabilir → diyabetik yara olasılığı yüksek."
5. Gemini modeli, bu sonucu kullanıcıya açıklayıcı bir dil ile sunar.

