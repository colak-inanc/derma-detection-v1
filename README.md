# Derma-Detection
Dermatologic Disease Detection Based on Visual and Medical Data

>[Scrum Tracking Board](https://miro.com/app/board/uXjVIMWBJzw=/)

## Project Details
1. <b>Moduls</b>
    1. <b>Diagnosis Based on Lab Test Data</b>
    - Analyzes test results such as blood or urine values using machine learning models.
    - Uses algorithms like decision trees, random forests, and XGBoost.
    - SHAP is applied to visualize feature importance and provide transparency.<br><br>

    2. <b>Dermatological Image Analysis</b>
    - Employs CNN-based models (e.g., ResNet50, EfficientNet) to detect skin conditions from images.
    - Grad-CAM is used to highlight important regions in the image.
    - Public datasets like HAM10000 and ISIC are used for model training.<br><br>

    3. <b>Hybrid Diagnosis System</b>
    - Accepts both an image and corresponding lab results (in PDF or tabular format) from the user.
    - Processes image and structured data separately and combines them for final disease prediction.
    - Mimics real-world scenarios where both visual and clinical data are used for diagnosis.<br><br><br>

2. <b>Technologies and Tools</b>

| Area                       | Libraries / Tools                      |
|----------------------------|----------------------------------------|
| Model Development          | scikit-learn, XGBoost, PyTorch         |
| Image Analysis             | EfficientNet-B0/B2, ResNet50, Grad-CAM |
| Tabular Data Processing    | pandas, SHAP, LightGBM                 |
| PDF Parsing                | pdfplumber, PyMuPDF                    |
| Explainability             | SHAP, Captum, Gemini(LLM)              |
| Interface (Demo)           | Streamlit                              |

<br><br>3. <b>Data sets for this project</b>
> Data sets to be used for Dermatologic Images
1. [Skin Cancer Malignant vs Benign Dataset](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
2. [Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
3. [Skin Disease Dataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset/data)
4. [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
5. [ISIC (International Skin Imaging Collaboration) Archive](https://api.isic-archive.com/collections/?pinned=true)

> Data sets to be used for testing or in case of insufficient dermatological data 
1. [Skin Disease Text Classification](https://www.kaggle.com/datasets/rafsunahmad/skin-disease-text-classification)
2. [20 Skin Diseases Dataset](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset)
3. [Skin Disease Classification Dataset](https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes)

> Data sets to be used for medical data
1. [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. [Health Test By Blood Dataset](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset)
3. [Disease Diagnosis Dataset](https://www.kaggle.com/datasets/s3programmer/disease-diagnosis-dataset)
4. [Medical Information Mart for Intensive Care](https://mimic.mit.edu)
5. [Complete Blood Count (CBC) Test](https://www.kaggle.com/datasets/ahmedelsayedtaha/complete-blood-count-cbc-test)

> ⚠️ **Note:** Due to data privacy regulations such as KVKK and HIPAA, publicly available datasets that include detailed blood test results and other biochemical values are limited. Existing datasets are often focused on a single disease (e.g., diabetes) or lack proper labeling and sufficient biochemical depth.  
>
> To overcome this limitation, **a large language model (LLM) such as Gemini** will be integrated into the system as a natural language interpretation layer. This component will help explain the predictions of the medical data model in human-friendly terms and compensate for potential data gaps by offering medically-informed reasoning and suggestions.

## <br><br>Example Use Case
1. The patient uploads an image of a wound on his/her arm and adds a PDF of a blood test.
2. The system extracts data from the PDF (e.g. glucose level, CRP, WBC, etc.).
3. With Image ResNet, the analysis data is analyzed separately with XGBoost.
4. The results are combined: “This wound may be associated with high glucose → high probability of diabetic wound.”
5. The Gemini model presents this result to the user in an explanatory language.