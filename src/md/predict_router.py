import joblib
import pandas as pd

# Hastalık tahmini (multi-class classification)
def predict_disease(input_dict):
    try:
        model = joblib.load('models/disease_model.pkl')
        df = pd.DataFrame([input_dict])
        return model.predict(df)[0]
    except Exception as e:
        return f"[HATA - Disease]: {str(e)}"

# Diyabet tahmini (binary classification)
def predict_diabetes(input_dict):
    try:
        model = joblib.load('models/diabetes_model.pkl')
        df = pd.DataFrame([input_dict])
        return model.predict(df)[0]
    except Exception as e:
        return f"[HATA - Diabetes]: {str(e)}"

# CBC kümeleme sonucu (clustering)
def predict_cbc(input_dict):
    try:
        model = joblib.load('models/cbc_model.pkl')
        scaler = joblib.load('models/cbc_scaler.pkl')
        df = pd.DataFrame([input_dict])
        df_scaled = scaler.transform(df)
        return int(model.predict(df_scaled)[0])
    except Exception as e:
        return f"[HATA - CBC]: {str(e)}"
