import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

# Dosya yolunu tanımla
csv_path = 'data-sets/Dataset/medical-data/disease_diagnosis.csv'

# Veriyi yükle
df = pd.read_csv(csv_path)

# Tüm sütunları küçük harfe çevirerek karşılaştırma yap
df.columns = df.columns.str.strip().str.lower()

# Gerçek hedef sütunu kontrol et
if 'diagnosis' not in df.columns:
    raise ValueError("Hedef sütun 'diagnosis' veri setinde bulunamadı.")

# Hedef değişken ve giriş değişkenlerini ayır
y = df['diagnosis']
X = df.drop(columns=['diagnosis', 'treatment_plan'])  # Tedavi planı çıktıya katkı sağlamaz

# Kategorik sütunları label encode et
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Hedef değişkeni de encode et (çok sınıflı olabilir)
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Eğitim ve test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost modelini tanımla ve eğit
model = xgboost_model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Tahmin ve doğruluk
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Disease Model Accuracy: {acc:.4f}")

# Modeli kaydet
joblib.dump(model, 'models/disease_model.pkl')
