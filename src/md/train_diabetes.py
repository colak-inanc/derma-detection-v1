import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Veri setini yükle
df = pd.read_csv('data-sets/Dataset/medical-data/diabetes.csv')

# Gereksiz sütunları at (Unnamed: 0 varsa)
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)

# Tüm nesne (object) türündeki sütunları sayısal değerlere çevir (örneğin Gender gibi)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Özellikler ve hedef değişkeni ayır
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost modelini oluştur ve eğit
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Tahmin yap ve başarı oranını göster
y_pred = model.predict(X_test)
print("Diabetes Model Accuracy:", accuracy_score(y_test, y_pred))

# Eğitilen modeli kaydet
joblib.dump(model, 'models/diabetes_model.pkl')
