from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_excel('data-sets/Dataset/medical-data/cbc.xlsx')

# Gerekli sayısal sütunları al
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# KMeans kümeleme
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Metrikler
sil_score = silhouette_score(X_scaled, labels)
inertia = kmeans.inertia_

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Inertia: {inertia:.2f}")
