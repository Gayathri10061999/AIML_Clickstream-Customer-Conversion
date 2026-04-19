import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from feature_engineering import create_features

df = pd.read_csv("data/train.csv")
df = create_features(df)

X = df.select_dtypes(include=["int64", "float64"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

joblib.dump(kmeans, "models/clustering.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Clustering model saved")
