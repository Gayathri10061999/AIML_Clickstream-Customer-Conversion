import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocessing import get_preprocessor
from feature_engineering import create_features

# Load data
df = pd.read_csv("data/train.csv")
df = create_features(df)

target = "conversion"
X = df.drop(columns=[target])
y = df[target]

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = get_preprocessor(num_features, cat_features)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(class_weight="balanced"))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "models/classifier.pkl")
print("Classifier saved")
