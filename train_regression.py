import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from preprocessing import get_preprocessor
from feature_engineering import create_features

df = pd.read_csv("data/train.csv")
df = create_features(df)

target = "revenue"
X = df.drop(columns=[target])
y = df[target]

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = get_preprocessor(num_features, cat_features)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor())
])

pipeline.fit(X, y)

joblib.dump(pipeline, "models/regressor.pkl")
print("Regressor saved")
