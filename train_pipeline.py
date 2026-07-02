import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import *
from src.feature_engineering import *

from src.classification import *
from src.regression import *
from src.clustering import *

# Load Dataset

df = load_data(
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/data/train.xlsx"
)

# Encoding

df, encoders = encode_features(df)

# Feature Engineering

session_df = create_session_features(df)

# Features

X = session_df.drop(
    [
        'session_id',
        'conversion',
        'revenue'
    ],
    axis=1
)

y_class = session_df['conversion']
y_reg = session_df['revenue']

# Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/scaler.pkl"
)

# Classification

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_class,
    test_size=0.2,
    random_state=42
)

clf = train_classifier(
    X_train,
    y_train
)

evaluate_classifier(
    clf,
    X_test,
    y_test
)

joblib.dump(
    clf,
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/classifier.pkl"
)

# Regression

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled,
    y_reg,
    test_size=0.2,
    random_state=42
)

reg = train_regressor(
    X_train_r,
    y_train_r
)

evaluate_regressor(
    reg,
    X_test_r,
    y_test_r
)

joblib.dump(
    reg,
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/regressor.pkl"
)

# Clustering

cluster_model, labels = train_cluster(
    X_scaled
)

evaluate_cluster(
    X_scaled,
    labels
)

joblib.dump(
    cluster_model,
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/cluster.pkl"
)

print("Training Completed")
