import streamlit as st
import pandas as pd
import joblib
from src.feature_engineering import create_features

st.title("E-Commerce Customer Intelligence")

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
cluster_model = joblib.load("models/clustering.pkl")
scaler = joblib.load("models/scaler.pkl")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = create_features(df)

    st.write("Input Data", df.head())

    # Predictions
    conversion = clf.predict(df)
    revenue = reg.predict(df)

    # Clustering
    X_cluster = df.select_dtypes(include=["int64", "float64"])
    X_scaled = scaler.transform(X_cluster)
    clusters = cluster_model.predict(X_scaled)

    df["Conversion"] = conversion
    df["Revenue"] = revenue
    df["Segment"] = clusters

    st.write("Results", df.head())
    st.bar_chart(df["Segment"].value_counts())
