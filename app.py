import streamlit as st
import numpy as np
import joblib

clf = joblib.load(
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/classifier.pkl"
)

reg = joblib.load(
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/regressor.pkl"
)

cluster = joblib.load(
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/cluster.pkl"
)

scaler = joblib.load(
    "C:/Users/gayat/AppData/Local/Programs/Python/Python313/models/scaler.pkl"
)

st.title(
    "Customer Conversion Analysis"
)

total_pages = st.number_input(
    "Total Pages"
)

avg_price = st.number_input(
    "Average Price"
)

max_order = st.number_input(
    "Max Order"
)

unique_locations = st.number_input(
    "Unique Locations"
)

country = st.number_input(
    "Country"
)

price_category = st.number_input(
    "Price Category"
)

if st.button("Predict"):

    data = np.array([[
        total_pages,
        avg_price,
        max_order,
        unique_locations,
        country,
        price_category
    ]])

    data = scaler.transform(data)

    conversion = clf.predict(data)[0]

    revenue = reg.predict(data)[0]

    segment = cluster.predict(data)[0]

    st.subheader("Results")

    st.write(
        "Conversion:",
        conversion
    )

    st.write(
        "Revenue:",
        round(revenue, 2)
    )

    st.write(
        "Cluster:",
        segment
    )
