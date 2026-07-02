import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    return pd.read_excel(path)


def encode_features(df):

    categorical_cols = [
        'country',
        'page1_main_category',
        'page2_clothing_model',
        'colour',
        'location',
        'model_photography',
        'price_2'
    ]

    encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le

    return df, encoders
