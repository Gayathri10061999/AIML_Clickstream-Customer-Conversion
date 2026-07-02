import pandas as pd


def create_session_features(df):

    session_df = df.groupby(
        'session_id'
    ).agg({

        'page':'count',
        'price':'mean',
        'order':'max',
        'location':'nunique',
        'country':'first',
        'price_2':'first'

    }).reset_index()

    session_df.columns = [

        'session_id',
        'total_pages',
        'avg_price',
        'max_order',
        'unique_locations',
        'country',
        'price_category'

    ]

    session_df['conversion'] = (
        session_df['total_pages'] > 5
    ).astype(int)

    session_df['revenue'] = (
        session_df['avg_price']
        *
        session_df['total_pages']
    )

    return session_df
