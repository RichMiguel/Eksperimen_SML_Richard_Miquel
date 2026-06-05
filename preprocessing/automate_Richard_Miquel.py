import pandas as pd

def encode_categorical(df):
    """
    One Hot Encoding kolom kategorikal
    """

    return pd.get_dummies(
        df,
        columns=['Neighborhood'],
        dtype=int
    )
    
def remove_outliers_iqr(df, columns):
    """
    Menghapus outlier menggunakan metode IQR
    """

    df_clean = df.copy()

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_clean = df_clean[
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ]

    return df_clean

def preprocess_data(df):
    """
    Pipeline preprocessing lengkap
    """

    numeric_cols = [
        'SquareFeet',
        'Bedrooms',
        'Bathrooms',
        'YearBuilt',
        'Price'
    ]

    df = remove_outliers_iqr(df, numeric_cols)

    df = encode_categorical(df)

    X = df.drop('Price', axis=1)
    y = df['Price']

    return X, y