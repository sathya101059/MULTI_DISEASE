import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_kidney_data(filepath):
    df = pd.read_csv(filepath)

    # Drop 'id' if present
    df.drop(columns=['id'], errors='ignore', inplace=True)

    # Replace '?' with np.nan
    df.replace('?', np.nan, inplace=True)

    # Encode object columns BEFORE converting to numeric
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
        df[col] = df[col].replace(-1, np.nan)  # Unknown categories → np.nan

    # Now ensure all are numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Impute missing values with SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Split features & target
    X = df_imputed.drop('classification', axis=1)
    y = df_imputed['classification'].replace({0: 0, 1: 1, 2: 1})

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

