import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_liver_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Replace target: 1 = liver patient, 2 = not liver patient → make 2 as 0
    df['Dataset'] = df['Dataset'].replace({2: 0})

    # Fill missing numeric values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode Gender: Male=1, Female=0
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

    # Features & target
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
