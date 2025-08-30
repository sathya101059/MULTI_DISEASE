import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_parkinsons_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Drop 'name' column — it's just an identifier
    df.drop('name', axis=1, inplace=True)

    # Features & target
    X = df.drop('status', axis=1)
    y = df['status']

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
