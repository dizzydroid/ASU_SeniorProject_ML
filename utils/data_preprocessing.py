# utils/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(data/data.csv)
    return data

def preprocess_data(df):
    """
    Preprocess the data:
    - Handle categorical variables
    - Encode labels
    - Feature scaling
    """
    # Drop irrelevant columns if any (e.g., 'Location' if not needed)
    # df = df.drop(['Location'], axis=1)

    # Handle categorical variables using Label Encoding or One-Hot Encoding
    categorical_cols = ['Country', 'Location', 'Age', 'Gender', 'Visited_Wuhan', 'From_Wuhan']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features and target
    X = df.drop('Result', axis=1)
    y = df['Result']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the data into training, validation, and testing sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
