import pandas as pd
import numpy as np

def process_model_input(input_json, label_encoders, scaler, expected_num_features=105):
    # Convert input JSON to DataFrame
    df = pd.DataFrame([input_json.dict()])

    # Remove 'Is_laundering' if present
    if 'Is_laundering' in df.columns:
        df = df.drop(columns=['Is_laundering'])

    # Apply label encoding, handling unseen labels
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)
        else:
            df[col] = -1  # Assign -1 for missing categorical columns

    # Convert 'Time' and 'Date' to numeric formats
    def time_string_to_seconds(time_str):
        try:
            parts = list(map(int, time_str.split(':')))
            return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else 0
        except (ValueError, IndexError):
            return 0

    if 'Time' in df.columns:
        df['Time'] = df['Time'].apply(time_string_to_seconds)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').apply(lambda x: x.toordinal() if pd.notnull(x) else 0)

    # Define and ensure all required features are present
    feature_order = ['Time', 'Date', 'Sender_account', 'Receiver_account', 'Amount',
                     'Payment_currency', 'Received_currency', 'Sender_bank_location',
                     'Receiver_bank_location', 'Payment_type', 'Laundering_type']
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with a default value (e.g., 0)

    # Select the columns in the correct order and ensure all data in X is numeric
    X = df[feature_order].values

    # Normalize feature values using the provided scaler
    X = scaler.transform(X)

    # Ensure the number of features matches the model's expected input
    num_features_missing = expected_num_features - X.shape[1]
    if num_features_missing > 0:
        # If there are missing features, add columns of zeros
        X = np.hstack([X, np.zeros((X.shape[0], num_features_missing))])

    return X