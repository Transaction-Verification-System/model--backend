import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_banking_data(data):
    try:
        print(f'Original Data: {data}')
        
        categorical_indices = [7, 14, 17, 24, 26]
        label_encoders = {}

        for index in categorical_indices:
            le = LabelEncoder()
            le.fit([data[index]])  
            data[index] = le.transform([data[index]])[0]
            label_encoders[index] = le

        print(f'Label Encoders: {label_encoders}')
        
        numeric_data = np.array(data, dtype=np.float32)

        expected_features = 54
        if len(numeric_data) < expected_features:
            numeric_data = np.pad(numeric_data, (0, expected_features - len(numeric_data)), 'constant')
        elif len(numeric_data) > expected_features:
            numeric_data = numeric_data[:expected_features]

        print(f'Numeric Data: {numeric_data}')

        return numeric_data  
    except Exception as e:
        print(f"Error: {e}")
        return None