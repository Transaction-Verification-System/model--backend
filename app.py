from validators.CreditFraudInput import CreditFraudInput
from validators.EcommerceFraud import TransactionData
from validators.BankingFraudGBMInput import FraudDetectionInput
from validators.AntiMoneyLaunderingInput import AntiMoneyLaunderingInput

from utils.preprocessAML import process_model_input
from utils.extractJson import extract_json
from fastapi import FastAPI
import joblib
import pandas as pd
import json


app = FastAPI()

credit_model = joblib.load('models/CreditFraudModel.joblib')
banking_model = joblib.load('models/BankingFraudModel.joblib')
ecommerce_model= joblib.load('models/logistic_regression_model.pkl')
banking_fraud_model = joblib.load('models/lightgbm_model.pkl')
lgbm_preprocessor = joblib.load('models/lgbm_preprocessor.pkl')
aml_model = joblib.load('models/AML.joblib')
aml_label_encoder = joblib.load('models/AML_encoder.pkl')
aml_scaler = joblib.load('models/AML_scaler.pkl')

@app.get("/")
def root():
    return {"message": "Hello from TIVS --model backend"}



@app.post('/aml/predict')
def predict_aml(input: AntiMoneyLaunderingInput):
    try:
        # Preprocess the input data
        preprocessed_data = process_model_input(input, aml_label_encoder, aml_scaler)


        print(f'-------\npreprocessed data: {preprocessed_data}---\n')
        
        # Ensure the preprocessed data is a 2D array
        if preprocessed_data.ndim == 1:
            preprocessed_data = preprocessed_data.reshape(1, -1)


        print(f'preprocessed data: {preprocessed_data}')
        
        # Make predictions
        prediction = aml_model.predict(preprocessed_data)

        return {
            'status': 'success',
            'model': 'anti-money-laundering',
            'isLaundering': int(prediction[0])
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }



@app.post('/credit-fraud/predict')
def predict_fraud(credit_card_input: CreditFraudInput):
    try:
        values = extract_json(credit_card_input)
        prediction = credit_model.predict([values])

        return {
            'status': 'success',
            'model': 'credit-card-fraud-detection',
            'isFraud': bool(prediction[0])
        }
    
    except Exception as e:
        return {
            'status': 'success',
            'error': str(e)
        }

@app.post('/ecommerce_fraud/predict')
def predict(transaction: TransactionData):
    try:
        # Extract the input data from the transaction object
        values = extract_json(transaction)
        
        # Make a prediction using the ecommerce fraud model
        prediction = ecommerce_model.predict([values])
        probability = ecommerce_model.predict_proba([values])
        
        # Determine if the transaction is considered fraudulent
        is_fraud = bool(prediction[0])

        return {
            'status': 'success',
            'prediction': int(prediction[0]),  # This is the raw model output (0 or 1)
            'probability': probability[0][int(prediction[0])],  # The probability of the predicted class
            'isFraud': is_fraud  # Boolean indicating if the transaction is fraudulent
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@app.post('/banking-fraud-gbm/predict')
def predict_new_data(request: FraudDetectionInput):
    try:
        # Convert the input request to a DataFrame
        user_data = request.json()
        data_dict = json.loads(user_data)
        user_data_df = pd.DataFrame([data_dict])

        # Preprocess the user data using the loaded preprocessor
        user_data_processed = lgbm_preprocessor.transform(user_data_df)

        # Make probability predictions on the preprocessed user data
        y_pred_new_proba = banking_fraud_model.predict(user_data_processed, num_iteration=banking_fraud_model.best_iteration)

        # Define the threshold for determining fraud
        threshold = 0.5

        # Convert probabilities to binary outcomes (0 or 1)
        y_pred_new = (y_pred_new_proba >= threshold).astype(int)

        # Determine if the transaction is considered fraudulent
        is_fraud = bool(y_pred_new[0])

        return {
            'status': 'success',
            'probabilities': y_pred_new_proba.tolist(),
            'predictions': y_pred_new.tolist(),
            'isFraud': is_fraud
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


   



# @app.post('/banking-fraud/predict')
# async def predict_banking_fraud(banking_fraud_input: BankingFraudInput):
#     try: 
#         values = extract_json(banking_fraud_input)
#         data_array = preprocess_banking_data(values)
#         print(f'processed data: {data_array}')
#         prediction = await banking_model.predict(data_array)

#         print(f'prediction: {prediction}')

#         return {
#             'status': 'success',
#             'model': 'banking-fraud-detection',
#             'isFraud': bool(prediction[0])
#         }
#     except Exception as e:
#         return {
#             'status': 'success',
#             'error': str(e)
#         }
    