from validators.CreditFraudInput import CreditFraudInput
from validators.BankingFraudInput import BankingFraudInput
from validators.EcommerceFraud import TransactionData
from validators.BankingFraudGBMInput import FraudDetectionInput
from utils.extractJson import extract_json
from utils.preprocessBankData import preprocess_banking_data
from fastapi import FastAPI
import joblib
import pandas as pd
from pandas import json_normalize
import json


app = FastAPI()

credit_model = joblib.load('models/CreditFraudModel.joblib')
banking_model = joblib.load('models/BankingFraudModel.joblib')
ecommerce_model= joblib.load('models/logistic_regression_model.pkl')
banking_fraud_model = joblib.load('models/lightgbm_model.pkl')
lgbm_preprocessor = joblib.load('models/lgbm_preprocessor.pkl')

@app.get("/")
def root():
    return {"message": "Hello from TIVS --model backend"}

@app.post('/credit-fraud/predict')
def predict_fraud(credit_card_input: CreditFraudInput):
    try:
        values = extract_json(credit_card_input)
        prediction = credit_model.predict(values)

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

@app.post('/banking-fraud/predict')
async def predict_banking_fraud(banking_fraud_input: BankingFraudInput):
    try: 
        values = extract_json(banking_fraud_input)
        data_array = preprocess_banking_data(values)
        print(f'processed data: {data_array}')
        prediction = await banking_model.predict(data_array)

        print(f'prediction: {prediction}')

        return {
            'status': 'success',
            'model': 'banking-fraud-detection',
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
        values = extract_json(transaction)
        prediction = ecommerce_model.predict(values)
        probability = ecommerce_model.predict_proba(values)
        return {
            'prediction': int(prediction[0]),
            'probability': probability[0][int(prediction[0])]
        }
    
    except Exception as e:
        return {
            'status': 'success',
            'error': str(e)
        }
    


@app.post('/banking-fraud-gbm/predict')
def predict_new_data(request:FraudDetectionInput):
   
    try:
        
        user_data = request.json()    
        data_dict = json.loads(user_data)    
        user_data_df = pd.DataFrame([data_dict])

        # Preprocess the user data
        user_data_processed = lgbm_preprocessor.transform(user_data_df)
        # Make probability predictions on the preprocessed user data
        y_pred_new_proba = banking_fraud_model.predict(user_data_processed, num_iteration=banking_fraud_model.best_iteration)
        
        # Convert probabilities to binary outcomes (0 or 1) based on a threshold (e.g., 0.5)
        threshold = 0.5
        y_pred_new = (y_pred_new_proba >= threshold).astype(int)
        
        return {
            'status': 'success',
            'probabilities': y_pred_new_proba.tolist(),
            'predictions': y_pred_new.tolist()
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

   