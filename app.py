from validators.CreditFraudInput import CreditFraudInput
from validators.BankingFraudInput import BankingFraudInput
from utils.extractJson import extract_json
from utils.preprocessBankData import preprocess_banking_data
from fastapi import FastAPI
import joblib
from pandas import json_normalize


app = FastAPI()

credit_model = joblib.load('models/CreditFraudModel.joblib')
banking_model = joblib.load('models/BankingFraudModel.joblib')


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