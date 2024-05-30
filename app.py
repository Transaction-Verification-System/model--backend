from CreditCardInput import CreditCardInput
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('credit_fraud_model.joblib')


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post('/credit-fraud/predict')
def predict_fraud(credit_card_input: CreditCardInput):

    try:

        values = [getattr(credit_card_input, attr) for attr in vars(credit_card_input)]

        data_array = np.array(values)


        prediction = model.predict([data_array])

        return {
            'fraud': bool(prediction[0])
        }
    
    except Exception as e:
        return {
            'error': str(e)
        }
