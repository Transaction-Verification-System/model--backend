import json

from validators.BankingFraudInput import BankingFraudInput

def extract_json(json_input: BankingFraudInput):
    try:
        data = json.loads(json_input.json())
        print(f'Data: {data}')

        attributes = [data[attr] for attr in data]
        print(f'Attributes: {attributes}')

        return attributes
    except Exception as e:
        print(f"Error: {e}")
        return None
