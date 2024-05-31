import json
import numpy as np

def extract_json(json_input):
    try:
        data = json.loads(json_input.model_dump_json())
        attributes = [data[attr] for attr in data]
        return [np.array(attributes)]

    except Exception as e:
        print("Error:", e)
        return None