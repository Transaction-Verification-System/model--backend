from validators.AntiMoneyLaunderingInput import AntiMoneyLaunderingInput
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

def process_model_input(input_data: AntiMoneyLaunderingInput, transformer: Pipeline) -> np.ndarray:
    """
    Preprocess the input data using the transformer.

    Parameters:
    - input_data: An instance of AntiMoneyLaunderingInput containing the transaction details.
    - transformer: A pre-loaded transformer (pipeline) for data preprocessing.

    Returns:
    - preprocessed_data: A numpy array containing the transformed data, ready for model prediction.
    """
    # Convert input data to a dictionary
    data_dict = input_data.dict()
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data_dict])
    
    # Ensure the transformer handles unknown categories properly
    for step_name, step in transformer.named_transformers_.items():
        if isinstance(step, Pipeline):
            for inner_step_name, inner_step in step.named_steps.items():
                if isinstance(inner_step, OrdinalEncoder):
                    # Set parameters to handle unknown categories gracefully
                    inner_step.set_params(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Preprocess the input data using the transformer
    preprocessed_data = transformer.transform(df)
    
    return preprocessed_data
