from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Create ColumnDropper class
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.drop(self.columns_to_drop,axis=1)
        return X_transformed