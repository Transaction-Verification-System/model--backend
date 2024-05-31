from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_banking_data(data):
    imputer = SimpleImputer(missing_values = np.nan , strategy='mean')
    onehotencoder = OneHotEncoder(categories = 'auto')
    sc = StandardScaler(with_mean = False)

    data = onehotencoder.fit_transform(data)
    print(f'onehotencoder:',data)

    data = imputer.fit_transform(data)
    print(f'imputer:',data)

    data = sc.fit_transform(data)
    print(f'sc:',data)



    return data
