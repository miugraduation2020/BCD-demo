import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess(data):
    X = data.drop([784], axis=1)
    X = X.to_numpy()
    X = X / 255

    ohe = OneHotEncoder(categories='auto')

    y = data[784]
    y = y.to_numpy()
    y = np.reshape(y, (-1, 1))
    y = ohe.fit_transform(y).toarray()

    return X, y