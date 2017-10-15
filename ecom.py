import numpy as np
import pandas as pd

def normalise_numerical_data(X, axis):
    return (X[:,axis] - X[:,axis].mean()) / X[:,axis].std()

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()

    print(df.head())

    X = data[:, :-1]
    Y = data[:, -1]

    # Normalise numerical columns
    X[:,1] = normalise_numerical_data(X,1)
    X[:,2] = normalise_numerical_data(X,2)


get_data()
