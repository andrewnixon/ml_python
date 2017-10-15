import numpy as np
import pandas as pd

def normalise_numerical_data(X, axis):
    return (X[:,axis] - X[:,axis].mean()) / X[:,axis].std()

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()

    X = data[:, :-1]
    Y = data[:, -1]

    # Normalise numerical columns
    X[:,1] = normalise_numerical_data(X,1)
    X[:,2] = normalise_numerical_data(X,2)

    # deal with catagorical data
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    # most of X is the same - just not catagorical data
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    # itterate over all rows and set new column based
    # on the content of the catagorical time column
    for n in xrange(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]

    return X2, Y2
