import pandas as pd
import numpy as np

X = np.random.RandomState(1104).normal(0,1, (15000, 20))

def standardize(data: np.array):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return np.divide((data - data_mean), data_std)


for alpha in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
    x = X.copy()
    x[:, 2] = (1-alpha)*x[:, 2] + (alpha)*x[:, 5]
    #x[:,5] = standardize(x[:,5])
    #x[:,2] = standardize(x[:,2])
    print(f'alpha: {alpha} corr: {np.corrcoef(x[:,2], x[:,5])[0][1]}')