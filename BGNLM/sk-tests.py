import pandas as pd
import numpy as np

def maxabs(data: np.array):
    x = data.copy()
    if len(x.shape) > 1:
        for i in range(x.shape[1]):
            x[:,i] = np.divide(x[:,i], np.max(np.abs(x[:,i])))
    else:
        x = np.divide(x, np.max(np.abs(x)))
    return x



df = pd.read_csv('slice_localization.csv', header=0)

x_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1]

y_df = maxabs(y_df)



x_df = x_df.loc[:, x_df.std(axis=0)>0.01]

# Split:
x_train = x_df.sample(frac=0.8, random_state=1104)
y_train = y_df.sample(frac=0.8, random_state=1104)

x_test = x_df[~x_df.index.isin(x_train.index)]
y_test = y_df[~y_df.index.isin(y_train.index)]

x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators = 500, n_jobs =4).fit(x_train, y_train)
pred = reg.predict(x_test)

rmse = np.sqrt(np.mean((y_test-pred)**2))

print(rmse)