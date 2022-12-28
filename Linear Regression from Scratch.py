# write your code here
import math

import numpy
import pandas as pd
import numpy as np

df = pd.DataFrame({
        'Capacity': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
        'Age': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
        'Cost/ton': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69],
})


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit(self, X, y):
        if self.fit_intercept:
            X.insert(loc=0, column='ones', value=[1 for _ in range(1, X.shape[0] + 1)])
            model_coef = np.dot(np.dot(np.linalg.inv(np.dot(numpy.transpose(X), X)), numpy.transpose(X)), y)
            self.intercept = model_coef[0]
            self.coefficient = model_coef[1:]
        else:
            model_coef = np.dot(np.dot(np.linalg.inv(np.dot(numpy.transpose(X), X)), numpy.transpose(X)), y)
            self.intercept = 0
            self.coefficient = model_coef
            pass

    def predict(self, X):
        if self.fit_intercept:
            X.insert(loc=0, column='ones', value=[1 for _ in range(1, X.shape[0]+1)])
            return np.dot(X, [self.intercept, *self.coefficient])
        else:
            return np.dot(X, self.coefficient)

    def r2_score(self, y, yhat):
        yhat_mean = np.mean(yhat)
        y_pairs = zip(y, yhat)
        return 1 - sum([(ys-yhats) ** 2 for ys, yhats in y_pairs]) / sum([(ys - yhat_mean) ** 2 for ys in y])

    def rmse(self, y, yhat):
        y_pairs = zip(y, yhat)
        mse = 1/y.shape[0] * sum([(ys - yhats) ** 2 for ys, yhats in y_pairs])
        return math.sqrt(mse)


reg = CustomLinearRegression(fit_intercept=True)
reg.fit(df[['Capacity', 'Age']], df['Cost/ton'])
y_pred = reg.predict(df[['Capacity', 'Age']])
r2 = reg.r2_score(df['Cost/ton'], y_pred)
rmse = reg.rmse(df['Cost/ton'], y_pred)
res = {
        'Intercept': reg.intercept,
        'Coefficient': reg.coefficient,
        'R2': r2,
        'RMSE': rmse
    }
print(res)
