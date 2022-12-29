# write your code here
import math
import numpy
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data_stage4.csv', sep=',')

# CUSTOM SOLUTION
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
reg.fit(df[['f1', 'f2', 'f3']], df['y'])
y_pred = reg.predict(df[['f1', 'f2', 'f3']])
r2 = reg.r2_score(df['y'], y_pred)
rmse = reg.rmse(df['y'], y_pred)
res = {
        'Intercept': reg.intercept,
        'Coefficient': reg.coefficient,
        'R2': r2,
        'RMSE': rmse
    }

# SKLEARN SOLUTION
regSci = LinearRegression(fit_intercept=True)
regSci.fit(df[['f1', 'f2', 'f3']], df['y'])
y_pred_sklearn = regSci.predict(df[['f1', 'f2', 'f3']])
r2_sklearn = r2_score(df['y'], y_pred_sklearn)
mse_sklearn = mean_squared_error(df['y'], y_pred_sklearn)
rmse_sklearn = math.sqrt(mse_sklearn)

# COMPARE MODELS
final_results = {
        'Intercept': regSci.intercept_ - reg.intercept,
        'Coefficient': regSci.coef_ - reg.coefficient,
        'R2': r2_sklearn - r2,
        'RMSE': rmse_sklearn - rmse
}

print(final_results)
