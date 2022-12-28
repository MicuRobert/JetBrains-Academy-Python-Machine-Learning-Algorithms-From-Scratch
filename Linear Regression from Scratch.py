# write your code here
import numpy
import pandas as pd
import numpy as np

df = pd.DataFrame(
    {   'x': [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0],
        'w': [1, -3, 2, 5, 0, 3, 6],
        'z': [11, 15, 12, 9, 18, 13, 16],
        'y': [33, 42, 45, 51, 53, 61, 62]
    })


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit(self, X, y):
        if self.fit_intercept:
            model_coef = np.dot(np.dot(np.linalg.inv(np.dot(numpy.transpose(X), X)), numpy.transpose(X)), y)
            self.intercept = model_coef[0]
            self.coefficient = model_coef[1:]
            return{ 'Intercept' : self.intercept, 'Coefficient' : self.coefficient }
        else:
            model_coef = np.dot(np.dot(np.linalg.inv(np.dot(numpy.transpose(X), X)), numpy.transpose(X)), y)
            self.intercept = 0
            self.coefficient = model_coef
            pass

    def predict(self, X):
        return np.dot(X, self.coefficient)


reg = CustomLinearRegression(fit_intercept=False)
reg.fit(df[['x', 'w', 'z']], df['y'])
y_pred = reg.predict(df[['x', 'w', 'z']])
print(y_pred)
