# write your code here
import numpy
import pandas as pd
import numpy as np

data = {
    'x' : [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0],
    'y' : [33, 42, 45, 51, 53, 61, 62]
}

df = pd.DataFrame(data)

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
        pass

X = pd.DataFrame({'ones': [1 for _ in range(1, df.shape[0]+1)], 'x': df['x'] })
model = CustomLinearRegression(fit_intercept=True)

print(model.fit(X, df['y']))
