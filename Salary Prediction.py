import os
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

# dict for power and results predictor
pw_res = {
    2 : [],
    3 : [],
    4 : []
}

for power in pw_res.keys():
    # X a dataframe with a predictor rating and y a series with a target salary
    X, y = data[['rating']] ** power, data['salary']

    # split into training and test parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict model on test data and calc MAPE
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    pw_res[power] = [round(model.intercept_,5), round(model.coef_[0],5), round(mape,5)]

# print best results
print(pw_res[min(pw_res, key = lambda k: pw_res[k][2])][2])
