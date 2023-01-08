import os
import numpy as np
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from itertools import combinations_with_replacement

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

# data correlation
data_corr = data.corr()

# correlations > 0.2
corr = list(data_corr.nlargest(4, 'salary').index)[1:]

# X a dataframe with a predictor rating and y a series with a target salary
X, y = data.loc[:, ['rating', 'draft_round', 'age', 'experience', 'bmi']], data['salary']

# split into training and test parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# linear regression model
model = LinearRegression()

# comb to drop
comb_to_drop = [np.unique(np.array(comb)) for comb in combinations_with_replacement(corr, 2)]

mape_res = []
combs = []
model.fit(X_train, y_train)
for drop_col in comb_to_drop:
    # fit model with the new subset
    model.fit(X_train.drop(drop_col, axis=1), y_train)

    # predict model on test data and calc MAPE
    y_pred = model.predict(X_test.drop(drop_col, axis=1))
    mape = round(mean_absolute_percentage_error(y_test, y_pred), 5)
    mape_res.append(mape)
    combs.append(drop_col)

# best comb to drop
best_comb = combs[np.argmin(mape_res)]

# fit model with the new subset
model.fit(X_train.drop(best_comb, axis=1), y_train)

# predict model on test data and calc MAPE
y_pred = pd.DataFrame(model.predict(X_test.drop(best_comb, axis=1)))

# mape without negative predictions
res = []
negative_to_zero = y_pred.clip(lower=0)
mape = round(mean_absolute_percentage_error(y_test, negative_to_zero), 5)
res.append(mape)
negative_to_median = y_pred.clip(lower=np.median(y_train))
mape = round(mean_absolute_percentage_error(y_test, negative_to_zero), 5)
res.append(mape)

# print lowest mape
print(min(res))
