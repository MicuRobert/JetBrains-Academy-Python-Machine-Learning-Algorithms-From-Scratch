import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


def clean_data(path):
    # Load data w/ pd
    data = pd.read_csv(path)

    # Parse the b_day and draft_year features as datetime objects;
    data['b_day'] = pd.to_datetime(data['b_day'], format='%m/%d/%y')
    data['draft_year'] = pd.to_datetime(data['draft_year'], format='%Y')

    # Replace the missing values in team feature with "No Team"
    data['team'].fillna('No Team', inplace=True)

    # Parse the height, weight, and salary features as floats;
    # Take the height feature in meters, the height feature contains metric and customary units;
    data['height'] = data['height'].apply(lambda x: float(x.split('/')[1]))

    # Take the weight feature in kg, the weight feature contains metric and customary units;
    data['weight'] = data['weight'].apply(lambda x: float(x.split('/')[1].replace('kg.', '')))

    # Remove the extraneous $ character from the salary feature
    data['salary'] = data['salary'].apply(lambda x: float(x[1:]))

    # Categorize the country feature as "USA" and "Not-USA";
    data['country'] = data['country'].apply(lambda x: x if x == 'USA' else 'Not-USA')

    # Replace the cells containing "Undrafted" in the draft_round feature with the string "0";
    data['draft_round'] = data['draft_round'].apply(lambda x: '0' if x == 'Undrafted' else x)

    return data


def feature_data(df):
    # Get the unique values in the version column of the DataFrame and parse as a datetime object;
    df['version'] = df['version'].apply(lambda x: x[3:].replace('k', '0'))
    df['version'] = pd.to_datetime(df['version'], format='%Y')

    # Engineer the age feature by subtracting b_day column from version. Calculate the value as year;
    df['age'] = pd.DatetimeIndex(df['version']).year - pd.DatetimeIndex(df['b_day']).year

    # Engineer the experience feature by subtracting draft_year column from version. Calculate the value as year;
    df['experience'] = pd.DatetimeIndex(df['version']).year - pd.DatetimeIndex(df['draft_year']).year

    # Engineer the bmi (body mass index) feature from weight (w) and height (h) columns. The formula is bmi = w / h^2
    df['bmi'] = df['weight'] / df['height'] ** 2

    # Drop the version, b_day, draft_year, weight, and height columns;
    df.drop(['version', 'b_day', 'draft_year', 'weight', 'height'], axis=1, inplace=True)

    # Remove the high cardinality features;
    for i in df.columns:
        if df[i].nunique() > 50 and i not in ['age', 'experience', 'bmi', 'salary']:
            df.drop(i, axis=1, inplace=True)
    return df

def multicol_data(df):
    # Drop multicollinear features from the DataFrame that you got from the feature_data;
    X = df.drop(columns='salary')
    y = df.salary
    m = X.corr(numeric_only=True)
    pairs = []
    for i in range(m.shape[0]):
        for j in range(i + 1, m.shape[0]):
            if abs(m.iloc[i][j]) > 0.5:
                pairs.append((i, j))
    for p in pairs:
        col0 = m.columns[p[0]]
        col1 = m.columns[p[1]]
        r0 = y.corr(X[col0])
        r1 = y.corr(X[col1])
        if r0 < r1:
            df = df.drop(columns=col0)
        else:
            df = df.drop(columns=col1)
    return df

def transform_data(df):
    # Transform numerical features in the DataFrame it got from multicol_data using StandardScaler;
    num_feat_df = df.select_dtypes('number').drop('salary', axis=1)
    scaler = StandardScaler()
    scaler.fit(num_feat_df)
    num_scaler_data = pd.DataFrame(scaler.transform(num_feat_df), columns=num_feat_df.columns)

    # Transform nominal categorical variables in the DataFrame using OneHotEncoder;
    cat_feat_df = df.select_dtypes('object')
    onehot = OneHotEncoder()
    onehot.fit(cat_feat_df)
    cols = [el for arr in onehot.categories_ for el in arr]
    cat_onehot_data = pd.DataFrame.sparse.from_spmatrix(onehot.transform(cat_feat_df), columns=cols)

    # Concatenate the transformed numerical and categorical features in the following order: numerical features, then nominal categorical features;
    df_res = pd.concat([num_scaler_data, cat_onehot_data], axis=1)
    return df_res, df['salary']
