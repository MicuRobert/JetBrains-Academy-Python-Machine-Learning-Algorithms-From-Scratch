import pandas as pd
import os
import requests

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
