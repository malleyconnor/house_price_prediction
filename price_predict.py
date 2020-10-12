import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing


def normalize_data(X, Y):
    # Normalizing features between 0 and 1
    mmScaler = preprocessing.MinMaxScaler()
    for column in X.columns:
        X[column] = mmScaler.fit_transform(X[column].to_numpy().reshape(-1, 1))

    # Normalizing labels as well
    Y = mmScaler.fit_transform(Y)
    Y_temp = pd.DataFrame()
    Y_temp['price'] = Y[:,0]
    Y = Y_temp

    return X, Y


def preprocess_data(path, drop_features=['date'], label='price'):
    # Splitting data into features/labels
    X = pd.read_csv(path)
    Y  = pd.DataFrame(X[label])

    # Dropping label and irrelevant features from X
    X.drop(label, inplace=True, axis=1)
    X.drop(drop_features, inplace=True, axis=1)

    X, Y = normalize_data(X, Y)

    return X, Y

typeEnum = {
    'binary'        : 0,
    'categorical'   : 1,
    'ordinal'       : 2,
    'continuous'    : 3
}

featureTypes = {
    'bedrooms'      : 'ordinal',
    'bathrooms'     : 'ordinal',
    'sqft_living'   : 'continuous',
    'floors'        : 'ordinal',
    'waterfront'    : 'binary',
    'view'          : 'ordinal',
    'condition'     : 'ordinal',
    'grade'         : 'ordinal',
    'sqft_above'    : 'continuous',
    'sqft_basement' : 'continuous',
    'yr_built'      : 'ordinal',
    'yr_renovated'  : 'ordinal',
    'zipcode'       : 'categorical',
    'lat'           : 'continuous',
    'long'          : 'continuous',
    'sqft_living15' : 'continuous',
    'sqft_lot15'    : 'continuous'
}


if __name__ == '__main__':
    X, Y = preprocess_data('./data/kc_house_data.csv')

    # Exporting normalized data to CSV
    X.to_csv('./data/X.csv')
    Y.to_csv('./data/Y.csv')

    print(X.head)
    print(Y.head)