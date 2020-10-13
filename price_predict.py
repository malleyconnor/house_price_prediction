import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import sys, getopt
import os

featureTypes = {
    'bedrooms'      : 'ordinal',
    'bathrooms'     : 'ordinal',
    'sqft_living'   : 'continuous',
    'sqft_lot'      : 'continuous',
    'floors'        : 'ordinal',
    'waterfront'    : 'binary',
    'view'          : 'ordinal',
    'condition'     : 'ordinal',
    'grade'         : 'ordinal',
    'sqft_above'    : 'continuous',
    'sqft_basement' : 'continuous',
    'yr_built'      : 'ordinal',
    'yr_renovated'  : 'ordinal/binary',
    'zipcode'       : 'categorical',
    'lat'           : 'continuous',
    'long'          : 'continuous',
    'sqft_living15' : 'continuous',
    'sqft_lot15'    : 'continuous'
}

typeEnum = {
    'binary'        : 0,
    'categorical'   : 1,
    'ordinal'       : 2,
    'continuous'    : 3
}

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


def preprocess_data(path, drop_features=['date'], label='price',
    save_dir=None):
    # Splitting data into features/labels
    X = pd.read_csv(path)
    Y  = pd.DataFrame(X[label])

    # Dropping label and irrelevant features from X
    X.drop(label, inplace=True, axis=1)
    X.drop(drop_features, inplace=True, axis=1)

    X, Y = normalize_data(X, Y)
    
    # Exporting normalized data to CSV
    if (save_dir != None):    
        X.to_csv('%s/X.csv' % (save_dir))
        Y.to_csv('%s/Y.csv' % (save_dir))

    return X, Y



# Plots histograms of all features in input dataframe
def plot_feature_histograms(X, save_dir):
    for column in X.columns:
        plt.hist(x=X[column], bins='auto', density=True, rwidth=0.95)
        plt.title('%s Probablity Density (%s)' % (column, featureTypes[column]))
        plt.ylabel('Probability Density')
        plt.xlabel('Feature value')
        plt.savefig('%s/%s_hist.png' % (save_dir, column), dpi=300)
        plt.clf()



if __name__ == '__main__':
    # Default input variables
    savePlots = False
    plotDir = './figures'

    # Command-line Option handler
    opts, args = getopt.getopt(sys.argv[1:], 'hp:', ['help', 'plot='])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Figure it out dummy.')
            sys.exit(2)
        elif opt in ('-p', 'plot'):
            plotDir = arg
            savePlots = True


    # Normalize on [0,1] and remove selected features    
    X, Y = preprocess_data('./data/kc_house_data.csv', 
           drop_features=['date', 'id'], save_dir='./data')


    # Plots histograms
    if (savePlots):
        if not(os.path.isdir(plotDir)):
            os.mkdir(plotDir)

        plot_feature_histograms(X, save_dir=plotDir)