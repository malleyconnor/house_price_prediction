import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import statistics
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

        feature_stats = get_feature_stats(X)
        if (feature_stats):
            statStrings = []
            for stat in feature_stats[column].keys():
                statStrings.append('%s = %f\n' % (stat, feature_stats[column][stat]))

            statString = ''.join(statStrings)

            # Displaying 
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) #From Stackoverflow
            plt.text(0.95, 0.95, statString, transform=plt.gca().transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.savefig('%s/%s_hist.png' % (save_dir, column), dpi=300)
        plt.clf()



# Gets stats for each feature like mean, stddev, etc...
def get_feature_stats(X):
    stats = {}
    for column in X.columns:
        stats[column] = {}
        stats[column]['mean'] = np.mean(X[column])
        stats[column]['median'] = np.median(X[column])
        stats[column]['std']  = np.std(X[column])
        stats[column]['var']  = np.var(X[column])

        # Gets mode (or None in case of continuous / equally probable values)
        try:
            stats[column]['mode'] = statistics.mode(X[column])
        except statistics.StatisticsError:
            mode = -1
        # TODO: Add frequency of mode
        # TODO: Add impurity / entropy measures

    return stats



if __name__ == '__main__':
    # Default input variables
    savePlots = False
    plotDir = './figures'

    # Command-line Option handler
    opts, args = getopt.getopt(sys.argv[1:], 'hp', ['help', 'plot='])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Figure it out dummy.')
            sys.exit(2)
        elif opt in ('-p', 'plot'):
            if (arg):
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