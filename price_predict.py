import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import statistics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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


# Gets pearson correlation coefficient between two variables
def get_correlation(X, Y):
    std_X = np.std(X)
    std_Y = np.std(Y)
    covariance = np.cov(X, Y)

    return covariance[0][1] / (std_X*std_Y)

# Gets list of correlations, sorted in reverse order by magnitude of correlation for each feature within a dataframe
def get_correlations(X, Y):
    correlations = []
    for i, column in enumerate(X.columns):
        correlations.append((column, get_correlation(X[column], Y[Y.columns[0]])))

    correlations = sorted(correlations, key=lambda tup: abs(tup[1]), reverse=True)
    return correlations


# Plots feature correlation with the label
def plot_feature_correlation(X, Y, save_dir):
    for column in X.columns:
        plt.scatter(x=X[column], y=Y[Y.columns[0]])
        plt.title('%s Feature Correlation (%s)' % (column, featureTypes[column]))
        plt.ylabel('Price')
        plt.xlabel('%s' % column)

        xy_correlation = get_correlation(X[column], Y[Y.columns[0]])

        # Displaying 
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) #From Stackoverflow
        plt.text(0.95, 0.95, 'Pearson Correlation: %f' % (xy_correlation), transform=plt.gca().transAxes, fontsize=10, 
        verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.savefig('%s/%s_correlation.png' % (save_dir, column), dpi=300)
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
        plot_feature_correlation(X, Y, save_dir=plotDir+"/correlation")


    # Ranks the correlations of each variable
    correlations = get_correlations(X, Y)
    print('Correlations of Features reverse sorted by magnitude:')
    for i in range(len(correlations)):
        print(correlations[i])



    # Ranking features using random forest (w/ MSE)
    rf = RandomForestRegressor(n_estimators=1000, max_depth=10, n_jobs=-1)
    rf.fit(X, Y[Y.columns[0]])
    feature_importances = zip(X.columns, rf.feature_importances_)
    feature_importances = sorted(feature_importances, key=lambda tup: abs(tup[1]), reverse=True)
    print('\nTop 10 Features reverse sorted by importance from random forest')
    for feature in feature_importances:
        print(feature)


    # TODO: Bin lat/long data into a 2-D grid and enumerate
    # TODO: Try mRMR for ranking features (see if it agrees with random forest)