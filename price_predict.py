import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import statistics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error
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

# Normalizes data columns between 0 and 1
def normalize_data(X_train, X_test, Y_train, Y_test, normalize_labels=True):
    # Normalizing features between 0 and 1
    mmScaler = preprocessing.MinMaxScaler()
    mmScaler.fit(X_train)
    X_train = pd.DataFrame(mmScaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test  = pd.DataFrame(mmScaler.transform(X_test), index=X_test.index, columns=X_test.columns)


    # Normalizing labels as well
    if (normalize_labels):
        mmScaler.fit(Y_train)
        Y_train = pd.DataFrame(mmScaler.transform(Y_train), index=X_train.index, columns=['price'])
        Y_test  = pd.DataFrame(mmScaler.transform(Y_test), index=X_test.index, columns=['price'])

    return X_train, X_test, Y_train, Y_test


# Preprocesses data by normalizing, dropping specified features, and splitting into train/test sets
def preprocess_data(path, drop_features=['date'], label='price',
    save_dir=None, test_size=0.2, normalize_labels=True):
    # Splitting data into features/labels
    X = pd.read_csv(path)
    Y = pd.DataFrame(X[label])

    # Dropping label and irrelevant features from X
    X.drop(label, inplace=True, axis=1)
    X.drop(drop_features, inplace=True, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, 
    shuffle=True, random_state=47)
    min_norm = np.min(Y_train)
    max_norm = np.max(Y_train)
    X_train, X_test, Y_train, Y_test = normalize_data(X_train, X_test, Y_train, Y_test, 
    normalize_labels)
    
    # Exporting normalized data to CSV
    if (save_dir != None):    
        X_train.to_csv('%s/X_train.csv' % (save_dir))
        Y_train.to_csv('%s/Y_train.csv' % (save_dir))

        X_test.to_csv('%s/X_test.csv' % (save_dir))
        Y_test.to_csv('%s/Y_test.csv' % (save_dir))

    return X_train, X_test, Y_train, Y_test, min_norm, max_norm



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
def get_correlations(X, Y, disp=False):
    correlations = []
    for i, column in enumerate(X.columns):
        correlations.append((column, get_correlation(X[column], Y[Y.columns[0]])))

    correlations = sorted(correlations, key=lambda tup: abs(tup[1]), reverse=True)

    if (disp):
        print('Correlations of Features reverse sorted by magnitude:')
        for i in range(len(correlations)):
            print(correlations[i])

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

    return stats

# Returns a label value to its unnormalized price
def unnormalize_val(y, min_val, max_val):
    return y * (max_val - min_val) + min_val

def unnormalize_arr(Y, min_val, max_val):
    for i in range(len(Y)):
        Y[i] = unnormalize_val(Y[i], min_val, max_val)

    return Y

# Plots a population histogram split by lat/long as specified in bins
def plot_lat_long_hist(X, bins=(5,5), save_path='./figures/latlong_hist.png'):
    # TODO: Bin lat/long data into a 2-D grid and enumerate
    # or just remove in favor of zip code
    plt.hist2d(X['long'], X['lat'], bins=(5,5))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Lat/Long Counts, Bins = %s' % (str(bins)))
    plt.colorbar()
    plt.savefig(save_path, dpi=300)
    plt.clf()

# Command-line Argument handler
def handle_cl_args():
    # Default input variables
    savePlots = False
    plotDir = './figures'

    opts, args = getopt.getopt(sys.argv[1:], 'hp', ['help', 'plot='])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Figure it out dummy.')
            sys.exit(2)
        elif opt in ('-p', 'plot'):
            if (arg):
                plotDir = arg
            savePlots = True

    return (savePlots, plotDir)

# Ranks the input features using a random forest algorithm (MSE)
def rf_rank(X, Y, n_estimators=100, disp=False):
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    rf.fit(X_train, Y_train['price'])
    feature_importances = zip(X_train.columns, rf.feature_importances_)
    feature_importances = sorted(feature_importances, key=lambda tup: abs(tup[1]), reverse=True)

    if (disp):
        print('\nTop 10 Features reverse sorted by importance from random forest')
        for feature in feature_importances:
            print(feature)
    
    return feature_importances

if __name__ == '__main__':
    savePlots, plotDir = handle_cl_args()

    # Initial Preprocessing of data
    test_size = 0.2
    X_train, X_test, Y_train, Y_test, min_norm, max_norm =\
    preprocess_data('./data/kc_house_data.csv', drop_features=['date', 'id'], 
    save_dir='./data', test_size=test_size, normalize_labels=False)

    # Plots histograms
    if (savePlots):
        if not(os.path.isdir(plotDir)):
            os.mkdir(plotDir)

        plot_feature_histograms(X, save_dir=plotDir)
        plot_feature_correlation(X, Y, save_dir=plotDir+"/correlation")
        plot_lat_long_hist(X)


    # TODO: Keep either zipcode or lat/long, possibly redundant 
    # (if keeping lat/long, then maybe bin it. Higher priced homes are likely to be on north west)

    # Ranks the correlations of each variable
    correlations = get_correlations(X_train, Y_train, disp=True)

    # Ranking features using random forest (w/ MSE)
    # TODO: Check as a function of n_estimators and max_depth, and see how it changes feature ranks or results
    feature_importances = rf_rank(X_train, Y_train, n_estimators=100, disp=True)

    # TODO: Try mRMR for removing redundant features (see if it agrees with random forest)


    # Training random forest regressors using restricted number of top features
    # TODO: Try exhaustive/randomized grid search with n_features, n_estimators, max_depth
    num_features = 7
    n_estimators = 100
    print(max_norm)
    feature_list = [feature[0] for feature in feature_importances[0:num_features]]
    max_depth = num_features
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, max_depth=max_depth)
    rf.fit(X_train[feature_list], Y_train['price'])
    Y_pred = rf.predict(X_test[feature_list])
    rf_error = mean_absolute_error(Y_test, Y_pred)
    print('MAE of Random Forest: %f' % (rf_error))

    # TODO: Use weights of feature importances for KNN


