import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
import scipy
import statistics
import sys, getopt
import os


from preprocess import *
from plotting import *
from cluster_model import * 

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


if __name__ == '__main__':
    savePlots, plotDir = handle_cl_args()

    test_size = 0.2
    print('Preprocessing data before clustering... (Just add option for no clustering)')
    preprocessed_data = DataPreprocessor('./data/kc_house_data.csv', drop_features=['date', 'id'], 
    save_dir='./data', test_size=test_size, normalize_labels=False, save_plots=False, plotDir=plotDir,
    omit_norm_features=['zipcode', 'lat', 'long'])

    doMRMR = False
    if (doMRMR):
        preprocessed_data.mRMR_KNN_test()

    # Initial Preprocessing of data
    X, Y, X_train, X_test, Y_train, Y_test = preprocessed_data.X, preprocessed_data.Y,\
    preprocessed_data.X_train, preprocessed_data.X_test, preprocessed_data.Y_train, preprocessed_data.Y_test 

    # Creating one specific type of cluster model
    print('Initializing clustering model...')
    cm = cluster_model(X, Y, X_train, X_test, Y_train, Y_test, cluster_type='latlong', 
    cluster_methods=['dbscan', 'kmeans'], regressors=['knn', 'lr'], plot_clusters=True)
    cm.evaluate()

    # Baseline mRMR + linear regression
    #ordered_features = preprocessed_data.mRMR(k=50, verbose=0, additive=False)

    #print("Features ordered my mRMR:\n", ordered_features)

    #best_lr = -1
    #best_lr_k = -1

    #best_knn = -1
    #best_knn_k = -1

    ## Find optimal number of features for linear regression
    #for k in range(3, len(ordered_features)):
    #    features = ordered_features[:k]
    #    reg = LinearRegression()
    #    reg.fit(X_train[features], Y_train[Y_train.columns[0]])
    #    score = reg.score(X_test[features], Y_test[Y_test.columns[0]])
    #    #print("Using ", k, "features with linear regression: ", score)

    #    if (score > best_lr):
    #        best_lr = score
    #        best_lr_k = k

    ## Find optimal number of features for KNN
    #for k in range(3, len(ordered_features)):
    #    features = ordered_features[:k]
    #    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    #    knn.fit(X_train[features], Y_train[Y_train.columns[0]])
    #    score = knn.score(X_test[features], Y_test[Y_test.columns[0]])
    #    #print("Using ", k, "features with knn: ", score)

    #    if (score > best_knn):
    #        best_knn = score
    #        best_knn_k = k

    ## Using optimal number of features, get average performance of Linear Regression over 10 trainings
    ## Add n-fold cross validation
    #best_lr_running = 0
    #for i in range(10):
    #    features = ordered_features[:best_lr_k]
    #    reg = LinearRegression()
    #    reg.fit(X_train[features], Y_train[Y_train.columns[0]])
    #    best_lr_running += reg.score(X_test[features], Y_test[Y_test.columns[0]])

    #best_lr = best_lr_running / 10

    ## Using optimal number of features, get average performance of KNN over 10 trainings
    #best_knn_running = 0
    #for i in range(10):
    #    features = ordered_features[:best_knn_k]
    #    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    #    knn.fit(X_train[features], Y_train[Y_train.columns[0]])
    #    best_knn_running += knn.score(X_test[features], Y_test[Y_test.columns[0]])
    
    #best_knn = best_knn_running / 10

    #print ("Best Linear Regression: Score =", best_lr, ", k =", best_lr_k)
    #print ("Best KNN: Score =", best_knn, ", k =", best_knn_k)

    ## Test polynomial regression
    #for p in range(2, 5):
    #    best = -1
    #    best_k = -1
    #    print ("  Degree =", p)
    #    for k in range(3, len(ordered_features)):
    #        features = ordered_features[:k]

    #        poly = PolynomialFeatures(degree=p)
    #        x_poly_train = poly.fit_transform(X_train[features])
    #        x_poly_test = poly.fit_transform(X_test[features])

    #        reg = LinearRegression()
    #        reg.fit(x_poly_train, Y_train[Y_train.columns[0]])
    #        score = reg.score(x_poly_test, Y_test[Y_test.columns[0]])

    #        print ("    K =", k, ": Score =", score)

    #        if (score > best):
    #            best = score
    #            best_k = k

    #    print ("For degree =", p, ": Best score =", best, ", best k =", best_k)

