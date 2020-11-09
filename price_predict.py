import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
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
    preprocessed_data = DataPreprocessor('./data/kc_house_data.csv', drop_features=['date', 'id'], 
    save_dir='./data', test_size=test_size, normalize_labels=False, save_plots=False, plotDir=plotDir)

    doMRMR = False
    if (doMRMR):
        preprocessed_data.mRMR_KNN_test()

    # Initial Preprocessing of data
    X, Y, X_train, X_test, Y_train, Y_test = preprocessed_data.X, preprocessed_data.Y,\
    preprocessed_data.X_train, preprocessed_data.X_test, preprocessed_data.Y_train, preprocessed_data.Y_test 

    # Creating one specific type of cluster model
    cm = cluster_model(X, Y, X_train, X_test, Y_train, Y_test, cluster_type='latlong', 
    cluster_methods=['dbscan'], regressors=['knn'], plot_clusters=False)
