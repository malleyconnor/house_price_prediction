import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
import scipy
import statistics
import seaborn as sns
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


    # KFold Split and Evaluation
    k = 5
    X_0 = pd.read_csv('./data/kc_house_data.csv')
    X_0 = X_0[X_0['bedrooms'] < 8]
    Y_0 = pd.DataFrame(X_0['price'].copy(deep=True))
    kf = KFold(n_splits=k)
    X_0.drop('price', inplace=True, axis=1)

    # Feature Engineering

    mean_r2_score = {}
    mean_rmse = {}
    r2_scores = {}
    rmse_scores = {}
    methods = ['dbscan', 'kmeans', 'none']
    regressors = ['knn', 'lr', 'adaboost', 'gradientboosting', 'randomforest', 'decisiontree', 'xgboost']
    regressor_names = ['KNN', 'LR', 'ADAB', 'GB', 'RF', 'DT', 'XGB']
    for k_iter, (train_inds, test_inds) in enumerate(kf.split(X_0)):
        print('Processing split %d' % (k_iter+1))
        X_train, X_test = X_0.iloc[train_inds].copy(), X_0.iloc[test_inds].copy()
        Y_train, Y_test = Y_0.iloc[train_inds].copy(), Y_0.iloc[test_inds].copy()

        # Initializing data for clustering
        #preprocessed_data = DataPreprocessor(drop_features=['date', 'id'], 
        #save_dir='./data', test_size=test_size, normalize_labels=False, save_plots=True, plotDir=plotDir,
        #omit_norm_features=['zipcode', 'lat', 'long'], xtrain=X_train, xtest=X_test,
        #ytrain=Y_train, ytest=Y_test, input_split=True)

        # Initial Preprocessing of data
        #X, Y, X_train, X_test, Y_train, Y_test = preprocessed_data.X, preprocessed_data.Y,\
        #preprocessed_data.X_train, preprocessed_data.X_test, preprocessed_data.Y_train, preprocessed_data.Y_test 

        # Creating one specific type of cluster model
        print('Initializing clustering model...')

        cm = cluster_model(X_0, Y_0, X_train, X_test, Y_train, Y_test, cluster_type='latlong',\
        cluster_methods=methods, regressors=regressors, plot_clusters=True, doMRMR=True)

        if savePlots:
            plot_train_test_split(X_train['long'], X_test['long'], X_train['lat'], X_test['lat'], k=k_iter+1)
            #plot_pearson_matrix(X_train[list(X_train.columns).remove('id')], Y_train, k=k_iter+1)

        cm.evaluate()

        # Mean evaluation scores
        for method in cm.cluster_methods:
            if method not in mean_r2_score.keys():
                mean_r2_score[method] = {}
                mean_rmse[method] = {}
                r2_scores[method] = {}
                rmse_scores[method] = {}
            for regressor in cm.regressors:
                if regressor not in mean_r2_score[method].keys():
                    mean_r2_score[method][regressor] = 0
                    mean_rmse[method][regressor] = 0
                    r2_scores[method][regressor] = []
                    rmse_scores[method][regressor] = []
                mean_r2_score[method][regressor] += cm.r2_score[method][regressor]
                mean_rmse[method][regressor] += cm.rmse[method][regressor]
                r2_scores[method][regressor].append(cm.r2_score[method][regressor])
                rmse_scores[method][regressor].append(cm.rmse[method][regressor])

    for method in methods:
        r2_list = []
        rmse_list = []

        for regressor in regressors:
            mean_r2_score[method][regressor] /= k
            mean_rmse[method][regressor] /= k

            r2_list.append(r2_scores[method][regressor])
            rmse_list.append(rmse_scores[method][regressor])

            print('Average R^2 score (%d-fold, %s, %s): %.4f' % (k, method, regressor, mean_r2_score[method][regressor]))
            print('Average RMSE (%d-fold, %s, %s): %.4f\n' % (k, method, regressor, mean_rmse[method][regressor]))
        
        scores = pd.DataFrame({
            'Model' : regressor_names,
            'R2' : r2_list,
            'RMSE' : rmse_list
        })
        scores = scores.explode('R2').explode('RMSE')

        r2_plot = sns.boxplot(by='Model', column='R2')
        r2_plot = r2_plot.get_figure()
        r2_plot.savefig('./figures/'+method+'_R2_plot.png')

        rmse_plot = sns.boxplot(by='Model', column='RMSE')
        rmse_plot = rmse_plot.get_figure()
        rmse_plot.savefig('./figures/'+method+'_RMSE_plot.png')

    '''
    pd.DataFrame({
        'Model' : ['KNN', 'Linear Regression', 'AdaBoosting', 'Gradient Boosting Regressor', 'Random Forest Regressor', 'Decision Tree Regressor'],
        'R2_kmeans' : []
    })
    '''


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

