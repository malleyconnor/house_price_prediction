import pandas as pd
import sklearn
import sys
from plotting import *
from preprocess import *
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error
import xgboost

class cluster_model(object):
    def __init__(self, X, Y, X_train, X_test, Y_train, Y_test, cluster_type='latlong', 
    cluster_methods=['dbscan', 'kmeans', 'none'], regressors=['knn'], plot_clusters=True, 
    plotDir='./figures', doMRMR=False, doRF=False):
        self.X = X.copy()
        self.Y = Y.copy()
        self.X_train = X_train.copy()
        self.X_test  = X_test.copy()
        self.Y_train = Y_train.copy()
        self.Y_test  = Y_test.copy()
        self.cluster_type = cluster_type
        self.cluster_methods = cluster_methods
        self.regressors = regressors
        self.plotDir = plotDir
        self.plot_clusters = plot_clusters
        self.test_size = 0.2
        self.doMRMR = doMRMR
        self.doRF = doRF

        if doRF and doMRMR:
            print('Set doMRMR=True or doRF=True, not both.')
            exit()

        if cluster_type == 'latlong':
            self.__latlong_cluster()
            
        self.models = self.__build_model()


    # Preprocesses clusters individually
    # Includes normalization and feature selection with mRMR
    def __preprocess_clusters(self):
        print('Preprocessing clusters individually...')
        for method in self.cluster_methods:
            clusters = self.__get_cluster_labels(method)
            for cluster in clusters:
                preprocessed_data = DataPreprocessor(xtrain=self.models[method][cluster]['X_train'], xtest=self.models[method][cluster]['X_test'], 
                ytrain=self.models[method][cluster]['Y_train'], ytest=self.models[method][cluster]['Y_test'], drop_features=[],
                save_dir='./data/'+str(method)+'/'+str(cluster), test_size=self.test_size, normalize_labels=False, save_plots=False, 
                plotDir=self.plotDir+'/'+str(method)+'/'+str(cluster), input_split=True, omit_norm_features=[])

                # In case a feature has the same value for every data point in a cluster 
                columns = preprocessed_data.X_train.columns
                for column in columns:
                    if len(list(set(preprocessed_data.X_train[column]))) <= 1:
                        preprocessed_data.X_train.drop(column, axis=1, inplace=True)
                        preprocessed_data.X_test.drop(column, axis=1, inplace=True)

                self.models[method][cluster]['preprocessed_data'] = preprocessed_data

                k = 8
                if (self.doMRMR):
                    selected_features = self.models[method][cluster]['preprocessed_data'].mRMR(k=k, verbose=0)
                elif (self.doRF):
                    selected_features = self.models[method][cluster]['preprocessed_data'].rf_rank(threshold=0.01)
                else:
                    selected_features = preprocessed_data.X_train.columns


                self.models[method][cluster]['X_train'] = preprocessed_data.X_train[selected_features[:k]].copy()
                self.models[method][cluster]['X_test']  = preprocessed_data.X_test[selected_features[:k]].copy()
                self.models[method][cluster]['Y_train'] = preprocessed_data.Y_train.copy()
                self.models[method][cluster]['Y_test']  = preprocessed_data.Y_test.copy()

                # Create separate sets of features for poly regression
                for regressor in self.regressors:
                    if regressor == 'pr2':
                        self.models[method][cluster]['pr2'] = {}
                        self.models[method][cluster]['pr2']['poly_transform'] = PolynomialFeatures(degree=2)
                        poly = self.models[method][cluster]['pr2']['poly_transform']
                        self.models[method][cluster]['pr2']['X_train'] = poly.fit_transform(self.models[method][cluster]['X_train'].copy())
                        self.models[method][cluster]['pr2']['X_test']  = poly.fit_transform(self.models[method][cluster]['X_test'].copy())
                    elif regressor == 'pr3':
                        self.models[method][cluster]['pr3'] = {}
                        self.models[method][cluster]['pr3']['poly_transform'] = PolynomialFeatures(degree=3)
                        poly = self.models[method][cluster]['pr3']['poly_transform']
                        self.models[method][cluster]['pr3']['X_train'] = poly.fit_transform(self.models[method][cluster]['X_train'].copy())
                        self.models[method][cluster]['pr3']['X_test']  = poly.fit_transform(self.models[method][cluster]['X_test'].copy())



    # Clustering based on lat long data
    def __latlong_cluster(self, eps_vals=None, core_neighbors_vals=None):
        print('Getting lat/long clusterings')
        X_long = np.array(self.X_train['long'])
        X_lat  = np.array(self.X_train['lat'])

        #if (self.plot_clusters):
        #    plot_latlong_clusters(self.X['long'], self.X['lat'], [0] * len(X_long), save_dir=self.plotDir, save_name="latlong_mapping")

        latlong = np.zeros((len(X_lat), 2))
        for i in range(len(X_lat)):
            latlong[i][0] = X_lat[i]
            latlong[i][1] = X_long[i]

        self.X_long = X_long
        self.X_lat  = X_lat

        self.cluster_features = latlong
        for method in self.cluster_methods:
            if method == 'kmeans':
                self.__find_best_kmeans()
            elif method == 'dbscan':
                self.__find_best_dbscan()

    def __find_best_dbscan(self, eps_vals=None, core_neighbors_vals=None, createPlots=True, precomputed=True, default_eps=0.0175, default_ms=100):
        print('Getting DBSCAN clustering')
        if not precomputed:
            max_k = 100
            k_nearest_distances = np.zeros((max_k, len(self.X_train)))

            # Getting distances between k-nearest neighbors in specified range
            cdist_matrix = distance.cdist(self.X_train[['lat', 'long']], self.X_train[['lat', 'long']])
            np.fill_diagonal(cdist_matrix, sys.maxsize)
            for i in range(len(self.X_train)):
                k_nearest_distances[:,i] = cdist_matrix[i, np.argpartition(cdist_matrix[i,:], max_k)[:max_k]]
            k_nearest_distances = np.sort(k_nearest_distances, axis=1)

            # Getting best eps for each value of k
            # (Maximize 2nd derivative to find best eps)
            optimal_eps = np.zeros(max_k)
            distances_2nd_diff = np.diff(k_nearest_distances, n=2, axis=1)
            max_diffs = np.argmax(distances_2nd_diff, axis=1)
            for i in range(max_k):
                optimal_eps[i] = k_nearest_distances[i, max_diffs[i]]

        
            if createPlots:
                save_dir = self.plotDir + '/DBSCAN/eps_neighbor_search/'
                os.makedirs(save_dir, exist_ok=True)
                
                def_eps = np.full(max_k, fill_value=default_eps)
                plot_eps_neighbor_search(k_nearest_distances, def_eps, save_dir)#optimal_eps, save_dir)

            
        # DBSCAN clustering
        # (Fixing min_samples = 50 here)
        core_neighbors_vals = [default_ms]
        eps_vals = [default_eps]
        for eps in eps_vals:
            for core_neighors in core_neighbors_vals:
                    self.dbscan = DBSCAN(eps=eps, min_samples=core_neighors).fit(self.cluster_features)
                    if (self.plot_clusters):
                        plot_latlong_clusters(self.X_train['long'], self.X_train['lat'], self.dbscan.labels_, 
                        save_dir=self.plotDir+"/DBSCAN", save_name=("latlong_DBSCAN_%s_%s" % (eps, core_neighors)))

        print('%d different clusters' % len(list(set(self.dbscan.labels_))))
        return self.dbscan

    # TODO: Change to actually return the best model
    def __find_best_kmeans(self, krange=None, precomputed=True, default_k=7):
        print('Finding best kmeans clustering...')
        if not precomputed:
            sse_vals = []
            if not krange:
                krange = range(2, 20)
                # KMeans clustering
                for nclusters in krange:
                    self.kmeans = KMeans(n_clusters=nclusters).fit(self.cluster_features)
                    if (self.plot_clusters):
                        plot_latlong_clusters(self.X_train['long'], self.X_train['lat'], self.kmeans.labels_, save_dir=self.plotDir+"/kmeans", 
                        save_name=("latlong_kmeans_%s_clusters" % nclusters))

                    sse_vals.append(self.kmeans.inertia_)

            plot_kmeans_sse(sse_vals)
        else:
            self.kmeans = KMeans(n_clusters=default_k).fit(self.cluster_features)
        return self.kmeans

    # Gets cluster train sets for DBSCAN model
    def __get_dbscan_train_sets(self, model):
        model['model'] = self.dbscan
        for label in self.dbscan.labels_:
            if label not in model.keys():
                model[label] = {}
                model[label]['X_train'] = self.X_train[self.X_train.columns][self.dbscan.labels_ == label]
                model[label]['Y_train'] = self.Y_train[self.dbscan.labels_ == label]
                model[label]['n_train'] = len(model[label]['X_train'])
        
        model['predictor'] = KNeighborsClassifier(n_neighbors=1)
        model['predictor'].fit(self.X_train[['lat', 'long']], self.dbscan.labels_)

    # Gets the test sets for a dbscan cluster model 
    def __get_dbscan_test_sets(self):
        predictions = np.array(self.models['dbscan']['predictor'].predict(self.X_test[['lat', 'long']]))
        cluster_labels = list(set(predictions))

        for label in cluster_labels:
            self.models['dbscan'][label]['X_test'] = self.X_test[self.X_test.columns][predictions == label]
            self.models['dbscan'][label]['Y_test'] = self.Y_test[predictions == label]
            self.models['dbscan'][label]['n_test'] = len(self.models['dbscan'][label]['X_test'])
            #print('cluster # = %d' % (label))
            #print('\tn_train = %d' % (self.models['dbscan'][label]['n_train']))
            #print('\tn_test = %d' % (self.models['dbscan'][label]['n_test']))


    # Fits the regressors for all clustering methods
    def __fit_regressors(self):
        print('Fitting regressors...')
        for method in self.cluster_methods:
            clusters = self.__get_cluster_labels(method)
            model = self.models[method]
            for label in clusters:
                for regressor in self.regressors:
                    # Training separate knn for each cluster
                    if regressor == 'knn':
                        model[label]['knn'] = {}
                        model[label]['knn']['model'] = KNeighborsRegressor(n_neighbors=5, weights='distance')
                        model[label]['knn']['model'].fit(model[label]['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'lr':
                        model[label]['lr'] = {}
                        model[label]['lr']['model'] = LinearRegression(normalize=True)
                        model[label]['lr']['model'].fit(model[label]['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'adaboost':
                        model[label]['adaboost'] = {}
                        model[label]['adaboost']['model'] = AdaBoostRegressor(n_estimators=100, learning_rate=0.2, loss='exponential')
                        model[label]['adaboost']['model'].fit(model[label]['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'gradientboosting':
                        model[label]['gradientboosting'] = {}
                        model[label]['gradientboosting']['model'] = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, loss='ls', max_depth=5, min_samples_split=2)
                        model[label]['gradientboosting']['model'].fit(model[label]['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'randomforest':
                        model[label]['randomforest'] = {}
                        model[label]['randomforest']['model'] = RandomForestRegressor(n_estimators=400)
                        model[label]['randomforest']['model'].fit(model[label]['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'decisiontree':
                        model[label]['decisiontree'] = {}
                        model[label]['decisiontree']['model'] = DecisionTreeRegressor()
                        model[label]['decisiontree']['model'].fit(model[label]['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'xgboost':
                        model[label]['xgboost'] = {}
                        model[label]['xgboost']['model'] = xgboost.XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=5)
                        model[label]['xgboost']['model'].fit(model[label]['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'pr2':
                        # Need to create and fit poly features in same function
                        model[label]['pr2']['model'] = LinearRegression(normalize=True)
                        model[label]['pr2']['model'].fit(model[label]['pr2']['X_train'], model[label]['Y_train']['price'])

                    elif regressor == 'pr3':
                        # Need to create and fit poly features in same function
                        model[label]['pr3']['model'] = LinearRegression(normalize=True)
                        model[label]['pr3']['model'].fit(model[label]['pr3']['X_train'], model[label]['Y_train']['price'])



    def __get_kmeans_train_and_test_sets(self, model):
        model['model'] = self.kmeans
        # Building training dataset and fitting regressor
        for label in self.kmeans.labels_:
            if  label not in model.keys():
                model[label] = {}
                model[label]['X_train'] = self.X_train[self.X_train.columns][self.kmeans.labels_ == label]
                model[label]['Y_train'] = self.Y_train[self.kmeans.labels_ == label]
                model[label]['n_train'] = len(model[label]['X_train'])

        # Building test set
        predictions = self.kmeans.predict(self.X_test[['lat', 'long']])
        cluster_labels = list(set(predictions))
        for label in cluster_labels:
            model[label]['X_test'] = self.X_test[self.X_test.columns][predictions == label]
            model[label]['Y_test'] = self.Y_test[predictions == label]
            model[label]['n_test'] = len(model[label]['X_test'])

        


    # Building entire cluster-based model (including fitting regressors)
    def __build_model(self):
        self.models = {}

        for method in self.cluster_methods:
            if method == 'dbscan':
                self.models['dbscan'] = {}
                self.__get_dbscan_train_sets(self.models['dbscan'])
                self.__get_dbscan_test_sets()
            elif method == 'kmeans':
                self.models['kmeans'] = {}
                self.__get_kmeans_train_and_test_sets(self.models['kmeans'])
            elif method == 'none':
                self.models['none'] = {}
                self.models[method]['model'] = None
                self.models[method][0] = {}
                self.models[method][0]['X_train'] = self.X_train.copy()
                self.models[method][0]['X_test']  = self.X_test.copy()
                self.models[method][0]['Y_train'] = self.Y_train.copy()
                self.models[method][0]['Y_test']  = self.Y_test.copy()

        self.__preprocess_clusters()
        self.__fit_regressors()
        return self.models


    def __get_cluster_labels(self, method):
        clusters = list(self.models[method].keys())
        clusters.remove('model')
        if (method == 'dbscan'):
            clusters.remove('predictor')
        
        return clusters


    # Evaluates the model on X_test set
    def evaluate(self, verbose=1):
        predictions = {}
        labels      = {}
        self.r2_score = {}
        self.rmse = {}

        # Getting prediction scores for each regressor in each cluster
        for method in self.cluster_methods:
            clusters = self.__get_cluster_labels(method)

            self.r2_score[method] = {}
            self.rmse[method] = {}
            for regressor in self.regressors:
                predictions[regressor] = []
                labels[regressor] = []

                if regressor != 'pr2' and regressor != 'pr3':
                    for cluster in clusters:
                        predictions[regressor].extend(self.models[method][cluster][regressor]['model'].predict(
                        self.models[method][cluster]['X_test']))
                        labels[regressor].extend(self.models[method][cluster]['Y_test']['price'].to_list())
                else:
                    for cluster in clusters:
                        predictions[regressor].extend(self.models[method][cluster][regressor]['model'].predict(
                        self.models[method][cluster][regressor]['X_test']))
                        labels[regressor].extend(self.models[method][cluster]['Y_test']['price'].to_list())

                # Getting total prediction score of the whole model
                score = r2_score(labels[regressor], predictions[regressor])
                rmse  = mean_squared_error(labels[regressor], predictions[regressor], squared=False) 

                if verbose:
                    print('R^2 score for %s clustering with %s regressor: %.4f' % (method, regressor, score))
                    print('RMSE for %s clustering with %s regressor: %.4f' % (method, regressor, rmse))

                self.r2_score[method][regressor] = score
                self.rmse[method][regressor]     = rmse





    # Predictions and score for each individual zipcode
    # (Will also analyze accuracy of whole dataset)
    #zip_models[zipcode]['predictions'] = np.array(zip_models[zipcode]['knn'].predict(zip_models[zipcode]['x_test']))
    #zip_models[zipcode]['r2_score'] = zip_models[zipcode]['knn'].score(zip_models[zipcode]['x_test'], zip_models[zipcode]['y_test'])
    #errors = np.subtract(zip_models[zipcode]['predictions'], zip_models[zipcode]['y_test'])
    #zip_models[zipcode]['MSE'] = np.sum(np.multiply(errors, errors)) / len(errors)
    #zip_models[zipcode]['MAE'] = np.sum(np.abs(errors)) / len(errors) 

    ## Gets average errors across test dataset
    #total_mse = 0
    #total_mae = 0
    #total_r2  = 0
    #weight_per_sample = 1 / len(Y_test)
    #for zipcode in zip_models.keys():
    #    total_mse += zip_models[zipcode]['MSE'] * weight_per_sample * zip_models[zipcode]['n_test']
    #    total_mae += zip_models[zipcode]['MAE'] * weight_per_sample * zip_models[zipcode]['n_test']
    #    total_r2 += zip_models[zipcode]['r2_score'] * weight_per_sample * zip_models[zipcode]['n_test']
    

    #print('Average MAE of KNN with zipcode based clusters: %f' % (total_mae))
    #print('Average MSE of KNN with zipcode based clusters: %f' % (total_mse))
    #print('Average R^2 score of KNN with zipcode based clusters: %f' % (total_r2))

    ## If a model hasn't been trained for a specific zipcode, we'll have to
    ## use a different method for those samples
    #train_zips = set(train_zips)
    #test_zips  = set(test_zips)
    #if not test_zips.issubset(train_zips):
    #    print('WARNING: Evaluation zipcodes is NOT a subset of train zipcodes')

    ## Gets min number of sets in zipcode
    #min_size = sys.maxsize
    #max_size = 0
    #for zipcode in zip_models.keys():
    #    if zip_models[zipcode]['n_train'] < min_size:
    #        min_size = zip_models[zipcode]['n_train']
    #    if zip_models[zipcode]['n_train'] > max_size:
    #        max_size = zip_models[zipcode]['n_train']
    #print('Min training set for zipcode clusters: %d' % (min_size))
    #print('Max training set for zipcode clusters: %d' % (max_size))

    #############################################################################################################

# Ranks the input features using a random forest algorithm (MSE)
def rf_rank(X, Y, n_estimators=25, max_depth=None, disp=False, threshold=None):
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X, Y)
    feature_importances = zip(X.columns, rf.feature_importances_)
    feature_importances = sorted(feature_importances, key=lambda tup: abs(tup[1]), reverse=True)

    if threshold:
        cutoff = len(feature_importances)
        for i, importance in enumerate(feature_importances):
            if importance[1] < threshold:
                cutoff = i
        feature_importances = feature_importances[:cutoff]

    if (disp):
        print('\nTop 10 Features reverse sorted by importance from random forest')
        for feature in feature_importances:
            print(feature)

    return feature_importances