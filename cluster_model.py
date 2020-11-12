import pandas as pd
import sklearn
import sys
from plotting import *
from preprocess import *
from sklearn.metrics import r2_score
from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error

class cluster_model(object):
    def __init__(self, X, Y, X_train, X_test, Y_train, Y_test, cluster_type='latlong', 
    cluster_methods=['dbscan', 'kmeans'], regressors=['knn'], plot_clusters=False, plotDir='./figures'):
        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.X_test  = X_test
        self.Y_train = Y_train
        self.Y_test  = Y_test
        self.cluster_type = cluster_type
        self.cluster_methods = cluster_methods
        self.regressors = regressors
        self.plotDir = plotDir
        self.plot_clusters = plot_clusters

        if cluster_type == 'latlong':
            self.__latlong_cluster()
            

        self.models = self.__build_model()


    # Clustering based on lat long data
    def __latlong_cluster(self, eps_vals=None, core_neighbors_vals=None):
        X_long = np.array(self.X_train['long'])
        X_lat  = np.array(self.X_train['lat'])

        if (self.plot_clusters):
            plot_latlong_clusters(self.X['long'], self.X['lat'], [0] * len(X_long), save_dir=self.plotDir, save_name="latlong_mapping")

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

    # TODO: Change to actually return the best model
    # TODO: Create plot which includes train/test split
    def __find_best_dbscan(self, eps_vals=None, core_neighbors_vals=None, createPlots=False):
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
            plot_eps_neighbor_search(k_nearest_distances, optimal_eps, save_dir)


        # DBSCAN clustering
        # (Fixing min_samples = 50 here)
        core_neighbors_vals = [50]
        eps_vals = [optimal_eps[core_neighbors_vals[0]-1]]
        for eps in eps_vals:
            for core_neighors in core_neighbors_vals:
                    self.dbscan = DBSCAN(eps=eps, min_samples=core_neighors).fit(self.cluster_features)
                    if (self.plot_clusters):
                        plot_latlong_clusters(self.X_train['long'], self.X_train['lat'], self.dbscan.labels_, 
                        save_dir=self.plotDir+"/DBSCAN", save_name=("latlong_DBSCAN_%s_%s" % (eps, core_neighors)))

        return self.dbscan

    # TODO: Change to actually return the best model
    def __find_best_kmeans(self, krange=None):
        if not krange:
            krange = range(2, 11)
            # KMeans clustering
            for nclusters in krange:
                self.kmeans = KMeans(n_clusters=nclusters).fit(self.cluster_features)
                if (self.plot_clusters):
                    plot_latlong_clusters(self.X_train['long'], self.X_train['lat'], self.kmeans.labels_, save_dir=self.plotDir+"/kmeans", 
                    save_name=("latlong_kmeans_%s_clusters" % nclusters))

        return self.kmeans


    def __build_dbscan_model(self, model):
        model['model'] = self.dbscan
        for label in self.dbscan.labels_:
            if  label not in model.keys():
                model[label] = {}
                model[label]['X_train'] = self.X_train[self.X_train.columns][self.dbscan.labels_ == label]
                model[label]['Y_train'] = self.Y_train[self.dbscan.labels_ == label]['price']
                model[label]['n_train'] = len(model[label]['X_train'])

                # MRMR
                model[label]['rf_ranking'] = rf_rank(model[label]['X_train'], model[label]['Y_train'])

                for regressor in self.regressors:
                    # Training separate knn for each cluster
                    if regressor == 'knn':
                        model[label]['knn'] = {}
                        model[label]['knn']['model'] = KNeighborsRegressor(n_neighbors=5, weights='distance')
                        model[label]['knn']['model'].fit(model[label]['X_train'], model[label]['Y_train'])
                    elif regressor == 'lr':
                        model[label]['lr'] = {}
                        model[label]['lr']['model'] = LinearRegression(normalize=True)
                        model[label]['lr']['model'].fit(model[label]['X_train'],
                            model[label]['Y_train'])


        model['predictor'] = KNeighborsClassifier(n_neighbors=1)
        model['predictor'].fit(self.X_train[['lat', 'long']], self.dbscan.labels_)


    # Gets the test sets for a dbscan cluster model 
    def __get_dbscan_test_sets(self):
        predictions = np.array(self.models['dbscan']['predictor'].predict(self.X_test[['lat', 'long']]))
        cluster_labels = list(set(predictions))

        for label in cluster_labels:
            self.models['dbscan'][label]['X_test'] = self.X_test[self.X_test.columns][predictions == label]
            self.models['dbscan'][label]['Y_test'] = self.Y_test[predictions == label]['price']
            self.models['dbscan'][label]['n_test'] = len(self.models['dbscan'][label]['X_test'])
            #print('cluster # = %d' % (label))
            #print('\tn_train = %d' % (self.models['dbscan'][label]['n_train']))
            #print('\tn_test = %d' % (self.models['dbscan'][label]['n_test']))

    # Builds training sets and tests sets for kmeans model. Also fits regressors.
    # (Doesn't need to be done in two parts like dbscan)
    def __build_kmeans_model(self, model):
        model['model'] = self.kmeans
        # Building training dataset and fitting regressor
        for label in self.kmeans.labels_:
            if  label not in model.keys():
                model[label] = {}
                model[label]['X_train'] = self.X_train[self.X_train.columns][self.kmeans.labels_ == label]
                model[label]['Y_train'] = self.Y_train[self.kmeans.labels_ == label]['price']
                model[label]['n_train'] = len(model[label]['X_train'])

                # MRMR
                model[label]['rf_ranking'] = rf_rank(model[label]['X_train'], model[label]['Y_train'])

                for regressor in self.regressors:
                    # Training separate knn for each cluster
                    if regressor == 'knn':
                        model[label]['knn'] = {}
                        model[label]['knn']['model'] = KNeighborsRegressor(n_neighbors=5, weights='distance')
                        model[label]['knn']['model'].fit(model[label]['X_train'], model[label]['Y_train'])
                    elif regressor == 'lr':
                        model[label]['lr'] = {}
                        model[label]['lr']['model'] = LinearRegression(normalize=True)
                        model[label]['lr']['model'].fit(model[label]['X_train'],
                            model[label]['Y_train'])

        # Building test set
        predictions = self.kmeans.predict(self.X_test[['lat', 'long']])
        cluster_labels = list(set(predictions))
        for label in cluster_labels:
            model[label]['X_test'] = self.X_test[self.X_test.columns][predictions == label]
            model[label]['Y_test'] = self.Y_test[predictions == label]['price']
            model[label]['n_test'] = len(model[label]['X_test'])



    # Building entire cluster-based model (including fitting regressors)
    def __build_model(self):
        self.models = {}

        for method in self.cluster_methods:
            self.models[method] = {}
            if method == 'dbscan':
                self.models['dbscan'] = {}
                self.__build_dbscan_model(self.models['dbscan'])
                self.__get_dbscan_test_sets()
            elif method == 'kmeans':
                self.models['kmeans'] = {}
                self.__build_kmeans_model(self.models['kmeans'])
            elif method == 'None':
                self.models[method]['model'] = None

        return self.models


    # Evaluates the model on X_test set
    def evaluate(self):
        predictions = {}
        labels      = {}

        # Getting prediction scores for each regressor in each cluster
        for method in self.cluster_methods:
            clusters = list(self.models[method].keys())
            clusters.remove('model')
            if (method == 'dbscan'):
                clusters.remove('predictor')

            for regressor in self.regressors:
                predictions[regressor] = []
                labels[regressor] = []
                for cluster in clusters:
                    predictions[regressor].extend(self.models[method][cluster][regressor]['model'].predict(
                    self.models[method][cluster]['X_test']))
                    labels[regressor].extend(self.models[method][cluster]['Y_test'].to_list())

                # Getting total prediction score of the whole model
                score = r2_score(labels[regressor], predictions[regressor])
                print('R^2 score for %s clustering with %s regressor: %f' % (method, regressor, score))




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