import pandas as pd
import sklearn
from plotting import *
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error

class cluster_model(object):
    def __init__(self, X, Y, X_train, X_test, Y_train, Y_test, cluster_type='latlong', 
    cluster_methods=['dbscan'], regressors=['knn'], plot_clusters=False, plotDir='./figures'):
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
            

        self.__build_model()

    # Clustering based on lat long data
    def __latlong_cluster(self, eps_vals=None, core_neighbors_vals=None):
        X_long = np.array(self.X['long'])
        X_lat  = np.array(self.X['lat'])

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
    def __find_best_dbscan(self, eps_vals=None, core_neighbors_vals=None):
        # DBSCAN clustering
        if not eps_vals:
            eps_vals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        if not core_neighbors_vals:
            core_neighbors_vals = [3, 5, 8, 10, 15, 30, 50]
        for eps in eps_vals:
            for core_neighors in core_neighbors_vals:
                    self.dbscan = DBSCAN(eps=eps, min_samples=core_neighors).fit(self.cluster_features)
                    if (self.plot_clusters):
                        plot_latlong_clusters(self.X['long'], self.X['lat'], self.dbscan.labels_, save_dir=self.plotDir+"/DBSCAN", 
                        save_name=("latlong_DBSCAN_%s_%s" % (eps, core_neighors)))

        return self.dbscan

    # TODO: Change to actually return the best model
    def __find_best_kmeans(self, krange=None):
        if not krange:
            krange = range(2, 11)
            # KMeans clustering
            for nclusters in krange:
                self.kmeans = KMeans(n_clusters=nclusters).fit(self.cluster_features)
                if (self.plot_clusters):
                    plot_latlong_clusters(self.X_long, self.X_lat, self.kmeans.labels_, save_dir=self.plotDir+"/kmeans", 
                    save_name=("latlong_kmeans_%s_clusters" % nclusters))

        return self.kmeans

    # Building entire cluster-based model (including fitting regressors)
    def __build_model(self):
        models = {}

        for method in self.cluster_methods:
            models[method] = {}
            if method == 'dbscan':
                models[method]['model'] = self.dbscan
            elif method == 'kmeans':
                models[method]['model'] = self.kmeans
            for label in self.dbscan.labels_:
                if  label not in models[method].keys():
                    models[method][label] = {}
                    models[method][label]['X_train'] = self.X_train[self.X_train.columns][self.dbscan.labels_ == label]
                    models[method][label]['Y_train'] = self.Y_train[self.dbscan.labels_ == label]
                    models[method][label]['n_train'] = len(self.X_train)
                    models[method][label]['rf_ranking'] = rf_rank(models[method][label]['X_train'])

                    for regressor in self.regressors:
                        # Training separate knn for each cluster
                        if regressor == 'knn':
                            models[method][label]['knn'] = KNeighborsRegressor(n_neighbors=5, weights='distance')
                            models[method][label]['knn'].fit(models[method][label]['X_train'], models[method][label]['Y_train'])


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