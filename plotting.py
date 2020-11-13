import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

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

# Plots histograms of all features in input dataframe
def plot_feature_histograms(X, feature_stats=None, save_dir='./figures'):
    for column in X.columns:
        plt.hist(x=X[column], bins='auto', density=True, rwidth=0.95)
        plt.title('%s Probablity Density (%s)' % (column, featureTypes[column]))
        plt.ylabel('Probability Density')
        plt.xlabel('Feature value')

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


# Plots feature correlation with the label
def plot_feature_correlation(X, Y, save_dir):
    for column in X.columns:
        plt.scatter(x=X[column], y=Y[Y.columns[0]])
        plt.title('%s Feature Correlation (%s)' % (column, featureTypes[column]))
        plt.ylabel('Price')
        plt.xlabel('%s' % column)

        xy_correlation = pearsonr(X[column], Y[Y.columns[0]])[0]

        # Displaying 
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) #From Stackoverflow
        plt.text(0.95, 0.95, 'Pearson Correlation: %f' % (xy_correlation), transform=plt.gca().transAxes, fontsize=10, 
        verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.savefig('%s/%s_correlation.png' % (save_dir, column), dpi=300)
        plt.clf()


def plot_latlong_clusters(X, Y, cluster, save_dir, background_dir="./map.png", save_name="latlong_clustering", marker_size=0.05):
    clusters = list(set(cluster))
    count = len(clusters)
    if (count == 0):
        print("lat-long cluster plot called with invalid arugments")
        return

    # Transform lat and long with map offset
    mc = [47.0451, -122.5736, 47.8116, -120.9609]
    y = Y.apply(lambda x: (x - mc[0]) / (mc[2] - mc[0]))
    x = X.apply(lambda x: (x - mc[1]) / (mc[3] - mc[1]))

    # Gets color map for clusters from colormap (to allow any # of clusters)
    cmap = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.min(clusters), vmax=np.max(clusters)), cmap='jet')

    # Create and save plot
    bg = plt.imread(background_dir)
    fig, ax = plt.subplots()
    ax.imshow(bg, extent=[0, 1, 0, 1])
    for clust in clusters:
        ax.scatter(x[cluster == clust], y[cluster == clust], color=cmap.to_rgba(clust), s=marker_size)
    ax.axis('off')
    plt.savefig('%s/%s.png' % (save_dir, save_name), dpi=300)
    plt.clf()
    fig.clf()
    plt.close('all')

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

# Plots Eps optimization plot for DBSCAN
def plot_eps_neighbor_search(k_nearest_distances, optimal_eps, save_dir):
    X = np.linspace(1, len(k_nearest_distances[0,:]), len(k_nearest_distances[0]), endpoint=True)
    for kval in range(1, len(k_nearest_distances[:,0])+1):
        plt.scatter(X, k_nearest_distances[kval-1,:])
        plt.xlabel('Sample #')
        plt.ylabel('Lat/Long Distance to %d-th nearest neighbor' % (kval))
        plt.title('DBSCAN Eps Optimization Plot')
        plt.hlines(optimal_eps[kval-1], xmin=0, xmax=len(X)+1, linestyles='dashed')
        plt.text(0, optimal_eps[kval-1]+0.02, 'Optimal Eps = %.3f' % (optimal_eps[kval-1]))
        plt.savefig(save_dir+'k_%d.png' % (kval), dpi=300)
        plt.clf()

def plot_train_test_split(long_train, long_test, lat_train, lat_test, save_dir='./figures', background_dir='./map.png', marker_size=0.05):

    # Transform lat and long with map offset
    mc = [47.0451, -122.5736, 47.8116, -120.9609]
    long_tr = long_train.apply(lambda x: (x - mc[1]) / (mc[3] - mc[1]))
    long_te  = long_test.apply(lambda x: (x - mc[1]) / (mc[3] - mc[1]))
    lat_tr  = lat_train.apply(lambda x: (x - mc[0]) / (mc[2] - mc[0]))
    lat_te   = lat_test.apply(lambda x: (x - mc[0]) / (mc[2] - mc[0]))

    # Gets color map for clusters from colormap (to allow any # of clusters)
    cmap = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap='bwr')

    # Create and save plot
    bg = plt.imread(background_dir)
    fig, ax = plt.subplots()
    ax.imshow(bg, extent=[0, 1, 0, 1])
    ax.scatter(long_tr, lat_tr, color=cmap.to_rgba(0), s=marker_size, label='Train')
    ax.scatter(long_te, lat_te, color=cmap.to_rgba(1), s=marker_size, label='Test')
    ax.axis('off')

    # Creating Legend
    legend_elements = [Line2D([0], [0], color=cmap.to_rgba(0), marker='o', markersize=8, label='Train'),
        Line2D([0], [0], color=cmap.to_rgba(1), marker='o', markersize=8, label='Test')]
    plt.legend(handles=legend_elements)
    

    plt.savefig('%s/train_test_split.png' % (save_dir), dpi=300)
    plt.clf()
    fig.clf()
    plt.close('all')