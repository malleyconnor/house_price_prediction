import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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


def plot_latlong_clusters(X, Y, cluster, save_dir, colors=["#F46036", "#2E294E", "#1B998B", "#E71D36", "#C5D86D", "#70D6FF", "#FF70A6", "#FF9770", "#FFD670", "#424242"], background_dir="./map.png", save_name="latlong_clustering", marker_size=0.05):
    count = min(len(X), min(len(Y), len(cluster)))
    if (count == 0 or len(colors) == 0):
        print("lat-long cluster plot called with invalid arugments")
        return

    # Transform lat and long with map offset
    mc = [47.0451, -122.5736, 47.8116, -120.9609]
    Y = Y.apply(lambda x: (x - mc[0]) / (mc[2] - mc[0]))
    X = X.apply(lambda x: (x - mc[1]) / (mc[3] - mc[1]))

    # Assign colors based on cluster number
    #point_colors = [colors[0]] * count
    #num_colors = len(colors)
    #for i in range(count):
    #    point_colors[i] = colors[cluster[i] % num_colors]

    # Gets color map for clusters from colormap (to allow any # of clusters)
    cmap = plt.cm.get_cmap('jet', count)

    # Create and save plot
    bg = plt.imread(background_dir)
    fig, ax = plt.subplots()
    ax.imshow(bg, extent=[0, 1, 0, 1])
    for i in range(len(X)):
        ax.scatter(x=X[i], y=Y[i], color=cmap(i), s=marker_size)
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