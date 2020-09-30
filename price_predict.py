import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing


if __name__ == '__main__':
    inputData = pd.read_csv('./data/kc_house_data.csv')
    inputLabels   = inputData['price']
    inputFeatures = inputData.drop('price', inplace=False, axis=1)

    print(inputFeatures.head)
    print(inputLabels.head)
