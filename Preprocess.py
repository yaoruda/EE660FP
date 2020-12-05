import pandas as pd
# from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
import cuml, cudf, cupy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sk_PCA
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss


class Preprocess:
    def __init__(self):
        pass

    """
    1. Standard the data use 0 mean and standard variance
    """
    def Standard(self, X):
        X = X.to_pandas()
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        return cudf.from_pandas(X)

    """
    2. apply PCA for the features
    """
    def PCA(self, X, cols, n_components):
        X = X.to_pandas()
        pca = sk_PCA(n_components=n_components)
        pca.fit(X[cols])
        X = pca.transform(X[cols])
        X = pd.DataFrame(X)
        print(f'PCA number of used components: {len(pca.explained_variance_ratio_)}')
        features = list(np.arange(len(pca.explained_variance_ratio_)))

        return cudf.from_pandas(X), features

    """
    3. extract some statistic features from data
    """
    def feature_statistic(self, df):
        # code cite from: https://www.kaggle.com/kento1993/nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp/notebook
        features_g = list([x for x in df.columns if x.startswith("g-")])
        features_c = list([x for x in df.columns if x.startswith("c-")])
        df = df.to_pandas()
        s = pd.DataFrame()
        s["g_sum"] = df[features_g].sum(axis=1)
        s["g_mean"] = df[features_g].mean(axis=1)
        s["g_median"] = df[features_g].median(axis=1)
        s["g_std"] = df[features_g].std(axis=1)
        s["g_kurt"] = df[features_g].kurtosis(axis=1)
        s["g_skew"] = df[features_g].skew(axis=1)
        s["c_sum"] = df[features_c].sum(axis=1)
        s["c_mean"] = df[features_c].mean(axis=1)
        s["c_std"] = df[features_c].std(axis=1)
        s["c_median"] = df[features_c].median(axis=1)
        s["c_kurt"] = df[features_c].kurtosis(axis=1)
        s["c_skew"] = df[features_c].skew(axis=1)
        s["gc_sum"] = df[features_g + features_c].sum(axis=1)
        s["gc_mean"] = df[features_g + features_c].mean(axis=1)
        s["gc_std"] = df[features_g + features_c].std(axis=1)
        s["gc_kurt"] = df[features_g + features_c].kurtosis(axis=1)
        s["gc_skew"] = df[features_g + features_c].skew(axis=1)
        s["gc_median"] = df[features_g + features_c].median(axis=1)

        return cudf.from_pandas(s), s.columns.values

