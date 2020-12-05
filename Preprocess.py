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
from sklearn.decomposition import PCA
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
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        return X

    """
    2. apply PCA for the features
    """
    def PCA(self, X, cols, n_components):
        pca = cuml.PCA(n_components=n_components)
        pca.fit(X[cols])
        X = pca.transform(X[cols])
        print(f'PCA number of used components: {len(pca.explained_variance_ratio_)}')

        return X

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

    """
    reduce data by different origin features:
    1. cp_time(24/48/72): 123
    2. cp_dose(D1/D2): 12
    3. g(0-771),c(0-99): gc
    cp_type(trt_cp/ctl_vehicle): pass, very unbalance.
    """
    def reduce_data(self, raw_data, cp_time, cp_dose, gc):
        data_time = pd.DataFrame(columns=raw_data.columns.values)
        if '1' in cp_time:
            data_time = data_time.append(raw_data[(raw_data['cp_time'] == 24)])
        if '2' in cp_time:
            data_time = data_time.append(raw_data[(raw_data['cp_time'] == 48)])
        if '3' in cp_time:
            data_time = data_time.append(raw_data[(raw_data['cp_time'] == 72)])
        data_dose = pd.DataFrame(columns=data_time.columns.values)
        if '1' in cp_dose:
            data_dose = data_dose.append(data_time[(data_time['cp_dose'] == 'D1')])
        if '2' in cp_dose:
            data_dose = data_dose.append(data_time[(data_time['cp_dose'] == 'D2')])
        if gc == 'gc':
            data_dose = data_dose.loc[:, 'g-0': 'c-99']
        elif gc == 'c':
            data_dose = data_dose.loc[:, 'c-0': 'c-99']
        elif gc == 'g':
            data_dose = data_dose.loc[:, 'g-0': 'g-771']

        return data_dose


    # read raw data from file
    def get_raw_data(self, cp_time, cp_dose, gc):
        label_data = pd.read_csv("./lish-moa/train_targets_scored.csv", index_col=0)
        raw_data = pd.read_csv("./lish-moa/train_features.csv", index_col=0)
        features_data = self.reduce_data(raw_data, cp_time, cp_dose, gc)
        X, y = [], []
        for index, row in features_data.iterrows():
            X.append(row)
            y.append(label_data.loc[index, :])
        print(f"Number of samples from files: {len(y)} .")

        return X, y







