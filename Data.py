from cudf import read_csv
from cuml.preprocessing.model_selection import train_test_split

"""
****************************************************
the following "Dataset_Methodology" function was modified from user Ahmet Erdem
from https://www.kaggle.com/aerdem4/moa-xgb-svm-solution
****************************************************
"""
class Data:

    def __init__(self):
        pass

    def Dataset_Methodology(self):
        train_features_df = read_csv('./data/train_features.csv')
        targets_scored_df = read_csv('./data/train_targets_scored.csv')
        targets_nonscored_df = read_csv('./data/train_targets_nonscored.csv')
        test_df = read_csv('./data/test_features.csv')  # not use here (just for future kaggle competition)

        gene_features = [col for col in train_features_df.columns if col.startswith('g-')]
        cell_features = [col for col in train_features_df.columns if col.startswith('c-')]
        targets_scored_col_name = [col for col in targets_scored_df.columns if col != 'sig_id']
        targets_nonscored_col_name = [col for col in targets_nonscored_df.columns if col != 'sig_id']
        features = ["cp_time", "cp_dose"] + gene_features + cell_features

        # y1, y2 merge
        targets_df = targets_scored_df.merge(targets_nonscored_df, on="sig_id")

        # only use trt_cp because other type cases number is too small
        all_df = train_features_df.merge(targets_df, on="sig_id")
        all_df = all_df[all_df['cp_type'] == 'trt_cp']
        train_features_df = all_df[['sig_id', 'cp_type']+features]  # remove
        targets_df = all_df[['sig_id']+targets_scored_col_name+targets_nonscored_col_name]  # seprate

        # preprocessing: deal with special features
        for data in [train_features_df, test_df]:
            data['cp_time'] = data['cp_time']/72
            data['cp_dose'] = 1.0 * (data['cp_dose'] == 'D1')  # D1->1, D2->0

        # use float32 dtype for GPU
        train_features_df[features] = train_features_df[features].astype('float32')

        # split test set
        X_rest_df, X_test, y_rest_df, y_test = train_test_split(train_features_df, targets_df, train_size=0.9, random_state=0)

        # split training and validation set
        X_train, X_val, y_train, y_val = train_test_split(X_rest_df, y_rest_df, train_size=0.9, random_state=0)

        return X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name
