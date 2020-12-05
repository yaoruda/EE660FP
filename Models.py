import numpy as np
from tqdm import tqdm
import cuml, cupy
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from cuml.ensemble import RandomForestClassifier as cuRFC

"""
1. SVC
"""
class SVC:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test
        self.features = features
        self.targets_scored_col_name = targets_scored_col_name

    def SVC_train(self, kernel):
        print('SVC Training:')
        # training SVM for each scored label
        self.y_pred = np.zeros((self.X_val.shape[0], len(self.targets_scored_col_name)))
        for i in tqdm(range(len(self.targets_scored_col_name))):
            this_target_col_name = self.targets_scored_col_name[i]
            # if samples < 5 in this label, pass
            if self.y_train[this_target_col_name].values.sum() < 5:
                self.y_pred[:, i] = np.zeros(len(self.X_val))
            #         print(f"111{this_y_pred.shape}")
            else:
                self.svc_model = cuml.SVC(kernel=kernel, C=100, cache_size=5000, probability=True)
                self.svc_model.fit(self.X_train[self.features], self.y_train[this_target_col_name])
                self.y_pred[:, i] = cupy.asnumpy(self.svc_model.predict_proba(self.X_val[self.features]).values)[:, 1]
        print('Training Finish.')

    def SVC_validation(self):
        print("SVC Validation Result:")
        y_real = self.y_val[self.targets_scored_col_name].values
        # In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
        self.acc = self.svc_model.score(self.X_val[self.features].values, y_real)
        self.loss = log_loss(cupy.asnumpy(y_real), self.y_pred)
        print(f'ACC{self.acc}')
        print(f'Loss{self.loss}')
        return [self.acc, self.loss]

    def SVC_test(self):
        print("SVC Testing Result:")
        y_real = self.y_test[self.targets_scored_col_name].values
        self.test_acc = self.svc_model.score(self.X_test[self.features].values, y_real)
        self.test_loss = log_loss(cupy.asnumpy(y_real), self.y_pred)
        print(f'ACC{self.test_acc}')
        print(f'Loss{self.test_loss}')
        return [self.test_acc, self.test_loss]

"""
2. Random Forest
"""
class RandomForest:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test
        self.features = features
        self.targets_scored_col_name = targets_scored_col_name

    def RandomForest_train(self, n_estimators):
        print('Random Forest Training:')
        # training AdaBoost for each scored label
        self.y_pred = np.zeros((self.X_val.shape[0], len(self.targets_scored_col_name)))
        for i in tqdm(range(len(self.targets_scored_col_name))):
            this_target_col_name = self.targets_scored_col_name[i]
            # if samples < 5 in this label, pass
            if self.y_train[this_target_col_name].values.sum() < 5:
                self.y_pred[:, i] = np.zeros(len(self.X_val))
            else:
                self.model = cuRFC(
                    n_estimators=n_estimators,  # Number of trees in the forest
                    max_depth=16,  # Maximum tree depth
                    max_features='auto',  # Ratio of number of features (columns) to consider per node split
                )
                self.model.fit(self.X_train[self.features], self.y_train[this_target_col_name])
                self.y_pred[:, i] = cupy.asnumpy(self.model.predict_proba(self.X_val[self.features]).values)[:, 1]
        print('Training Finish.')


    def RandomForest_validation(self):
        print("AdaBoost Validation Result:")
        y_real = self.y_val[self.targets_scored_col_name].values
        # In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
        self.acc = self.model.score(self.X_val[self.features].values, y_real)
        self.loss = log_loss(cupy.asnumpy(y_real), self.y_pred)
        print(f'ACC{self.acc}')
        print(f'Loss{self.loss}')
        return [self.acc, self.loss]

    def RandomForest_test(self):
        print("AdaBoost Testing Result:")
        y_real = self.y_test[self.targets_scored_col_name].values
        self.test_acc = self.model.score(self.X_test[self.features].values, y_real)
        self.test_loss = log_loss(cupy.asnumpy(y_real), self.y_pred)
        print(f'ACC{self.test_acc}')
        print(f'Loss{self.test_loss}')
        return [self.test_acc, self.test_loss]
