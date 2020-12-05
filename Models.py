import numpy as np
from tqdm import tqdm
import cuml, cupy
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

"""
1. SVC
"""
class SVC:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test
        self.features = features
        self.targets_scored_col_name = targets_scored_col_name

    def SVC_train(self):
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
                self.svc_model = cuml.SVC(C=100, cache_size=5000, probability=True)
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

    def SVC_test(self):
        print("SVC Testing Result:")
        y_real = self.y_test[self.targets_scored_col_name].values
        self.test_acc = self.svc_model.score(self.X_test[self.features].values, y_real)
        self.test_loss = log_loss(cupy.asnumpy(y_real), self.y_pred)
        print(f'ACC{self.test_acc}')
        print(f'Loss{self.test_loss}')

"""
1. Ada
"""
class AdaBoost:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test
        self.features = features
        self.targets_scored_col_name = targets_scored_col_name

    def AdaBoost_train(self):
        print('AdaBoost Training:')
        # training AdaBoost for each scored label
        self.y_pred = np.zeros((self.X_val.shape[0], len(self.targets_scored_col_name)))
        for i in tqdm(range(len(self.targets_scored_col_name))):
            this_target_col_name = self.targets_scored_col_name[i]
            # if samples < 5 in this label, pass
            if self.y_train[this_target_col_name].values.sum() < 5:
                self.y_pred[:, i] = np.zeros(len(self.X_val))
            else:
                self.model = AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=1),
                    algorithm="SAMME",
                    n_estimators=200
                )
                self.model.fit(self.X_train[self.features], self.y_train[this_target_col_name])
                self.y_pred[:, i] = cupy.asnumpy(self.model.predict_proba(self.X_val[self.features]).values)[:,
                                    1]
        print('Training Finish.')


    def AdaBoost_validation(self):
        print("AdaBoost Validation Result:")
        y_real = self.y_val[self.targets_scored_col_name].values
        # In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
        self.acc = self.model.score(self.X_val[self.features].values, y_real)
        self.loss = log_loss(cupy.asnumpy(y_real), self.y_pred)
        print(f'ACC{self.acc}')
        print(f'Loss{self.loss}')

    def AdaBoost_test(self):
        print("AdaBoost Testing Result:")
        y_real = self.y_test[self.targets_scored_col_name].values
        self.test_acc = self.model.score(self.X_test[self.features].values, y_real)
        self.test_loss = log_loss(cupy.asnumpy(y_real), self.y_pred)
        print(f'ACC{self.test_acc}')
        print(f'Loss{self.test_loss}')

    """
    1. SVC with original features, implement binary relevance method by hand
    """
    def s1_1(self, X_raw, y_raw, kernel_values, C_values):

        for C in C_values:
            for kernel in kernel_values:
                # Training H
                no_svc_label = []
                svc_list = []
                for label_index in tqdm(range(len(y_train[0]))):
                    X_svc_train = []
                    y_svc_train = []
                    for samples_index in range(len(y_train)):
                        X_svc_train.append(X_train[samples_index])
                        y_svc_train.append(y_train[samples_index][label_index])
                    if np.array(y_svc_train).max() == 0:
                        no_svc_label.append(label_index)
                        svc_list.append(None)
                        continue
                    # svc = svm.SVC(C=C, kernel=kernel, gamma='scale', decision_function_shape='ovr', cache_size=1000, class_weight='balanced')
                    svc = svm.SVC(C=C, kernel=kernel, cache_size=1000)
                    svc.fit(X_svc_train, y_svc_train)
                    svc_list.append(svc)
                # Use Validation data to get hg(H)
                label_predict = []
                for label_index in range(scored_label_len):
                    if label_index in no_svc_label:
                        label_predict.append(np.zeros(len(X_val)))
                    else:
                        label_predict.append(svc_list[label_index].predict(X_val))
                label_predict = np.array(label_predict).T
                true_amount = 0
                for val_sample_index in range(len(y_val)):
                    if list(label_predict[val_sample_index]) == y_val[val_sample_index][:scored_label_len].tolist():
                        true_amount += 1
                print(f'Model: C={C}, kernel={kernel}')
                print('---Accuracy: {:.2%}'.format(true_amount / len(y_val)))
                loss = log_loss(y_val[:, :scored_label_len], label_predict)
                print('---Log loss: {:.5}'.format(loss))

"""
Stage 2: Use OneVsRestClassifier to solve the multi class single label problem
"""
def SVC_all(X_raw, y_raw, kernel_values, C_values):
    result = {}
    # Preprocess (only use standard scaler)
    X_scaler = preprocessing_standard_scaler(X_raw)
    X, y = X_scaler, y_raw
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    clf = OneVsRestClassifier(SVC(probability=True), n_jobs=4)
    clf.fit(X_train, y_train)
    # Use Validation data to get hg(H)
    y_pred = clf.predict_proba(X_val)
    print(log_loss(y_val, y_pred))
    print(clf.score(X_val, y_val))

# SVC_all(X_raw, y_raw, ['rbf'], [5])
