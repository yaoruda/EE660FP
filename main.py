from Preprocess import Preprocess
from Data import Data
from Models import SVC, AdaBoost
from sklearn.model_selection import train_test_split

preprocess_cls = Preprocess()
data_cls = Data()

# load data
X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name = data_cls.Dataset_Methodology()


# training model
model = 'SVC'

# (1) SVC
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

all_result = {}
if model=='SVC':
# (1.1) SVC with original features
    # preprocessing
    X_train = preprocess_cls.Standard(X_train)
    X_val = preprocess_cls.Standard(X_val)
    X_test = preprocess_cls.Standard(X_test)
    result = {}
    for k in kernels:
        svc = SVC(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
        svc.SVC_train(k)
        r = svc.SVC_validation()
        result[k] = r
    print(k)
    print(result)
    all_result['1.1'] = result

# if model=='1-2':
# (1.2) SVC with PCA features
    # preprocessing
    n_component = 300
    X_train = preprocess_cls.PCA(X_train, features, n_component)
    X_val = preprocess_cls.PCA(X_val, features, n_component)
    X_test = preprocess_cls.PCA(X_test, features, n_component)
    result = {}
    for k in kernels:
        svc = SVC(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
        svc.SVC_train(k)
        r = svc.SVC_validation()
        result[k] = r
    print(k)
    print(result)
    all_result['1.2'] = result

# if model=='1-3':
# (1.3) SVC with statistic features
    # preprocessing
    X_train, features = preprocess_cls.feature_statistic(X_train)
    X_val, features = preprocess_cls.feature_statistic(X_val)
    X_test, features = preprocess_cls.feature_statistic(X_test)
    result = {}
    for k in kernels:
        svc = SVC(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
        svc.SVC_train(k)
        r = svc.SVC_validation()
        result[k] = r
    print(k)
    print(result)
    all_result['1.3'] = result
    print(all_result)

if model=='2-1':
# (2) AdaBoost with original features
    svc = AdaBoost(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
    svc.AdaBoost_train()
    svc.AdaBoost_validation()

# Stage 2
# transfor to the multi class (single label) question
# def transform_multi_class(y):
#     new_y = []
#     for value in y:
#         labels = np.nonzero(list(y[value]))
#         pass
#     return


# testing the best model


