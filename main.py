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
# kernels = ['poly', 'rbf', 'sigmoid']
kernels = ['rbf']

all_result = {}
if model=='SVC':
# (1.1) SVC with original features
    if False:
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
    result = {}
    n_component = [0.7]
    for n_c in n_component:
        this_X_train, new_features = preprocess_cls.PCA(X_train, features, n_c)
        # this_X_val, new_features = preprocess_cls.PCA(X_val, features, n_c)
        # this_X_test, new_features = preprocess_cls.PCA(X_test, features, n_c)

        for k in kernels:
            svc = SVC(this_X_train, X_val, X_test, y_train, y_val, y_test, new_features, targets_scored_col_name)
            svc.SVC_train(k)
            r = svc.SVC_validation()
            result[f'{k,n_c}'] = r
        print(k)
        print(result)
    all_result['1.2'] = result

# if model=='1-3':
# (1.3) SVC with statistic features
    # preprocessing
    this_X_train, new_features = preprocess_cls.feature_statistic(X_train)
    this_X_val, new_features = preprocess_cls.feature_statistic(X_val)
    this_X_test, new_features = preprocess_cls.feature_statistic(X_test)
    result = {}
    for k in kernels:
        svc = SVC(this_X_train, this_X_val, this_X_test, y_train, y_val, y_test, new_features, targets_scored_col_name)
        svc.SVC_train(k)
        r = svc.SVC_validation()
        result[k] = r
    print(k)
    print(result)
    all_result['1.3'] = result
    print(all_result)



# (2) Random Forest
all_result = {}
if model=='RF':
# (2.1) RF with original features
    result = {}
    for n in number_trees:
        rfc = RandomForest(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
        rfc.RandomForest_train(n)
        r = rfc.RandomForest_validation()
        result[n] = r
    print(result)
    all_result['2.1'] = result

# if model=='2-2':
# (2.2) RF with PCA features
    # preprocessing
    result = {}
    n_component = [0.5, 0.9]
    for n_c in n_component:
        this_X_train, new_features = preprocess_cls.PCA(X_train, features, n_c)
        this_X_val, new_features = preprocess_cls.PCA(X_val, features, n_c)
        this_X_test, new_features = preprocess_cls.PCA(X_test, features, n_c)

        for n in number_trees:
            rfc = RandomForest(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
            rfc.RandomForest_train(n)
            r = rfc.RandomForest_validation()
            result[n] = r
        print(k)
        print(result)
    all_result['2.2'] = result

# if model=='2-3':
# (2.3) RF with statistic features
    # preprocessing
    this_X_train, new_features = preprocess_cls.feature_statistic(X_train)
    this_X_val, new_features = preprocess_cls.feature_statistic(X_val)
    this_X_test, new_features = preprocess_cls.feature_statistic(X_test)
    result = {}
    for n in number_trees:
        rfc = RandomForest(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
        rfc.RandomForest_train(n)
        r = rfc.RandomForest_validation()
        result[n] = r
    print(k)
    print(result)
    all_result['2.3'] = result
    print(all_result)

# testing the best model


