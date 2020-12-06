from Preprocess import Preprocess
from Data import Data
from Models import SVC, RandomForest


def training(model_name, X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name):
    # (1) SVC
    result = {}
    kernels = ['rbf', 'poly', 'sigmoid']
    if model_name=='SVC':
    # (1.1) SVC with original features
        for k in kernels:
            svc = SVC(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
            svc.SVC_train(k)
            r = svc.SVC_validation()
            result_name = 'SVC-' + k
            result[result_name] = r
        print(result)

    # (1.2) SVC with PCA features
        # preprocessing
        n_component = [0.5, 0.7, 0.9]
        for n_c in n_component:
            this_X_train, new_features = preprocess_cls.PCA(X_train, features, n_c)
            this_X_val, new_features = preprocess_cls.PCA(X_val, features, len(new_features))
            this_X_test, new_features = preprocess_cls.PCA(X_test, features, len(new_features))
            for k in kernels:
                svc = SVC(this_X_train, this_X_val, this_X_test, y_train, y_val, y_test, new_features, targets_scored_col_name)
                svc.SVC_train(k)
                r = svc.SVC_validation()
                result_name = f'PCA({n_c}-SVC-)' + k
                result[result_name] = r
        print(result)

    # (1.3) SVC with statistic features
        # preprocessing
        this_X_train, new_features = preprocess_cls.feature_statistic(X_train)
        this_X_val, new_features = preprocess_cls.feature_statistic(X_val)
        this_X_test, new_features = preprocess_cls.feature_statistic(X_test)
        for k in kernels:
            svc = SVC(this_X_train, this_X_val, this_X_test, y_train, y_val, y_test, new_features, targets_scored_col_name)
            svc.SVC_train(k)
            r = svc.SVC_validation()
            result_name = 'Statistic-SVC-' + k
            result[result_name] = r
        print(result)



    # (2) Random Forest
    number_trees = [200, 300, 400]
    if model_name=='RF':
    # (2.1) RF with original features
        for n in number_trees:
            rfc = RandomForest(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
            rfc.RandomForest_train(n)
            r = rfc.RandomForest_validation()
            result_name = f'RF-{n}'
            result[result_name] = r
        print(result)

    # (2.2) RF with PCA features
        # preprocessing
        n_component = [0.5, 0.9]
        for n_c in n_component:
            this_X_train, new_features = preprocess_cls.PCA(X_train, features, n_c)
            this_X_val, new_features = preprocess_cls.PCA(X_val, features, n_c)
            this_X_test, new_features = preprocess_cls.PCA(X_test, features, n_c)

            for n in number_trees:
                rfc = RandomForest(this_X_train, this_X_val, this_X_test, y_train, y_val, y_test, new_features, targets_scored_col_name)
                rfc.RandomForest_train(n)
                r = rfc.RandomForest_validation()
                result_name = f'PCA({n_c}-RF-{n})'
                result[result_name] = r
        print(result)

    # if model=='2-3':
    # (2.3) RF with statistic features
        # preprocessing
        this_X_train, new_features = preprocess_cls.feature_statistic(X_train)
        this_X_val, new_features = preprocess_cls.feature_statistic(X_val)
        this_X_test, new_features = preprocess_cls.feature_statistic(X_test)
        for n in number_trees:
            rfc = RandomForest(this_X_train, this_X_val, this_X_test, y_train, y_val, y_test, new_features, targets_scored_col_name)
            rfc.RandomForest_train(n)
            r = rfc.RandomForest_validation()
            result_name = f'Statistic-RF-{n}'
            result[result_name] = r
        print(result)
    return result

# testing the best model
if __name__ == "__main__":
    # load the data preprocess tools and data tools
    preprocess_cls = Preprocess()
    data_cls = Data()

    # load the data
    X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name = data_cls.Dataset_Methodology()

    # training and validate different models
    if False:  # only outputs test result for the submission
        for model_name in ['SVC', 'RF']:
            result = training(model_name, X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
            print(result)

    # test the best model
    """
    After the training process, the best model is SVC model with kernel = 'rbf'.
    """
    svc = SVC(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
    acc, loss = svc.SVC_test('rbf')




