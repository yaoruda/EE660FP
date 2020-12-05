from Preprocess import Preprocess
from Data import Data
from Models import SVC
from sklearn.model_selection import train_test_split

preprocess_cls = Preprocess()
data_cls = Data()
# (1) SVC with original features
# load data
X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name = data_cls.Dataset_Methodology()
# preprocessing
# X_train = preprocess_cls.preprocessing_standard_scaler(X_train)
# X_val = preprocess_cls.preprocessing_standard_scaler(X_val)
# X_test = preprocess_cls.preprocessing_standard_scaler(X_test)
# training model
svc = SVC(X_train, X_val, X_test, y_train, y_val, y_test, features, targets_scored_col_name)
svc.SVC_train()
svc.SVC_validation()

# Stage 2
# transfor to the multi class (single label) question
# def transform_multi_class(y):
#     new_y = []
#     for value in y:
#         labels = np.nonzero(list(y[value]))
#         pass
#     return


# Final Stage
"""
Need the use of GPU with 'cupy' and 'cuml' library! (GPU version of sklearn)
"""

