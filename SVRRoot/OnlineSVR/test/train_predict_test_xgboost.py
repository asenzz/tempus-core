from sklearn import datasets
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pyarma, random
from numba import *


def conv_mat_to_ndarray(mat):
    res = np.asarray(mat)
    res.reshape(mat.n_rows, mat.n_cols)
    #res = np.ndarray(arr, shape=[mat.n_rows, mat.n_cols])
    #for i in range(mat.n_rows):
    #    for j in range(mat.n_cols):
    #        res[i, j] = mat[i, j]
    #print(str(res))
    return res


X_data = pyarma.mat()
X_data.load(str("/mnt/faststore/features_dataset_100_q_svrwave_eurusd_avg_3600_bid_level_10_adjacent_levels_31_lag_400_call_14.csv"))
y_data = pyarma.mat()
y_data.load(str("/mnt/faststore/labels_dataset_100_q_svrwave_eurusd_avg_3600_bid_level_10_adjacent_levels_31_lag_400_call_14.csv"))

X = conv_mat_to_ndarray(X_data)
X_min = np.abs(np.min(X))
y_data += X_min
X += X_min
X_train, X_test, Y_train, Y_test = train_test_split(X, y_data, test_size=0.01)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)
random.seed(0)
np.random.seed(0)


steps = 20  # The number of training iterations
model = xgb.XGBRegressor(objective='reg:pseudohubererror', gpu_id=0, n_jobs=-1,
                         max_depth=4, seed=0, learning_rate=0.01)
#model = xgb.XGBRFRegressor(objective='reg:squarederror', gpu_id=0, n_jobs=-1, max_depth=6, seed=42
                           # min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0, n_estimators=50, tree_method='hist'
#                           )
model.fit(X_train, Y_train)

preds = model.predict(X_test)
mae = 0.

print(preds)
print(Y_test)
for i in range(len(preds)):
    mae += abs(preds[i] - Y_test[i])
mae = mae / len(preds)
mape = 100. * mae / pyarma.mean(y_data)
print("MAE " + str(mae) + " MAPE " + str(mape) + " len " + str(len(preds)))
