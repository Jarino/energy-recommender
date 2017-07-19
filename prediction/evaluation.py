import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import utils.metrics as metrics

warnings.filterwarnings('error')

class Evaluation():
    def __init__(self, target, predicted, df_test, df_train):
        y_true = df_test[target]
        y_train = df_train[target]
        self.mape = metrics.mape(y_true, predicted)
        try:
            self.mase = metrics.mase(y_train, y_true, predicted)
        except RuntimeWarning:
            # this happens when all data are zero:
            self.mase = 1
        self.mse  = mean_squared_error(y_true, predicted)
        self.rmse = np.sqrt(self.mse)
        self.mae  = mean_absolute_error(y_true, predicted)

    def __str__(self):
       return 'MAPE: %f, MASE %f, MSE: %f, RMSE: %f, MAE: %f' % (self.mape, self.mase, self.mse, self.rmse, self.mae)
