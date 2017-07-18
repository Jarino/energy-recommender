import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import utils.data as dutils

class Evaluation():
    def __init__(self, target, predicted, df_test):
        y_true = df_test[target]
        self.mape = dutils.mape(y_true, predicted)
        self.mse  = mean_squared_error(y_true, predicted)
        self.rmse = np.sqrt(self.mse)
        self.mae  = mean_absolute_error(y_true, predicted)

    def __str__(self):
       return 'MAPE: %f, MSE: %f, RMSE: %f, MAE: %f' % (self.mape, self.mse, self.rmse, self.mae)
