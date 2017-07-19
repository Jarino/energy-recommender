import numpy as np
from sklearn.metrics import mean_absolute_error

def mape(y_true, y_pred):
    """
    Returns the mean average percentage error

    Args:
        y_true (numpy.array): vector of actual values
        y_pred (numpy.array): vector of predicted values
    Returns:
        you would never guess

    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mase(y_train, y_true, y_pred):
    n = y_train.shape[0]
    mae = mean_absolute_error(y_true, y_pred)
    d = np.abs(np.diff(y_train)).sum()/(n-1)
    
    return mae/d