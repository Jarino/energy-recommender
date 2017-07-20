"""Module encapsulating the training and evaluating the predictions via SVR"""
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn import preprocessing
import numpy as np

import utils.data as dutils

def get_mean_and_std(evals):
    return np.mean(evals), np.std(evals)

def print_evaluations(eval_list):
    mase_mean, mase_std = get_mean_and_std([x.mase for x in eval_list])
    mse_mean, mse_std = get_mean_and_std([x.mse for x in eval_list])
    rmse_mean, rmse_std = get_mean_and_std([x.rmse for x in eval_list])
    mae_mean, mae_std = get_mean_and_std([x.mae for x in eval_list])
    output = 'MASE:\t%f,\t%f\nMSE:\t%f,\t%f\nRMSE:\t%f,\t%f\nMAE:\t%f,\t%f' % (
        mase_mean, mase_std,
        mse_mean, mse_std,
        rmse_mean, rmse_std,
        mae_mean, mae_std
    )
    print(output)


def train_and_eval(df, eval_day, target, features, days_to_train, config=None):
    """
    Train and eval with the SVR

    Args:
        df (pandas.Dataframe): dataframe on which to perform
        eval_day (datetime.datetime): starting evaluation day
        features (list): features
        days_to_train (int): number of training days before the evaluation day
        stantardize (bool): whether to use standardization for data
    Returns:
        (tuple) the predicted values, test values and MAE error
    """
    training_subset = dutils.select_range(df, eval_day - timedelta(days=days_to_train + 1), eval_day - timedelta(days=1))

    X = training_subset[features].as_matrix()
    y = training_subset[target].as_matrix()

    if config:
        clf = svm.SVR(**config)
    else:
        clf = svm.SVR()
    clf.fit(X, y)

    testing_subset = dutils.select_range(df, eval_day, eval_day)

    X_test = testing_subset[features]
    y_test = testing_subset[target]

    y_pred = clf.predict(X_test)

    return y_pred, y_test, dutils.mape(y_test, y_pred), np.sqrt(mean_squared_error(y_pred, y_test))


def eval_on_multiple_days(df, eval_day, target, features, days_to_train, evals):
    """
    Train and eval with the SVR for multiple days

    Args:
        df (pandas.Dataframe): dataframe on which to perform
        eval_day (datetime.datetime): starting evaluation day
        features (list): features
        days_to_train (int): number of training days before the evaluation day
        stantardize (bool): whether to use standardization for data
        evals (int): number of evaluation days (a.k.a. the train and evaluation iterations)
    Returns:
        the average MAPE
    """
    mape_errors = []
    mse_errors = []

    for _ in range(0, evals):
        eval_day += timedelta(days=1)
        
        _, _, mape_error, mse_error = train_and_eval(df, eval_day, target, features, days_to_train)

        mape_errors.append(mape_error)
        mse_errors.append(mse_error)

    return np.mean(mape_errors), np.mean(mse_errors)
