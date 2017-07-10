"""Module encapsulating the training and evaluating the predictions via SVR"""

from datetime import datetime, timedelta

from sklearn import svm
import numpy as np

import utils.data as dutils


def train_and_eval(df, eval_day, features, days_to_train, evals):
    """
    Train and eval with the SVR

    Args:
        df (pandas.Dataframe): dataframe on which to perform
        eval_day (datetime.datetime): starting evaluation day
        features (list): features
        days_to_train (int): number of training days before the evaluation day
        evals (int): number of evaluation days (a.k.a. the train and evaluation iterations)
    Returns:
        the average MAPE
    """
    errors = []

    for _ in range(0, evals):
        eval_day += timedelta(days=1)
        
        training_subset = dutils.select_range(df, eval_day - timedelta(days=days_to_train + 1), eval_day - timedelta(days=1))

        X = training_subset[features].as_matrix()
        y = training_subset['spotreba'].as_matrix()

        clf = svm.SVR()
        clf.fit(X, y)

        testing_subset = dutils.select_range(df, eval_day, eval_day)

        X_test = testing_subset[features]
        y_test = testing_subset['spotreba']

        y_pred = clf.predict(X_test)

        errors.append(dutils.mape(y_pred, y_test))

    return np.mean(errors)