"""Utility functions for manipulating with Pecan Street dataset"""

import numpy as np

def select_range(df, start_date, end_date):
    """
    Select subset of dataset according to start and end date.

    Args:
        df (pandas.Dataframe): dataframe from which to select
        param1 (datetime): start datetime
        param2 (datetime): end datetime

    Returns:
        pandas.Dataframe: subset of dataframe

    """

    start_index = df[(df['year'] == start_date.year)
                     & (df['month'] == start_date.month)
                     & (df['day'] == start_date.day)].index[0]

    end_index = df[(df['year'] == end_date.year)
                   & (df['month'] == end_date.month)
                   & (df['day'] == end_date.day)].index[-1]

    return df.iloc[start_index:end_index + 1]


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

