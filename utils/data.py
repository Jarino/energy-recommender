"""Utility functions for manipulating with Pecan Street dataset"""
import os
from datetime import datetime

import numpy as np
import pandas as pd


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

def load_all_users(data_path, group_by, features):
    files = os.listdir(data_path)

    data = pd.DataFrame()

    for file in files:
        full_path = os.path.join(data_path, file)
        df = pd.read_csv(full_path)
        df = select_range(df, datetime(2014,1,1), datetime(2015,12,31))
        data = data.append(df.groupby(group_by).mean()[features])
        
    data = data.reset_index(drop=True)

    return data

def distance_matrix(data, metric):
    dist_mat = np.zeros([len(data), len(data)])
    indices = np.triu_indices(len(data),1)
    for x,y in zip(indices[0], indices[1]):
        dist_mat[x,y] = metric(data[x,:], data[y,:])[0]
        
    dist_mat += dist_mat.T

    return dist_mat