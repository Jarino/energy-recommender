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
    """
    Compute a distance matrix from input data based on provided distance metric
    """
    dist_mat = np.zeros([len(data), len(data)])
    indices = np.triu_indices(len(data),1)
    for x,y in zip(indices[0], indices[1]):
        dist_mat[x,y] = metric(data[x,:], data[y,:])[0]
        
    dist_mat += dist_mat.T

    return dist_mat


def add_weekdays(dataframe):
    """
    Computes weekdays (0-6) and weekends (0-1), input dataframe has to contain columns
    year, month and day
    Args:
        dataframe (pandas.DataFrame)
    Return:
        copy of input pandas.DataFrame containing two added columns
    """
    datetimes = [datetime(int(vals['year']),
                          int(vals['month']), 
                          int(vals['day'])) for index, vals in dataframe.iterrows()]
    dataframe['weekday'] = [x.weekday() for x in datetimes]
    dataframe['weekend'] = [0 if not x.weekday() in [5,6] else 1 for x in datetimes]
    return dataframe


def add_artificial_features(data, target_feature):
    """
    Add first and second order difference, as well as previos consumption up
    to three hours.
    Args:
        data (pandas.DataFrame): input dataframe
        target_feature (string): name of column for feature creation
    Returns:
        pandas.DataFrame: copy of dataframe with added features
    """

    # first difference
    diff = np.diff(data[target_feature])
    data['diff'] = [0,0, *diff[:-1]]

    # second difference
    diff2 = np.diff(data[target_feature], n=2)
    data['diff2'] = [0, 0, 0, *diff2[:-1]]

    # consumption in previous hour
    data['prev_con'] = [0, *data[target_feature][:-1]]

    # consumption in pre-previous hour
    data['prev_con2'] = [0, 0, *data[target_feature][:-2]]

    data['prev_con3'] = [0, 0, 0,*data[target_feature][:-3]]
    
    return data