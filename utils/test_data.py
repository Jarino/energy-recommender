""" Unit tests for data utils"""

from datetime import datetime

import pandas as pd
import numpy as np

import utils.data as data_utils

def test_select_range():
    df = pd.read_csv('./utils/sample_processed_data.csv')

    start = datetime(2014,1,2)
    end   = datetime(2014,1,2) # select single day

    selected_df = data_utils.select_range(df, start, end)

    assert  len(selected_df) == 24

    start = datetime(2014, 1, 3)
    end   = datetime(2014, 1, 5)

    selected_df = data_utils.select_range(df, start, end)

    assert len(selected_df) == 24 * 3

def test_mape():
    actual = np.array([1,2,3,4,5])
    predicted = np.array([2,1,3,4,6])
    error = data_utils.mape(actual, predicted)
    assert error == 34