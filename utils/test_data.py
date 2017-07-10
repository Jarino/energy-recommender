""" Unit tests for data utils"""

from datetime import datetime

import pandas as pd

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
