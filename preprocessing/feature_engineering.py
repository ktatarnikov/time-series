import logging
import os
import sys
from datetime import date
from datetime import timedelta as td

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tsfresh
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_selection.selection import select_features
from tsfresh.utilities.dataframe_functions import impute


class TimeSeriesFeatureEngineering:
    '''
    Time series feature engineering using tsfresh.
    Contains a set of preprocessing utilities for creating a dataset from
    pandas dataframe.

    Parameters
    ----------
    ts_variable: str
        timestamp variable name in the dataframe
    '''
    def __init__(self,
                 x_columns,
                 roll_shift=10,
                 ts_variable='timestamp',
                 y_column="Y"):
        self.ts_variable = ts_variable
        self.y_column = y_column
        self.x_columns = x_columns
        self.roll_shift = roll_shift
        self.random_state = 42
        self.smote_neighbours = 3
        self.smote_ratio = 1.0
        # Muting feature selection warnings
        logging.getLogger(
            "tsfresh.feature_selection.significance_tests").setLevel(
                logging.ERROR)
        logging.getLogger("tsfresh.feature_selection.relevance").setLevel(
            logging.ERROR)
        logging.getLogger("tsfresh.utilities.dataframe_functions").setLevel(
            logging.ERROR)

    def make_features(self, frame):
        frame.sort_values(by=[self.ts_variable], inplace=True)
        rolled_frame = self.roll_dataset(frame, self.y_column)
        labels = self.aggregate_labels(rolled_frame, self.y_column)
        features_frame = self.extract(rolled_frame, labels, self.x_columns,
                                      self.y_column)
        features_frame[self.y_column] = labels.loc[features_frame.index][
            self.y_column]
        return features_frame

    def extract(self, frame, labels, x_columns, y_column):
        extract_columns = ['tsfresh_id', 'sort'] + x_columns
        features = tsfresh.extract_features(frame[extract_columns],
                                            column_id='tsfresh_id',
                                            column_sort='sort',
                                            disable_progressbar=True)
        impute(features)
        selected_features = select_features(features,
                                            labels[y_column],
                                            ml_task="classification")
        return selected_features

    def roll_dataset(self, frame, y_column):
        frame['tsfresh_id'] = 1
        rolled_series = tsfresh.utilities.dataframe_functions.roll_time_series(
            df_or_dict=frame,
            column_id='tsfresh_id',
            column_sort=None,
            column_kind=None,
            rolling_direction=+1,
            max_timeshift=self.roll_shift - 1)
        return rolled_series

    def aggregate_labels(self, rolled_series, y_column):
        class_and_roll = rolled_series[["tsfresh_id", y_column]].copy()
        the_max_class = class_and_roll.groupby(by=["tsfresh_id"],
                                               as_index=False).max()
        the_max_class.set_index(keys="tsfresh_id", inplace=True)
        return the_max_class

    def class_imbalance_fix(self, X, Y, ratio=None):
        smote = SMOTE( \
            random_state = self.random_state, \
            k_neighbors = self.smote_neighbours, \
            sampling_strategy = self.smote_ratio if ratio is None else ratio)
        X_res, Y_res = smote.fit_sample(X, Y)
        return X_res, Y_res
