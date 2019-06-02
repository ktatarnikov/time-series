import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta as td
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# TODO imputation

class TimeSeriesPreprocessor:
    def __init__(self, window_size_seconds = 7200, window_shift = 3600, horizon_shift_seconds = 300, scaler = StandardScaler(), \
            ts_variable = 'timestamp'):
        self.window_size_seconds = window_size_seconds
        self.ts_variable = ts_variable
        self.window_shift = window_shift
        self.scaler = scaler
        self.horizon_shift_seconds = horizon_shift_seconds

    def make_dataset_from_series_and_labels(self, series, input_vars, output_vars, numeric_vars, auto_impute = []):
        series = self.clean(series, input_vars, output_vars, auto_impute)
        series = self.impute(series, input_vars, output_vars, auto_impute)
        series = self.scale(series, numeric_vars)
        return self.split_into_windows(series, input_vars, output_vars)

    def split_into_windows(self, series, input_variables, output_variables):
        series = series.sort_values(by = self.ts_variable)
        start_date = pd.Timestamp(series[self.ts_variable][0]) - pd.Timedelta(seconds = self.window_shift)
        end_date = pd.Timestamp(series[self.ts_variable][0]) + pd.Timedelta(seconds = self.window_shift)
        series_begin = pd.Timestamp(series[self.ts_variable][0])
        series_end = pd.Timestamp(series[self.ts_variable][len(series[self.ts_variable]) - 1]) - pd.Timedelta(seconds = self.horizon_shift_seconds)

        nanos_in_second = 1000 * 1000 * 1000
        total_seconds = int((series_end - series_begin).delta / nanos_in_second)
        number_of_windows = int((total_seconds) / self.window_shift)
        result = []

        for window_num in range(0, number_of_windows):
            start_date = start_date + pd.Timedelta(seconds = self.window_shift)
            end_date = end_date + pd.Timedelta(seconds = self.window_shift)
            horizon_shift_date = end_date + pd.Timedelta(seconds = self.horizon_shift_seconds)

            input_mask = (series[self.ts_variable] >= start_date) & (series[self.ts_variable] < end_date)
            input_cut = series.loc[input_mask].copy()
            input = input_cut[input_variables]

            output_mask = (series[self.ts_variable] >= end_date) & (series[self.ts_variable] < horizon_shift_date)
            output_cut = series.loc[output_mask].copy()
            output = output_cut[output_variables]
            result.append([input, output])
        return result

    def scale(self, series, numeric_vars):
        if self.scaler is not None:
            series[numeric_vars] = (series[numeric_vars] - series[numeric_vars].mean())/series[numeric_vars].std(ddof=0)
        return series

    def impute(self, series, input_variables, output_variables, auto_impute):
        # series.groupby(self.ts_variable)
        return series

    def clean(self, series, input_variables, output_variables, auto_impute):
        missing_values = sum(series[auto_impute].isnull().sum().values)
        if missing_values > 0:
            series.sort_values(self.ts_variable)
            to_fill = series[auto_impute]
            to_fill.fillna(method='ffill', inplace=True)
            to_fill.fillna(method='bfill', inplace=True)
            series[auto_impute] = to_fill
        return series
