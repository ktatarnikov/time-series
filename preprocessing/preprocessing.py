import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta as td
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

class TimeSeriesPreprocessor:
    def __init__(self,
            window_size_seconds = 7200,
            window_shift = 3600,
            horizon_shift_seconds = 300,
            scaler = StandardScaler(),
            probe_period_seconds = 300,
            ts_variable = 'timestamp'):
        self.window_size_seconds = window_size_seconds
        self.ts_variable = ts_variable
        self.window_shift = window_shift
        self.scaler = scaler
        self.probe_period_seconds = probe_period_seconds
        self.horizon_shift_seconds = horizon_shift_seconds

    def make_dataset_from_series_and_labels(self, series, input_vars, output_vars, numeric_vars, auto_impute = []):
        series = self.time_alignment(series, input_vars, output_vars)
        series = self.impute_missing(series, input_vars, output_vars, auto_impute)
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

    def time_alignment(self, series, input_variables, output_variables):
        all_vars = list(set(input_variables).union(output_variables))

        series.sort_values(self.ts_variable, inplace=True)
        start_date = pd.Timestamp(series.iloc[0][self.ts_variable])
        end_date = pd.Timestamp(series.iloc[series.shape[0] - 1][self.ts_variable])
        half_window_delta = pd.Timedelta(seconds = self.probe_period_seconds/2)
        window_delta = pd.Timedelta(seconds = self.probe_period_seconds)

        current_start = start_date - half_window_delta
        current_end = current_start + window_delta

        nanos_in_second = 1000 * 1000 * 1000
        total_seconds = int((end_date - current_end).delta / nanos_in_second)
        num_windows = int(total_seconds / window_delta.seconds)

        to_append = []
        for i in range(0, num_windows):
            mask = (series[self.ts_variable] >= current_start) & (series[self.ts_variable] < current_end)
            number_of_rows_in_window = series.loc[mask].shape[0]
            if number_of_rows_in_window == 0:
                missing_row = {self.ts_variable: current_start + half_window_delta}
                for var in input_variables:
                    missing_row[var] = np.nan
                for var in output_variables:
                    missing_row[var] = np.nan
                to_append.append(missing_row)
            elif number_of_rows_in_window > 1:
                means = series.loc[mask, all_vars].mean(axis = 0)
                for var in all_vars:
                    series.loc[mask, var] = means[var]
                window_index = series.loc[mask].index
                for idx in range(1, len(window_index)):
                    series = series.drop(window_index[idx])
            current_start = current_start + window_delta
            current_end = current_end + window_delta
        if len(to_append) > 0:
            series = series.append(to_append, ignore_index=True)
        return series

    def impute_missing(self, series, input_variables, output_variables, auto_impute):
        missing_values = sum(series[auto_impute].isnull().sum().values)
        if missing_values > 0:
            series.sort_values(self.ts_variable, inplace=True)
            to_fill = series[auto_impute]
            to_fill.fillna(method='ffill', inplace=True)
            to_fill.fillna(method='bfill', inplace=True)
            series[auto_impute] = to_fill
        return series
