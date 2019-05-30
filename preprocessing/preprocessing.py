import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta as td
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# TODO imputation
# TODO std

class TimeSeriesPreprocessor:
    def __init__(self, window_size_seconds = 7200, window_shift = 3600, label_shift_seconds = 300, scaler = StandardScaler(), \
            normalize = True, response_variable = 'y', ts_variable = 'timestamp', label_variable = 'label'):
        self.window_size_seconds = window_size_seconds
        self.response_variable = response_variable
        self.ts_variable = ts_variable
        self.window_shift = window_shift
        self.label_variable = label_variable
        self.scaler = scaler
        self.normalize = normalize
        self.label_shift_seconds = label_shift_seconds

    def make_dataset_from_series_and_labels(self, series):
        series = self.clean(series)
        series = self.impute(series)
        series = self.scale(series)
        return self.split_into_windows(series)

    def split_into_windows(self, series):
        series = series.sort_values(by = self.ts_variable)
        start_date = pd.Timestamp(series[self.ts_variable][0]) - pd.Timedelta(seconds = self.window_shift)
        end_date = pd.Timestamp(series[self.ts_variable][0]) + pd.Timedelta(seconds = self.window_shift)
        series_begin = pd.Timestamp(series[self.ts_variable][0])
        series_end = pd.Timestamp(series[self.ts_variable][len(series[self.ts_variable]) - 1])

        nanos_in_second = 1000 * 1000 * 1000
        total_seconds = int((series_end - series_begin).delta / nanos_in_second)
        number_of_windows = int((total_seconds) / self.window_shift)
        result = []

        for window_num in range(0, number_of_windows):
            start_date = start_date + pd.Timedelta(seconds = self.window_shift)
            end_date = end_date + pd.Timedelta(seconds = self.window_shift)

            mask = (series[self.ts_variable] >= start_date) & (series[self.ts_variable] < end_date)
            subseries = series.loc[mask].copy()
            result.append(subseries)
        return result

    def scale(self, series):
        if self.normalize is not None:
            series[self.response_variable] = (series[self.response_variable] - series[self.response_variable].mean())/series[self.response_variable].std(ddof=0)
        return series

    def impute(self, series):
        return series

    def clean(self, series):
        return series
