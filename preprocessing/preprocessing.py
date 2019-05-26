import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta as td
sys.path.append("..")
from lsh.similarity import TimeSeriesLSH


class TimeSeriesPreprocessor:
    def __init__(self, window_size_seconds = 7200, window_shift = 3600, \
            response_variable = 'y', ts_variable = 'timestamp', label_variable = 'label'):
        self.window_size_seconds = window_size_seconds
        self.response_variable = response_variable
        self.ts_variable = ts_variable
        self.window_shift = window_shift
        self.label_variable = label_variable

    def make_dataset(self, series, labels):
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
            mask = (labels[self.ts_variable] >= start_date) & (labels[self.ts_variable] < end_date)
            sublabels = labels.loc[mask].copy()
            subseries.loc[subseries.index,self.label_variable] = 0
            subseries[self.label_variable] = sublabels
            result.append(subseries)
        return result

    def group_by_hour(self, metric):
        grouped_by_hour = metric.groupby(
             [metric['timestamp'].map(lambda x : x.year).rename('year'),
              metric['timestamp'].map(lambda x : x.month).rename('month'),
              metric['timestamp'].map(lambda x : x.day).rename('day'),
              metric['timestamp'].map(lambda x : x.hour).rename('hour')]).count()
        return grouped_by_hour
