import os
import sys
from datetime import date
from datetime import timedelta as td

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


class TimeSeriesPreprocessor:
    '''
    Time series preprocessor.
    Contains a set of preprocessing utilities for creating a dataset from
    pandas dataframe.

    Parameters
    ----------
    window_size_seconds : int
        input sample window size
    window_shift : int
        the window shift seconds
    horizon_shift_seconds: int
        the output or prediction horizon
    scaler: instance of StandardScaler or MinMaxScaler
    probe_period_seconds: int
        the period during which it is expected that at least 1 data point exists
    ts_variable: str
        timestamp variable name in the dataframe
    '''
    def __init__(self,
                 window_size_seconds=7200,
                 window_shift=3600,
                 horizon_shift_seconds=300,
                 scaler=StandardScaler(),
                 probe_period_seconds=300,
                 ts_variable='timestamp'):
        self.window_size_seconds = window_size_seconds
        self.ts_variable = ts_variable
        self.window_shift = window_shift
        self.scaler = scaler
        self.probe_period_seconds = probe_period_seconds
        self.horizon_shift_seconds = horizon_shift_seconds

    def eval_prediction(self,
                        model,
                        series,
                        input_vars,
                        output_vars,
                        prediction_horizon_multiplier=1):
        input_duration = pd.Timedelta(seconds=self.window_size_seconds)
        output_duration = pd.Timedelta(seconds=self.horizon_shift_seconds)
        prediction_horizon_duration = pd.Timedelta(
            seconds=prediction_horizon_multiplier * self.horizon_shift_seconds)
        shift_duration = pd.Timedelta(seconds=self.window_shift)

        series.sort_values(by=[self.ts_variable], inplace=True)
        series_begin = pd.Timestamp(series.iloc[0][self.ts_variable])
        series_end = pd.Timestamp(series.iloc[-1][
            self.ts_variable]) - prediction_horizon_duration - input_duration

        start = pd.Timestamp(series.iloc[0][
            self.ts_variable]) - prediction_horizon_duration + output_duration

        nanos_in_second = 1000 * 1000 * 1000
        total_seconds = int(
            (series_end - series_begin).delta / nanos_in_second)
        number_of_windows = int(total_seconds /
                                prediction_horizon_duration.seconds)

        # Reserving variables
        result = series.copy()
        prediction_vars = [(v, f'output_{v}') for v in output_vars]
        for _, (_, output_var) in enumerate(prediction_vars):
            result[output_var] = np.nan

        for _ in range(0, number_of_windows):
            start = start + prediction_horizon_duration
            end = start + input_duration

            input_mask = (result[self.ts_variable] >=
                          start) & (result[self.ts_variable] < end)
            input_cut = result.loc[input_mask].copy()
            input = input_cut[input_vars].to_numpy()

            input_start = start - output_duration
            # iterate over predictions recursively feeding the output to the input
            for _ in range(0, prediction_horizon_multiplier):
                input_start = input_start + output_duration
                input_end = input_start + input_duration
                output_date = input_end + output_duration

                prediction = model.predict(np.array([input]))
                output_mask = (result[self.ts_variable] >= input_end) & (
                    result[self.ts_variable] < output_date)

                prediction_as_input = prediction[0]
                output = prediction_as_input.transpose()
                input_last = input.shape[0]

                input = np.append(input,
                                  np.zeros(shape=(prediction_as_input.shape[0],
                                                  input.shape[1])),
                                  axis=0)

                # Set output
                for (output_idx, (_, var)) in enumerate(prediction_vars):
                    result.loc[output_mask, var] = output[output_idx]
                # reset input
                for output_idx, (orig_var, _) in enumerate(prediction_vars):
                    if orig_var in input_vars:
                        input_idx = input_vars.index(orig_var)
                        input[input_last:,
                              input_idx] = prediction_as_input[:, output_idx]
                input = input[-input_last:, :]

        series.sort_values(by=[self.ts_variable], inplace=True)
        series.reset_index(drop=True, inplace=True)
        return result

    def make_dataset_from_series_and_labels(self,
                                            series,
                                            input_vars,
                                            output_vars,
                                            numeric_vars,
                                            auto_impute=[]):
        '''
        Executes a number of preprocessing steps on input series.
            1. Time Alignment
                Ensures that at least one data point exists for every self.probe_period_seconds
            2. Imputation
                Impute missing for the variables specified in input_vars, output_vars
            3. Scaling
                Scale variables specified in numeric_vars
            4. Splitting into samples
                Makes a list of samples [<input>, <output>]
        Parameters
        ----------
        series : pd.DataFrame
            input data frame
        input_vars: list[string]
            the list of input variables
        output_vars: list[string]
            the list of output variables
        numeric_vars: list[string]
            the list holding numeric variables, those will be scaled
        auto_impute: list[string]
            the list holding the variables to impute
        Returns
        -------
        a list of samples
            - each sample is a list holding pair [input, output]
            - input (output) is a pd.TimeSeries holding input (output) matrix
            - input (output) matrix has
                - columns that correspond to input_vars (output_vars)
                - rows that correspond to timestamped value
        '''
        series_prepared = self.prepare_series(series, input_vars, output_vars,
                                              numeric_vars, auto_impute)
        return self.split_into_samples(series_prepared, input_vars,
                                       output_vars)

    def prepare_series(self, series, input_vars, output_vars, numeric_vars,
                       auto_impute):
        series = self.time_alignment(series, input_vars, output_vars)
        series = self.impute_missing(series, auto_impute)
        series = self.scale(series, numeric_vars)
        return series

    def split_into_samples(self, series, input_variables, output_variables):
        '''
        Split input series into dataset.

        Parameters
        ----------
        series : pd.DataFrame
            input data frame
        input_variables: list[string]
            the input variable names
        output_variables: list[string]
            the output variable names
        Returns
        -------
            The dataset is a list of pairs [<input>, <output>].
            Input and output are pandas frames that contain input and prediction windows
            with data points for each column.
        '''
        input_duration = pd.Timedelta(seconds=self.window_size_seconds)
        output_duration = pd.Timedelta(seconds=self.horizon_shift_seconds)
        shift_duration = pd.Timedelta(seconds=self.window_shift)

        series = series.sort_values(by=self.ts_variable)
        series_begin = pd.Timestamp(series.iloc[0][self.ts_variable])
        series_end = pd.Timestamp(series.iloc[-1][
            self.ts_variable]) - input_duration - output_duration

        input_start = pd.Timestamp(
            series.iloc[0][self.ts_variable]) - shift_duration
        input_end = input_start + input_duration

        nanos_in_second = 1000 * 1000 * 1000
        total_seconds = int(
            (series_end - series_begin).delta / nanos_in_second)
        number_of_windows = int(total_seconds / self.window_shift)
        result = []

        for window_num in range(0, number_of_windows):
            input_start = input_start + shift_duration
            input_end = input_start + input_duration
            output_date = input_end + output_duration

            input_mask = (series[self.ts_variable] >=
                          input_start) & (series[self.ts_variable] < input_end)
            input_cut = series.loc[input_mask].copy()
            input = input_cut[input_variables]

            output_mask = (series[self.ts_variable] >= input_end) & (
                series[self.ts_variable] < output_date)
            output_cut = series.loc[output_mask].copy()
            output = output_cut[output_variables]
            result.append([input, output])
        return result

    def scale(self, series, numeric_vars):
        if self.scaler is not None:
            series[numeric_vars] = self.scaler.fit_transform(
                series[numeric_vars])
        return series

    def time_alignment(self, series, input_variables, output_variables):
        '''
        Ensure that there is one and only one point exists for the self.probe_period_seconds.
        If the point does not exists then it creates the point with NaN value.
        If there are multiple points exists it downsample the point with mean value.

        Parameters
        ----------
        series : pd.DataFrame
            input data frame
        input_variables: list[string]
            the input variable names
        output_variables: list[string]
            the output variable names
        Returns
        -------
            time-aligned series
        '''
        all_vars = list(set(input_variables).union(output_variables))

        series = series.sort_values(self.ts_variable)
        start_date = pd.Timestamp(series.iloc[0][self.ts_variable])
        end_date = pd.Timestamp(series.iloc[series.shape[0] -
                                            1][self.ts_variable])
        half_window_delta = pd.Timedelta(seconds=self.probe_period_seconds / 2)
        window_delta = pd.Timedelta(seconds=self.probe_period_seconds)

        current_start = start_date - half_window_delta
        current_end = current_start + window_delta

        nanos_in_second = 1000 * 1000 * 1000
        total_seconds = int((end_date - current_end).delta / nanos_in_second)
        num_windows = int(total_seconds / window_delta.seconds)
        to_append = []
        for i in range(0, num_windows):
            mask = (series[self.ts_variable] >=
                    current_start) & (series[self.ts_variable] < current_end)
            number_of_rows_in_window = series.loc[mask].shape[0]
            if number_of_rows_in_window == 0:
                missing_row = {
                    self.ts_variable: current_start + half_window_delta
                }
                for var in input_variables:
                    missing_row[var] = np.nan
                for var in output_variables:
                    missing_row[var] = np.nan
                to_append.append(missing_row)
            elif number_of_rows_in_window > 1:
                means = series.loc[mask, all_vars].mean(axis=0)
                for var in all_vars:
                    series.loc[mask, var] = means[var]

                window_index = series.loc[mask].index
                downsampled_row = {
                    self.ts_variable: current_start + half_window_delta
                }
                for var in input_variables:
                    downsampled_row[var] = series.loc[window_index[0], var]
                for var in output_variables:
                    downsampled_row[var] = series.loc[window_index[0], var]
                to_append.append(downsampled_row)

                for idx in range(0, len(window_index)):
                    series = series.drop(window_index[idx])

            current_start = current_start + window_delta
            current_end = current_end + window_delta
        if len(to_append) > 0:
            series = series.append(to_append, ignore_index=True)
        series.sort_values(by=[self.ts_variable], inplace=True)
        series.reset_index(drop=True, inplace=True)
        return series

    def impute_missing(self, series, auto_impute):
        '''
        Impute missing values for the variables specified in the auto_impute.

        Parameters
        ----------
        series : pd.DataFrame
            input data frame
        auto_impute: list[string]
            the list holding the variable names to impute
        Returns
        -------
            time series with back and forward filled values
        '''
        missing_values = sum(series[auto_impute].isnull().sum().values)
        if missing_values > 0:
            series.sort_values(self.ts_variable, inplace=True)
            to_fill = series[auto_impute]
            to_fill.fillna(method='ffill', inplace=True)
            to_fill.fillna(method='bfill', inplace=True)
            series[auto_impute] = to_fill
        return series
