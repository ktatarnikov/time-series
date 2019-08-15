import tempfile
import unittest

import numpy as np
import pandas as pd
from numpy import testing

from preprocessing.helper import TimeseriesHelper
from preprocessing.preprocessing import TimeSeriesPreprocessor
from preprocessing.test_common import make_labels, make_series


class TimeSeriesPreprocessorTest(unittest.TestCase):
    def test_make_dataset(self):
        preprocessor = TimeSeriesPreprocessor(window_size_seconds=300,
                                              window_shift=180,
                                              horizon_shift_seconds=300,
                                              probe_period_seconds=60)
        labels_list = [0, 1, 30, 45, 59]
        input = make_series([i for i in range(0, 60)])

        label_index = input.iloc[labels_list, 0].index
        labels = make_labels(input, indices=label_index)
        input['label'] = labels['label']

        input_variables = ['y']
        output_variables = ['y', 'label']

        dataset = preprocessor.make_dataset_from_series_and_labels(
            series=input,
            input_vars=input_variables,
            output_vars=output_variables,
            numeric_vars=["y"])

        self.assertEqual(len(dataset), 16)
        total_count = 0
        last = 0
        for idx, window in enumerate(dataset):
            input = window[0]
            output = window[1]
            # checking length of input and output
            self.assertEqual(5, len(input))
            self.assertEqual(5, len(output))

            testing.assert_array_equal(input_variables, input.columns)
            testing.assert_array_equal(output_variables, output.columns)

            index = (output.index.intersection(labels.index))
            self.assertEqual((output.loc[index]['label'] == 1).sum(),
                             len(label_index.intersection(index)))

            total_count += (output.loc[index]['label'] == 1).sum()
            last = index[len(output) - 1]
        self.assertEqual(last, 54)
        self.assertEqual(total_count, 4)

    def test_eval_prediction_one_var(self):
        preprocessor, input = self._create_preprocessor()
        model = self._create_mock_model()

        input_variables = ['y']
        output_variables = ['y', 'label']

        result = preprocessor.eval_prediction(model=model,
                                              series=input,
                                              input_vars=input_variables,
                                              output_vars=output_variables,
                                              prediction_horizon_multiplier=2)

        self.assertEqual(result['output_y'].sum(), 880.0)
        self.assertEqual(result['output_label'].sum(), 880.0)

    def _create_preprocessor(self):
        preprocessor = TimeSeriesPreprocessor(window_size_seconds=300,
                                              window_shift=180,
                                              horizon_shift_seconds=300,
                                              probe_period_seconds=60)
        labels_list = [0, 1, 30, 45, 59]
        input = make_series([i for i in range(0, 60)])

        label_index = input.iloc[labels_list, 0].index
        labels = make_labels(input, indices=label_index)
        input['label'] = labels['label']
        return preprocessor, input

    def _create_mock_model(self):
        class mockmodel:
            def predict(self, x):
                xs = x.squeeze()
                if len(xs.shape) > 1:
                    xs = xs[:, 0]
                v = np.array(
                    [np.array([xs.transpose(), xs.transpose()]).transpose()])
                return v

        model = mockmodel()
        return model

    def test_eval_prediction_multiple_vars(self):
        preprocessor, input = self._create_preprocessor()
        model = self._create_mock_model()

        input_variables = ['y', 'label']
        output_variables = ['y', 'label']

        result = preprocessor.eval_prediction(model=model,
                                              series=input,
                                              input_vars=input_variables,
                                              output_vars=output_variables,
                                              prediction_horizon_multiplier=2)
        self.assertEqual(result['output_y'].sum(), 880.0)
        self.assertEqual(result['output_label'].sum(), 880.0)

    def test_impute_series(self):
        preprocessor = TimeSeriesPreprocessor(window_size_seconds=300,
                                              window_shift=150,
                                              horizon_shift_seconds=150,
                                              probe_period_seconds=60)
        input_variables = ['y', 'label']
        output_variables = ['y', 'label']

        helper = TimeseriesHelper()

        input = make_series([i for i in range(0, 60)])

        labels_list = [0, 15, 30, 45, 59]
        label_index = input.iloc[labels_list, 0].index
        labels = make_labels(input, indices=label_index)
        input['label'] = labels['label']

        input_variables = ['y']
        output_variables = ['y', 'label']

        input.loc[[10, 11, 20, 40], 'y'] = np.nan

        windows = preprocessor.make_dataset_from_series_and_labels(
            series=input,
            input_vars=input_variables,
            output_vars=output_variables,
            numeric_vars=["y"],
            auto_impute=["y"])

        self.assertEqual(len(windows), 20)
        for w in windows:
            self.assertEqual(0, w[0].isnull().sum()[0])
            self.assertEqual(0, w[1].isnull().sum()[0])

    def test_time_alignment(self):

        input_variables = ['y', 'label']
        output_variables = ['y', 'label']
        probe_period_seconds = 60

        preprocessor = TimeSeriesPreprocessor(
            window_size_seconds=300,
            window_shift=120,
            horizon_shift_seconds=120,
            probe_period_seconds=probe_period_seconds)

        input = make_series([i for i in range(0, 60)])

        labels_list = [0, 15, 30, 45, 59]
        label_index = input.iloc[labels_list, 0].index
        labels = make_labels(input, indices=label_index)
        input['label'] = labels['label']

        input = input.drop(10)
        input = input.drop(11)
        input = input.drop(35)

        input = input.append(input.loc[5].copy(), ignore_index=True)
        input = input.append(input.loc[27].copy(), ignore_index=True)
        input.loc[5, 'y'] = 3
        input.loc[27, 'y'] = 3

        windows = preprocessor.make_dataset_from_series_and_labels(
            series=input,
            input_vars=input_variables,
            output_vars=output_variables,
            numeric_vars=["y"],
            auto_impute=["y", "label"])

        self.assertEqual(26, len(windows))
        for input_and_output in windows:
            input = input_and_output[0]
            output = input_and_output[1]
            self.assertEqual(0, input.isnull().sum()[0])
            self.assertEqual(0, output.isnull().sum()[0])
            self.assertEqual(10, input.count().sum())
            self.assertEqual(4, output.count().sum())
