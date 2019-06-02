import unittest
import tempfile
import numpy as np
from numpy import testing
import pandas as pd
from .preprocessing import TimeSeriesPreprocessor
from .helper import TimeseriesHelper
from .test_common import make_labels, make_series

class TimeSeriesPreprocessorTest(unittest.TestCase):

  def test_make_dataset(self):
      preprocessor = TimeSeriesPreprocessor(window_size_seconds = 300, window_shift=150)
      labels_list = [0, 1, 30, 45, 59]
      input = make_series([i for i in range(0, 60)])

      label_index = input.iloc[labels_list, 0].index
      labels = make_labels(input, indices = label_index)
      input['label'] = labels['label']

      input_variables = ['y']
      output_variables = ['y', 'label']

      dataset = preprocessor.make_dataset_from_series_and_labels(
        series = input,
        input_vars = input_variables,
        output_vars = output_variables,
        numeric_vars = ["y"])

      self.assertEqual(len(dataset), 21)
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
          self.assertEqual((output.loc[index]['label'] == 1).sum(), len(label_index.intersection(index)))

          total_count += (output.loc[index]['label'] == 1).sum()
          last = index[len(output) - 1]
      self.assertEqual(last, 59)
      self.assertEqual(total_count, 5)

  def test_impute_series(self):
      preprocessor = TimeSeriesPreprocessor(window_size_seconds = 300, window_shift=150, horizon_shift_seconds=150)
      input_variables = ['y', 'label']
      output_variables = ['y', 'label']

      helper = TimeseriesHelper()

      input = make_series([i for i in range(0, 60)])

      labels_list = [0, 15, 30, 45, 59]
      label_index = input.iloc[labels_list, 0].index
      labels = make_labels(input, indices = label_index)
      input['label'] = labels['label']

      input_variables = ['y']
      output_variables = ['y', 'label']

      input.loc[[10, 11, 20, 40], 'y'] = np.nan

      windows = preprocessor.make_dataset_from_series_and_labels(
        series = input,
        input_vars = input_variables,
        output_vars = output_variables,
        numeric_vars = ["y"],
        auto_impute = ["y"])

      self.assertEqual(len(windows), 22)
      for w in windows:
          self.assertEqual(0, w[0].isnull().sum()[0])
          self.assertEqual(0, w[1].isnull().sum()[0])

  def test_time_alignment(self):

      input_variables = ['y', 'label']
      output_variables = ['y', 'label']
      probe_period_seconds = 60

      preprocessor = TimeSeriesPreprocessor(
        window_size_seconds = 300,
        window_shift=120,
        horizon_shift_seconds=120,
        probe_period_seconds = probe_period_seconds)

      input = make_series([i for i in range(0, 60)])

      labels_list = [0, 15, 30, 45, 59]
      label_index = input.iloc[labels_list, 0].index
      labels = make_labels(input, indices = label_index)
      input['label'] = labels['label']

      input = input.drop(10)
      input = input.drop(11)
      input = input.drop(35)

      windows = preprocessor.make_dataset_from_series_and_labels(
        series = input,
        input_vars = input_variables,
        output_vars = output_variables,
        numeric_vars = ["y"],
        auto_impute = ["y", "label"])

      self.assertEqual(16, len(windows))
      for input_and_output in windows:
          input = input_and_output[0]
          output = input_and_output[1]
          self.assertEqual(0, input.isnull().sum()[0])
          self.assertEqual(0, output.isnull().sum()[0])
          self.assertEqual(8, input.count().sum())
          self.assertEqual(4, output.count().sum())
