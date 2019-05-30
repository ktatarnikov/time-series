import unittest
import tempfile
import numpy as np
from numpy import testing
import pandas as pd
from .preprocessing import TimeSeriesPreprocessor
from .test_common import make_labels, make_series

class TimeSeriesPreprocessorTest(unittest.TestCase):

  def test_make_dataset(self):
      preprocessor = TimeSeriesPreprocessor(window_size_seconds = 300, window_shift=150)
      labels_list = [0, 1, 30, 45, 59]
      input = make_series([i for i in range(0, 60)])

      label_index = input.iloc[labels_list, 0].index
      labels = make_labels(input, indices = label_index)

      dataset = preprocessor.make_dataset_from_series_and_labels(input, labels)

      self.assertEqual(len(dataset), 23)
      total_count = 0
      last = 0
      for idx, window in enumerate(dataset):
          index = (window.index.intersection(labels.timestamp.index))
          self.assertEqual((window.loc[index]['label'] == 1).sum(), len(label_index.intersection(index)))
          total_count += (window.loc[index]['label'] == 1).sum()
          last = index[len(window) - 1]
      self.assertEqual(last, 59)
      self.assertEqual(total_count, 7)
