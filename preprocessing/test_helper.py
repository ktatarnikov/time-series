import unittest
import tempfile
import numpy as np
from numpy import testing
import pandas as pd
from .helper import TimeseriesHelper
from .test_common import make_labels, make_series

class TimeseriesHelperTest(unittest.TestCase):

  def test_find_max_discord(self):
      helper = TimeseriesHelper()
      input1 = make_series([1, 2, 3, 4, 3, 2, 1])
      input2 = make_series([1, 2, 3, 3, 3, 2, 1])
      max_idx, max_window = helper.find_window_discords([input1, input2])
      self.assertEqual(max_idx, 0)

  def test_find_min_discord(self):
      helper = TimeseriesHelper()
      input1 = make_series([1, 2, 3, 4, 3, 2, 1])
      input2 = make_series([1, 2, 3, -10, 3, 2, 1])
      max_idx, max_window = helper.find_window_discords([input1, input2])
      self.assertEqual(max_idx, 1)

  def test_load_labeled_series(self):
      helper = TimeseriesHelper()
      data_frame = helper.load_labeled_series("realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv")
      testing.assert_array_equal(['timestamp', 'y', 'label'], data_frame.columns)
