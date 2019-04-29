import unittest
import tempfile
import numpy as np
from numpy import testing
import pandas as pd
from .similarity import TimeSeriesLSH


class TimeSeriesLSHTest(unittest.TestCase):

  def test_to_shingle_str(self):
      hash = TimeSeriesLSH(W = 4, shingle_size = 5)
      result = hash._to_shingle_str([0.5, -0.5, 0.5, -0.5, 0.1, -0.9])
      testing.assert_array_equal(int('0b010101', 2), result)

  def test_to_shingle_map(self):
      hash = TimeSeriesLSH(W = 4, shingle_size = 3)
      result = hash._bits_to_shingles([0.5, -0.5, -0.5, 0.1, -0.9])
      testing.assert_array_equal([[1, 1], [4, 1], [2, 1]], result)

  def test_series_shingles(self):
      hash = TimeSeriesLSH(W = 4, shingle_size = 3, sigma = 1)
      hash.R = np.array([0.5, -0.5, 0.5, -0.5])
      series = pd.Series([0.5, -0.5, -0.5, 0.1, -0.9, 0.5, 0.4, 0.5, -0.5, 0.5, -0.5, -0.5, 0.1])
      result = hash._series_shingles(series)
      testing.assert_array_equal([[3, 1], [5, 4], [2, 3]], result)

  def test_fit_query(self):
      hash = TimeSeriesLSH(W = 4, shingle_size = 3, sigma = 1)
      hash.R = np.array([0.5, -0.5, 0.5, -0.5])
      # query = pd.Series([0.51, -0.49, -0.3, 0.11, -0.91, 1.31, 0.41, 1.29, -0.49, 0.51, -0.49, -0.55, 0.12])
      # query = pd.Series([0.5, -0.5, -0.5, 0.1, -0.9, 1.3, 0.4, 1.3, -0.5, 0.5, -0.5, -0.5, 0.1])
      query = pd.Series([0.6, 0.49, 0.3, 0.11, 0.91, 1.31, 0.41, 1.29, 0.49, 0.51, -0.49, -0.55, 0.12])

      series1 = pd.Series([0.5, -0.5, -0.5, 0.1, -0.9, 1.3, 0.4, 1.3, -0.5, 0.5, -0.5, -0.5, 0.1])
      series2 = pd.Series([0.0, 0.0, -2.5, 1.13, 0.1, -0.7, 0.4, 1.3, -0.5, 0.5, -0.5, -0.5, 0.1])
      series3 = pd.Series([1.5, 0.0, 2.5, -1.13, 0.1, 0.0, 0.0, 1.3, 15.5, 10.0, 10.5, 3., 0.1])
      series4 = pd.Series([1.5, 10.0, -2.5, -1.13, 0.1, 0.0, 0.0, 1.3, -15.5, -10.0, 10.5, 0.5, 0.1])
      series5 = pd.Series([1.5, 0.0, 2.5, -1.13, 0.1, 0.0, 0.0, 1.3, 15.5, -10.0, 10.5, -0.5, 0.1])
      series6 = pd.Series([1.5, 10.0, 12.5, -1.13, 0.1, 0.0, 0.0, 1.3, -15.5, 10.0, 10.5, 0.5, 0.1])
      series7 = pd.Series([1.5, 30.0, 2.5, -1.13, 0.1, 0.0, 0.0, 1.3, 15.5, -10.0, 10.5, 0.5, 0.1])
      input = [series1, series2, series3, series4, series5, series6, series7]
      hash.fit(input)
      related = hash.query(query)
      related.sort(key = lambda v: v["similarity"], reverse = True)
      for item in related:
          print(f"idx: {item['object']}, similarity: {item['similarity']}.")
      # testing.assert_array_equal([ 0.51, -0.49, -0.3 ,  0.11, -0.91,  1.31,  0.41,  1.29, -0.49,
      #   0.51, -0.49, -0.55,  0.12], related)
