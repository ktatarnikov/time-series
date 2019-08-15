import tempfile
import unittest

import numpy as np
import pandas as pd
from numpy import testing
from sklearn.preprocessing import StandardScaler

from preprocessing.feature_engineering import TimeSeriesFeatureEngineering
from preprocessing.helper import TimeseriesHelper
from preprocessing.test_common import make_labels, make_series


class TimeSeriesFeatureEngineeringTest(unittest.TestCase):
    def test_make_dataset(self):
        feature_engineering = TimeSeriesFeatureEngineering(
            x_columns=["x"],
            roll_shift=20,
            ts_variable='timestamp',
            y_column="y")
        labels_list = [0, 1, 30, 45, 59]
        input = make_series([i
                             for i in range(0, 60)]).rename(columns={"y": "x"})

        label_index = input.iloc[labels_list, 0].index
        labels = make_labels(input, indices=label_index)
        input['y'] = labels['label']
        input[['x']] = StandardScaler().fit_transform(input[['x']])

        feature_dataset = feature_engineering.make_features(input)
        self.assertEqual(feature_dataset.shape, (60, 20))
        self.assertEqual(feature_dataset['y'].shape, (60, ))
