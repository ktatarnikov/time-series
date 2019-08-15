import unittest

import numpy as np
import pandas as pd
from numpy import testing
from sklearn.model_selection import train_test_split

from models.common import HyperParams
from models.lstm import LSTMAutoencoder, LSTMAutoencoderParams
from models.xgboost import XGBoostModel
from preprocessing.feature_engineering import TimeSeriesFeatureEngineering
from preprocessing.helper import TimeseriesHelper
from preprocessing.preprocessing import TimeSeriesPreprocessor
from preprocessing.test_common import make_labels, make_series


class XGBoostIntegrationTest(unittest.TestCase):
    def test_pipeline(self):
        input_variables = ['y']
        output_variable = 'label'
        train_test_split_ratio = 0.2
        train_valid_split_ratio = 0.2
        seed = 42

        helper = TimeseriesHelper()
        ec2_request_latency_system_failure_file = "realKnownCause/ec2_request_latency_system_failure.csv"
        ec2_cpu_utilization_failure_file = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
        Twitter_volume_AMZN_file = "realTweets/Twitter_volume_AMZN.csv"
        machine_temperature_system_failure_file = "realKnownCause/machine_temperature_system_failure.csv"

        series_name = "machine_temperature_system_failure"
        roll_shift = 20

        data_frame = helper.load_labeled_series(
            f"realKnownCause/{series_name}.csv")
        print(f"Calculating for {roll_shift}")
        preprocessor = TimeSeriesPreprocessor(window_size_seconds=7200,
                                              window_shift=3600,
                                              horizon_shift_seconds=3600,
                                              probe_period_seconds=300)

        feature_engineering = TimeSeriesFeatureEngineering(
            x_columns=input_variables,
            roll_shift=roll_shift,
            ts_variable='timestamp',
            y_column=output_variable)

        series = preprocessor.prepare_series(data_frame,
                                             input_vars=input_variables,
                                             output_vars=[output_variable],
                                             numeric_vars=input_variables,
                                             auto_impute=["y"])

        dataset = feature_engineering.make_features(series)

        input_features = list(dataset.columns)
        input_features.remove(output_variable)

        X_train, X_test, y_train, y_test = train_test_split(
            dataset[input_features],
            dataset[output_variable],
            test_size=train_test_split_ratio,
            random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=train_valid_split_ratio,
            random_state=seed)

        model = XGBoostModel()
        print("class distribution in original training data.\n",
              y_train.groupby(by=lambda v: y_train.loc[v]).count())
        X_upsampled, Y_upsampled = feature_engineering.class_imbalance_fix(
            X_train, y_train)
        elements, counts_elements = np.unique(Y_upsampled, return_counts=True)
        print("class distribution in upsampled training data.")
        print(elements)
        print(counts_elements)

        model.fit(X_upsampled, Y_upsampled, X_valid.as_matrix(),
                  y_valid.as_matrix())

        Y_train_pred = model.predict(X_train.as_matrix())
        helper.print_results("Training",
                             helper.evaluate(y_train, Y_train_pred))

        Y_valid_pred = model.predict(X_valid.as_matrix())
        helper.print_results("Validation",
                             helper.evaluate(y_valid, Y_valid_pred))

        y_test_pred = model.predict(X_test.as_matrix())
        evaluation_result = helper.evaluate(y_test, y_test_pred)
        helper.print_results("Test", evaluation_result)
        self.assertGreater(evaluation_result["f1_score"], 0.0)
