import unittest

import numpy as np
import pandas as pd
from numpy import testing
from sklearn.model_selection import train_test_split

from models.common import HyperParams
from models.lstm import LSTMAutoencoder, LSTMAutoencoderParams
from preprocessing.helper import TimeseriesHelper
from preprocessing.preprocessing import TimeSeriesPreprocessor
from preprocessing.test_common import make_labels, make_series


class LSTMAutoencoderTest(unittest.TestCase):
    def test_fit_transform(self):
        input_variables = ['y', 'label']
        output_variables = ['y', 'label']

        helper = TimeseriesHelper()
        # data_frame = helper.load_labeled_series("realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv")
        data_frame = helper.load_labeled_series(
            "realKnownCause/ec2_request_latency_system_failure.csv")

        preprocessor = TimeSeriesPreprocessor(window_size_seconds=7200,
                                              window_shift=3600,
                                              horizon_shift_seconds=3600,
                                              probe_period_seconds=300)

        windows = preprocessor.make_dataset_from_series_and_labels(
            data_frame,
            input_vars=input_variables,
            output_vars=output_variables,
            numeric_vars=["y"],
            auto_impute=["y", 'label'])

        input = np.stack([w[0].values for w in windows])
        output = np.stack([w[1].values for w in windows])

        n_in_features = len(input_variables)
        n_out_features = len(output_variables)
        timesteps_back = input.shape[1]
        timesteps_forward = output.shape[1]
        seed = 42
        epoch_count = 1
        learning_rate = 0.0005
        batch_size = 128
        train_test_split_ratio = 0.2
        train_valid_split_ratio = 0.3

        X_train, X_test, y_train, y_test = train_test_split(
            input, output, test_size=train_test_split_ratio, random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=train_valid_split_ratio,
            random_state=seed)

        lstm_params = LSTMAutoencoderParams(
            timesteps_back=timesteps_back,
            timesteps_forward=timesteps_forward,
            encoder_layers=[{
                "size": 16
            }, {
                "size": 8
            }],
            decoder_layers=[{
                "size": 8
            }, {
                "size": 16
            }],
            n_in_features=n_in_features,
            n_out_features=n_out_features,
            seed=seed)
        hyper_params = HyperParams(epoch_count=epoch_count,
                                   learning_rate=learning_rate,
                                   batch_size=batch_size)
        lstm = LSTMAutoencoder(lstm_params=lstm_params,
                               hyper_params=hyper_params)
        lstm.fit(X_train, y_train, X_valid, y_valid)

        Y_pred = lstm.predict(X_test)

        df_prepared = preprocessor.prepare_series(data_frame,
                                                  input_vars=input_variables,
                                                  output_vars=output_variables,
                                                  numeric_vars=["y"],
                                                  auto_impute=["y", "label"])

        df_predicted = preprocessor.eval_prediction(lstm, df_prepared,
                                                    input_variables,
                                                    output_variables)

        self.assertTrue(df_predicted['output_label'].sum() != 0)
        self.assertTrue(df_predicted['output_y'].sum() != 0)
