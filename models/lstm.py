import pandas as pd

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed

from tensorflow import set_random_seed

class LSTMParams:
    def __init__(self, timesteps, n_features, seed = 42):
        self.seed = seed
        self.timesteps = timesteps
        self.n_features = n_features

class LSTM:
    def __init__(self, lstm_params, hyper_params, response_var = 'y', timestamp_var = 'ts'):
        self.response_var = response_var
        self.timestamp_var = timestamp_var
        self.lstm_params = lstm_params
        self.hyper_params = hyper_params
        self.model = self._create_model(lstm_params);

    def _create_model(self, lstm_params):
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(LSTM(self.lstm_params.timesteps, activation='relu', input_shape=(self.lstm_params.timesteps, self.lstm_params.n_features), return_sequences=True))
        lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(1, activation='relu'))
        lstm_autoencoder.add(RepeatVector(self.lstm_params.timesteps))
        # Decoder
        lstm_autoencoder.add(LSTM(timesteps, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
        lstm_autoencoder.add(TimeDistributed(Dense(self.lstm_params.n_features)))

        lstm_autoencoder.summary()
        return lstm_autoencoder

    def fit(self, x_train, x_valid):
        adam = optimizers.Adam(lr)
        self.model.compile(loss = 'mse', optimizer = adam)
        checkpoint = ModelCheckpoint(filepath="lstm_autoencoder.h5", save_best_only=True, verbose=0)
        tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        model_history = self.model.fit(X_train, X_train,
                                        epochs = self.hyper_params.epoch_count,
                                        batch_size = self.hyper_params.batch,
                                        validation_data = (x_valid, x_valid),
                                        verbose=2).history




    def transform(self, window):
        return []
