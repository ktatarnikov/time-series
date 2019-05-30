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

class LSTMAutoencoderParams:
    '''
    The LSTM Autoencoder parameter holder

    Parameters
    ----------
    timesteps_back : int
        the number of steps back

    timesteps_forward : int
        the number of time step forward

    n_features : int
        the number of features

    seed : int
        random seed
    '''
    def __init__(self, timesteps_back, timesteps_forward, n_features, seed):
        self.seed = seed
        self.timesteps_back = timesteps_back
        self.n_features = n_features
        self.timesteps_forward = timesteps_forward

class LSTMAutoencoder:
    '''
    Creates LSTM Autoencoder model with lstm_parameters

    Parameters
    ----------
    lstm_params : LSTMAutoencoderParams
        the lstm configuration

    hyper_params : HyperParams
        the hyper parameters
    '''
    def __init__(self, lstm_params, hyper_params):
        self.lstm_params = lstm_params
        self.hyper_params = hyper_params
        self.model = self._create_model(lstm_params);

    def _create_model(self, lstm_params):
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(LSTM(self.lstm_params.timesteps_back, activation='relu', input_shape=(self.lstm_params.timesteps_back, self.lstm_params.n_features), return_sequences=True))
        lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(1, activation='relu'))
        lstm_autoencoder.add(RepeatVector(self.lstm_params.timesteps_forward))
        # Decoder
        lstm_autoencoder.add(LSTM(self.lstm_params.timesteps_forward, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
        lstm_autoencoder.add(TimeDistributed(Dense(self.lstm_params.n_features)))

        lstm_autoencoder.summary()
        return lstm_autoencoder

    def fit(self, X_train, X_valid):

        adam = optimizers.Adam(self.hyper_params.learning_rate)
        self.model.compile(loss = 'mse', optimizer = adam)
        checkpoint = ModelCheckpoint(filepath="lstm_autoencoder.h5", save_best_only=True, verbose=0)
        tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        model_history = self.model.fit(X_train, X_train,
            epochs = self.hyper_params.epoch_count,
            batch_size = self.hyper_params.batch_size,
            validation_data = (X_valid, X_valid),
            callbacks=[tensor_board],
            verbose=2).history
        return model_history

    def predict(self, X_input):
        return self.model.predict(X_input)
