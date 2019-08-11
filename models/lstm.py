import pandas as pd
import tensorflow as tf
from keras import Sequential, optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Model, load_model
from keras.utils import plot_model
from numpy.random import seed
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_recall_fscore_support, recall_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    n_in_features : int
        the number of input features

    n_output_features : int
        the number of output features

    encoder_layers : list of layer_config

    decoder_layers : list of layer_config

    seed : int
        random seed
    '''
    def __init__(self, timesteps_back, timesteps_forward, n_in_features,
                 n_out_features, encoder_layers, decoder_layers, seed):
        self.seed = seed
        self.timesteps_back = timesteps_back
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.timesteps_forward = timesteps_forward
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers


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
        self.model = self._create_model(lstm_params)

    def _create_model(self, lstm_params):
        """Create Autonecoder model using lstm_params.
        Args:
            lstm_params: parameters for lstm model
        Returns:
            keras lstm model
        """
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(
            LSTM(self.lstm_params.timesteps_back,
                 activation='relu',
                 input_shape=(self.lstm_params.timesteps_back,
                              self.lstm_params.n_in_features),
                 return_sequences=True))
        for idx, layer_config in enumerate(self.lstm_params.encoder_layers):
            if idx != len(self.lstm_params.encoder_layers) - 1:
                lstm_autoencoder.add(
                    LSTM(layer_config["size"],
                         activation='relu',
                         return_sequences=True))
            else:
                lstm_autoencoder.add(
                    LSTM(layer_config["size"], activation='relu'))
        lstm_autoencoder.add(RepeatVector(self.lstm_params.timesteps_forward))
        # Decoder
        lstm_autoencoder.add(
            LSTM(self.lstm_params.timesteps_forward,
                 activation='relu',
                 return_sequences=True))
        for layer_config in self.lstm_params.decoder_layers:
            lstm_autoencoder.add(
                LSTM(layer_config["size"],
                     activation='relu',
                     return_sequences=True))
        lstm_autoencoder.add(
            TimeDistributed(Dense(self.lstm_params.n_out_features)))

        lstm_autoencoder.summary()
        return lstm_autoencoder

    def save(self, file_name):
        self.model.save(file_name)

    def load(self, file_name):
        self.model = load_model(file_name)

    def fit(self, X_train, y_train, X_valid, y_valid):
        """Fit lstm model using adam optimizer.
        Args:
            X_train: ndarray of train input
            y_train: ndarray of train output
            X_valid: ndarray of validation input
            y_valid: ndarray of validation output
        Returns:
            model training history
        """
        adam = optimizers.Adam(self.hyper_params.learning_rate)
        self.model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath="lstm_autoencoder.h5",
                                     save_best_only=True,
                                     verbose=0)
        tensor_board = TensorBoard(log_dir='./logs',
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=True)
        model_history = self.model.fit(X_train,
                                       y_train,
                                       epochs=self.hyper_params.epoch_count,
                                       batch_size=self.hyper_params.batch_size,
                                       validation_data=(X_valid, y_valid),
                                       callbacks=[tensor_board],
                                       verbose=2).history
        return model_history

    def predict(self, X_input):
        """Prediction.
        Args:
            X_input: ndarray of test input
        Returns:
            prediction
        """
        return self.model.predict(X_input)
