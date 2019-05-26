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

class VAEParams:
    def __init__(self, original_dim, seed = 42):
        self.seed = seed
        self.input_shape = (original_dim, )
        self.intermediate_dim = 512
        self.batch_size = 128
        self.latent_dim = 2

class VAE:
    def __init__(self, encoder_params, hyper_params, response_var = 'y', timestamp_var = 'ts'):
        self.response_var = response_var
        self.timestamp_var = timestamp_var
        self.encoder_params = encoder_params
        self.hyper_params = hyper_params
        self.model = self._create_model(encoder_params);

    def _create_model(self, params):
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape = params.input_shape, name='encoder_input')
        x = Dense(params.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(params.latent_dim, name='z_mean')(x)
        z_log_var = Dense(params.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(params.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(params.latent_dim,), name='z_sampling')
        x = Dense(params.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(params.original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(params.latent_inputs, outputs, name='decoder')
        decoder.summary()

        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        return vae

    def fit(self, x_train, x_valid):
        self.model.fit(x_train,
            epochs = self.hyper_params.epoch_count,
            batch_size = self.hyper_params.batch_size,
            validation_data=(x_valid, None))

    def transform(self, window):
        return []


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
