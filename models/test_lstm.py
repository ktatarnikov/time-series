from models.lstm import LSTM, LSTMParams
from models.common import HyperParams

import unittest
import numpy as np
from numpy import testing
import pandas as pd
from preprocessing.test_common import make_labels, make_series

class LSTMTest(unittest.TestCase):

  def test_fit_transform(self):
      vae = LSTM(lstm_params = LSTMParams(), hyper_params = HyperParams(epoch_count=2))
