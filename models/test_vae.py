from models.vae import VAE, VAEParams
from models.common import HyperParams

import unittest
import numpy as np
from numpy import testing
import pandas as pd
from preprocessing.test_common import make_labels, make_series

class VAETest(unittest.TestCase):

  def test_fit_transform(self):
      vae = VAE(encoder_params = VAEParams(), hyper_params = HyperParams(epoch_count=2))