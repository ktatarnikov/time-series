import collections
import copy

import numpy as np


class ConsistentWeightedSamplingError(Exception):
    pass


class ConsistentWeightedSampling(object):
    '''
    Consistent weighted sampling based on the following paper:

    "Improved consistent sampling, weighted minhash and l1 sketching.", by Ioffe, Sergey.
    Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.

    Parameters
    ----------
    dimension : int
        the dimension of the object,
        for document similarity this indicates the size of vocabulary, the number of unique words.
    sample_size : int
        the number of hashes in the sample
    seed: int
        random seed
    '''
    def __init__(self, dimension, sample_size=128, seed=42):
        self.dimension = dimension
        self.sample_size = sample_size
        self.seed = seed
        generator = np.random.RandomState(seed=seed)
        self.rgamma = generator.gamma(2, 1, (sample_size, dimension)).astype(
            np.float32)
        self.ln_cgamma = np.log(generator.gamma(
            2, 1, (sample_size, dimension))).astype(np.float32)
        self.beta_uniform = generator.uniform(
            0, 1, (sample_size, dimension)).astype(np.float32)

    def hash(self, input):
        """Calculates weighted minhash.
        Args:
            input: the vector of weights
        Returns:
            np array of weighted min hash pairs
        """
        if not len(input) == self.dimension:
            raise ConsistentWeightedSamplingError(
                f"Expecting the array of size {self.dimension}.")
        hashvalues = np.zeros((self.sample_size, 2), dtype=np.int)
        input_zeros = (input == 0)
        if input_zeros.all():
            raise ConsistentWeightedSamplingError(
                "Expected nonzero vector of weights.")
        input[input_zeros] = np.nan
        input_log = np.log(input)
        for i in range(self.sample_size):
            t = np.floor((input_log / self.rgamma[i]) + self.beta_uniform[i])
            ln_y = (t - self.beta_uniform[i]) * self.rgamma[i]
            ln_a = self.ln_cgamma[i] - ln_y - self.rgamma[i]
            k = np.nanargmin(ln_a)
            hashvalues[i][0], hashvalues[i][1] = k, int(t[k])
        return hashvalues
