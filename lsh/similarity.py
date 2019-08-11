import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from lsh.hashindex import HashIndex
from lsh.minhash import Minhash


class TimeSeriesLSHException(Exception):
    pass


class TimeSeriesLSH:
    '''
    Time series hashing algorithm based on the following paper:

    [NIPS Time Series Workshop 2016] SSH (Sketch, Shingle, & Hash) for Indexing Massive-Scale Time Series.
    by Chen Luo, Anshumali Shrivastava (https://arxiv.org/abs/1610.07328)

    Parameters
    ----------
    W : int
        the number of hashtables in the index
    sigma : int
        sliding window step
    shingle_size: int
        the size of shingle
    hash_tables: int
        the number of hash tables in the hash
    response_variables: _encode_string
        the name of response variable column
    random_seed: int
        the random seed
    '''
    def __init__(self,
                 W,
                 sigma=1,
                 shingle_size=15,
                 hash_tables=128,
                 response_variable='y',
                 random_seed=42):
        np.random.seed(random_seed)
        self.W = W
        self.R = np.random.rand(W)
        self.sigma = sigma
        self.shingle_size = shingle_size
        self.index = HashIndex(hash_tables=hash_tables)
        self.minhash = Minhash(permutation_count=hash_tables)
        self.response_variable = response_variable

    def fit(self, time_series):
        """Fit function that perform indexing of timeseries.
        Args:
            time_series: the array of timeseries pandas frames
        Returns:
            None
        """
        for idx, ts in enumerate(time_series):
            shingles = self._series_shingles(ts)
            hash = self._hash_shingles(shingles)
            self.index.index({"series": ts, "idx": idx}, hash)

    def _series_shingles(self, series):
        """Makes shingles out of pandas frame
        Args:
            series: pandas frame with time series
        Returns:
            list of shingles
        """
        znorm = StandardScaler().fit_transform(
            series[self.response_variable].values.reshape(-1, 1))
        bits = self._series_to_bit_string(znorm.squeeze())
        return self._bits_to_shingles(bits)

    def _series_to_bit_string(self, series):
        """Makes a list of bit strings from timeseries.
        Args:
            series: extracts from time series list of bit strings
        Returns:
            list of bit strings
        """
        result = []
        for i in range(0, len(series) - self.W + 1, self.sigma):
            window = series[i:(i + self.W)]
            result.append(np.sign(window.dot(self.R)))
        return result

    def _bits_to_shingles(self, bits):
        """Makes a weighted list of shingles out of list of bit strings by dedublicating the elements in the list.
           This is analogous to getting the frequencies of words from the document.
        Args:
            bits: extracts from time series list of bit strings
        Returns:
            list of pairs [shingle, occurence_count]
        """
        result = {}
        for i in range(0, len(bits) - self.shingle_size + 1, 1):
            shingle = self._to_shingle_str(bits[i:(i + self.shingle_size)])
            if shingle not in result:
                result[shingle] = 1
            else:
                result[shingle] += 1
        arr = []
        for key, value in result.items():
            arr.append([key, value])
        return arr

    def _to_shingle_str(self, shingle_window):
        """Converts shingle window into compact bit string. (Assuming the shingle window is less than 32)
        Args:
            shingle_window: window of [1, -1, -1, ... ]
        Returns:
            integer value keeping bit representation of the shingle window where
            bit set to 1 if the corresponding value in the shingle window is >= 0,
            or set to 0 - otherwise
        """
        if len(shingle_window) > 32:
            raise TimeSeriesLSHException(
                "Expected shingle window of size < 32")
        result = 0
        for idx, c in enumerate(shingle_window):
            result = self.set_bit(result, idx, c >= 0)
        return result

    def set_bit(self, value, index, x):
        """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value.
        Args:
            value: the value where bit is set
            index: index of the bit
            x: bit value
        Returns:
            integer value with bit set
        """
        mask = 1 << index  # Compute mask, an integer with just bit 'index' set.
        value &= ~mask  # Clear the bit indicated by the mask (if x is False)
        if x:
            value |= mask  # If x was True, set the bit indicated by the mask.
        return value

    def _hash_shingles(self, shingles):
        """Calculates LSH value using consistent weighted sampling schema.
        Args:
            singles: list of pairs [shingle, occurence_count]
        Returns:
            np array of hash pairs
        """
        return self.minhash.weighted_minhash(shingles,
                                             np.power(2, self.shingle_size))

    def query(self, query):
        """Query similar time series using weighted jaccard similarity.
        Args:
            query: pandas frame with time series
        Returns:
            a list of objects {"object": series, "similarity": similarity}
        """
        query_shingles = self._series_shingles(query)
        query_hash = self._hash_shingles(query_shingles)
        similar_items = self.index.query(query_hash)
        result = []
        for value in similar_items:
            series = value["object"]
            hash = value["hash"]
            similarity = self.minhash.weighted_jaccard(hash, query_hash)
            result.append({"object": series, "similarity": similarity})
        return result
