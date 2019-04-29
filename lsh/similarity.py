import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from lsh.hash_func import sha1_hash32
from lsh.minhash import Minhash
from lsh.hashindex import HashIndex

class TimeSeriesLSH:
    def __init__(self, W, sigma = 1, shingle_size = 15, hash_tables = 128):
        self.W = W
        self.sigma = sigma
        self.shingle_size = shingle_size
        self.index = HashIndex(hash_tables = hash_tables)
        self.minhash = Minhash(permutation_count = hash_tables)

    def fit(self, time_series):
        self.time_series = time_series
        for idx, ts in enumerate(self.time_series):
            shingles = self._series_shingles(ts)
            hash = self._hash_shingles(shingles)
            self.index.index(ts, hash)
        return self

    def _series_shingles(self, series):
        znorm = StandardScaler().fit_transform(series.values.reshape(-1, 1))
        bits = self._series_to_bit_string(znorm.squeeze())
        return self._bits_to_shingles(bits)

    def _series_to_bit_string(self, series):
        B = []
        for i in range(0, len(series) - self.W + 1, self.sigma):
            window = series[i : (i + self.W)]
            B.append(np.sign(window.dot(self.R)))
        return B

    def _bits_to_shingles(self, bits):
        result = {}
        for i in range(0, len(bits) - self.shingle_size + 1, 1):
            shingle = self._to_shingle_str(bits[i : (i + self.shingle_size)])
            if shingle not in result:
                result[shingle] = 1
            else:
                result[shingle] += 1
        arr = []
        for key, value in result.items():
            arr.append([key, value])
        return arr

    def _to_shingle_str(self, shingle_window):
        result = 0
        for idx, c in enumerate(shingle_window):
            result = self.set_bit(result, idx, c >= 0)
        return result

    def set_bit(self, value, index, x):
        """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
        mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
        value &= ~mask          # Clear the bit indicated by the mask (if x is False)
        if x:
            value |= mask         # If x was True, set the bit indicated by the mask.
        return value

    def _hash_shingles(self, shingles):
        return self.minhash.weighted_minhash(shingles, np.power(2, self.shingle_size))

    def query(self, query):
        query_shingles = self._series_shingles(query)
        query_hash = self._hash_shingles(query_shingles)
        similar_items = self.index.query_all(query_hash)
        result = []
        for value in similar_items:
            series = value["object"]
            hash = value["hash"]
            similarity = self.minhash.jaccard_weighted(hash, query_hash)
            result.append({"object": series, "similarity": similarity})
        return result
