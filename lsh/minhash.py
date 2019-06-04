import numpy as np
import sys
import hashlib
import math
import struct
from lsh.cws import ConsistentWeightedSampling

_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)

class MinhashError(Exception):
    pass

def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.
    Args:
        data (bytes): the data to generate 32-bit integer hash from.
    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

class Minhash:
    '''
    Minhash

    Parameters
    ----------
    permutation_count : int
        the number of hashtables in the index

    '''
    def __init__(self, permutation_count = 128, hashfunc = sha1_hash32, seed = 42):
        self.seed = seed
        self.permutation_count = permutation_count
        self.hashfunc = hashfunc
        generator = np.random.RandomState(self.seed)
        self.permutations = np.array(
            [(generator.randint(1, _mersenne_prime, dtype=np.uint64),
              generator.randint(0, _mersenne_prime, dtype=np.uint64)) for _ in range(self.permutation_count)],
              dtype=np.uint64).T

    def jaccard(self, values1, values2):
        if len(values1) != len(values2):
            raise MinhashError(f"the length of hashvalues are different: {len(length1)} vs {len(length2)}.")
        return np.float(np.count_nonzero(values1 == values2)) / np.float(len(values1))

    def minhash_values(self, values):
        result = np.ones(self.permutation_count, dtype = np.uint64) * _max_hash
        for value in values:
            result = np.minimum(result, self.minhash_value(value))
        return result

    def minhash_value(self, value):
        hash = self.hash(value)
        a, b = self.permutations
        return np.bitwise_and((a * hash + b) % _mersenne_prime, np.uint64(_max_hash))

    def hash(self, value):
        return self.hashfunc(self._encode_string(value))

    def weighted_minhash(self, weighted_values, dim):
        sampler = ConsistentWeightedSampling(dim)
        weights = np.array(weighted_values)
        dense_array = np.zeros(dim, dtype = np.float64)
        dense_array[weights[:,0]] = weights[:,1]
        return sampler.hash(dense_array)

    def jaccard_weighted(self, values, other):
        # Check how many pairs of (k, t) hashvalues are equal
        intersection = 0
        for this, that in zip(values, other):
            if np.array_equal(this, that):
                intersection += 1
        return float(intersection) / float(len(values))

    def _encode_string(self, value):
        if isinstance(value, str):
            return value.encode("utf-8")
        else:
            return value
