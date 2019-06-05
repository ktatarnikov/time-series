import numpy as np
import sys
import hashlib
import math
import struct
from lsh.cws import ConsistentWeightedSampling

class MinhashError(Exception):
    pass

class Minhash:
    '''
    Minhash and weighted minhash calculator.

    Parameters
    ----------
    permutation_count : int
        the number of hashtables in the index
    seed: int
        random seed
    '''
    def __init__(self, permutation_count = 128, seed = 42):
        self.seed = seed
        self.hash_prime = (1 << 61) - 1
        self.max_hash = (1 << 32) - 1
        self.hash_range = (1 << 32)

        self.permutation_count = permutation_count
        generator = np.random.RandomState(self.seed)
        self.permutations = np.array(
            [(generator.randint(1, self.hash_prime, dtype = np.uint64),
              generator.randint(0, self.hash_prime, dtype = np.uint64)) for _ in range(self.permutation_count)],
              dtype=np.uint64).T

    def jaccard(self, values1, values2):
        """Calculates jaccard similarity between 2 arrays of hashes.
        Args:
            values1: the array of hash values
            values2: the array of hash values
        Returns:
            the approximated jaccard similarity
        """
        if len(values1) != len(values2):
            raise MinhashError(f"the length of hashvalues are different: {len(length1)} vs {len(length2)}.")
        return np.float(np.count_nonzero(values1 == values2)) / np.float(len(values1))

    def minhash(self, values):
        """Calculates minhash of values.
        Args:
            values: array values to be hashed
        Returns:
            np array of hashed values
        """
        result = np.ones(self.permutation_count, dtype = np.uint64) * self.max_hash
        for value in values:
            result = np.minimum(result, self._minhash_value(value))
        return result

    def _minhash_value(self, value):
        """Calculates hash value.
        Args:
            value: value to be hashed
        Returns:
            hash
        """
        hash = self._sha_hash(self._encode_string(value))
        a, b = self.permutations
        return np.bitwise_and((a * hash + b) % self.hash_prime, np.uint64(self.max_hash))

    def weighted_jaccard(self, values1, values2):
        """ Approximated jaccard similarity.
            Check how many pairs of (k, t) hashvalues are equal.
        Args:
            values1: array of weighted minhash
            values2: array of weighted minhash
        Returns:
            jaccard similarity
        """
        intersection = 0
        for this, that in zip(values1, values2):
            if np.array_equal(this, that):
                intersection += 1
        return float(intersection) / float(len(values1))

    def weighted_minhash(self, weighted_values, dimension):
        """Calculates jaccard similarity of weighted array.
        Args:
            weighted_values: array of weights
            dimension: the dimension - max value of weight data type, e.g. 2^16
        Returns:
            weighted minhash
        """
        sampler = ConsistentWeightedSampling(dimension = dimension, seed = self.seed)
        weights = np.array(weighted_values)
        dense_array = np.zeros(dimension, dtype = np.float64)
        dense_array[weights[:,0]] = weights[:,1]
        return sampler.hash(dense_array)

    def _encode_string(self, value):
        """encodes string into utf-8 if necessary.
        Args:
            value:
        Returns:
            encoded string
        """
        if isinstance(value, str):
            return value.encode("utf-8")
        else:
            return value

    def _sha_hash(self, data):
        """A 32-bit hash function based on SHA1.
        Args:
            data (bytes): the data to generate 32-bit integer hash from.
        Returns:
            int: an integer hash value that can be encoded using 32 bits.
        """
        return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]
