import unittest
import numpy as np
from numpy import testing
from .minhash import Minhash

class MinhashTest(unittest.TestCase):
    def test_different_objects(self):
        minhash = Minhash()
        hashes1 = minhash.minhash_values(["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.minhash_values(["26", "27", "28", "29", "30", "31"])
        testing.assert_array_equal(0, minhash.jaccard(hashes1, hashes2))

    def test_different_objects(self):
        minhash = Minhash()
        hashes1 = minhash.minhash_values(["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.minhash_values(["13", "14", "15", "29", "30", "31"])
        testing.assert_array_equal(0.3125, minhash.jaccard(hashes1, hashes2))

    def test_the_same_objects(self):
        minhash = Minhash()
        hashes1 = minhash.minhash_values(["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.minhash_values(["10", "11", "12", "13", "14", "15"])
        testing.assert_array_equal(1.0, minhash.jaccard(hashes1, hashes2))

    # def test_weighted_different_objects(self):
    #     minhash = Minhash()
    #     hashes1 = minhash.weighted_minhash(["10", "11", "12", "13", "14", "15"])
    #     hashes2 = minhash.weighted_minhash(["26", "27", "28", "29", "30", "31"])
    #     testing.assert_array_equal(0, minhash.jaccard(hashes1, hashes2))
    #
    # def test_weighted_different_objects(self):
    #     minhash = Minhash()
    #     hashes1 = minhash.weighted_minhash([[10, 0.1], [11, 0.1], [12, 0.1], [13, 0.1], [14, 0.1], [15, 0.1]])
    #     hashes2 = minhash.weighted_minhash([[16, 0.1], [17, 0.1], [18, 0.1], [19, 0.1], [20, 0.1], [21, 0.1]])
    #     testing.assert_equal(0.0, minhash.jaccard_weighted(hashes1, hashes2))
    #
    # def test_weighted_the_same_objects(self):
    #     minhash = Minhash()
    #     hashes1 = minhash.weighted_minhash([[10, 0.5], [11, 0.5], [12, 0.5], [13, 0.5], [14, 0.5], [15, 0.5]])
    #     hashes2 = minhash.weighted_minhash([[10, 0.5], [11, 0.5], [12, 0.5], [13, 0.5], [14, 0.5], [15, 0.5]])
    #     testing.assert_equal(1.0, minhash.jaccard_weighted(hashes1, hashes2))
    #
    # def test_weighted_the_same_objects_differnt_weights(self):
    #     minhash = Minhash()
    #     hashes1 = minhash.weighted_minhash([[10, 0.1], [11, 0.1], [12, 0.1], [13, 0.1], [14, 0.1], [15, 0.1]])
    #     hashes2 = minhash.weighted_minhash([[10, 0.5], [11, 0.5], [12, 0.5], [13, 0.5], [14, 0.5], [15, 0.5]])
    #     testing.assert_equal(0.0, minhash.jaccard_weighted(hashes1, hashes2))
    #
    #     hashes1 = minhash.weighted_minhash([[10, 0.4], [11, 0.5], [12, 0.6], [13, 0.1], [14, 0.1], [15, 0.1]])
    #     hashes2 = minhash.weighted_minhash([[10, 0.5], [11, 0.5], [12, 0.5], [13, 0.5], [14, 0.5], [15, 0.5]])
    #     testing.assert_equal(0.109375, minhash.jaccard_weighted(hashes1, hashes2))
    #
    # def test_weighted_the_same_objects(self):
    #     minhash = Minhash()
    #     hashes1 = minhash.minhash_values(["10", "11", "12", "13", "14", "15"])
    #     hashes2 = minhash.minhash_values(["10", "11", "12", "13", "14", "15"])
    #     testing.assert_array_equal(1.0, minhash.jaccard(hashes1, hashes2))

    def test_weighted_the_same_objects_differnt_weights(self):
        minhash = Minhash()
        hashes1 = minhash.weighted_minhash([[10, 0], [11, 0], [12, 1], [13, 0], [14, 1], [15, 0]], 20)
        hashes2 = minhash.weighted_minhash([[10, 1], [11, 2], [12, 0], [13, 1], [14, 0], [15, 1]], 20)
        testing.assert_equal(0.0, minhash.jaccard_weighted(hashes1, hashes2))
