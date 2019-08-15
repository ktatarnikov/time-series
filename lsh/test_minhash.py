import unittest

import numpy as np
from numpy import testing

from lsh.minhash import Minhash


class MinhashTest(unittest.TestCase):
    def test_different_objects(self):
        minhash = Minhash()
        hashes1 = minhash.minhash(["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.minhash(["26", "27", "28", "29", "30", "31"])
        testing.assert_array_equal(0, minhash.jaccard(hashes1, hashes2))

    def test_different_objects(self):
        minhash = Minhash()
        hashes1 = minhash.minhash(["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.minhash(["13", "14", "15", "29", "30", "31"])
        testing.assert_array_equal(0.3125, minhash.jaccard(hashes1, hashes2))

    def test_the_same_objects(self):
        minhash = Minhash()
        hashes1 = minhash.minhash(["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.minhash(["10", "11", "12", "13", "14", "15"])
        testing.assert_array_equal(1.0, minhash.jaccard(hashes1, hashes2))

    def test_weighted_different_objects(self):
        minhash = Minhash()
        hashes1 = minhash.weighted_minhash(
            ["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.weighted_minhash(
            ["26", "27", "28", "29", "30", "31"])
        testing.assert_array_equal(0, minhash.jaccard(hashes1, hashes2))

    def test_weighted_different_objects(self):
        minhash = Minhash()
        hashes1 = minhash.weighted_minhash(
            [[10, 1], [11, 1], [12, 1], [13, 1], [14, 1], [15, 1]], 32)
        hashes2 = minhash.weighted_minhash(
            [[16, 1], [17, 1], [18, 1], [19, 1], [20, 1], [21, 1]], 32)
        testing.assert_equal(0.0, minhash.weighted_jaccard(hashes1, hashes2))

    def test_weighted_the_same_objects(self):
        minhash = Minhash()
        hashes1 = minhash.weighted_minhash(
            [[10, 5], [11, 5], [12, 5], [13, 5], [14, 5], [15, 5]], 16)
        hashes2 = minhash.weighted_minhash(
            [[10, 5], [11, 5], [12, 5], [13, 5], [14, 5], [15, 5]], 16)
        testing.assert_equal(1.0, minhash.weighted_jaccard(hashes1, hashes2))

    def test_weighted_the_same_objects_differnt_weights(self):
        minhash = Minhash()
        hashes1 = minhash.weighted_minhash(
            [[10, 1], [11, 1], [12, 1], [13, 1], [14, 1], [15, 1]], 16)
        hashes2 = minhash.weighted_minhash(
            [[10, 5], [11, 5], [12, 5], [13, 5], [14, 5], [15, 5]], 16)
        testing.assert_equal(minhash.weighted_jaccard(hashes1, hashes2),
                             0.15625)

        hashes1 = minhash.weighted_minhash(
            [[10, 4], [11, 5], [12, 6], [13, 1], [14, 1], [15, 1]], 16)
        hashes2 = minhash.weighted_minhash(
            [[10, 5], [11, 5], [12, 5], [13, 5], [14, 5], [15, 5]], 16)
        testing.assert_equal(minhash.weighted_jaccard(hashes1, hashes2),
                             0.515625)

    def test_weighted_the_same_objects(self):
        minhash = Minhash()
        hashes1 = minhash.minhash(["10", "11", "12", "13", "14", "15"])
        hashes2 = minhash.minhash(["10", "11", "12", "13", "14", "15"])
        testing.assert_array_equal(1.0, minhash.jaccard(hashes1, hashes2))

    def test_weighted_the_same_objects_different_weights(self):
        minhash = Minhash()
        hashes1 = minhash.weighted_minhash(
            [[10, 0], [11, 0], [12, 1], [13, 0], [14, 1], [15, 0]], 20)
        hashes2 = minhash.weighted_minhash(
            [[10, 1], [11, 2], [12, 0], [13, 1], [14, 0], [15, 1]], 20)
        testing.assert_equal(0.0, minhash.weighted_jaccard(hashes1, hashes2))
