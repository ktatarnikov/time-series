import tempfile
import unittest

import numpy as np
from numpy import testing

from .sax import SAX


class TestSAX(unittest.TestCase):
    def test_to_windows(self):
        sax = SAX(alpha=3, window_size=3)
        actual = sax._to_windows(np.array([1, 1, 1, 2, 2], dtype=np.float64))
        expected = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 2]],
                            dtype=np.float64)
        testing.assert_array_equal(expected, actual)

    def test_to_std(self):
        sax = SAX(alpha=3, window_size=3)
        actual = sax._to_std(
            np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float64))
        expected = np.array(
            [[-1.224745, 0, 1.224745], [-1.224745, 0, 1.224745],
             [-1.224745, 0, 1.224745]],
            dtype=np.float64)
        testing.assert_array_almost_equal(expected, actual)

    def test_to_paa(self):
        sax = SAX(alpha=3, paa_size=3)
        actual = sax._to_paa_windows(
            np.array([[-2, 0, 2, 0, -1], [2, 3, 4, 4, 3, 2]]))
        expected = np.array([[-1.2, 1.2, -0.6], [2.5, 4, 2.5]],
                            dtype=np.float64)
        testing.assert_array_almost_equal(expected, actual)

    def test_encode_success(self):
        sax = SAX(alpha=3, window_size=5, paa_size=3)
        expected = np.array(["abc", "abc", "abc", "abc", "abc"])
        actual = sax.encode(
            np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.float64))
        testing.assert_array_equal(expected, actual)

        sax = SAX(alpha=4, window_size=5, paa_size=3)
        expected = np.array(
            ["abd", "acd", "acd", "abd", "acd", "acd", "abd", "acd"])
        actual = sax.encode(
            np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.float64))
        testing.assert_array_equal(expected, actual)
