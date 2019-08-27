import tempfile
import unittest

import numpy as np
from numpy import testing

from mf.factor import TRMF
from preprocessing.test_common import (make_labels, make_multi_series,
                                       make_series, sinwave)


class TestTRMF(unittest.TestCase):
    def test_factorize_matrix(self):
        window_millis = 60 * 5 * 1000
        M = 3
        N = 100
        Y = np.random.rand(M, N)
        for i in range(0, M):
            Y[i, :] = np.array(sinwave(freq=(i + 1) * 5, samples=N))
        np.random.seed(42)
        trmf = TRMF(M=M, N=N, factors=3, lambda_f=0.1, alpha=0.001)
        prev_loss = 1
        for i in range(1, 10000):
            loss = trmf.loss(Y)
            if i % 50 == 0:
                print(f"loss: {loss}, diff: {np.abs(loss-prev_loss)}")
            prev_loss = loss
            trmf.update_F(Y)
            trmf.update_X(Y)
            if loss < 0.01:
                break
        # print()

        # expected = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 2]],
        #                     dtype=np.float64)
        # testing.assert_array_equal(expected, actual)
