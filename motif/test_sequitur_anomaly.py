import tempfile
import unittest

import numpy as np
from numpy import testing

from motif.sequitur import Sequitur
from motif.sequitur_anomaly import SequiturAnomaly


class TestSequiturAnomaly(unittest.TestCase):
    def test_encode_success(self):
        input = ["abc", "bcd", "cde", "cdr", "abc", "bcd", "xxx", "cde", "cdr"]
        seq = Sequitur()
        seq.induce(input)
        anomaly = SequiturAnomaly(seq)
        anomalies = anomaly.detect(input)
        testing.assert_array_equal([[6, 7]], anomalies)
