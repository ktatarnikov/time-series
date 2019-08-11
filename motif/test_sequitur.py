import tempfile
import unittest

import numpy as np
from numpy import testing

from .sequitur import Sequitur


class TestSequitur(unittest.TestCase):
    def test_encode_success(self):
        seq = Sequitur()
        seq.induce("abcdbc")
        testing.assert_array_equal(['R0001 -> b:c : 2'], seq.get_rules())

    def test_encode_complex(self):
        seq = Sequitur()
        input = "abcabdabcabd"
        seq.induce([w for w in input])
        # abcabdabcab
        # AcAdabcab
        # AcAdAcab
        # BAdBab
        # BAdBA
        # CdC
        testing.assert_array_equal([
            'R0001 -> a:b : 4', 'R0002 -> R0001:c : 2',
            'R0003 -> R0002:R0001 : 2', 'R0004 -> R0003:d : 2'
        ], seq.get_rules())
