import collections
import copy
import numpy as np

class WeightedMinHashGenerator(object):
    def __init__(self, dim, sample_size=128, seed=1):
        self.dim = dim
        self.sample_size = sample_size
        self.seed = seed
        generator = np.random.RandomState(seed=seed)
        self.rs = generator.gamma(2, 1, (sample_size, dim)).astype(np.float32)
        self.ln_cs = np.log(generator.gamma(2, 1, (sample_size, dim))).astype(np.float32)
        self.betas = generator.uniform(0, 1, (sample_size, dim)).astype(np.float32)

    def minhash(self, v):
        if not isinstance(v, collections.Iterable):
            raise TypeError("Input vector must be an iterable")
        if not len(v) == self.dim:
            raise ValueError("Input dimension mismatch, expecting %d" % self.dim)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        elif v.dtype != np.float32:
            v = v.astype(np.float32)
        hashvalues = np.zeros((self.sample_size, 2), dtype=np.int)
        vzeros = (v == 0)
        if vzeros.all():
            raise ValueError("Input is all zeros")
        v[vzeros] = np.nan
        vlog = np.log(v)
        for i in range(self.sample_size):
            t = np.floor((vlog / self.rs[i]) + self.betas[i])
            ln_y = (t - self.betas[i]) * self.rs[i]
            ln_a = self.ln_cs[i] - ln_y - self.rs[i]
            k = np.nanargmin(ln_a)
            hashvalues[i][0], hashvalues[i][1] = k, int(t[k])
        return hashvalues
