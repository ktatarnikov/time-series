from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import numpy as np

class SAX:
    def __init__(self, alpha, symbol_size, paa_size = None, window_size = None):
        self.alpha = alpha
        self.symbol_size = symbol_size
        self.window_size = window_size
        self.paa_size = paa_size
        # 1/(α −1),2/(α −1),...,(α −2)/(α −1)
        # (for α = 3, breakpoints = [−0.43,0.43]; for α = 4, breakpoints = [−0.67,0,0.67])
        self.breakpoints = [norm.ppf(float(i)/float(alpha)) for i in range(1, alpha)]
        self.symbols = "abcdefghijklmnopqrstuvwxyz"

    def encode(self, series: np.array):
        # series = StandardScaler().fit_transform(series.reshape(-1, 1)).squeeze()
        windows = self._to_windows(series)
        std = self._to_std(windows)
        paa = self._to_paa_windows(std)
        return self._quantize_windows(paa)

    def _to_windows(self, series):
        result = []
        for i in range(0, len(series) - self.window_size + 1, 1):
            window = series[i : (i + self.window_size)]
            result.append(window)
        return np.array(result)

    def _to_std(self, windows):
        results = []
        for window in windows:
            result = StandardScaler().fit_transform(window.reshape(-1, 1))
            result = result.squeeze()
            results.append(result)
        return np.array(results)

    def _to_paa_windows(self, windows):
        results = []
        for window in windows:
            results.append(self._to_paa(window))
        return np.array(results)

    def _to_paa(self, series):
        series_len = len(series)
        if (series_len == self.paa_size):
            return np.copy(series)
        else:
            res = np.zeros(self.paa_size)
            if (series_len % self.paa_size == 0):
                inc = series_len // self.paa_size
                for i in range(0, series_len):
                    idx = i // inc
                    np.add.at(res, idx, series[i])
                return res / inc
            else:
                for i in range(0, self.paa_size * series_len):
                    idx = i // series_len
                    pos = i // self.paa_size
                    np.add.at(res, idx, series[pos])
                return res / series_len

    def _quantize_windows(self, windows):
        results = []
        for window in windows:
            results.append(self._quantize(window))
        return np.array(results)

    def _quantize(self, series: np.array):
        word = ""
        for point in series:
            symbol = self._point_to_symbol(point)
            word += symbol
        return word

    def _point_to_symbol(self, point):
        for idx, value in enumerate(self.breakpoints):
            if point < value:
                return self.symbols[idx]
        return self.symbols[len(self.breakpoints)]
