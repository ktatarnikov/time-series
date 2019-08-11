import numpy as np


class DynamicTimeWarping:
    '''
    Calculates dynamic time warping distance.
    https://en.wikipedia.org/wiki/Dynamic_time_warping

    TODO tests

    '''
    def dist(self, x, y):
        """Calculates distance.
        Args:
            x, y:
                point args
        Returns:
            squared distance
        """
        return (x - y) * (x - y)

    def distance(self, series1, series2, r_size):
        """DTW distance.
        Args:
            series1: numpy array of timeseries1
            series1: numpy array of timeseries2
            r_size: size of Sakoe-Chiba warping band
        Returns:
            dtw distance
        """
        cost = np.full(2 * r_size + 1, np.Infinity, dtype=np.float32)
        cost_prev = np.full(2 * r_size + 1, np.Infinity, dtype=np.float32)
        m = len(series)
        k = 0
        for i in range(0, m):
            k = max(0, r_size - i)
            min_cost = np.Infinity
            for j in range(max(0, i - r_size), min(m - 1, i + r_size) + 1):
                if i == 0 and j == 0:
                    cost[k] = self.dist(series1[0], series2[0])
                    min_cost = cost[k]
                else:
                    if j - 1 < 0 or k - 1 < 0:
                        y = np.Infinity
                    else:
                        y = cost[k - 1]

                    if i - 1 < 0 or k + 1 > 2 * r_size:
                        x = np.Infinity
                    else:
                        x = cost_prev[k + 1]

                    if i - 1 < 0 or j - 1 < 0:
                        z = np.Infinity
                    else:
                        z = cost_prev[k]

                    cost[k] = np.min(np.min(x, y), z) + self.dist(
                        series1[i], series2[j])
                    if cost[k] < min_cost:
                        min_cost = cost[k]
                k += 1

            cost_tmp = cost
            cost = cost_prev
            cost_prev = cost_tmp
        k -= 1
        return cost_prev[k]
