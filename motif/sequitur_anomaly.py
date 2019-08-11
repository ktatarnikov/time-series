import numpy as np


class SequiturAnomaly:
    '''
    Anomaly detection using Sequitur algorithm.

    Time series anomaly discovery with grammar-based compression (http://csdl.ics.hawaii.edu/techreports/2014/14-05/14-05.pdf)
    Senin, P., Lin, J., Wang, X., Oates, T., Gandhi, S., Boedihardjo, A.P., Chen, C., Frankenstein, S., EDBT 2015.

    Parameters
    ----------
    sequitur : Sequitur
        the instance of sequitur algorithm
    density_threshold : int
        the threshold that specifies the anomalous limit of rule density
    anomaly_duration : int
        the number of anomalous points below density threshold that indicate anomaly
    '''
    def __init__(self, sequitur, density_threshold=1, anomaly_duration=1):
        self.sequitur = sequitur
        self.density_threshold = density_threshold
        self.anomaly_duration = anomaly_duration

    def detect(self, words):
        """Detect anomalies by calculating the rule density curve.
           Low rule density indicates irregular pattern in the timeseries and
           idicates anomaly.
        Args:
            words: sequitur words
        Returns:
            np array of anomalies
        """
        self.density_curve = self._calculate_density(words)
        anomalies = self._detect_anomalies(self.density_curve)
        return anomalies

    def _calculate_density(self, words):
        """Calculate rule density using instance of sequitur.
        Args:
            words: a list sequitur words
        Returns:
            np array - rule density curve
        """
        results = np.zeros(len(words))
        rules = self.sequitur.get_grammar().rules.values()
        for rule in rules:
            for pos in rule.get_positions():
                start, end = pos
                for i in range(start, end):
                    results[start:end + 1] += 1
        return results

    def _detect_anomalies(self, density_curve):
        """Detects the anomalies in the density_curve.
        Args:
            density_curve: np.array
                the rule density curve
        Returns:
            list of anomaly points, each is a list of format [start, end]
        """
        result = []
        in_anomaly = False
        start = -1
        for i, density in enumerate(density_curve):
            if density < self.density_threshold:
                in_anomaly = True
                if start == -1:
                    start = i
            else:
                if in_anomaly and start + self.anomaly_duration <= i:
                    result.append([start, i])
                start = -1
                in_anomaly = False
        return result
