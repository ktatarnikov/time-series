import numpy as np


class SequiturAnomaly:
    def __init__(self, sequitur, density_threshold = 1, anomaly_duration = 1):
        self.sequitur = sequitur
        self.density_threshold = density_threshold
        self.anomaly_duration = anomaly_duration

    def detect(self, words):
        self.density_curve = self._calculate_density(words)
        anomalies = self._detect_anomalies(self.density_curve)
        return anomalies

    def _calculate_density(self, words):
        results = np.zeros(len(words))
        rules = self.sequitur.get_grammar().rules.values()
        for rule in rules:
            for pos in rule.get_positions():
                start, end = pos
                for i in range(start, end):
                    results[start:end+1] += 1
        return results

    def _detect_anomalies(self, density_curve):
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
