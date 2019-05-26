import pandas as pd
import matplotlib.pyplot as plt

class TimeseriesHelper:
    def __init__(self, response_variable = 'y', ts_variable = 'timestamp'):
        self.response_variable = response_variable
        self.ts_variable = ts_variable

    def show_metric(self, windows, num):
        metric_window = windows[num]
        self.plot_metric(windows[num])

    def plot_metric(self, metric):
        plt.plot(metric[self.ts_variable], metric[self.response_variable])
        plt.xticks(rotation='vertical')

    def find_window_discord(self, windows):
        max_response = 0
        max_window = None
        max_idx = 0
        for idx, window in enumerate(windows):
            current_max = window[self.response_variable].max()
            if current_max > max_response:
                max_response = current_max
                max_window = window
                max_idx = idx
        return max_idx, max_window
