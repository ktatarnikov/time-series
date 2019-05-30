import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

class TimeseriesHelper:
    def __init__(self, response_variable = 'y', ts_variable = 'timestamp', label_variable = 'label'):
        self.response_variable = response_variable
        self.ts_variable = ts_variable
        self.label_variable = label_variable

    def show_metric(self, windows, num):
        metric_window = windows[num]
        self.plot_metric(windows[num])

    def plot_metric(self, metric):
        plt.plot(metric[self.ts_variable], metric[self.response_variable])
        plt.xticks(rotation='vertical')

    def load_labeled_series(self, series_path):
        '''
        Loads dataseries and attach labels.

        Parameters
        ----------
        series_path : str
            The path to timeseries relative to 'data/NAB' directory

        Returns
        -------
        pd.DataFrame
            with the following columns
            - self.response_variable ('y') as response variable
            - self.label_variable ('label') as label column
            - self.ts_variable ('timestamp') as timestamp column.
        '''
        data_frame = pd.read_csv(f"data/NAB/{series_path}").rename(columns={"value": "y"})
        data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'], infer_datetime_format=True)

        labels_json = self._load("labels/combined_labels.json")
        labels = json.loads(labels_json)
        data_frame[self.label_variable] = data_frame[self.ts_variable].apply(lambda x: 1 if x in labels[series_path] else 0)
        return data_frame

    def _load(self, file_name):
        with open (file_name, "r") as file:
            return file.read()

    def group_by_hour(self, metric):
        grouped_by_hour = metric.groupby(
             [metric['timestamp'].map(lambda x : x.year).rename('year'),
              metric['timestamp'].map(lambda x : x.month).rename('month'),
              metric['timestamp'].map(lambda x : x.day).rename('day'),
              metric['timestamp'].map(lambda x : x.hour).rename('hour')]).count()
        return grouped_by_hour

    def find_window_discords(self, windows):
        max_response = 0
        max_window = None
        max_idx = 0
        mean = pd.concat([w[self.response_variable] for w in windows]).mean()
        for idx, window in enumerate(windows):
            abs_max = window[self.response_variable].apply(lambda v: np.abs(v - mean)).max()
            if abs_max > max_response:
                max_response = abs_max
                max_window = window
                max_idx = idx
        return max_idx, max_window

    def plot_model_history(self, model_history):
        plt.plot(model_history['loss'], linewidth=2, label='Train')
        plt.plot(model_history['val_loss'], linewidth=2, label='Valid')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    def plot_curve(self, X_test, Y_test, X_prediction):
        mse = np.mean(np.power(flatten(X_test) - flatten(X_prediction), 2), axis=1)
        error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': Y_test.tolist()})
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
        plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
        plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.show()
