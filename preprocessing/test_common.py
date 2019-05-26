import numpy as np
from numpy import testing
import pandas as pd

def make_labels(input, indices):
  index = pd.date_range('1/1/2011', periods=len(input), freq='T')
  labels = pd.DataFrame(input, columns=["label"])
  labels['label'] = 0.0
  labels.loc[indices, 'label'] = 1
  labels['timestamp'] = index
  return labels

def make_series(array):
  index = pd.date_range('1/1/2011', periods=len(array), freq='T')
  df = pd.DataFrame(array, columns=["y"])
  df['timestamp'] = index
  return df
