from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy import testing


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


def make_multi_series(frame: Dict[str, Sequence[Any]],
                      freq_millis: int,
                      start_date: str = '1/1/1970') -> pd.DataFrame:
    freq = pd.Timedelta(value=freq_millis, unit="ms")
    max_length = max([len(value) for _, value in frame.items()])
    index = pd.date_range(start_date, periods=max_length, freq=freq)
    df = pd.DataFrame(frame, columns=list(frame.keys()))
    df.index = index
    return df


def sinwave(freq: int = 20, sample_rate: int = 100,
            samples: int = 100) -> List[float]:
    result = [0] * samples
    for n in range(samples):
        result[n] = np.sin(2 * np.pi * freq * n / sample_rate)
    return result
