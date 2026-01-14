import numpy as np
import pandas as pd

def make_labels(df):
    future_return = np.log(df['Close'].shift(-1) / df['Close'])
    labels = (future_return > 0).astype(int)
    return labels

