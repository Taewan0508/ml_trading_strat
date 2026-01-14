import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(df):
    features = pd.DataFrame(index=df.index)

    # Returns
    features['ret_1'] = np.log(df['Close'] / df['Close'].shift(1))
    features['ret_5'] = features['ret_1'].rolling(5).sum()
    features['ret_10'] = features['ret_1'].rolling(10).sum()

    # Volatility
    features['vol_5'] = features['ret_1'].rolling(5).std()
    features['vol_20'] = features['ret_1'].rolling(20).std()

    # RSI
    features['rsi_14'] = compute_rsi(df['Close'], 14)

    # Moving averages
    ma_10 = df['Close'].rolling(10).mean()
    ma_50 = df['Close'].rolling(50).mean()

    features['price_ma10'] = df['Close'] / ma_10
    features['ma10_ma50'] = ma_10 / ma_50

    features.dropna(inplace=True)
    return features

