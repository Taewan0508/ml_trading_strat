import pandas as pd
import yfinance as yf

def load_price_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

