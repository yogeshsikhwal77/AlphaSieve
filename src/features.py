# technical indicator engineering

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional
from src.data_loader import fetch_ohlcv


def features(ticker: str,start: str):
    ticker = ticker
    df = fetch_ohlcv(ticker=ticker,start=start)
    df.reset_index(inplace=True)

    # log return feature 1
    df['log_return'] = np.log((df['close'])/df['close'].shift(1))

    # Moveing average distance feature 2
    df["MA_Distance"] = (
        df["close"] / 
        df["close"].rolling(window=50).mean()
    ) - 1
 
    # RSI (relative strength index) feature 3
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    
    avg_gain = gain.ewm(com=13,adjust=False).mean()
    avg_loss = loss.ewm(com=13,adjust=False).mean()
    
    rs = avg_gain/avg_loss
    df['RSI'] = 100 - (100/(1+rs))

    # avg true Range (ATR) feature 4
    TR = pd.concat([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift(1)),
        np.abs(df['low'] - df['close'].shift(1))
    ],axis=1).max(axis=1)
    
    df['ATR'] = TR.ewm(alpha=1/14,adjust=False).mean()

    # On Balance Volume (OBV) feature 5
    direction = np.sign(df['close'].diff())
    dir_volume = df['volume'] * direction
    df['OBV'] = dir_volume.fillna(0).cumsum()

    # Rolling Z-Score feature 6
    mean = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['Z_score'] = (df['close'] - mean)/std
    
    # features data store
    FEATURES_DIR = Path(__file__).resolve().parent.parent /"data"/"features"
    FEATURES_DIR.mkdir(parents=True,exist_ok=True)

    save_path = FEATURES_DIR / f"{ticker}_features.parquet"
    df.to_parquet(save_path)
    
    return df

    
    


if __name__ == "__main__":
    features('AAPL',"2020-01-01")