import yfinance as yf
import pandas as pd
import numpy as np
import ccxt
from jugaad_data.nse import stock_df
from datetime import datetime
from typing import Optional
from pathlib import Path

# Match README structure: just /data instead of /data/raw
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_ohlcv(ticker: str, start: str = "2018-01-01", end: Optional[str] = None, force_refresh: bool = False) -> pd.DataFrame:
    """Fetches US Equities via yfinance"""
    cache_path = DATA_DIR / f"{ticker}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)
        
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    df.to_parquet(cache_path)
    return df

def fetch_crypto_ohlcv(symbol: str = 'BTC/USDT', timeframe: str = '1d', limit: int = 1000) -> pd.DataFrame:
    """Fetches Crypto via ccxt"""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    
    cache_path = DATA_DIR / f"{symbol.replace('/', '')}.parquet"
    df.to_parquet(cache_path)
    return df

def fetch_nse_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetches Indian Equities via jugaad-data"""
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    df = stock_df(symbol=symbol, from_date=start_dt, to_date=end_dt, series="EQ")
    df = df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].rename(columns=str.lower)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    cache_path = DATA_DIR / f"{symbol}.parquet"
    df.to_parquet(cache_path)
    return df

def sanity_check(df: pd.DataFrame, ticker: str) -> dict:
    n_missing = int(df.isna().sum().sum())
    gaps = df.index.to_series().diff().dt.days.dropna()
    max_gap = int(gaps.max()) if not gaps.empty else 0
    return {"ticker": ticker, "rows": len(df), "missing_values": n_missing, "max_gap_days": max_gap}

def fetch_universe(tickers: list, **kwargs) -> dict:
    return {t: fetch_ohlcv(t, **kwargs) for t in tickers}