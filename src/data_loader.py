# fetches OHLCV data

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" /"raw"
DATA_DIR.mkdir(parents=True,exist_ok=True)


def fetch_ohlcv(
        ticker: str,                      # stock names short form
        start: str = "2018-01-01",        # 
        end: Optional[str] = None,
        force_refresh: bool = False,      
) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{ticker}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)
    
    df = yf.download(ticker,start=start,end=end,auto_adjust=True,progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} - check symbol and internet connection")
    
    # isinstance( object , class )  isinstance(x,int) -> true
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns = str.lower)
    df.index.name = "date"
    df.to_parquet(cache_path)
    return df

def sanity_check(df: pd.DataFrame,ticker: str) -> dict:
    n_missing = int(df.isna().sum().sum())
    gaps = df.index.to_series().diff().dt.days.dropna()
    max_gap = int(gaps.max()) if not gaps.empty else 0
    return {"ticker": ticker, "rows": len(df), "missing_values": n_missing, "max_gap_days": max_gap}

def fetch_universe(tickers: list, **kwargs) -> dict:
    return {t: fetch_ohlcv(t,**kwargs) for t in tickers}