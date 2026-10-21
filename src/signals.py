import pandas as pd
import numpy as np

class RuleBasedSignals:
    """
    Generates baseline rule-based trading signals (1: Buy, -1: Sell, 0: Hold).
    These raw signals serve as the foundation for the AlphaSieve ML classifier.
    """
    
    @staticmethod
    def moving_average_crossover(df: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.Series:
        """
        Momentum Strategy: Golden Cross / Death Cross
        Buys when the short-term MA crosses above the long-term MA.
        Sells when the short-term MA crosses below the long-term MA.
        """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")
            
        sma_short = df['Close'].rolling(window=short_window, min_periods=1).mean()
        sma_long = df['Close'].rolling(window=long_window, min_periods=1).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # 1 when short > long, -1 when short < long
        signals[sma_short > sma_long] = 1
        signals[sma_short < sma_long] = -1
        
        # Only trigger on actual crossovers (shift logic)
        crossover_signals = signals.diff()
        
        # Return 1 for upward crossover, -1 for downward crossover, 0 otherwise
        final_signals = pd.Series(0, index=df.index)
        final_signals[crossover_signals > 0] = 1
        final_signals[crossover_signals < 0] = -1
        
        return final_signals

    @staticmethod
    def rsi_mean_reversion(df: pd.DataFrame, period: int = 14, overbought: int = 70, oversold: int = 30) -> pd.Series:
        """
        Mean-Reversion Strategy: Relative Strength Index (RSI)
        Buys when RSI drops below the oversold threshold.
        Sells when RSI rises above the overbought threshold.
        """
        if 'RSI' not in df.columns:
            # Calculate RSI if not already provided by features.py
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
        signals = pd.Series(0, index=df.index)
        
        # Buy when RSI crosses below oversold, Sell when RSI crosses above overbought
        signals[df['RSI'] < oversold] = 1
        signals[df['RSI'] > overbought] = -1
        
        # Filter for the exact crossover day to avoid consecutive identical signals
        signal_changes = signals.diff()
        final_signals = pd.Series(0, index=df.index)
        final_signals[(signals == 1) & (signal_changes != 0)] = 1
        final_signals[(signals == -1) & (signal_changes != 0)] = -1
        
        return final_signals

    @staticmethod
    def macd_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.Series:
        """
        Momentum Strategy: MACD 
        Buys when the MACD line crosses above the Signal line.
        Sells when the MACD line crosses below the Signal line.
        """
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[macd_line > signal_line] = 1
        signals[macd_line < signal_line] = -1
        
        crossover = signals.diff()
        
        final_signals = pd.Series(0, index=df.index)
        final_signals[crossover > 0] = 1
        final_signals[crossover < 0] = -1
        
        return final_signals

    @classmethod
    def generate_composite_signal(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates all signals and adds them to the DataFrame.
        """
        df = df.copy()
        df['Signal_MACross'] = cls.moving_average_crossover(df)
        df['Signal_RSI'] = cls.rsi_mean_reversion(df)
        df['Signal_MACD'] = cls.macd_crossover(df)
        
        # Example of a basic combined rule-based signal (Optional)
        # e.g., only buy if both momentum and mean-reversion agree
        df['Combined_Raw_Signal'] = 0
        df.loc[(df['Signal_MACross'] == 1) | (df['Signal_RSI'] == 1), 'Combined_Raw_Signal'] = 1
        df.loc[(df['Signal_MACross'] == -1) | (df['Signal_RSI'] == -1), 'Combined_Raw_Signal'] = -1
        
        return df

    @staticmethod
    def detect_market_regime(df: pd.DataFrame, short_lookback: int = 50, long_lookback: int = 200) -> pd.Series:
        """
        Classifies the market regime based on moving averages.
        1: Bull Market (Price > 50 SMA > 200 SMA)
        -1: Bear Market (Price < 50 SMA < 200 SMA)
        0: Sideways/Transition Market
        """
        sma_short = df['Close'].rolling(window=short_lookback, min_periods=1).mean()
        sma_long = df['Close'].rolling(window=long_lookback, min_periods=1).mean()
        
        regime = pd.Series(0, index=df.index)
        
        # Bull Regime
        regime[(df['Close'] > sma_short) & (sma_short > sma_long)] = 1
        # Bear Regime
        regime[(df['Close'] < sma_short) & (sma_short < sma_long)] = -1
        
        return regime