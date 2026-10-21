import pandas as pd
import numpy as np
import vectorbt as vbt

class BacktestEngine:
    """
    Vectorized backtesting engine using vectorbt.
    Applies realistic transaction costs, slippage, and calculates the portfolio equity curve.
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001, slippage: float = 0.0005):
        """
        :param initial_capital: Starting cash in dollars.
        :param commission: Broker commission per trade (e.g., 0.001 = 0.1%).
        :param slippage: Assumed price slippage per trade (e.g., 0.0005 = 0.05%).
        """
        self.initial_capital = initial_capital
        # Combine commission and slippage into a single fee metric for vectorbt
        self.fees = commission + slippage

    def run(self, df: pd.DataFrame, signal_col: str = 'Filtered_Signal', price_col: str = 'Close') -> pd.DataFrame:
        """
        Executes the backtest on the provided DataFrame using vectorbt.
        Assumes signals are 1 (Buy), -1 (Sell), or 0 (Hold).
        """
        df = df.copy()
        
        # 1. Map signals to vectorbt entry and exit arrays
        entries = df[signal_col] == 1
        exits = df[signal_col] == -1
        
        # 2. Build the Strategy Portfolio
        pf = vbt.Portfolio.from_signals(
            close=df[price_col],
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            freq='1D', # Assuming daily data
            sl_stop=0.05,  # 5% Stop Loss
            tp_stop=0.10
        )
        
        # 3. Build the Benchmark Portfolio (Buy & Hold)
        pf_bench = vbt.Portfolio.from_holding(
            close=df[price_col],
            init_cash=self.initial_capital,
            fees=self.fees,
            freq='1D'
        )
        
        # 4. Reconstruct the output columns required by metrics.py and app.py
        df['Strategy_Equity'] = pf.value()
        df['Benchmark_Equity'] = pf_bench.value()
        
        df['Net_Return'] = pf.returns()
        df['Market_Return'] = pf_bench.returns()
        
        # Drawdown is calculated as a negative percentage in vectorbt
        df['Drawdown'] = pf.drawdown()
        
        # Clean up starting rows (fill NaNs)
        df['Strategy_Equity'] = df['Strategy_Equity'].fillna(self.initial_capital)
        df['Benchmark_Equity'] = df['Benchmark_Equity'].fillna(self.initial_capital)
        df['Net_Return'] = df['Net_Return'].fillna(0)
        df['Market_Return'] = df['Market_Return'].fillna(0)
        df['Drawdown'] = df['Drawdown'].fillna(0)
        
        return df

# Example usage stringing the pipeline together:
if __name__ == "__main__":
    print("BacktestEngine initialized with vectorbt. Ready to process filtered signals.")