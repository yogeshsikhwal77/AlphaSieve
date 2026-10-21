# Sharpe, drawdown, CAGR, etc.
import pandas as pd
import numpy as np

class PerformanceMetrics:
    """
    Calculates key performance indicators (KPIs) for the backtested strategy.
    Assumes daily data with 252 trading days per year.
    """
    
    @staticmethod
    def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        """Calculates Compound Annual Growth Rate."""
        if len(equity_curve) == 0:
            return 0.0
        
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        num_years = len(equity_curve) / periods_per_year
        
        if total_return <= 0 or num_years == 0:
            return 0.0
            
        return (total_return ** (1 / num_years)) - 1

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculates annualized Sharpe Ratio."""
        if returns.std() == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())

    @staticmethod
    def calculate_max_drawdown(drawdown_series: pd.Series) -> float:
        """Calculates Maximum Drawdown from the pre-calculated drawdown series."""
        return drawdown_series.min()

    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """
        Calculates the percentage of winning trades/days.
        Only counts days where the strategy was active (return != 0).
        """
        winning_days = returns[returns > 0].count()
        total_active_days = returns[returns != 0].count()
        
        if total_active_days == 0:
            return 0.0
        return winning_days / total_active_days

    @classmethod
    def generate_report(cls, df: pd.DataFrame) -> dict:
        """
        Generates a comprehensive performance report comparing the Strategy to Buy & Hold.
        Expects 'Strategy_Equity', 'Benchmark_Equity', 'Net_Return', 'Market_Return', and 'Drawdown'.
        """
        # Calculate benchmark drawdown on the fly
        peak_bench = df['Benchmark_Equity'].cummax()
        bench_dd = (df['Benchmark_Equity'] - peak_bench) / peak_bench

        report = {
            "Strategy": {
                "CAGR": cls.calculate_cagr(df['Strategy_Equity']),
                "Sharpe Ratio": cls.calculate_sharpe_ratio(df['Net_Return']),
                "Max Drawdown": cls.calculate_max_drawdown(df['Drawdown']),
                "Win Rate": cls.calculate_win_rate(df['Net_Return'])
            },
            "Buy & Hold": {
                "CAGR": cls.calculate_cagr(df['Benchmark_Equity']),
                "Sharpe Ratio": cls.calculate_sharpe_ratio(df['Market_Return']),
                "Max Drawdown": bench_dd.min(),
                "Win Rate": cls.calculate_win_rate(df['Market_Return'])
            }
        }
        return report
        
    @staticmethod
    def format_report(report: dict) -> pd.DataFrame:
        """Formats the report dictionary into a clean Pandas DataFrame for frontend display."""
        df_report = pd.DataFrame(report).T
        df_report['CAGR'] = (df_report['CAGR'] * 100).map("{:.2f}%".format)
        df_report['Sharpe Ratio'] = df_report['Sharpe Ratio'].map("{:.2f}".format)
        df_report['Max Drawdown'] = (df_report['Max Drawdown'] * 100).map("{:.2f}%".format)
        df_report['Win Rate'] = (df_report['Win Rate'] * 100).map("{:.2f}%".format)
        
        # Reorder columns to match the README structure
        return df_report[['CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']]