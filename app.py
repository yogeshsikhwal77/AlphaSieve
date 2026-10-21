import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import os

# Import custom modules
from src.signals import RuleBasedSignals
from src.model import MLFilter
from src.backtest import BacktestEngine
from src.metrics import PerformanceMetrics
from src.features import features as generate_features

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AlphaSieve Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark-themed KPI cards
st.markdown("""
    <style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AlphaSieve 🔍📈")
st.markdown("### An ML-Filtered Algorithmic Trading Backtester")
st.markdown("Filters rule-based technical signals using machine learning to eliminate low-probability setups.")
st.divider()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📊 Data Selection", expanded=True):
        selected_ticker = st.text_input("Asset Ticker (e.g., AAPL, TSLA, INFY, SPY)", value="AAPL").upper().strip()
        
    with st.expander("📈 Strategy Selection", expanded=True):
        strategy_choice = st.selectbox(
            "Base Signal (Trade Frequency)",
            [
                "RSI Mean Reversion (High Frequency)",
                "MACD Crossover (Medium Frequency)",
                "MA Crossover (Low Frequency)"
            ],
            index=0,
            help="RSI generates frequent dip-buying trades; MA crossover trades rarely over long trends."
        )
        
    with st.expander("🤖 ML Filter Settings", expanded=True):
        model_choice = st.selectbox("Model Architecture", ["xgboost", "random_forest", "logistic_regression"])
        prob_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.50, 
            max_value=0.90, 
            value=0.55, 
            step=0.01,
            help="Minimum ML probability required to allow a Buy trade. Lower values increase trade frequency."
        )
        
    with st.expander("💰 Execution Costs", expanded=True):
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
        commission = st.number_input("Commission Rate (0.001 = 0.1%)", value=0.0010, format="%.4f")
        slippage = st.number_input("Slippage Rate (0.0005 = 0.05%)", value=0.0005, format="%.4f")
        
    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# --- PIPELINE EXECUTION ---
if run_button:
    if not selected_ticker:
        st.error("Please enter a valid ticker symbol.")
        st.stop()
        
    with st.spinner(f"Running AlphaSieve Pipeline for {selected_ticker}..."):
        try:
            # 1. Load or Fetch Data
            file_path = f"data/{selected_ticker}_features.parquet"
            
            if not os.path.exists(file_path):
                st.info(f"Data for {selected_ticker} not found locally. Fetching from API...")
                try:
                    df = generate_features(selected_ticker, start="2018-01-01")
                except Exception as fetch_error:
                    st.error(f"Failed to fetch data for {selected_ticker}. Check symbol and connection: {fetch_error}")
                    st.stop()
            else:
                df = pd.read_parquet(file_path)
            
            # Ensure Date is used as datetime index for clean timeline charts
            date_col = next((col for col in ['date', 'Date', 'datetime', 'Datetime'] if col in df.columns), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass
            
            # Standardize 'Close' column name
            df = df.rename(columns=lambda x: 'Close' if str(x).lower() in ['close', 'adj close'] else x)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
            # 2. Generate Rule-Based Signals & Market Regimes
            df = RuleBasedSignals.generate_composite_signal(df)
            if 'Market_Regime' not in df.columns:
                df['Market_Regime'] = RuleBasedSignals.detect_market_regime(df)
            
            # Map user's UI choice to target signal column
            strategy_col_map = {
                "RSI Mean Reversion (High Frequency)": "Signal_RSI",
                "MACD Crossover (Medium Frequency)": "Signal_MACD",
                "MA Crossover (Low Frequency)": "Signal_MACross"
            }
            signal_col_name = strategy_col_map[strategy_choice]
            
            # Fallback check if column casing differs
            if signal_col_name not in df.columns:
                lower_lookup = {col.lower(): col for col in df.columns}
                if signal_col_name.lower() in lower_lookup:
                    signal_col_name = lower_lookup[signal_col_name.lower()]
                else:
                    st.error(f"Signal column '{signal_col_name}' could not be computed.")
                    st.stop()
            
            # 3. Apply ML Filter
            ml_features = ['MA_Distance', 'RSI', 'ATR', 'OBV', 'Z_score', 'Market_Regime']
            for feat in ml_features:
                if feat not in df.columns:
                    df[feat] = 0.0
                df[feat] = df[feat].bfill().fillna(0)
                
            ml_filter = MLFilter(model_type=model_choice)
            
            df_filtered = ml_filter.train_and_filter(
                df=df, 
                feature_cols=ml_features, 
                signal_col=signal_col_name, 
                prob_threshold=prob_threshold
            )
            
            # 4. Run Backtest
            engine = BacktestEngine(initial_capital, commission, slippage)
            results_df = engine.run(df_filtered)
            
            # 5. Calculate Metrics
            report_dict = PerformanceMetrics.generate_report(results_df)
            
            # --- DASHBOARD VISUALIZATION ---
            st.subheader(f"🏆 Strategy Performance vs. Buy & Hold ({selected_ticker})")
            st.caption(f"Strategy: **{strategy_choice}** | Filter: **{model_choice.upper()}** (Threshold: {prob_threshold:.2f})")
            
            col1, col2, col3, col4 = st.columns(4)
            strat = report_dict['Strategy']
            bench = report_dict['Buy & Hold']
            
            with col1:
                st.metric("CAGR", f"{strat['CAGR']*100:.2f}%", f"{(strat['CAGR'] - bench['CAGR'])*100:.2f}% vs B&H")
            with col2:
                st.metric("Sharpe Ratio", f"{strat['Sharpe Ratio']:.2f}", f"{strat['Sharpe Ratio'] - bench['Sharpe Ratio']:.2f} vs B&H")
            with col3:
                dd_delta = bench['Max Drawdown'] - strat['Max Drawdown'] 
                st.metric("Max Drawdown", f"{strat['Max Drawdown']*100:.2f}%", f"{dd_delta*100:.2f}% vs B&H", delta_color="inverse")
            with col4:
                st.metric("Win Rate", f"{strat['Win Rate']*100:.2f}%")

            st.write("")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Signals", "📊 Equity & Drawdown", "🧠 ML Interpretability", "🗄️ Raw Data"])
            
            with tab1:
                st.markdown(f"### {selected_ticker} Price Action & Executed Trades")
                fig_price = go.Figure()
                
                fig_price.add_trace(go.Scatter(
                    x=results_df.index, y=results_df['Close'], 
                    mode='lines', name='Price', line=dict(color='#888888', width=1.2)
                ))
                
                buys = results_df[results_df['Filtered_Signal'] == 1]
                fig_price.add_trace(go.Scatter(
                    x=buys.index, y=buys['Close'], mode='markers', name='Buy Signal', 
                    marker=dict(symbol='triangle-up', color='lime', size=11, line=dict(width=1, color='darkgreen'))
                ))
                
                sells = results_df[results_df['Filtered_Signal'] == -1]
                fig_price.add_trace(go.Scatter(
                    x=sells.index, y=sells['Close'], mode='markers', name='Sell Signal', 
                    marker=dict(symbol='triangle-down', color='red', size=11, line=dict(width=1, color='darkred'))
                ))
                
                fig_price.update_layout(template="plotly_dark", hovermode="x unified", height=500, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_price, use_container_width=True)

            with tab2:
                st.markdown("### Portfolio Equity Curve")
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=results_df.index, y=results_df['Strategy_Equity'], 
                    mode='lines', name='AlphaSieve Strategy', line=dict(color='#00ff00', width=2)
                ))
                fig_equity.add_trace(go.Scatter(
                    x=results_df.index, y=results_df['Benchmark_Equity'], 
                    mode='lines', name='Buy & Hold', line=dict(color='#888888', dash='dot')
                ))
                fig_equity.update_layout(template="plotly_dark", hovermode="x unified", height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_equity, use_container_width=True)
                
                st.markdown("### Drawdown Profile")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=results_df.index, y=results_df['Drawdown'] * 100, 
                    mode='lines', name='Strategy Drawdown', fill='tozeroy', line=dict(color='#ff4444')
                ))
                fig_dd.update_layout(template="plotly_dark", hovermode="x unified", height=300, yaxis_title="Drawdown (%)", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_dd, use_container_width=True)
                
            with tab3:
                st.markdown("### Feature Importance (SHAP)")
                st.markdown("SHAP values explain which technical indicators drove the machine learning model's trade filtering decisions.")
                
                X_train = df_filtered[ml_features].dropna()
                
                if not X_train.empty and len(X_train) > 5:
                    with st.spinner("Calculating SHAP values..."):
                        explainer, shap_values = ml_filter.get_shap_values(X_train)
                        fig_shap, ax = plt.subplots(figsize=(10, 5))
                        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                        
                        fig_shap.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.tick_params(axis='x', colors='white')
                        ax.tick_params(axis='y', colors='white')
                        
                        st.pyplot(fig_shap)
                        plt.clf()
                else:
                    st.warning("Insufficient valid samples to generate SHAP visualizations.")
                    
            with tab4:
                st.markdown("### Trade Activity Log")
                display_cols = [col for col in ['Close', 'Market_Regime', signal_col_name, 'ML_Probability', 'Filtered_Signal', 'Net_Return', 'Strategy_Equity'] if col in results_df.columns]
                st.dataframe(results_df[display_cols].tail(100), use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
            st.exception(e)
else:
    st.info("👈 Select your asset, strategy, and ML parameters in the sidebar, then click **Run Backtest**.")