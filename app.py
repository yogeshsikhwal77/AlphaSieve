import streamlit as st
import pandas as pd
from datetime import datetime
# Import both your raw loader and your brand new features engineering pipeline
from src.data_loader import fetch_ohlcv
from src.features import features

# Page Configuration
st.set_page_config(page_title="AlphaSieve Dashboard", page_icon="📈", layout="wide")

# Title and Header
st.title("AlphaSieve 🔍📈")
st.caption("Advanced Technical Indicator & Feature Engineering Dashboard")
st.markdown("---")

# 1. SIDEBAR FOR CONTROLS
st.sidebar.header("Dashboard Configuration")
ticker = st.sidebar.text_input("Ticker Symbol:", value="AAPL").upper()

# Date Picker for flexibility
start_date = st.sidebar.date_input("Start Date:", datetime(2020, 1, 1))
start_date_str = start_date.strftime("%Y-%m-%d")

# Add a toggle to let the user choose between Raw Data or Engineered Features
data_type = st.sidebar.radio("Data Mode:", ["Engineered Features", "Raw Data Only"])

fetch_button = st.sidebar.button("Load and Process Data", use_container_width=True)

# 2. MAIN DASHBOARD LOGIC
if fetch_button or 'df_loaded' in st.session_state:
    # Use session state so the app doesn't re-download data when switching tabs
    st.session_state['df_loaded'] = True
    
    with st.spinner(f"Retrieving and processing data for {ticker}..."):
        try:
            if data_type == "Engineered Features":
                # Call your newly created features function!
                df = features(ticker=ticker, start=start_date_str)
            else:
                df = fetch_ohlcv(ticker=ticker, start=start_date_str)
                df.reset_index(inplace=True)

            st.success(f"Successfully loaded {len(df)} rows for {ticker}!")

            # 3. TOP KPI METRICS (Only display if features exist)
            if "RSI" in df.columns and not df.empty:
                latest = df.iloc[-1]
                prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
                price_chg = latest['close'] - prev_close
                price_chg_pct = (price_chg / prev_close) * 100

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latest Close", f"${latest['close']:.2f}", f"{price_chg_pct:+.2f}%")
                m2.metric("RSI (14)", f"{latest['RSI']:.2f}", "Overbought > 70 | Oversold < 30")
                m3.metric("ATR (14)", f"${latest['ATR']:.2f}", "Volatility Metric")
                m4.metric("Z-Score (20)", f"{latest['Z_score']:.2f}", "Deviations from Mean")
                st.markdown("---")

            # 4. TABS FOR CLEAN NAVIGATION
            tab1, tab2, tab3 = st.tabs(["📊 Price & Volume Analysis", "⚙️ Technical Indicators", "📋 Inspected Data Table"])

            with tab1:
                st.subheader("Price Evolution")
                st.line_chart(df.set_index('date')['close'])
                
                st.subheader("Trading Volume")
                st.bar_chart(df.set_index('date')['volume'])

            with tab2:
                if data_type == "Engineered Features":
                    st.subheader("Momentum & Volatility Indicators")
                    
                    # Row 1: RSI and Rolling Z-Score side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Relative Strength Index (RSI)**")
                        st.line_chart(df.set_index('date')['RSI'])
                    with col2:
                        st.write("**Rolling Z-Score (20 Days)**")
                        st.line_chart(df.set_index('date')['Z_score'])
                    
                    # Row 2: ATR and MA Distance side by side
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("**Average True Range (ATR)**")
                        st.area_chart(df.set_index('date')['ATR'])
                    with col4:
                        st.write("**Moving Average Distance (50 Days)**")
                        st.line_chart(df.set_index('date')['MA_Distance'])
                else:
                    st.info("Switch to 'Engineered Features' mode in the sidebar to visualize technical indicators.")

            with tab3:
                st.subheader("Recent Historical Data Records")
                st.dataframe(df.tail(100), use_container_width=True)

        except Exception as e:
            st.error(f"Error processing dashboard: {e}")