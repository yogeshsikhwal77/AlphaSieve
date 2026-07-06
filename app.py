# Streamlit dashboard

import streamlit as st
from src.data_loader import fetch_ohlcv

st.set_page_config(page_title="AlphaSieve Data Tester", page_icon="📈")

st.title("AlphaSieve 🔍📈")
st.subheader("Data Loader Test")

# Text input for the ticker
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT, SPCX):", value="AAPL")

# Button to trigger data fetching
if st.button("Fetch Data"):
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Call your custom data_loader function
            df = fetch_ohlcv(ticker=ticker)
            
            st.success(f"Successfully loaded {len(df)} rows for {ticker}!")
            
            # Show the last 5 rows of the dataframe
            st.write("### Recent Data (Last 5 Days)")
            st.dataframe(df.tail())
            
            # Plot the closing price using Streamlit's native line chart
            st.write("### Closing Price")
            if 'close' in df.columns:
                st.line_chart(df['close'])
            elif 'Close' in df.columns:
                st.line_chart(df['Close'])
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")