import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Crypto Strategy Dashboards", layout="wide")

# --- SHARED DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_historical_fng():
    url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=kr"
    response = requests.get(url, timeout=10)
    data = response.json()
    fng_df = pd.DataFrame(data['data'])
    fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'])
    fng_df.set_index('timestamp', inplace=True)
    fng_df.sort_index(ascending=True, inplace=True)
    fng_df['value'] = fng_df['value'].astype(float)
    return fng_df[['value', 'value_classification']]

@st.cache_data(ttl=900)
def get_btc_data_fng():
    btc_df = yf.Ticker("BTC-USD").history(start="2018-02-01", auto_adjust=False)
    if btc_df.index.tz is not None:
        btc_df.index = btc_df.index.tz_localize(None)
    return btc_df

@st.cache_data(ttl=900)
def get_mstr_data():
    df = yf.Ticker("MSTR").history(start="2020-08-01", auto_adjust=False)
    if df.empty:
        df = yf.download("MSTR", start="2020-08-01")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

@st.cache_data(ttl=900)
def get_btc_price_latest():
    try:
        btc_df = yf.Ticker("BTC-USD").history(period="1d")
        if not btc_df.empty:
            return btc_df['Close'].iloc[-1]
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_fear_and_greed_latest():
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            val = data["data"][0]["value"]
            classification = data["data"][0]["value_classification"]
            return val, classification
    except Exception:
        pass
    return "N/A", "N/A"


# --- FEAR & GREED CADENCE VIEW ---
def render_fng_cadence():
    st.title("Fear & Greed Cadence Filter (OCM Circuit Breaker)")
    st.markdown("This dashboard calculates 90-day sentiment momentum overlaid directly on BTC price action with a 200-day trend filter to protect LEAPS capital during macro bear traps.")

    st.sidebar.header("Strategy Settings")
    cadence_len = st.sidebar.number_input("Cadence Lookback (Days)", value=90, min_value=14, max_value=200, key="fng_cadence_len")
    smooth_len = st.sidebar.number_input("Cadence Smoothing (EMA Days)", value=14, min_value=3, max_value=None, key="fng_smooth_len")
    ma_len = st.sidebar.number_input("Trend Filter (SMA)", value=200, min_value=50, max_value=300, key="fng_ma_len")
    
    fng_df = get_historical_fng()
    btc_df = get_btc_data_fng()

    df = btc_df.join(fng_df, how='inner')
    df.rename(columns={'value': 'FnG_Score', 'value_classification': 'FnG_Class'}, inplace=True)

    if len(df) > ma_len:
        df['SMA_200'] = df['Close'].rolling(window=ma_len).mean()
        df['Bull_Regime'] = df['Close'] > df['SMA_200']
        
        df['FnG_Cadence'] = df['FnG_Score'].diff(cadence_len)
        df['FnG_Cadence_Smooth'] = df['FnG_Cadence'].ewm(span=smooth_len, adjust=False).mean()
        
        df['Cadence_Prev'] = df['FnG_Cadence'].shift(1)
        df['Cadence_Cross_Up'] = (df['FnG_Cadence'] > 0) & (df['Cadence_Prev'] <= 0)
        df['Cadence_Cross_Dn'] = (df['FnG_Cadence'] < 0) & (df['Cadence_Prev'] >= 0)
        
        df['Valid_Buy'] = df['Cadence_Cross_Up'] & df['Bull_Regime'] & (df['FnG_Score'] > 20)
        df['Valid_Sell'] = df['Cadence_Cross_Dn']
        
        df['Signal'] = 'Cash ⚪'
        in_position = False
        
        for i in range(1, len(df)):
            idx = df.index[i]
            if df['Valid_Buy'].iloc[i] and not in_position:
                in_position = True
                df.at[idx, 'Signal'] = 'Buy 🟢'
            elif df['Valid_Sell'].iloc[i] and in_position:
                in_position = False
                df.at[idx, 'Signal'] = 'Sell 🔴'
            elif in_position:
                if not df['Bull_Regime'].iloc[i]:
                    in_position = False
                    df.at[idx, 'Signal'] = 'Bear Trap Stop 🔴'
                else:
                    df.at[idx, 'Signal'] = 'Hold 🟡'
            else:
                df.at[idx, 'Signal'] = 'Cash ⚪'

        st.sidebar.markdown("---")
        st.sidebar.header("📥 Export Data to Excel")
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        start_date = st.sidebar.date_input("From Date", value=min_date, min_value=min_date, max_value=max_date, key="fng_start")
        end_date = st.sidebar.date_input("To Date", value=max_date, min_value=min_date, max_value=max_date, key="fng_end")

        filtered_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
        
        export_df = pd.DataFrame(index=filtered_df.index)
        export_df['Date'] = filtered_df.index.strftime('%Y-%m-%d')
        export_df['BTC Price ($)'] = filtered_df['Close'].round(2)
        export_df['200D SMA ($)'] = filtered_df['SMA_200'].round(2)
        export_df['Raw Fear & Greed'] = filtered_df['FnG_Score']
        export_df['90D Cadence (pts)'] = filtered_df['FnG_Cadence'].round(2)
        export_df['Smoothed Cadence (pts)'] = filtered_df['FnG_Cadence_Smooth'].round(2)
        export_df['Signal'] = filtered_df['Signal']

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name='FnG_Cadence_Data', index=False)
        excel_data = excel_buffer.getvalue()

        st.sidebar.download_button(
            label="📥 Download Excel File",
            data=excel_data,
            file_name=f"FnG_Cadence_Data_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="fng_dl"
        )

        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BTC Live Price", f"${latest['Close']:,.2f}")
        col2.metric("Master Trend (200D)", "BULL 🟢" if latest['Bull_Regime'] else "BEAR 🔴")
        col3.metric("F&G Cadence (90D)", f"{latest['FnG_Cadence']:+.1f} pts")
        col4.metric("Current State", latest['Signal'])

        st.markdown("---")
        st.subheader("🛡️ LEAPS Circuit Breaker Status")
        
        if not latest['Bull_Regime']:
            st.error(f"**BLOCKED:** Price is below the {ma_len}-Day SMA. You are likely in a Macro B-Wave or C-Wave. **Do NOT buy LEAPS.** 100% options cash is required.")
        elif latest['FnG_Cadence'] < 0:
            st.warning(f"**CAUTION:** Price is in a Bull Regime, but Sentiment Cadence is negative. Wait for momentum to cross zero before executing Z-Score LEAPS buys.")
        else:
            st.success(f"**ARMED & ACTIVE:** Price is above the {ma_len}-Day SMA and Sentiment Cadence is positive. Z-Score LEAPS buy signals are cleared for execution.")
            
        st.markdown("---")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        df['Hover_Date'] = df.index.strftime('%d %B %Y')
        cadence_colors = np.where(df['FnG_Cadence'] >= 0, 'rgba(59, 130, 246, 0.45)', 'rgba(239, 68, 68, 0.45)')
        
        fig.add_trace(go.Bar(x=df.index, y=df['FnG_Cadence'], name="90D Cadence", marker_color=cadence_colors, customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate="<b>Date:</b> %{customdata[0]}<br><b>90D Cadence:</b> %{y:+.1f} pts<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['FnG_Cadence_Smooth'], name=f"Cadence Trend ({smooth_len}D EMA)", line=dict(color='#00e5ff', width=2.5, shape='spline'), customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate="<b>Date:</b> %{customdata[0]}<br><b>Smoothed Trend:</b> %{y:+.1f} pts<extra></extra>"), secondary_y=False)
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(255, 255, 255, 0.6)", secondary_y=False)

        price_hovertemplate = "<b>Date:</b> %{customdata[0]}<br><b>BTC Price:</b> $%{y:,.2f}<extra></extra>"
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="BTC Price", line=dict(color='white', width=2), customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate=price_hovertemplate), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name="200D SMA (Trend Filter)", line=dict(color='orange', width=1.5), customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate="<b>Date:</b> %{customdata[0]}<br><b>200D SMA:</b> $%{y:,.2f}<extra></extra>"), secondary_y=True)
        
        buy_dates = df[df['Signal'] == 'Buy 🟢'].index
        buy_prices = df.loc[buy_dates, 'Close']
        buy_hover = df.loc[buy_dates, 'Hover_Date']
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name="Cadence Buy", marker=dict(color='lime', size=12, symbol='triangle-up', line=dict(color='black', width=1)), customdata=np.stack((buy_hover,), axis=-1), hovertemplate="<b>BUY</b><br>" + price_hovertemplate), secondary_y=True)
        
        sell_dates = df[df['Signal'].isin(['Sell 🔴', 'Bear Trap Stop 🔴'])].index
        sell_prices = df.loc[sell_dates, 'Close']
        sell_hover = df.loc[sell_dates, 'Hover_Date']
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name="Cadence Sell", marker=dict(color='red', size=12, symbol='triangle-down', line=dict(color='black', width=1)), customdata=np.stack((sell_hover,), axis=-1), hovertemplate="<b>SELL</b><br>" + price_hovertemplate), secondary_y=True)

        fig.update_layout(height=650, template="plotly_dark", hovermode="x unified", title_text="Composite Chart: BTC Price & 90-Day F&G Cadence Trend", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="Cadence Momentum (pts)", secondary_y=False, showgrid=False)
        fig.update_yaxes(title_text="BTC Price ($)", secondary_y=True, type="log")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Historical Cadence Signal Log")
        log_df = df[df['Signal'].isin(['Buy 🟢', 'Sell 🔴', 'Bear Trap Stop 🔴'])].copy()
        if not log_df.empty:
            log_export = pd.DataFrame(index=log_df.index)
            log_export['Date'] = log_df.index.strftime('%Y-%m-%d')
            log_export['BTC Price ($)'] = log_df['Close'].apply(lambda x: f"${x:,.2f}")
            log_export['Raw F&G'] = log_df['FnG_Score']
            log_export['90D Cadence (pts)'] = log_df['FnG_Cadence'].round(2)
            log_export['Smoothed Cadence (pts)'] = log_df['FnG_Cadence_Smooth'].round(2)
            log_export['Signal'] = log_df['Signal']
            st.dataframe(log_export, use_container_width=True)
        else:
            st.info("No active signals triggered in current dataset range.")
    else:
        st.warning("Not enough data to calculate metrics.")


# --- MSTR Z-SCORE VIEW ---
def render_mstr_zscore():
    st_autorefresh(interval=900000, key="data_refresh")
    st.title("MSTR Composite Price & Z-Score Dashboard")
    st.markdown("Price action with 50D/200D SMAs overlaid with a daily Z-Score Bar Chart.")

    st.sidebar.header("Chart Settings")
    z_view = st.sidebar.radio("Z-Score View (Bar Chart)", ["Tactical (50D)", "Macro (200D)"], key="mstr_z_view")

    st.sidebar.header("Backtest Settings")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, step=1000, key="mstr_init_cap")

    st.sidebar.header("Strategy Settings")
    short_len = st.sidebar.number_input("Tactical Length", value=50, key="mstr_short_len")
    macro_len = st.sidebar.number_input("Macro Length", value=200, key="mstr_macro_len")

    entry_z = st.sidebar.slider("Momentum Entry Z", -2.0, 3.0, 1.0, 0.1, key="mstr_entry_z")
    bull_dip_z = st.sidebar.slider("Bull Dip Z", -2.0, 3.0, 0.0, 0.1, key="mstr_dip_z")
    cycle_low_z = st.sidebar.slider("Macro Cycle Low Z", -4.0, 0.0, -2.0, 0.1, key="mstr_low_z")
    cycle_high_z = st.sidebar.slider("Macro Top Exit Z", 1.0, 4.0, 2.5, 0.1, key="mstr_high_z")
    exit_z = st.sidebar.slider("Standard Exit Z", -2.0, 2.0, 0.0, 0.1, key="mstr_exit_z")
    use_macro = st.sidebar.checkbox("Bear Market Protection", value=True, key="mstr_use_macro")

    df = get_mstr_data()

    if len(df) > macro_len:
        df['Log_Close'] = np.log(df['Close'])
        df['Short_Base'] = df['Log_Close'].rolling(short_len).mean()
        df['Short_Dev'] = df['Log_Close'].rolling(short_len).std()
        df['Short_Z'] = np.where(df['Short_Dev']==0, 0, (df['Log_Close'] - df['Short_Base']) / df['Short_Dev'])
        df['Macro_Base'] = df['Log_Close'].rolling(macro_len).mean()
        df['Macro_Dev'] = df['Log_Close'].rolling(macro_len).std()
        df['Macro_Z'] = np.where(df['Macro_Dev']==0, 0, (df['Log_Close'] - df['Macro_Base']) / df['Macro_Dev'])
        df['Bull_Regime'] = df['Log_Close'] > df['Macro_Base']
        df['SMA_50'] = df['Close'].rolling(window=short_len).mean()
        df['SMA_200'] = df['Close'].rolling(window=macro_len).mean()
        df['Hover_Date'] = df.index.strftime('%d %B %Y')
        
        df['Short_Z_Prev'] = df['Short_Z'].shift(1)
        df['Macro_Z_Prev'] = df['Macro_Z'].shift(1)
        df['Cross_Zero'] = (df['Short_Z'] > 0) & (df['Short_Z_Prev'] <= 0)
        df['Cross_Dip'] = (df['Short_Z'] > bull_dip_z) & (df['Short_Z_Prev'] <= bull_dip_z)
        df['Cross_Entry'] = (df['Short_Z'] > entry_z) & (df['Short_Z_Prev'] <= entry_z)
        
        df['Buy_Macro'] = False
        df['Buy_Dip'] = False
        df['Buy_Mom'] = False
        df['Master_Buy'] = False
        df['Master_Sell'] = False
        df['Exit_Std'] = False
        df['Exit_Top'] = False
        df['Exit_Bear'] = False
        df['Signal'] = 'Cash ⚪'
        
        armed_cycle_low = False
        bars_since_armed = 0
        in_position = False
        
        for i in range(1, len(df)):
            current_idx = df.index[i]
            if df['Macro_Z'].iloc[i] <= cycle_low_z or df['Short_Z'].iloc[i] <= cycle_low_z:
                armed_cycle_low = True
                bars_since_armed = 0
            elif armed_cycle_low:
                bars_since_armed += 1
                if bars_since_armed > short_len:
                    armed_cycle_low = False
                    
            buy_macro = armed_cycle_low and df['Cross_Zero'].iloc[i]
            buy_dip = df['Bull_Regime'].iloc[i] and df['Cross_Dip'].iloc[i] and not buy_macro
            
            if use_macro:
                buy_mom = df['Bull_Regime'].iloc[i] and df['Cross_Entry'].iloc[i]
            else:
                buy_mom = df['Cross_Entry'].iloc[i]
                
            exit_std = (df['Short_Z'].iloc[i] < exit_z) and (df['Short_Z_Prev'].iloc[i] >= exit_z)
            exit_top = ((df['Macro_Z'].iloc[i] < cycle_high_z) and (df['Macro_Z_Prev'].iloc[i] >= cycle_high_z)) or                        ((df['Short_Z'].iloc[i] < cycle_high_z) and (df['Short_Z_Prev'].iloc[i] >= cycle_high_z))
            
            exit_bear = False
            if use_macro and not df['Bull_Regime'].iloc[i]:
                exit_bear = (df['Short_Z'].iloc[i] < -0.5) and (df['Short_Z_Prev'].iloc[i] >= -0.5)
                
            if (buy_macro or buy_dip or buy_mom) and not in_position:
                in_position = True
                df.at[current_idx, 'Master_Buy'] = True
                df.at[current_idx, 'Signal'] = 'Buy 🟢'
                if buy_macro: 
                    df.at[current_idx, 'Buy_Macro'] = True
                    armed_cycle_low = False 
                elif buy_dip: df.at[current_idx, 'Buy_Dip'] = True
                else: df.at[current_idx, 'Buy_Mom'] = True
            elif (exit_std or exit_top or exit_bear) and in_position:
                in_position = False
                df.at[current_idx, 'Master_Sell'] = True
                df.at[current_idx, 'Signal'] = 'Sell 🔴'
                if exit_top: df.at[current_idx, 'Exit_Top'] = True
                elif exit_bear: df.at[current_idx, 'Exit_Bear'] = True
                else: df.at[current_idx, 'Exit_Std'] = True
            else:
                if in_position:
                    df.at[current_idx, 'Signal'] = 'Hold 🟡'
                else:
                    df.at[current_idx, 'Signal'] = 'Cash ⚪'

        st.sidebar.markdown("---")
        st.sidebar.header("📥 Export Data to Excel")
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        
        start_date = st.sidebar.date_input("From Date", value=min_date, min_value=min_date, max_value=max_date, key="mstr_start")
        end_date = st.sidebar.date_input("To Date", value=max_date, min_value=min_date, max_value=max_date, key="mstr_end")
        filtered_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()

        export_full_df = pd.DataFrame(index=filtered_df.index)
        export_full_df['Date'] = filtered_df.index.strftime('%Y-%m-%d')
        export_full_df['MSTR Price ($)'] = filtered_df['Close'].round(2)
        export_full_df['50D SMA ($)'] = filtered_df['SMA_50'].round(2)
        export_full_df['200D SMA ($)'] = filtered_df['SMA_200'].round(2)
        export_full_df['Tactical Z-Score (50D)'] = filtered_df['Short_Z'].round(4)
        export_full_df['Macro Z-Score (200D)'] = filtered_df['Macro_Z'].round(4)
        export_full_df['Signal'] = filtered_df['Signal']

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            export_full_df.to_excel(writer, sheet_name='MSTR_ZScore_Data', index=False)
        excel_data = excel_buffer.getvalue()

        st.sidebar.download_button(
            label="📥 Download Excel File",
            data=excel_data,
            file_name=f"MSTR_Data_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="mstr_dl"
        )

        latest = df.iloc[-1]
        btc_price = get_btc_price_latest()
        fng_val, fng_class = get_fear_and_greed_latest()
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("MSTR Live Price", f"${latest['Close']:.2f}")
        col2.metric("Regime", "BULL 🟢" if latest['Bull_Regime'] else "BEAR 🔴")
        col3.metric("Current Signal", latest['Signal'])
        col4.metric("BTC Live Price", f"${btc_price:,.2f}" if btc_price else "N/A")
        col5.metric("Crypto F&G", f"{fng_val}", fng_class, delta_color="off")
        col6.metric("Tactical Z (50D)", f"{latest['Short_Z']:.2f}σ")
        col7.metric("Macro Z (200D)", f"{latest['Macro_Z']:.2f}σ")
        
        st.markdown("---")
        st.subheader("💡 Current Strategy Action Plan")
        
        def render_action_callout(z_score, timeframe_label):
            if z_score <= -2.0:
                st.success(f"**{timeframe_label} ({z_score:.2f}σ) — Deep Discount:** Accumulate Spot Shares / Buy Deep ITM LEAPS.")
            elif z_score <= -1.0:
                st.info(f"**{timeframe_label} ({z_score:.2f}σ) — Undervalued:** Watch for reversal. Hold shares.")
            elif z_score <= 0.0:
                st.warning(f"**{timeframe_label} ({z_score:.2f}σ) — Mean Reverting:** Wait in Cash. Avoid holding options (Theta/Vega risk).")
            elif z_score <= 1.0:
                st.info(f"**{timeframe_label} ({z_score:.2f}σ) — Fair Value:** Hold existing trend. Do not execute new swing trades.")
            elif z_score < 2.0:
                st.success(f"**{timeframe_label} ({z_score:.2f}σ) — Momentum Zone:** Active Buy for Shares and ATM LEAPS.")
            else:
                st.error(f"**{timeframe_label} ({z_score:.2f}σ) — Extreme Euphoria:** Distribute / Scale Out 25%–50% to Cash. Sell LEAPS.")

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            render_action_callout(latest['Short_Z'], "Tactical Status (50D)")
        with col_stat2:
            render_action_callout(latest['Macro_Z'], "Macro Status (200D)")
            
        st.markdown("---")
        st.subheader(f"Composite Chart: Price & {z_view} Bars")
        
        z_colorscale = [[0.0, 'rgba(239, 68, 68, 0.8)'], [0.25, 'rgba(249, 115, 22, 0.8)'], [0.5, 'rgba(255, 255, 255, 0.8)'], [0.75, 'lightgreen'], [1.0, 'darkgreen']]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if z_view == "Tactical (50D)":
            bar_y, bar_name = df['Short_Z'], "Tactical Z-Score (50D)"
            z_hovertemplate = "<b>Date:</b> %{customdata[0]}<br><b>Z-Score (50D):</b> %{y:.2f}σ<extra></extra>"
        else:
            bar_y, bar_name = df['Macro_Z'], "Macro Z-Score (200D)"
            z_hovertemplate = "<b>Date:</b> %{customdata[0]}<br><b>Z-Score (200D):</b> %{y:.2f}σ<extra></extra>"

        fig.add_trace(go.Bar(x=df.index, y=bar_y, name=bar_name, marker=dict(color=bar_y, colorscale=z_colorscale, cmin=-3.0, cmax=3.0, showscale=False), customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate=z_hovertemplate), secondary_y=False)
        
        price_hovertemplate = "<b>Date:</b> %{customdata[0]}<br><b>Price:</b> $%{y:,.2f}<extra></extra>"
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="MSTR Price", line=dict(color='white', width=2), customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate=price_hovertemplate), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="50D SMA", line=dict(color='#3b82f6', width=1.5), customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate="<b>Date:</b> %{customdata[0]}<br><b>50D SMA:</b> $%{y:,.2f}<extra></extra>"), secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name="200D SMA", line=dict(color='orange', width=1.5), customdata=np.stack((df['Hover_Date'],), axis=-1), hovertemplate="<b>Date:</b> %{customdata[0]}<br><b>200D SMA:</b> $%{y:,.2f}<extra></extra>"), secondary_y=True)
        
        buy_dates = df[df['Master_Buy']].index
        buy_prices = df[df['Master_Buy']]['Close']
        buy_hover = df[df['Master_Buy']]['Hover_Date']
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name="Buy Signal", marker=dict(color='lime', size=12, symbol='triangle-up', line=dict(color='black', width=1)), customdata=np.stack((buy_hover,), axis=-1), hovertemplate="<b>BUY</b><br>" + price_hovertemplate), secondary_y=True)
                                       
        sell_dates = df[df['Master_Sell']].index
        sell_prices = df[df['Master_Sell']]['Close']
        sell_hover = df[df['Master_Sell']]['Hover_Date']
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name="Sell Signal", marker=dict(color='red', size=12, symbol='triangle-down', line=dict(color='black', width=1)), customdata=np.stack((sell_hover,), axis=-1), hovertemplate="<b>SELL</b><br>" + price_hovertemplate), secondary_y=True)

        fig.add_hline(y=2.0, line_dash="dash", line_color="darkgreen", secondary_y=False, annotation_text="Extreme Overvalued (+2.0σ)", annotation_position="top left", annotation_font=dict(color="darkgreen", size=10))
        fig.add_hline(y=1.0, line_dash="dot", line_color="lightgreen", secondary_y=False, annotation_text="Overvalued (+1.0σ)", annotation_position="top left", annotation_font=dict(color="lightgreen", size=10))
        fig.add_hline(y=0.0, line_dash="solid", line_color="rgba(255, 255, 255, 0.6)", secondary_y=False, annotation_text="Fair Price (0.0σ)", annotation_position="top left", annotation_font=dict(color="white", size=10))
        fig.add_hline(y=-1.0, line_dash="dot", line_color="rgba(249, 115, 22, 0.5)", secondary_y=False, annotation_text="Undervalued (-1.0σ)", annotation_position="bottom left", annotation_font=dict(color="orange", size=10))
        fig.add_hline(y=-2.0, line_dash="dash", line_color="rgba(239, 68, 68, 0.7)", secondary_y=False, annotation_text="Extreme Undervaluation (-2.0σ)", annotation_position="bottom left", annotation_font=dict(color="red", size=10))

        fig.update_layout(height=650, template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="Z-Score (σ)", secondary_y=False, showgrid=False)
        fig.update_yaxes(title_text="MSTR Price ($)", secondary_y=True, type="log")

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Z-Score Valuation Framework Reference"):
            st.markdown('''Based on statistical distribution principles, Z-score standard deviation ranges correspond to these valuation tiers:
            * **Extreme Overvaluation (+2.0σ or higher):** Price trades 2+ standard deviations above its rolling average. Historically occurs under ~2.5% of the time.
            * **Overvalued (+1.0σ to +2.0σ):** Price is trading at a notable statistical premium to its rolling average.
            * **Fair Price (-1.0σ to +1.0σ):** Price is within 1 standard deviation of its mean (~68% of historical observations).
            * **Undervalued (-2.0σ to -1.0σ):** Price is trading at a notable discount relative to its recent trend.
            * **Extreme Undervaluation (-2.0σ or lower):** Price sits 2+ standard deviations below its rolling mean (~2.5% tail event), representing rare market dislocation or cycle bottoms.''')

        st.subheader("Historical Z-Score Trade Log")
        actions = []
        for _, row in df.iterrows():
            if row['Buy_Macro']: actions.append('Macro Bottom Buy')
            elif row['Buy_Dip']: actions.append('Bull Dip Buy')
            elif row['Buy_Mom']: actions.append('Momentum Buy')
            elif row['Exit_Std']: actions.append('Momentum Sell (Mean Reversion)')
            elif row['Exit_Top']: actions.append('Macro Top Exit')
            elif row['Exit_Bear']: actions.append('Bear Regime Exit')
            else: actions.append('-')
            
        df['Script Execution'] = actions
        log_df = df[df['Script Execution'] != '-'].copy()
        
        export_df = pd.DataFrame(index=log_df.index)
        export_df['Price (Adj)'] = log_df['Close'].apply(lambda x: f"${x:,.2f}")
        export_df['Tactical Z-Score (50D)'] = log_df['Short_Z'].round(2)
        export_df['Macro Z-Score (200D)'] = log_df['Macro_Z'].round(2)
        export_df['Script Execution'] = log_df['Script Execution']
        st.dataframe(export_df, use_container_width=True)
    else:
        st.warning("Not enough data to calculate 200-day macro baseline.")

# --- MAIN CONTROLLER ---
st.sidebar.title("Navigation")
view_selection = st.sidebar.radio("Select Dashboard View", ["MSTR Z-Score Strategy", "Fear & Greed Cadence"])

if view_selection == "Fear & Greed Cadence":
    render_fng_cadence()
else:
    render_mstr_zscore()
