import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(180deg, #0b2447 0%, #1b3b73 100%);
        color: #f2f7ff;
    }
    .stButton>button {
        background-color: #f7b733;
        color: #0b2447;
        font-weight: 700;
    }
    .stSlider>div>div>div>div {
        background: #f7b733;
    }
    .st-bf {
        color: #f2f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='padding: 1rem 1rem 0; border-radius: 18px; background: rgba(255,255,255,0.05);'>
        <h1 style='color: #ffdd57; margin-bottom: 0.2rem;'>Stock Price Prediction Tool</h1>
        <p style='font-size:1.05rem; color:#e6eefc;'>Explore historical price trends and forecast future close values using polynomial regression.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Welcome! Choose your data source, enter parameters, then click 'Generate prediction' to see the forecast.")

with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Select data source:", ["Yahoo Finance (Online)", "Upload CSV"])
    
    st.header("Input Settings")
    
    if data_source == "Yahoo Finance (Online)":
        ticker = st.text_input("Stock ticker", value="AAPL").upper()
        today = date.today()
        start_date = st.date_input("Start date", today - timedelta(days=365 * 2))
        end_date = st.date_input("End date", today)
    else:
        ticker = None
        start_date = None
        end_date = None
        st.info("Upload a CSV file with columns: Date, Close (required). Additional columns: Open, High, Low, Volume (optional).")
    
    degree = st.slider("Polynomial degree", min_value=1, max_value=5, value=3)
    forecast_days = st.slider("Forecast horizon (days)", min_value=5, max_value=30, value=10)
    show_raw = st.checkbox("Show raw data", value=False)
    st.write("**Note:** After changing any parameters, click 'Generate prediction' to update the forecast.")

# Check if inputs have changed
current_inputs = (data_source, ticker, start_date, end_date, degree, forecast_days, show_raw)
if "previous_inputs" not in st.session_state or st.session_state.previous_inputs != current_inputs:
    st.session_state.prediction_ready = False
st.session_state.previous_inputs = current_inputs

# Handle CSV upload
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
else:
    uploaded_file = None

def fetch_stock_data(ticker_symbol, start, end):
    try:
        data = yf.download(
            ticker_symbol,
            start=start,
            end=end + timedelta(days=1),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        return data
    except Exception as exc:
        st.error(f"Error fetching data for {ticker_symbol}: {exc}")
        return pd.DataFrame()

def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Check required columns
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("CSV must contain 'Date' and 'Close' columns.")
            return None
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as exc:
        st.error(f"Error reading CSV file: {exc}")
        return None


def display_prediction(data, forecast_df, fig, ticker_symbol, forecast_days, degree, show_raw):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{ticker_symbol} Price Chart")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Forecast Summary")
        latest_close = float(data["Close"].iloc[-1])
        next_price = float(forecast_df["Forecast"].iloc[-1])
        change = next_price - latest_close
        change_pct = (change / latest_close) * 100
        st.metric("Most Recent Close", f"${latest_close:,.2f}")
        st.metric(f"Forecast in {forecast_days} days", f"${next_price:,.2f}", f"{change_pct:+.2f}%")
        st.write("### Model details")
        st.write(f"Polynomial degree: **{degree}**")
        st.write(f"Data window: **{len(data)} days**")

    st.markdown("---")
    st.write("### Forecast table")
    st.dataframe(forecast_df.set_index("Date"))

    if show_raw:
        st.markdown("---")
        st.write("### Raw historical data")
        # Only show columns that exist in the data
        cols_to_show = ["Date"]
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                cols_to_show.append(col)
        st.dataframe(data[cols_to_show].set_index("Date"))


if "prediction_ready" not in st.session_state:
    st.session_state.prediction_ready = False

if st.button("Generate prediction"):
    if data_source == "Yahoo Finance (Online)":
        if not ticker or ticker.strip() == "":
            st.error("Please enter a stock ticker symbol.")
        elif start_date >= end_date:
            st.error("Start date must come before end date.")
        else:
            try:
                with st.spinner(f"Fetching data for {ticker}..."):
                    raw_data = fetch_stock_data(ticker, start_date, end_date)
                if raw_data.empty:
                    st.error(f"No data found for {ticker} in the selected date range. Please check:\n- Ticker symbol is correct (e.g., AAPL, GOOGL, MSFT)\n- Date range is valid\n- Or try uploading your own CSV file instead.")
                    st.session_state.prediction_ready = False
                else:
                    data = raw_data.reset_index()
                    data["DayIndex"] = np.arange(len(data))
                    X = data[["DayIndex"]]
                    y = data["Close"]

                    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    model.fit(X, y)

                    forecast_index = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
                    forecast_values = model.predict(forecast_index).ravel()

                    future_dates = [data["Date"].iloc[-1] + timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
                    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast_values})

                    plot_df = pd.concat(
                        [
                            data[["Date", "Close"]].rename(columns={"Close": "Price"}).assign(Series="Actual"),
                            forecast_df.rename(columns={"Forecast": "Price"}).assign(Series="Forecast"),
                        ],
                        ignore_index=True,
                    )

                    fig = px.line(
                        plot_df,
                        x="Date",
                        y="Price",
                        color="Series",
                        markers=True,
                        labels={"Price": "Price (USD)", "Date": "Date"},
                        title=f"{ticker} Actual vs Forecast",
                    )
                    fig.update_traces(
                        marker=dict(size=6),
                        line=dict(width=3),
                        hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}<extra></extra>",
                    )
                    fig.update_layout(
                        plot_bgcolor="#081f3a",
                        paper_bgcolor="#081f3a",
                        font_color="#f2f7ff",
                        hovermode="x unified",
                        legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.14)", borderwidth=1),
                        margin=dict(l=20, r=20, t=60, b=20),
                        title=dict(font=dict(size=20, color="#ffdd57")),
                    )
                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor="#0c2a53",
                        gridwidth=1,
                        zeroline=False,
                        tickformat="%b %Y",
                        tickfont=dict(color="#c5d4ff"),
                        showline=True,
                        linecolor="#264770",
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridcolor="#0c2a53",
                        gridwidth=1,
                        zeroline=False,
                        tickfont=dict(color="#c5d4ff"),
                        showline=True,
                        linecolor="#264770",
                    )

                    st.session_state.prediction_ready = True
                    st.session_state.ticker = ticker
                    st.session_state.data = data
                    st.session_state.forecast_df = forecast_df
                    st.session_state.fig = fig
                    st.session_state.forecast_days = forecast_days
                    st.session_state.degree = degree
                    st.session_state.show_raw = True
            except Exception as exc:
                st.error(f"Unable to process stock data: {exc}")
    else:  # CSV Upload
        if uploaded_file is None:
            st.error("Please upload a CSV file.")
        else:
            data = load_csv_data(uploaded_file)
            if data is not None:
                try:
                    data["DayIndex"] = np.arange(len(data))
                    X = data[["DayIndex"]]
                    y = data["Close"]

                    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    model.fit(X, y)

                    forecast_index = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
                    forecast_values = model.predict(forecast_index).ravel()

                    future_dates = [data["Date"].iloc[-1] + timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
                    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast_values})

                    plot_df = pd.concat(
                        [
                            data[["Date", "Close"]].rename(columns={"Close": "Price"}).assign(Series="Actual"),
                            forecast_df.rename(columns={"Forecast": "Price"}).assign(Series="Forecast"),
                        ],
                        ignore_index=True,
                    )

                    fig = px.line(
                        plot_df,
                        x="Date",
                        y="Price",
                        color="Series",
                        markers=True,
                        labels={"Price": "Price (USD)", "Date": "Date"},
                        title="Uploaded Data: Actual vs Forecast",
                    )
                    fig.update_traces(
                        marker=dict(size=6),
                        line=dict(width=3),
                        hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}<extra></extra>",
                    )
                    fig.update_layout(
                        plot_bgcolor="#081f3a",
                        paper_bgcolor="#081f3a",
                        font_color="#f2f7ff",
                        hovermode="x unified",
                        legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.14)", borderwidth=1),
                        margin=dict(l=20, r=20, t=60, b=20),
                        title=dict(font=dict(size=20, color="#ffdd57")),
                    )
                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor="#0c2a53",
                        gridwidth=1,
                        zeroline=False,
                        tickformat="%b %Y",
                        tickfont=dict(color="#c5d4ff"),
                        showline=True,
                        linecolor="#264770",
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridcolor="#0c2a53",
                        gridwidth=1,
                        zeroline=False,
                        tickfont=dict(color="#c5d4ff"),
                        showline=True,
                        linecolor="#264770",
                    )

                    st.session_state.prediction_ready = True
                    st.session_state.ticker = "Uploaded Data"
                    st.session_state.data = data
                    st.session_state.forecast_df = forecast_df
                    st.session_state.fig = fig
                    st.session_state.forecast_days = forecast_days
                    st.session_state.degree = degree
                    st.session_state.show_raw = True
                except Exception as exc:
                    st.error(f"Unable to process uploaded data: {exc}")

if st.session_state.prediction_ready:
    display_prediction(
        st.session_state.data,
        st.session_state.forecast_df,
        st.session_state.fig,
        st.session_state.ticker,
        st.session_state.forecast_days,
        st.session_state.degree,
        st.session_state.show_raw,
    )

st.markdown(
    """
    <div style='margin-top: 2rem; padding: 1rem; border-radius: 18px; background: rgba(255,255,255,0.05);'>
        <h2 style='color: #ffdd57;'>How it works</h2>
        <ul style='color: #dbe7ff;'>
            <li><strong>Online Mode:</strong> Fetches historical stock prices using Yahoo Finance (e.g., AAPL, GOOGL, MSFT).</li>
            <li><strong>Upload Mode:</strong> Upload your own CSV file with Date and Close columns.</li>
            <li>Trains a polynomial regression model on closing prices.</li>
            <li>Displays a clean chart with actual vs forecast values.</li>
            <li>Lets you adjust the forecast horizon and model complexity.</li>
        </ul>
        <h3 style='color: #ffdd57;'>CSV Format Example</h3>
        <p style='color: #dbe7ff; font-size: 0.9rem;'>Date,Open,High,Low,Close,Volume<br/>2024-01-01,150.5,152.3,150.0,151.8,1000000<br/>2024-01-02,151.8,153.5,151.5,152.9,1200000</p>
    </div>
    """,
    unsafe_allow_html=True,
)
