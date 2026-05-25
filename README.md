# Stock Price Prediction Tool

A Python-based stock price prediction app with a polished Streamlit UI.

## Features

- Enter any stock ticker symbol (for example, `AAPL`, `MSFT`, `TSLA`).
- Fetches historical price data automatically from Yahoo Finance.
- Trains a polynomial regression model on closing prices.
- Displays actual vs forecast charts with responsive UI.
- Adjustable forecast horizon and polynomial degree.

## Setup

1. Install Python 3.9+.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notes

- This demo is for learning and visualization only.
- Polynomial regression is a simple baseline, not a production trading model.
