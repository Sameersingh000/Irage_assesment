import pandas as pd
import numpy as np
import yfinance as yf

def calculate_var_es(ticker, start_date, end_date, confidence_level=0.95, investment=100000):
    
    # Fetch data
    data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    
    # Compute log returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # Compute 1-day 95% VaR and ES
    VaR = data['log_return'].quantile(1 - confidence_level)
    ES = data['log_return'][data['log_return'] <= VaR].mean()

    # INR value
    VaR_money = abs(VaR) * investment
    ES_money = abs(ES) * investment if not np.isnan(ES) else 0  # Avoid NaN issue

    return VaR, ES, VaR_money, ES_money


if __name__ == "__main__":
    #parameters
    ticker = "RELIANCE.NS"
    start_date = "2023-01-01"
    end_date = "2025-01-01"
    confidence_level = 0.95
    investment = 100000

    # function calling
    VaR, ES, VaR_money, ES_money = calculate_var_es(ticker, start_date, end_date, confidence_level, investment)

    # Print results with formatting
    print("=" * 50)
    print(f"Stock: {ticker}")
    print(f"Confidence Level: {confidence_level:.0%}")
    print("-" * 50)
    print(f"1-Day 95% VaR: {VaR:.2%} (₹{VaR_money:,.2f})")
    print(f"1-Day 95% Expected Shortfall (ES): {ES:.2%} (₹{ES_money:,.2f})")
    print("=" * 50)
