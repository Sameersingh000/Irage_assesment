import pandas as pd
import numpy as np
import yfinance as yf

# Fetch data
ticker = "RELIANCE.NS"
data = yf.download(ticker, start="2023-01-01", end="2025-01-01")[['Close']]

# Compute log returns
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Confidence level
confidence_level = 0.95

# Calculate 1-day 95% VaR and ES (Historical Method)
VaR = data['log_return'].quantile(1 - confidence_level)  # No need for negative sign
ES = data['log_return'][data['log_return'] <= VaR].mean()  # Use VaR directly

# Assume an investment amount
investment = 100000

# Convert to monetary values
VaR_money = abs(VaR) * investment
ES_money = abs(ES) * investment

# Print results with better formatting
print("=" * 50)
print(f"Stock: {ticker}")
print(f"Confidence Level: {confidence_level:.0%}")
print("-" * 50)
print(f"1-Day 95% VaR: {VaR:.4%} (₹{VaR_money:,.2f})")
print(f"1-Day 95% Expected Shortfall (ES): {ES:.4%} (₹{ES_money:,.2f})")
print("=" * 50)
