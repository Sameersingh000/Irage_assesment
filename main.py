import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameters
trading_days = 252  # number of trading days in a year
t = np.linspace(0, trading_days, trading_days)

# Given CAGR values (as multipliers over one year)
# Mean Reversion: 1096.59% => final multiplier ≈ 11.9659
# Directional: 1748.69% => final multiplier ≈ 18.4869
# Semi-Directional: 1423.85% => final multiplier ≈ 15.2385

# Calculate daily drift using: drift = exp(log(final_multiplier)/trading_days)-1
drift_mean_rev = np.exp(np.log(11.9659)/trading_days) - 1
drift_directional = np.exp(np.log(18.4869)/trading_days) - 1
drift_semi = np.exp(np.log(15.2385)/trading_days) - 1

# Assume some volatility (estimated) for simulation
vol_mean_rev = 0.02  # 2% daily volatility
vol_directional = 0.03
vol_semi = 0.025

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic daily returns using geometric Brownian motion
def generate_equity_curve(drift, vol, days):
    # Generate daily returns
    daily_returns = drift + vol * np.random.randn(days)
    # Start at 1 and take cumulative product to simulate equity growth
    equity = np.cumprod(1 + daily_returns)
    return equity

equity_mean_rev = generate_equity_curve(drift_mean_rev, vol_mean_rev, trading_days)
equity_directional = generate_equity_curve(drift_directional, vol_directional, trading_days)
equity_semi = generate_equity_curve(drift_semi, vol_semi, trading_days)

# Combined portfolio as an equally weighted average of the three strategies
combined_equity = (equity_mean_rev + equity_directional + equity_semi) / 3

# Plot individual equity curves and the combined portfolio curve
plt.figure(figsize=(12, 6))
plt.plot(t, equity_mean_rev, label="Mean Reversion Strategy", linestyle="--", alpha=0.7)
plt.plot(t, equity_directional, label="Directional Strategy", linestyle="--", alpha=0.7)
plt.plot(t, equity_semi, label="Semi-Directional Strategy", linestyle="--", alpha=0.7)
plt.plot(t, combined_equity, label="Combined Portfolio", linewidth=2, color='black')

plt.xlabel("Trading Days")
plt.ylabel("Normalized Equity Value")
plt.title("Combined Portfolio Equity Curve")
plt.legend()
plt.grid(True)
plt.show()
