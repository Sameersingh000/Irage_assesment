import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('nifty_options_realistic.csv')

# Fix: Handle date parsing for ISO/mixed formats
df['EXPIRY_DT'] = pd.to_datetime(df['EXPIRY_DT'], errors='coerce')  # Auto-detect format
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')   # Auto-detect format

# Drop rows with invalid dates (if any)
df = df.dropna(subset=['EXPIRY_DT', 'TIMESTAMP'])

# Sort data
df = df.sort_values(['EXPIRY_DT', 'STRIKE_PR', 'TIMESTAMP'])

# Proceed with the rest of your strategy...
# [Your existing code for MA, ATR, backtest, etc.]

# Calculate Moving Averages (50-period and 200-period)
df['MA50'] = df['CLOSE'].rolling(window=50, min_periods=1).mean()
df['MA200'] = df['CLOSE'].rolling(window=200, min_periods=1).mean()

# Generate Directional Strategy signals
df['Signal_Dir'] = np.where(df['MA50'] > df['MA200'], 'Buy Call', 
                            np.where(df['MA50'] < df['MA200'], 'Buy Put', 'Hold'))

# Backtesting Parameters
initial_capital = 100000  # ₹1L initial capital
position_size = 1  # Nifty options multiplier (adjust based on capital)
slippage = 0.001  # 0.1% slippage per trade
transaction_cost = 20  # ₹20 transaction cost per order

# Backtesting Function
def backtest(df, signal_column):
    trades = []
    position = None
    capital = initial_capital
    capital_history = [capital]  # Track capital for visualization
    
    for i, row in df.iterrows():
        signal = row[signal_column]
        
        # Open position
        if signal == 'Buy Call' and position is None:
            entry_price = row['CLOSE']
            position = {'type': 'CE', 'entry': entry_price, 'time': row['TIMESTAMP']}
        elif signal == 'Buy Put' and position is None:
            entry_price = row['CLOSE']
            position = {'type': 'PE', 'entry': entry_price, 'time': row['TIMESTAMP']}
        
        # Simplified exit condition (exit at 3 PM or trend reversal)
        if position:
            if row['TIMESTAMP'].hour == 15 or (
                (position['type'] == 'CE' and row['MA50'] < row['MA200']) or 
                (position['type'] == 'PE' and row['MA50'] > row['MA200'])
            ):
                exit_price = row['CLOSE']
                
                # Apply slippage and transaction cost
                pnl = (exit_price - position['entry']) * position_size * (1 - slippage)
                pnl -= transaction_cost  # Subtract transaction cost
                
                capital += pnl  # Update capital
                capital_history.append(capital)  # Track capital history
                trades.append({'PnL': pnl, 'Duration': (row['TIMESTAMP'] - position['time']).seconds / 60})
                position = None

    return pd.DataFrame(trades), capital, capital_history

# Run backtest for Directional Strategy
trades_dir, final_capital_dir, capital_history_dir = backtest(df, 'Signal_Dir')

# Compute Performance Metrics
def compute_metrics(trades, final_capital, initial_capital):
    years = len(df['TIMESTAMP'].dt.year.unique()) if len(df['TIMESTAMP'].dt.year.unique()) > 0 else 1
    total_pnl = trades['PnL'].sum()
    cagr = (final_capital / initial_capital) ** (1 / years) - 1
    mdd = (trades['PnL'].cumsum().cummax() - trades['PnL'].cumsum()).max()
    sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(252) if trades['PnL'].std() != 0 else 0
    win_rate = len(trades[trades['PnL'] > 0]) / len(trades) if len(trades) > 0 else 0
    avg_pnl = trades['PnL'].mean()

    return cagr, mdd, sharpe, win_rate, avg_pnl

# Compute metrics for Directional Strategy
metrics_dir = compute_metrics(trades_dir, final_capital_dir, initial_capital)

# Print results for Directional Strategy
print(f"Directional Strategy - CAGR: {metrics_dir[0]:.2%}, MDD: ₹{metrics_dir[1]:.2f}, Sharpe: {metrics_dir[2]:.2f}, Win Rate: {metrics_dir[3]:.2%}, Avg PnL: ₹{metrics_dir[4]:.2f}")

# Plot Capital Growth Over Trades
plt.figure(figsize=(10, 5))
plt.plot(range(len(capital_history_dir)), capital_history_dir, marker='o', linestyle='-')

# Labels and title
plt.xlabel("Number of Trades")
plt.ylabel("Capital (₹)")
plt.title("Capital Growth Over Trades - Directional Strategy")
plt.grid(True)

# Show the plot
plt.show()
