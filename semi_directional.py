import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('nifty_options_realistic.csv')

# Convert date columns
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce', dayfirst=True)
df['EXPIRY_DT'] = pd.to_datetime(df['EXPIRY_DT'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['TIMESTAMP'])  # Ensure no NaT values

# Compute Moving Average & Bollinger Bands
df['MA20'] = df['CLOSE'].rolling(window=20, min_periods=1).mean()
df['UpperBand'] = df['MA20'] + 1.5 * df['CLOSE'].rolling(window=20, min_periods=1).std()
df['LowerBand'] = df['MA20'] - 1.5 * df['CLOSE'].rolling(window=20, min_periods=1).std()

# Calculate ATR (Average True Range)
df['High-Low'] = df['HIGH'] - df['LOW']
df['High-Close'] = abs(df['HIGH'] - df['CLOSE'].shift(1))
df['Low-Close'] = abs(df['LOW'] - df['CLOSE'].shift(1))
df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()  # 14-period ATR

# Parameters for execution issues
slippage_pct = 0.01  # 1% slippage per trade
bid_ask_spread_pct = 0.005  # 0.5% bid-ask spread per trade
commission_per_trade = 20  # Flat commission per trade

def backtest_strategy_realistic(df):
    initial_capital = 100000
    capital = initial_capital
    trades = []
    capital_history = []
    timestamps = []
    position = None
    lot_size = 25  # Number of shares per contract
    risk_per_trade = 0.01  # Risk only 2% of capital per trade

    for _, row in df.iterrows():
        current_time = row['TIMESTAMP']
        if pd.isna(current_time) or row['ATR'] == 0:
            continue  # Skip invalid timestamps

        # Calculate position size based on risk
        position_size = (capital * risk_per_trade) / (1.5 * row['ATR'])  # Stop loss at 1.5 ATR
        position_size = max(1, int(position_size / lot_size)) * lot_size  # Round to lot size

        # Adjusted entry price considering bid-ask spread
        entry_price = row['CLOSE'] * (1 + bid_ask_spread_pct)

        if abs(row['CLOSE'] - row['MA20']) > row['CLOSE'] * 0.03 and position is None:
            stop_loss = entry_price - 1.5 * row['ATR']
            take_profit = entry_price + 2 * row['ATR']
            position = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': current_time,
                'size': position_size
            }

        if position is not None:
            # Adjusted exit price considering slippage
            exit_price = row['CLOSE'] * (1 - slippage_pct) if row['CLOSE'] <= position['stop_loss'] else row['CLOSE'] * (1 + slippage_pct)
            
            if row['CLOSE'] <= position['stop_loss'] or row['CLOSE'] >= position['take_profit']:
                pnl = ((exit_price - position['entry_price']) * position['size']) - commission_per_trade
                capital += pnl
                trade_duration = (current_time - position['entry_time']).total_seconds() / 60  # in minutes
                trades.append({'PnL': pnl, 'Duration': trade_duration})
                position = None

        capital_history.append(capital)
        timestamps.append(current_time)

    return pd.DataFrame(trades), capital, capital_history, timestamps

def compute_metrics(trades, final_capital, initial_capital, timestamps):
    timestamps = [t for t in timestamps if pd.notna(t)]
    if len(timestamps) < 2:
        return -1, 0, 0, 0, 0

    start_date, end_date = timestamps[0], timestamps[-1]
    years = max((end_date - start_date).days / 365.25, 1)
    
    cagr = ((final_capital / max(initial_capital, 1)) ** (1 / years)) - 1 if final_capital > 0 else -1

    if not trades.empty:
        equity_curve = trades['PnL'].cumsum()
        max_drawdown = (equity_curve.cummax() - equity_curve).max()
        sharpe = trades['PnL'].mean() / trades['PnL'].std() * np.sqrt(252) if trades['PnL'].std() > 0 else 0
        win_rate = len(trades[trades['PnL'] > 0]) / len(trades) if len(trades) > 0 else 0
        avg_pnl = trades['PnL'].mean()
    else:
        max_drawdown, sharpe, win_rate, avg_pnl = 0, 0, 0, 0
    
    return cagr, max_drawdown, sharpe, win_rate, avg_pnl

# Running the backtest
trades, final_capital, capital_history, timestamps = backtest_strategy_realistic(df)
cagr, mdd, sharpe, win_rate, avg_pnl = compute_metrics(trades, final_capital, 100000, timestamps)

print(f"Final Realistic Strategy - CAGR: {cagr*100:.2f}%, MDD: ₹{mdd:.2f}, Sharpe: {sharpe:.2f}, Win Rate: {win_rate*100:.2f}%, Avg PnL: ₹{avg_pnl:.2f}")
print("Final Capital:", final_capital)

# Plot the updated capital curve
# Convert timestamps to a numerical index for better x-axis formatting
plt.figure(figsize=(12, 6))
plt.plot(timestamps, capital_history, label='Realistic Capital Over Time', color='red', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Capital')
plt.title('Realistic Strategy - Capital Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# Equity curve (Cumulative PnL)
if not trades.empty:
    trades['Cumulative PnL'] = trades['PnL'].cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(trades.index, trades['Cumulative PnL'], label='Cumulative PnL', color='blue', linewidth=2)
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative P&L')
    plt.title('Equity Curve - Cumulative P&L Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

