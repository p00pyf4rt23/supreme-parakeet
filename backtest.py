import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from calibrator import load_tickers, log_returns, calibrate_mjd_mle, method_of_moments, MIN_RETURNS
from engine import Engine

# --- Backtest Settings (Hourly) ---
LOOKBACK_DAYS = 252               # days for lookback history
BACKTEST_DAYS = 5                 # business days in test
CAPITAL = 100_000                 # starting capital
ETA = 0.002                       # minimum edge threshold
COS_DOMAIN = (-0.05, 0.05)        # domain for COS inversion
COS_N = 64                        # COS grid size
HOURS_PER_DAY = 6.5               # trading hours per business day
DT = 1.0 / (252 * HOURS_PER_DAY)  # hourly dt in annualized terms
RISK_MULTIPLIER = 5               # fraction of Kelly to use
MAX_EXPOSURE = 1.0                # max fraction of capital per trade
TX_COST = 0.0002                  # transaction cost per share (round-trip 2 bps)
N_MAX = 20                        # max iterations for MJD calibration
# Stop-loss and take-profit thresholds per stock
STOP_LOSS_PCT = 0.02              # stop loss at -2% intra-hour
TAKE_PROFIT_PCT = 0.03            # take profit at +3% from entry
# Shorting toggle
ALLOW_SHORTS = True            # set False to disable short positions


# Load tickers and initialize engine
tickers = load_tickers("allowed_stocks.txt")
env = Engine(capital=CAPITAL,
             dt=DT,
             domain=COS_DOMAIN,
             N=COS_N,
             delta_u=0.01,
             delta_d=0.01,
             eta=0.0,
             risk_multiplier=RISK_MULTIPLIER,
             max_exposure=MAX_EXPOSURE)

# Fetch hourly market data
now = pd.Timestamp.now().floor('h')
history_start = now - pd.Timedelta(days=LOOKBACK_DAYS + BACKTEST_DAYS + 5)
hist = yf.download(
    tickers,
    start=history_start,
    end=now + pd.Timedelta(hours=1),
    interval='60m',
    progress=False
)
prices = hist['Close'] if 'Close' in hist else hist
# Restrict to US market hours (13:30–20:00 UTC)
market_hours = prices.between_time('13:30', '20:00')

# Build business-day list and select last BACKTEST_DAYS days
business_days = sorted(market_hours.index.normalize().unique())
if len(business_days) < BACKTEST_DAYS:
    raise ValueError("Insufficient business days for backtest period")
selected_days = business_days[-BACKTEST_DAYS:]
periods = [selected_days]

# Initialize state
capital = CAPITAL
history = []
# Track open positions: ticker -> {'entry_price': float, 'shares': int, 'side': 'long' or 'short'}
positions = {}
# Define single period and load params
period = periods[0]
# Load parameters from params.csv (skip week header)
import os
if os.path.exists('params.csv'):
    params = pd.read_csv('params.csv', skiprows=1, index_col=0).to_dict(orient='index')
else:
    raise FileNotFoundError('params.csv not found; run calibration first')

# Simulate each hour in the period
capital = CAPITAL
history = []
# Track open positions: ticker -> {'entry_price': float, 'shares': int, 'side': 'long' or 'short'}
positions = {}
for day in period:
    hours = market_hours[market_hours.index.normalize() == day]
    for ts in hours.index:
        next_ts = ts + pd.Timedelta(hours=1)
        if next_ts not in market_hours.index:
            continue
        hourly_profit = 0.0
        env.capital = capital
        # iterate tickers for open/entry logic
        for t, p in params.items():
            price_t = market_hours.at[ts, t]
            price_n = market_hours.at[next_ts, t]
            if not np.isfinite(price_t) or not np.isfinite(price_n) or price_t <= 0:
                continue
            # if position open, check exit
            if t in positions:
                entry = positions[t]['entry_price']
                shares = positions[t]['shares']
                side = positions[t]['side']
                if side == 'long':
                    ret = price_n/entry - 1
                else:
                    ret = entry/price_n - 1
                if ret <= -STOP_LOSS_PCT or ret >= TAKE_PROFIT_PCT:
                    if side == 'long':
                        proceeds = shares * price_n
                    else:
                        proceeds = shares * (2*entry - price_n)
                    cost = shares * entry * TX_COST
                    profit = proceeds - shares * entry - cost
                    hourly_profit += profit
                    print(f"{t}: closing {side} at ret={ret:.2%}, profit={profit:.2f}")
                    del positions[t]
                # after exit or hold, skip to next ticker
                continue
            # no open position: decide entry
            signal = env.generate_signal(price_t, p, capital)
            f_star = signal['f_star']
            if f_star > 0:
                # long entry
                alloc = capital * f_star
                shares = int(np.floor(alloc / price_t))
                if shares > 0:
                    positions[t] = {'entry_price': price_t, 'shares': shares, 'side': 'long'}
                    print(f"{t}: opening long at {price_t:.2f}, shares={shares}")
            elif f_star < 0 and ALLOW_SHORTS:
                # short entry
                alloc = capital * abs(f_star)
                shares = int(np.floor(alloc / price_t))
                if shares > 0:
                    positions[t] = {'entry_price': price_t, 'shares': shares, 'side': 'short'}
                    print(f"{t}: opening short at {price_t:.2f}, shares={shares}")
            # else no action

        if not np.isfinite(hourly_profit):
            hourly_profit = 0.0
        capital += hourly_profit
        history.append({'timestamp': ts, 'capital': capital, 'profit': hourly_profit})
        print(f"{ts}: profit={hourly_profit:.2f}, capital={capital:.2f}")

# Summary
if history:
    df = pd.DataFrame(history).set_index('timestamp')
    print("Simulation Results:")
    print(df)
    print(f"Start={CAPITAL:.2f}, End={capital:.2f}, Return={capital/CAPITAL-1:.4f}")
else:
    print("No trades executed; check data or parameters.")
if history:
    df = pd.DataFrame(history).set_index('timestamp')
    print("\nSimulation Results:")
    print(df)
    print(f"Start={CAPITAL:.2f}, End={capital:.2f}, Return={capital/CAPITAL-1:.4f}")
else:
    print("No trades executed; check data or parameters.")
