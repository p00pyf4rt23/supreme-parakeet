import random
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from calibrator import load_tickers, fetch_prices, log_returns, calibrate_mjd_mle, method_of_moments
from engine import lEngine

# --- Backtest Settings ---
LOOKBACK_DAYS = 252         # days for calibration window
WEEKS_BACK = 12            # look for a week within last 12 weeks
CAPITAL = 100_000          # starting capital
DELTA_U = 0.01
DELTA_D = 0.01
ETA = 0.1
COS_DOMAIN = (-0.2, 0.2)
COS_N = 128
DT = 1/252
N_MAX = 20

# Load allowed tickers
tickers = load_tickers("allowed_stocks.txt")

def select_random_week(end_date: pd.Timestamp):
    # generate list of business days up to end_date
    start_limit = end_date - pd.DateOffset(weeks=WEEKS_BACK)
    all_bd = pd.bdate_range(start_limit, end_date)
    # choose start of week ensuring 5 bd days exist
    possible_starts = all_bd[:-4]
    start = random.choice(possible_starts)
    week = pd.bdate_range(start, periods=5)
    return week

# Fetch full history to cover lookback + week + next-day prices
today = pd.Timestamp.today().normalize()
history_start = today - pd.DateOffset(days=LOOKBACK_DAYS + WEEKS_BACK*7)
hist_data = yf.download(tickers, start=history_start, end=today + pd.Timedelta(days=1), interval="1d", progress=False)
prices = pd.DataFrame({t: hist_data['Close'][t] for t in tickers})

# Select a random week
week = select_random_week(prices.index[-1])
print(f"Backtesting week: {week[0].date()} to {week[-1].date()}")

# Initialize engine
engine = MJDKellySignalEngine(
    capital=CAPITAL,
    dt=DT,
    domain=COS_DOMAIN,
    N=COS_N,
    delta_u=DELTA_U,
    delta_d=DELTA_D,
    eta=ETA
)

# DataFrame to store daily returns
results = []

for day in week:
    # Skip if market closed
    if day not in prices.index:
        continue
    # Calibration window up to previous day
    calib_end = day - pd.Timedelta(days=1)
    hist_window = prices.loc[:calib_end].dropna()
    # Ensure window length
    if len(hist_window) < LOOKBACK_DAYS:
        print(f"Insufficient history for {day.date()}, skipping")
        continue
    # For each ticker, calibrate parameters
    params_map = {}
    for t in tickers:
        series = hist_window[t].dropna()
        r = np.log(series / series.shift(1)).dropna().values
        if len(r) < MIN_RETURNS:
            continue
        try:
            mu, sigma, lam, m, delta = calibrate_mjd_mle(r, DT, N_MAX)
        except:
            mu, sigma, lam, m, delta = method_of_moments(r, DT)
        params_map[t] = {'mu':mu,'sigma':sigma,'lambda':lam,'m':m,'delta':delta}

    # Generate signals and compute next-day returns
    day_close = prices.loc[day]
    next_day = day + pd.Timedelta(days=1)
    if next_day not in prices.index:
        print(f"No next-day price for {next_day.date()}, skipping P&L")
        break
    next_close = prices.loc[next_day]

    # Portfolio return = sum(f* * (P_next/P_today - 1))
    daily_ret = 0.0
    for t, pr in params_map.items():
        price_t = day_close[t]
        signal = engine.generate_signal(price_t, pr)
        f_star = signal['f_star']
        # return of asset
        R = (next_close[t] / price_t - 1)
        daily_ret += f_star * R

    results.append({'date': day.date(), 'return': daily_ret})

# Present results
df_res = pd.DataFrame(results).set_index('date')
print("Daily returns for backtest week:")
print(df_res)

# Compute summary
print(f"Average daily return: {df_res['return'].mean():.4f}")
print(f"Cumulative return: {(1+df_res['return']).prod()-1:.4f}")
