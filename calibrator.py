import os
import time
import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- Hand-Tunable Settings ---
ALLOWED_TICKERS_FILE = "allowed_stocks.txt"
PARAM_FILE = "params.csv"
HIST_DAYS = 252
DT = 1/252
N_MAX = 20
MIN_RETURNS = 30

SMOOTHING_ALPHA = 0.3
CHANGE_THRESHOLD = 0.1

# --- Helpers ---
def load_tickers(path: str) -> list:
    with open(path) as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    print(f"[DEBUG] Loaded {len(tickers)} tickers: {tickers}")
    return tickers

def fetch_prices(tickers: list, days: int) -> pd.DataFrame:
    data = yf.download(tickers, period=f"{days}d", interval="1d", progress=False)
    if isinstance(data, pd.DataFrame) and 'Close' in data:
        prices = data['Close']
    else:
        prices = pd.DataFrame({tickers[0]: data['Close']})
    print(f"[DEBUG] Fetched price data with shape {prices.shape}")
    return prices.dropna(how='all')

def log_returns(series: pd.Series) -> np.ndarray:
    return np.log(series / series.shift(1)).dropna().values

def merton_nll(params, r, dt, n_max):
    mu, sigma, lamb, m, delta = params
    if sigma <= 0 or delta <= 0 or lamb < 0:
        return np.inf
    k = math.exp(m + 0.5*delta**2) - 1
    lamdt = lamb * dt
    total = 0.0
    for x in r:
        mix = 0.0
        for n in range(n_max+1):
            w = math.exp(-lamdt) * lamdt**n / math.factorial(n)
            a_n = (mu - 0.5*sigma**2 - lamb*k)*dt + n*m
            s2 = sigma**2*dt + n*delta**2
            mix += w * (1.0/math.sqrt(2*math.pi*s2)) * math.exp(-0.5*(x - a_n)**2/s2)
        total += math.log(mix + 1e-12)
    return -total

def calibrate_mjd_mle(r, dt, n_max):
    mu0 = np.mean(r)/dt
    sigma0 = np.std(r)/np.sqrt(dt)
    lam0 = min(0.5, np.mean(np.abs(r))*dt)
    m0 = 0.0
    delta0 = sigma0/2
    x0 = [mu0, sigma0, lam0, m0, delta0]
    bounds = [(-1,1), (1e-6,2), (1e-6,50), (-1,1), (1e-6,1)]
    res = minimize(merton_nll, x0, args=(r,dt,n_max), bounds=bounds)
    return tuple(res.x) if res.success else method_of_moments(r,dt)

def method_of_moments(r, dt):
    mean_r = np.mean(r)
    var_r = np.var(r)
    sigma_s = np.sqrt(var_r)
    thr = 3*sigma_s
    jumps = np.sum(np.abs(r) > thr)
    lam = jumps / len(r) / dt if len(r)>0 else 1e-6
    delta2 = np.var(r[np.abs(r)>thr]) if jumps>0 else sigma_s**2/4
    sigma = np.sqrt(max(var_r - lam*delta2, 1e-6)/dt)
    m = np.mean(r[np.abs(r)>thr]) if jumps>0 else 0.0
    mu = (mean_r + lam*(math.exp(m+0.5*delta2)-1)*dt)/dt
    return mu, sigma, lam, m, math.sqrt(delta2)

# New helper to find Monday
def get_week_start():
    today = datetime.today()
    return today - timedelta(days=today.weekday())

def run_calibration():
    tickers = load_tickers(ALLOWED_TICKERS_FILE)
    prices = fetch_prices(tickers, HIST_DAYS)

    week_start = get_week_start().strftime("%m-%d-%Y")

    # Prepare file and write header
    with open(PARAM_FILE, 'w') as f:
        f.write(f"Week of {week_start}\n")
        header = "ticker,mu,sigma,lambda,m,delta\n"
        f.write(header)

    start = time.time()
    for t in tickers:
        mu, sigma, lam, m, delta = 0.0, 1e-6, 1e-6, 0.0, 1e-6

        if t in prices.columns:
            r = log_returns(prices[t])
            if len(r) >= MIN_RETURNS:
                try:
                    new = calibrate_mjd_mle(r, DT, N_MAX)
                except Exception as e:
                    print(f"[DEBUG] MLE failed for {t}: {e}")
                    new = method_of_moments(r, DT)

                def blend(nv, ov):
                    if ov and abs(nv-ov)/abs(ov) < CHANGE_THRESHOLD:
                        return ov
                    return SMOOTHING_ALPHA*nv + (1-SMOOTHING_ALPHA)*ov

                mu, sigma, lam, m, delta = map(blend, new, [mu, sigma, lam, m, delta])

        line = f"{t},{mu},{sigma},{lam},{m},{delta}\n"
        with open(PARAM_FILE, 'a') as f:
            f.write(line)
        print(f"[DEBUG] Wrote {t}: mu={mu:.4f}, sigma={sigma:.4f}, lambda={lam:.4f}, m={m:.4f}, delta={delta:.4f}")

    print(f"Calibration done in {time.time()-start:.1f}s")

if __name__ == "__main__":
    run_calibration()
