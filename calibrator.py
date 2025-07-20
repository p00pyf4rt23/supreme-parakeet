import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import time

# --- Hand-Tunable Settings ---
ALLOWED_TICKERS_FILE = "allowed_stocks.txt"  # file with one ticker per line to track and calibrate
PARAM_FILE = "params.csv"     # CSV to read/write parameters
HIST_DAYS = 252                # lookback window for returns (days)
DT = 1/252                     # one trading day in years
N_MAX = 20                     # truncate Poisson sum at n_max
MIN_RETURNS = 30               # minimum # of returns required for calibration

# Overfitting Prevention Settings
SMOOTHING_ALPHA = 0.3          # smoothing factor for parameter updates (0 = keep old, 1 = full new)
CHANGE_THRESHOLD = 0.1         # relative change threshold below which old params are retained

# --- Helper Functions ---
def load_tickers(path: str) -> list:
    """Load ticker symbols from a text file (one per line)."""
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def fetch_prices(tickers: list, days: int) -> pd.DataFrame:
    """
    Download historical daily close prices for given tickers.
    Returns DataFrame indexed by date, columns=tickers.
    """
    data = yf.download(
        tickers,
        period=f"{days}d",
        interval="1d",
        progress=False,
        group_by='ticker'
    )
    prices = pd.DataFrame({
        t: data[t]['Close'] if t in data else pd.Series(dtype=float)
        for t in tickers
    })
    return prices.dropna(how='all')


def log_returns(prices: pd.Series) -> np.ndarray:
    """Compute daily log returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna().values


def merton_nll(params, r, dt, n_max):
    """
    Negative log-likelihood for Merton jump-diffusion on returns r.
    params = [mu, sigma, lamb, m, delta]
    """
    mu, sigma, lamb, m, delta = params
    if sigma <= 0 or delta <= 0 or lamb < 0:
        return np.inf
    k = np.exp(m + 0.5 * delta**2) - 1
    lamdt = lamb * dt
    total = 0.0
    for x in r:
        mix = 0.0
        for n in range(n_max + 1):
            w = np.exp(-lamdt) * lamdt**n / np.math.factorial(n)
            a_n = (mu - 0.5*sigma**2 - lamb*k)*dt + n*m
            s_n2 = sigma**2*dt + n*delta**2
            mix += w * (1.0/np.sqrt(2*np.pi*s_n2)) * np.exp(-0.5*(x - a_n)**2/s_n2)
        total += np.log(mix + 1e-12)
    return -total


def calibrate_mjd_mle(r: np.ndarray, dt: float, n_max: int) -> tuple:
    """
    Calibrate Merton parameters via MLE on returns r.
    Returns (mu, sigma, lambda, m, delta).
    """
    mu0 = np.mean(r)/dt
    sigma0 = np.std(r)/np.sqrt(dt)
    lam0 = min(0.5, np.mean(np.abs(r)) * dt)
    m0 = 0.0
    delta0 = sigma0/2
    x0 = [mu0, sigma0, lam0, m0, delta0]
    bounds = [(-1, 1), (1e-6, 2), (1e-6, 5), (-1, 1), (1e-6, 1)]
    res = minimize(merton_nll, x0, args=(r, dt, n_max), bounds=bounds)
    if res.success:
        return tuple(res.x)
    else:
        return method_of_moments(r, dt)


def method_of_moments(r: np.ndarray, dt: float) -> tuple:
    """
    Approximate MJD parameters via method-of-moments and outlier counts.
    Returns (mu, sigma, lambda, m, delta).
    """
    mean_r = np.mean(r)
    var_r = np.var(r)
    sigma_s = np.sqrt(var_r)
    thr = 3*sigma_s
    jumps = np.sum(np.abs(r) > thr)
    lam = jumps / len(r) / dt
    delta2 = np.var(r[np.abs(r) > thr]) if jumps>0 else sigma_s**2/4
    sigma = np.sqrt(max(var_r - lam*delta2, 1e-6)/dt)
    m = np.mean(r[np.abs(r) > thr]) if jumps>0 else 0.0
    mu = (mean_r + lam*(np.exp(m+0.5*delta2)-1)*dt)/dt
    return mu, sigma, lam, m, np.sqrt(delta2)

# --- Main Calibration Loop ---
def run_calibration():
    tickers = load_tickers(TICKER_FILE)
    prices = fetch_prices(tickers, HIST_DAYS)

    # Load previous parameters for smoothing
    if os.path.exists(PARAM_FILE):
        old_df = pd.read_csv(PARAM_FILE).set_index('ticker')
    else:
        old_df = pd.DataFrame()

    records = []
    start = time.time()
    for t in tickers:
        if t not in prices.columns:
            print(f"No data for {t}, skipping")
            continue
        r = log_returns(prices[t])
        if len(r) < MIN_RETURNS:
            print(f"Insufficient data for {t}, skipping")
            continue
        try:
            new_mu, new_sigma, new_lam, new_m, new_delta = calibrate_mjd_mle(r, DT, N_MAX)
        except Exception as e:
            print(f"Calibration failed for {t}: {e}")
            new_mu, new_sigma, new_lam, new_m, new_delta = method_of_moments(r, DT)

        # Overfitting prevention via smoothing + threshold
        if t in old_df.index:
            old = old_df.loc[t]
            def smooth(param_new, param_old):
                if param_old != 0 and abs(param_new - param_old)/abs(param_old) < CHANGE_THRESHOLD:
                    return param_old
                return SMOOTHING_ALPHA * param_new + (1 - SMOOTHING_ALPHA) * param_old
            mu    = smooth(new_mu,    old['mu'])
            sigma = smooth(new_sigma, old['sigma'])
            lam   = smooth(new_lam,   old['lambda'])
            m     = smooth(new_m,     old['m'])
            delta = smooth(new_delta, old['delta'])
        else:
            mu, sigma, lam, m, delta = new_mu, new_sigma, new_lam, new_m, new_delta

        records.append({
            'ticker': t,
            'mu': mu,
            'sigma': sigma,
            'lambda': lam,
            'm': m,
            'delta': delta
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(PARAM_FILE, index=False)
    print(f"Calibration completed in {time.time()-start:.2f}s")

if __name__ == '__main__':
    run_calibration()
