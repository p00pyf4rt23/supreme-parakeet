import time
import pandas as pd
import yfinance as yf
from engine import Engine

# --- Configuration ---
PARAM_FILE = "params.csv"   # CSV with columns: ticker,mu,sigma,lambda,m,delta
TICKER_COL = "ticker"
CAPITAL = 100_000          # total capital for sizing
DT = 1/252
DOMAIN = (-0.2, 0.2)
N_COEFF = 128
DELTA_U = 0.01
DELTA_D = 0.01
ETA = 0.1
UPDATE_INTERVAL = 60       # seconds between updates

# --- Load precomputed parameters ---
def load_parameters(path: str) -> pd.DataFrame:
    """
    Read CSV of precomputed model parameters.
    Expected columns: ticker, mu, sigma, lambda, m, delta
    """
    df = pd.read_csv(path)
    df = df.set_index(TICKER_COL)
    return df

# --- Real-time runner ---
def run_realtime_loop():
    # Load parameters once at start (reload periodically if needed)
    params_df = load_parameters(PARAM_FILE)
    # Initialize signal engine
    engine = Engine(
        capital=CAPITAL,
        dt=DT,
        domain=DOMAIN,
        N=N_COEFF,
        delta_u=DELTA_U,
        delta_d=DELTA_D,
        eta=ETA
    )

    tickers = params_df.index.tolist()
    print(f"Starting real-time signal runner for {len(tickers)} symbols...")

    while True:
        start_time = time.time()
        signals = {}

        # 1) Fetch real-time prices (1-min) in batch
        # Using yfinance: get latest 'regularMarketPrice'
        data = yf.download(
            tickers,
            period="1d",
            interval="1m",
            progress=False,
            threads=True
        )
        # Data is a MultiIndex columns: ('Close', ticker)
        for ticker in tickers:
            try:
                price = data['Close'][ticker].iloc[-1]
            except Exception:
                # Fallback to Ticker.info
                info = yf.Ticker(ticker).info
                price = info.get('regularMarketPrice', None)

            if price is None or pd.isna(price):
                print(f"Warning: could not fetch price for {ticker}")
                continue

            # 2) Retrieve parameters for this ticker
            row = params_df.loc[ticker]
            params = {
                'mu': row['mu'],
                'sigma': row['sigma'],
                'lambda': row['lambda'],
                'm': row['m'],
                'delta': row['delta']
            }

            # 3) Generate signal
            sig = engine.generate_signal(price, params)
            sig['price'] = price
            signals[ticker] = sig

        # Output determinations
        timestamp = pd.Timestamp.now()
        print(f"\n== Signals @ {timestamp} ==")
        for ticker, sig in signals.items():
            side = sig['side']
            shares = sig['shares']
            if side != 'HOLD' and shares > 0:
                print(f"{side} {shares} shares of {ticker} at {sig['price']:.2f} | f*={sig['f_star']:.3f} | p_up={sig['p_up']:.3f}, p_down={sig['p_down']:.3f}")
        print(f"Cycle took {time.time()-start_time:.2f}s")

        # Wait until next interval
        elapsed = time.time() - start_time
        sleep_time = max(0, UPDATE_INTERVAL - elapsed)
        time.sleep(sleep_time)

if __name__ == '__main__':
    run_realtime_loop()
