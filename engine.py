import numpy as np

class Engine:
    """
    Merton Jump-Diffusion signal engine using the COS method for density
    inversion and Kelly criterion for position sizing.

    Now supports a risk multiplier, max exposure cap, and dynamic capital inputs.
    """
    def __init__(self,
                 capital,
                 dt=1/252,
                 domain=(-0.2, 0.2),
                 N=128,
                 delta_u=0.01,
                 delta_d=0.01,
                 eta=0.1,
                 risk_multiplier=1.0,
                 max_exposure=1.0):
        # Trading parameters
        self.capital = capital  # default capital for position sizing
        self.dt = dt
        self.a, self.b = domain
        self.N = N
        self.delta_u = delta_u
        self.delta_d = delta_d
        self.eta = eta
        # Scale Kelly fraction and cap exposure
        self.risk_multiplier = risk_multiplier
        self.max_exposure = max_exposure
        # Precompute COS frequencies
        self.u_k = np.arange(N) * np.pi / (self.b - self.a)

    def characteristic_function(self, u, params):
        mu = params['mu']
        sigma = params['sigma']
        lam = params['lambda']
        m = params['m']
        delta = params['delta']
        k = np.exp(m + 0.5 * delta**2) - 1
        phi_Y = np.exp(1j * u * m - 0.5 * delta**2 * u**2)
        exponent = (
            1j * u * (mu - 0.5 * sigma**2 - lam * k) * self.dt
            - 0.5 * sigma**2 * u**2 * self.dt
            + lam * self.dt * (phi_Y - 1)
        )
        return np.exp(exponent)

    def cos_coefficients(self, params):
        phi_vals = self.characteristic_function(self.u_k, params)
        A = np.empty(self.N, dtype=float)
        A[0] = (1.0 / (self.b - self.a)) * np.real(phi_vals[0] * np.exp(-1j * self.u_k[0] * self.a))
        factor = 2.0 / (self.b - self.a)
        for k in range(1, self.N):
            A[k] = factor * np.real(phi_vals[k] * np.exp(-1j * self.u_k[k] * self.a))
        return A

    def tail_probabilities(self, A):
        a, b, N = self.a, self.b, self.N
        x_up = np.log(1 + self.delta_u)
        x_down = np.log(1 - self.delta_d)
        p_up = A[0] * (b - x_up)
        p_down = A[0] * (x_down - a)
        for k in range(1, N):
            coeff = A[k] * (b - a) / (k * np.pi)
            p_up   += -coeff * np.sin(k * np.pi * (x_up - a) / (b - a))
            p_down +=  coeff * np.sin(k * np.pi * (x_down - a) / (b - a))
        return float(p_up), float(p_down)

    def kelly_fraction(self, p_up, p_down):
        # basic Kelly
        R_plus = self.delta_u
        R_minus = self.delta_d
        num = p_up * R_plus - p_down * R_minus
        den = R_plus * R_minus
        f = num / den
        # apply risk multiplier and cap exposure
        f *= self.risk_multiplier
        return float(np.clip(f, -self.max_exposure, self.max_exposure))

    def generate_signal(self, price, params, capital=None):
        # allow dynamic capital input
        cap = capital if capital is not None else self.capital
        A = self.cos_coefficients(params)
        p_up, p_down = self.tail_probabilities(A)
        diff = p_up - p_down
        # threshold check
        if abs(diff) < self.eta:
            return {'side': 'HOLD', 'shares': 0, 'f_star': 0.0,
                    'p_up': p_up, 'p_down': p_down}
        # scaled Kelly fraction
        f_star = self.kelly_fraction(p_up, p_down)
        side = 'BUY' if f_star > 0 else 'SELL'
        # compute share count based on provided capital
        alloc = abs(f_star) * cap
        shares = int(np.floor(alloc / price))
        return {'side': side,
                'shares': shares,
                'f_star': f_star,
                'p_up': p_up,
                'p_down': p_down}
