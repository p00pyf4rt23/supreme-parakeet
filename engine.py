import numpy as np

class Engine:
    """
    Merton Jump-Diffusion signal engine using the COS method for density
    inversion and Kelly criterion for position sizing.
    """
    def __init__(self,
                 capital,
                 dt=1/252,
                 domain=(-0.2, 0.2),
                 N=128,
                 delta_u=0.01,
                 delta_d=0.01,
                 eta=0.1):
        # Trading parameters
        self.capital = capital
        self.dt = dt
        self.a, self.b = domain
        self.N = N
        self.delta_u = delta_u
        self.delta_d = delta_d
        self.eta = eta
        # Precompute COS u_k
        self.u_k = np.arange(N) * np.pi / (self.b - self.a)

    def characteristic_function(self, u, params):
        """
        Characteristic function phi_X(u) of log-return X over dt under
        physical measure using Merton jump-diffusion parameters.
        params: dict with keys mu, sigma, lam, m, delta
        """
        mu = params['mu']
        sigma = params['sigma']
        lam = params['lambda']
        m = params['m']
        delta = params['delta']
        k = np.exp(m + 0.5 * delta**2) - 1
        # jump component characteristic
        phi_Y = np.exp(1j * u * m - 0.5 * delta**2 * u**2)
        exponent = (
            1j * u * (mu - 0.5 * sigma**2 - lam * k) * self.dt
            - 0.5 * sigma**2 * u**2 * self.dt
            + lam * self.dt * (phi_Y - 1)
        )
        return np.exp(exponent)

    def cos_coefficients(self, params):
        """
        Compute COS series coefficients A_k for density approximation.
        """
        phi_vals = self.characteristic_function(self.u_k, params)
        # A_0 separately
        A = np.empty(self.N)
        A[0] = (1.0 / (self.b - self.a)) * np.real(phi_vals[0] * np.exp(-1j * self.u_k[0] * self.a))
        # A_k for k>=1
        factor = 2.0 / (self.b - self.a)
        for k in range(1, self.N):
            A[k] = factor * np.real(phi_vals[k] * np.exp(-1j * self.u_k[k] * self.a))
        return A

    def tail_probabilities(self, A):
        """
        Compute p_up and p_down from COS coefficients A_k.
        """
        a, b = self.a, self.b
        N = self.N
        delta_u, delta_d = self.delta_u, self.delta_d
        x_up = np.log(1 + delta_u)
        x_down = np.log(1 - delta_d)
        # p_up
        p_up = A[0] * (b - x_up)
        # p_down
        p_down = A[0] * (x_down - a)
        # add k>=1 terms
        for k in range(1, N):
            coeff = A[k] * (b - a) / (k * np.pi)
            p_up   += -coeff * np.sin(k * np.pi * (x_up - a) / (b - a))
            p_down +=  coeff * np.sin(k * np.pi * (x_down - a) / (b - a))
        return float(p_up), float(p_down)

    def kelly_fraction(self, p_up, p_down):
        """
        Compute Kelly-optimal fraction f* given tail probabilities and thresholds.
        """
        R_plus = self.delta_u
        R_minus = self.delta_d
        num = p_up * R_plus - p_down * R_minus
        den = R_plus * R_minus
        f = num / den
        # Clamp f to [-1, 1] for safety
        return float(np.clip(f, -1, 1))

    def generate_signal(self, price, params):
        """
        Given current price and model params, compute order.
        Returns a dict: {{ 'f_star', 'p_up', 'p_down', 'shares', 'side' }}
        """
        # 1) compute COS coefficients
        A = self.cos_coefficients(params)
        # 2) get tail probabilities
        p_up, p_down = self.tail_probabilities(A)
        # 3) decide direction
        diff = p_up - p_down
        if abs(diff) < self.eta:
            return {'side': 'HOLD', 'shares': 0, 'f_star': 0.0,
                    'p_up': p_up, 'p_down': p_down}
        # 4) Kelly fraction
        f_star = self.kelly_fraction(p_up, p_down)
        side = 'BUY' if f_star > 0 else 'SELL'
        # 5) compute share count
        alloc = abs(f_star) * self.capital
        shares = int(np.floor(alloc / price))
        return {'side': side,
                'shares': shares,
                'f_star': f_star,
                'p_up': p_up,
                'p_down': p_down}
