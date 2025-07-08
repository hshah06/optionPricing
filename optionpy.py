# option_pricing_engine.py

import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S        # Spot price
        self.K = K        # Strike price
        self.T = T        # Time to maturity (in years)
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / \
               (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

# Example usage:
if __name__ == "__main__":
    model = BlackScholesModel(S=100, K=105, T=1, r=0.05, sigma=0.2)
    print(f"Call Option Price: ${model.call_price():.2f}")
    print(f"Put Option Price:  ${model.put_price():.2f}")
