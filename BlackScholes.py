import math
import random
import numpy as np
from scipy.stats import norm

S0 = 50
K = 50
r = 0.1
q = 0.05
sigma = 0.4
T = 0.5
type = "put"
# type = "call"

class BS:
    def __init__(self, S0, K, r, q, sigma, T, type):
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.type = type

    def option_price(self):
        Nd1 = norm.cdf((math.log(self.S0 / self.K) + (self.r - self.q + self.sigma ** 2 / 2) * self.T) / (self.sigma * (self.T ** 0.5)))
        Nd2 = norm.cdf((math.log(self.S0 / self.K) + (self.r - self.q - self.sigma ** 2 / 2) * self.T) / (self.sigma * (self.T ** 0.5)))
        if type == "call":
            ans = self.S0 * math.exp(-self.q*self.T) * Nd1 - self.K * math.exp(-self.r*self.T) * Nd2
            print("----------"*20)
            print(f'S0: {self.S0}, K: {self.K}, r: {self.r}, q: {self.q}, sigma: {self.sigma}, T: {self.T}, type; {self.type} ')
            print("BS call value: ", ans)
            print("")

            return ans
        elif type == "put":
            ans = self.K * math.exp(-self.r*self.T) * (1 - Nd2) - self.S0 * math.exp(-self.q*self.T) * (1 - Nd1)
            print("----------" * 20)
            print(f'S0: {self.S0}, K: {self.K}, r: {self.r}, q: {self.q}, sigma: {self.sigma}, T: {self.T}, type; {self.type} ')
            print("BS put value: ", ans)
            print("")
            return ans


BS(S0, K, r, q, sigma, T, type).option_price()

