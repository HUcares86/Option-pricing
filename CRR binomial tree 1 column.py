import math
import random
import numpy as np
from scipy.stats import norm
import copy
import random
import numpy as np
from scipy.stats import norm
# S0 = float(input("initial stock price:"))
# K = float(input("K:"))
# r = float(input("interest rate:"))
# q = float(input("dividend rate:"))
# sigma = float(input("sigma:"))
# T = float(input("T:"))
# type = input()
# print(" ")
S0, K, r, q, sigma, T = 50, 50, 0.1, 0.05, 0.4, 0.5


class CRR_1_column:
    def __init__(self, S0, K, r, q, sigma, T, type_call_put, type_EU_NA, n):
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.type_call_put = type_call_put
        self.type_EU_NA = type_EU_NA
        self.n = n
        self.dt = T / n
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = math.exp(-sigma * math.sqrt(self.dt))
        self.p = (math.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    def payoff(self, S1, type):
        if type == "call":
            return max(S1 - K, 0)
        elif type == "put":
            return max(K - S1, 0)

    def option_price(self):
        option_payoff = []
        for i in range(self.n + 1):
            the_payoff = self.payoff(self.S0 * (self.u ** (self.n - i)) * (self.d ** i), self.type_call_put)
            option_payoff.append(the_payoff)
        for i in range(self.n):
            # print(f"now in branch {n-i}")
            for j in range(self.n - i):
                option_payoff[j] = math.exp(-r * self.dt) * (self.p * option_payoff[j] + (1 - self.p) * option_payoff[j + 1])
                if self.type_EU_NA == "NA":
                    option_payoff[j] = max(option_payoff[j],
                                           self.payoff(self.S0 * (self.u ** (self.n - 1 - i - j)) * (self.d ** j), self.type_call_put))

        # print("----------" * 20)
        print(f'S0: {self.S0}, K: {self.K}, r: {self.r}, q: {self.q}, sigma: {self.sigma}, T: {self.T}, type_call_put; {self.type_call_put}, type_EU_NA; {self.type_EU_NA}, n: {self.n}')
        print(f"CRR {self.type_EU_NA} {self.type_call_put} option price is: {option_payoff[0]}")
        print("")



CRR_1_column(S0, K, r, q, sigma, T, "call", "EU", 100).option_price()
