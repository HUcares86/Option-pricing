import math
import copy
import random
import numpy as np
from scipy.stats import norm


S0 = 100
K = 100
r = 0.05
q = 0.01
sigma = 0.3
T = 1
type_call_put = "put"
type_EU_NA = "EU"
# "NA", "EU"
n = 100

def exp(x):
    return math.exp(x)
def sqrt(x):
    return math.sqrt(x)

dt = T / n
u = exp(sigma * sqrt(dt))
d = exp(-sigma * sqrt(dt))
p = (exp((r - q) * dt) - d) / (u - d)


def payoff(S, type_call_put):
    if type_call_put == "call":
        return max(S - K, 0)
    elif type_call_put == "put":
        return max(K - S, 0)

option_payoff = []
for i in range(n+1):
    the_payoff = payoff(S0*(u**(n-i))*(d**i), type_call_put)
    option_payoff.append(the_payoff)
for i in range(n):
    # print(f"now in branch {n-i}")
    for j in range(n-i):
        option_payoff[j] = exp(-r*dt) * ( p * option_payoff[j] + (1-p) * option_payoff[j+1] )
        if type_EU_NA == "NA":
            option_payoff[j] = max(option_payoff[j], payoff(S0*(u**(n-1-i-j))*(d**j), type_call_put))

print(f"The option price is: {option_payoff[0]}")
