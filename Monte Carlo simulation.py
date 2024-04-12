import math
import random
import numpy as np
from scipy.stats import norm

# S0 = float(input())
# K = float(input())
# r = float(input())
# q = float(input())
# sigma = float(input())
# T = float(input())
# type = input()
# simul = int(input())
# repet = int(input())
# print("")


S0 = 50
K = 50
r = 0.1
q = 0.05
sigma = 0.4
T = 0.5
type = "put"
# type = "call"
simul = 10000
repet = 20


def payoff(S, type):
    if type == "call":
        if S - K > 0:
            payoff = S - K
        else:
            payoff = 0
    if type == "put":
        if S - K > 0:
            payoff = 0
        else:
            payoff = K - S
    return payoff

repet_price_list = []
for rep in range(repet):
    simul_payoff_list = []
    for si in range(simul):
        Z = random.gauss(0, 1)
        S = S0 * math.exp((r-q-sigma**2/2) * T + sigma * (T**0.5) * Z)
        simul_payoff_list.append(payoff(S, type))
    repet_price_list.append(math.exp(-r*T) * np.mean(simul_payoff_list))
mean = np.mean(repet_price_list)
print(mean)
repet_price_arr = np.array(repet_price_list)
std = np.std(repet_price_arr)
print(f"[{mean-2*std},{mean+2*std}]")