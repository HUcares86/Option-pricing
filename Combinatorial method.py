import math
import copy
import random
import numpy as np
from scipy.stats import norm

# S0 = float(input())
# K = float(input())
# r = float(input())
# q = float(input())
# sigma = float(input())
# T = float(input())
# type_EU_NA = input()
# type_call_put = input()
# n = int(input())

S0 = 100
K = 100
r = 0.05
q = 0.01
sigma = 0.3
T = 1
type_EU_NA = "EU"
type_call_put = "put"
n = 100

def fac(x):
    return math.factorial(x)
def exp(x):
    return math.exp(x)
def ln(x):
    return math.log(x)
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

def ln_one_to(x):
    sum = 0
    for i in range(1,x+1):
        sum += math.log(i)
    return sum

option_value = 0
for j in range(n+1):
    AAA = ln_one_to(n) - ln_one_to(n-j) - ln_one_to(j) + (n-j) * ln(p) + j * ln(1-p)
    option_value += exp(-r*T) * exp(AAA) * payoff(S0*(u**(n-j))*(d**j), type_call_put)
print(option_value)