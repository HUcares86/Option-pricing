import math, random
import numpy as np


class Extra_bonus_2:
    def __init__(self, S0, K, r, q, sigma, T, m, n, Smin, Smax, type_1, NA_EU):
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.m = m
        self.n = n
        self.Smin = Smin
        self.Smax = Smax
        self.Smin = Smin
        self.Smax = Smax
        self.type_1 = type_1
        self.NA_EU = NA_EU
        self.dt = T / n

    def payoff(self, s):
        if self.type_1 == "call":
            return max(s - self.K, 0)
        if self.type_1 == "put":
            return max(self.K - s, 0)


    def a_explicit(self, j):
        return 1 / (1 + self.r * self.dt) * (-0.5 * (self.r - self.q) * j * self.dt + 0.5 * self.sigma ** 2 * j ** 2 * self.dt)

    def b_explicit(self, j):
        return 1 / (1 + self.r * self.dt) * (1 - self.sigma ** 2 * j ** 2 * self.dt)

    def c_explicit(self, j):
        return 1 / (1 + self.r * self.dt) * (0.5 * (self.r - self.q) * j * self.dt + 0.5 * self.sigma ** 2 * j ** 2 * self.dt)

    def a_implicit(self, j):
        return (self.r - self.q) / 2 * j * self.dt - 0.5 * self.sigma ** 2 * j ** 2 * self.dt

    def b_implicit(self, j):
        return 1 + self.sigma ** 2 * j ** 2 * self.dt + self.r * self.dt

    def c_implicit(self, j):
        return -(self.r - self.q) / 2 * j * self.dt - 0.5 * self.sigma ** 2 * j ** 2 * self.dt

    def explicit(self):
        S_arr = np.zeros((self.m + 1, self.n + 1))
        for i in range(self.m + 1):
            Sij = (self.Smax / self.m) * i
            S_arr[self.m - i][self.n] = self.payoff(Sij)

        # 生成差分關係矩陣A
        A = np.zeros((self.m - 1, self.m + 1))

        for i in range(self.m - 1):
            A[i][i] = self.c_explicit(self.m - 1 - i)
            A[i][i + 1] = self.b_explicit(self.m - 1 - i)
            A[i][i + 2] = self.a_explicit(self.m - 1 - i)

        for i in range(self.n):
            # print(f"now in time {self.n - i}")
            fi = np.matmul(A, S_arr[:, self.n - i].T)
            for j in range(self.m - 1):
                if self.NA_EU == "NA":
                    fi[j] = max(fi[j], self.payoff(self.Smax / self.m * (self.m - 1 - j)), 0)
            S_arr[1:self.m, self.n - i - 1] = fi
            S_arr[0, self.n - i - 1] = self.payoff(self.Smax)
            S_arr[self.m, self.n - i - 1] = self.payoff(self.Smin)

        print(f"By explicit method, the option price is {S_arr[self.m // 2][0]}")
        return S_arr[self.m // 2 - 1][0]

    def implicit(self):
        S_arr = np.zeros((self.m - 1, self.n + 1))
        for i in range(1, self.m):
            Sij = (self.Smax / self.m) * i
            S_arr[self.m - 1 - i][self.n] = self.payoff(Sij)

        # print(S_arr)

        # 生成差分關係矩陣A
        A = np.zeros((self.m - 1, self.m - 1))


        A[0][0] = self.b_implicit(self.m - 1)
        A[0][1] = self.a_implicit(self.m - 1)
        A[self.m - 2][self.m - 3] = self.c_implicit(1)
        A[self.m - 2][self.m - 2] = self.b_implicit(1)
        for i in range(1, self.m - 2):
            A[i][i - 1] = self.c_implicit(self.m - 1 - i)
            A[i][i] = self.b_implicit(self.m - 1 - i)
            A[i][i + 1] = self.a_implicit(self.m - 1 - i)
        A_inv = np.linalg.inv(A)

        for i in range(self.n):
            # print(f"now in time {self.n - i}")
            B = S_arr[:, self.n - i].T
            B[0] -= self.c_implicit(self.m - 1) * self.payoff(self.Smax)
            B[self.m - 2] -= self.a_implicit(1) * self.payoff(self.Smin)
            fi = np.matmul(A_inv, S_arr[:, self.n - i].T)
            for j in range(self.m - 1):
                if self.NA_EU == "NA":
                    fi[j] = max(fi[j], self.payoff(self.Smax / self.m * (self.m - 1 - j)))
            S_arr[:, self.n - i - 1] = fi

        print(f"By implicit method, the option price is {S_arr[self.m // 2 - 1][0]}")
        return S_arr[self.m // 2 - 1][0]



# (S0, K, r, q, sigma, T, m, n, Smin, Smax, type_1, NA_EU)
I_1 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 400, 100, 0, 100, "put", "EU").implicit()
print("")
I_2 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 400, 100, 0, 100, "put", "NA").implicit()
print("")
I_3 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 400, 100, 0, 100, "call", "EU").implicit()
print("")
I_4 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 400, 100, 0, 100, "call", "NA").implicit()
print("")
print("------------------------------------------------------------")
E_1 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 100, 1000, 0, 100, "put", "EU").explicit()
print("")
E_2 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 100, 1000, 0, 100, "put", "NA").explicit()
print("")
E_3 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 100, 1000, 0, 100, "call", "EU").explicit()
print("")
E_4 = Extra_bonus_2(50, 50, 0.05, 0.01, 0.4, 0.5, 100, 1000, 0, 100, "call", "NA").explicit()
print("")




