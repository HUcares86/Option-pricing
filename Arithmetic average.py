import math, random
import numpy as np


class MC:
    def __init__(self, S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, rep, sim):
        self.S_t = S_t
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.T_minus_t = T_minus_t
        self.M = M
        self.n = n
        self.S_ave_t = S_ave_t
        self.rep = rep
        self.sim = sim

    def option_price(self):
        T = self.t + self.T_minus_t
        dt = (T - self.t) / self.n
        z = self.n * (self.t / self.T_minus_t)
        rep_list = []
        for i in range(self.rep):
            # print("now in repetition " + str(i + 1))
            sim_list = []
            for j in range(self.sim):
                S_now = self.S_t
                sum_S = 0
                for k in range(self.n):
                    rn = random.gauss(0, 1)
                    S_now = S_now * math.exp((self.r - self.q - self.sigma ** 2 / 2) * dt + self.sigma * np.sqrt(dt) * rn)
                    sum_S += S_now
                ave_S = ((z + 1) * self.S_ave_t + sum_S) / (z + self.n + 1)
                payoff = max(ave_S - self.K, 0)
                sim_list.append(payoff)
            rep_list.append(np.mean(sim_list) * math.exp(-self.r * (T - self.t)))

        std = np.std(rep_list)
        mean = np.mean(rep_list)
        print(f"The option price by Monti Carlo is {mean},  95% CI is [{mean - 2 * std},{mean + 2 * std}]")
        return mean


class binomial:
    def __init__(self, S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, NA_EU):
        self.S_t = S_t
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.T_minus_t = T_minus_t
        self.M = M
        self.n = n
        self.S_ave_t = S_ave_t
        self.NA_EU = NA_EU


    def option_price(self):

        T = self.t + self.T_minus_t
        dt = self.T_minus_t / self.n
        u = math.exp(self.sigma * np.sqrt(dt))
        d = math.exp(-self.sigma * np.sqrt(dt))
        p = (math.exp((self.r - self.q) * dt) - d) / (u - d)
        q = 1 - p
        z = self.n * (self.t / self.T_minus_t)

        # 生成 Average price tree
        Ave_Call_arr = np.zeros((self.n + 1, self.n + 1, self.M + 1, 2))
        Ave_Call_arr[0][0][0][0] = self.S_ave_t
        for i in range(1, self.n + 1):
            for j in range(i + 1):
                Amax = (self.S_ave_t * (z + 1) +self.S_t * u * (1 - u ** (i - j)) / (1 - u) + self.S_t * u ** (i - j) * d * (
                            1 - d ** j) / (1 - d)) / (z + i + 1)
                Amin = (self.S_ave_t * (z + 1) + self.S_t * d * (1 - d ** j) / (1 - d) + self.S_t * d ** j * u * (1 - u ** (i - j)) / (
                            1 - u)) / (z + i + 1)
                for k in range(self.M + 1):
                    Aijk = (self.M - k) / self.M * Amax + k / self.M * Amin
                    Ave_Call_arr[i][j][k][0] = Aijk

        # 生成 Call payoff tree
        for j in range(self.n + 1):
            for k in range(self.M + 1):
                Ave_Call_arr[self.n][j][k][1] = max(Ave_Call_arr[self.n][j][k][0] - self.K, 0)
        for ii in range(self.n):
            i = self.n - 1 - ii
            for j in range(i + 1):
                index_u = 1
                index_d = 1
                for k in range(self.M + 1):
                    Au = ((z + i + 1) * Ave_Call_arr[i][j][k][0] + self.S_t * u ** (i + 1 - j) * d ** j) / (z + i + 2)
                    if j == 0:
                        Cu = Ave_Call_arr[i + 1][j][k][1]
                    else:
                        if Au > Ave_Call_arr[i + 1][j][0][0]:
                            Cu = Ave_Call_arr[i + 1][j][0][1]

                        elif Au < Ave_Call_arr[i + 1][j][-1][0]:
                            Cu = Ave_Call_arr[i + 1][j][-1][1]
                        else:
                            for lu in range(index_u, self.M + 1):
                                if Ave_Call_arr[i + 1][j][lu][0] <= Au <= Ave_Call_arr[i + 1][j][lu - 1][0]:
                                    Wu = (Ave_Call_arr[i + 1][j][lu - 1][0] - Au) / (
                                                Ave_Call_arr[i + 1][j][lu - 1][0] - Ave_Call_arr[i + 1][j][lu][0])
                                    Cu = Wu * Ave_Call_arr[i + 1][j][lu][1] + (1 - Wu) * Ave_Call_arr[i + 1][j][lu - 1][
                                        1]
                                    index_u = lu
                                    break
                    Ad = ((z + i + 1) * Ave_Call_arr[i][j][k][0] + self.S_t * u ** (i - j) * d ** (j + 1)) / (z + i + 2)
                    if j == i:
                        Cd = Ave_Call_arr[i + 1][j + 1][k][1]
                    else:
                        if Ad > Ave_Call_arr[i + 1][j + 1][0][0]:
                            Cd = Ave_Call_arr[i + 1][j + 1][0][1]
                        elif Ad < Ave_Call_arr[i + 1][j + 1][-1][0]:
                            Cd = Ave_Call_arr[i + 1][j + 1][-1][1]
                        else:
                            for ld in range(index_d, self.M + 1):
                                if Ave_Call_arr[i + 1][j + 1][ld][0] <= Ad <= Ave_Call_arr[i + 1][j + 1][ld - 1][0]:
                                    Wd = (Ave_Call_arr[i + 1][j + 1][ld - 1][0] - Ad) / (
                                                Ave_Call_arr[i + 1][j + 1][ld - 1][0] - Ave_Call_arr[i + 1][j + 1][ld][
                                            0])
                                    Cd = Wd * Ave_Call_arr[i + 1][j + 1][ld][1] + (1 - Wd) * \
                                         Ave_Call_arr[i + 1][j + 1][ld - 1][1]
                                    index_d = ld
                                    break
                    Cijk = (p * Cu + q * Cd) * math.exp(-self.r * dt)

                    if self.NA_EU == "NA":
                        Cijk = max(Ave_Call_arr[i][j][k][0] -self. K, Cijk)
                    Ave_Call_arr[i][j][k][1] = Cijk
        print(f"The option price by binomial is {Ave_Call_arr[0][0][0][1]}")
        return Ave_Call_arr[0][0][0][1]

print("--------------------------------------------------------------------")
print("")
# MC(S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, NA_EU).option_price()
b_1 = binomial(50, 50, 0.1, 0.05, 0.8, 0, 0.25, 100, 100, 50, "EU").option_price()
print("")
b_2 = binomial(50, 50, 0.1, 0.05, 0.8, 0, 0.25, 100, 100, 50, "NA").option_price()
print("")
b_3 = binomial(50, 50, 0.1, 0.05, 0.8, 0.25, 0.25, 100, 100, 50, "EU").option_price()
print("")
b_4 = binomial(50, 50, 0.1, 0.05, 0.8, 0.25, 0.25, 100, 100, 50, "NA").option_price()


print("--------------------------------------------------------------------")
print("")
# MC(S_t, K, r, q, sigma, t, T_minus_t, M, n, S_ave_t, rep, sim).option_price()
mc_1 = MC(50, 50, 0.1, 0.05, 0.8, 0, 0.25, 100, 100, 50, 20, 10000).option_price()
print("")
mc_2 = MC(50, 50, 0.1, 0.05, 0.8, 0.25, 0.25, 100, 100, 50, 20, 10000).option_price()
print("")
