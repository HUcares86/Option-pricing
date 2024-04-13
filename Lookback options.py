import math
import numpy as np
import random


class lookback_MC:
    def __init__(self,S_t, r, q, T, t, sigma, Smax_t, sim, rep, n):
        # self.S_t = S_t
        # self.r = r
        # self.q = q
        # self.T = T
        # self.t = t
        # self.sigma = sigma
        # self.Smax_t = Smax_t
        # self.sim = sim
        # self.rep = rep
        # self.n = n
        dt = (T - t) / n
        dis_rep_list = []
        rep_list = []

        for i in range(rep):
            # print(f"Now in repetition {i+1}")
            sim_list = []
            for j in range(sim):
                maxi = Smax_t
                S_now = S_t
                for k in range(n):
                    z = random.gauss(0, 1)
                    S_now = S_now * math.exp((r-q-sigma**2/2)*dt + sigma * math.sqrt(dt) * z)
                    if S_now > maxi:
                        maxi = S_now
                payoff = maxi - S_now
                sim_list.append(payoff)
            rep_list.append(np.mean(sim_list))
            dis_rep_list.append(np.mean(sim_list) * math.exp(-r*(T-t)))
        rep_arr = np.array(dis_rep_list)
        std = np.std(rep_arr)
        mean = np.mean(rep_arr)

        print(f"The option price S_max = {Smax_t}, n = {n}, by Monti Carlo is {mean}, 95% CI is [{mean-2*std},{mean+2*std}]")
        print("")


class lookback_binomial:
    def __init__(self, S_t, r, q, T, t, sigma, Smax_t, n, type1):
        self.S_t = S_t
        self.r = r
        self.q = q
        self.T = T
        self.t = t
        self.sigma = sigma
        self.Smax_t = Smax_t
        self.n = n
        self.type = type1

        self.S_arr = [[] for i in range(self.n + 1)]
        self.Smax_arr = [[[] for j in range(self.n + 1)] for i in range(self.n + 1)]
        self.Put_arr = [[[] for j in range(self.n + 1)] for i in range(self.n + 1)]
        self.dt = (self.T - self.t) / self.n
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = math.exp(-sigma * math.sqrt(self.dt))
        self.p = (math.exp((r - q) * self.dt) - self.d) / (self.u - self.d)
        self.q = 1 - self.p

    def option_price(self):
        S_t = self.S_t
        r = self.r
        Smax_t = self.Smax_t
        n = self.n
        S_arr = self.S_arr
        Smax_arr = self.Smax_arr
        Put_arr = self.Put_arr
        dt = self.dt
        u = self.u
        d = self.d
        p = self.p
        q = self.q

        # 生成Stock tree
        for j in range(n + 1):
            S_arr[n].append(S_t * (u ** (n - j)) * (d ** j))

        for j in range(n):
            S_arr[n - 1].append(S_t * (u ** (n - 1 - j)) * (d ** j))

        for i in range(n - 1):
            for j in range(n - i - 1):
                S_arr[n - i - 2].append(S_arr[n - i][j + 1])
        # print(S_arr)

        # 生成Stock max tree
        Smax_arr[0][0].append(Smax_t)
        # 上下
        for i in range(1, n + 1):
            if Smax_t < S_arr[i][0]:
                Smax_arr[i][0].append(S_arr[i][0])
            else:
                Smax_arr[i][0].append(Smax_t)
            Smax_arr[i][i].append(Smax_t)
        #   中間
        for i in range(2, n + 1):
            for j in range(1, i):
                for k in range(len(Smax_arr[i - 1][j - 1])):
                    # smax  大不在裡面
                    if Smax_arr[i - 1][j - 1][k] >= S_arr[i][j] and Smax_arr[i - 1][j - 1][k] not in Smax_arr[i][j]:
                        Smax_arr[i][j].append(Smax_arr[i - 1][j - 1][k])
                    # s 大不在裡面
                    elif Smax_arr[i - 1][j - 1][k] < S_arr[i][j] and S_arr[i][j] not in Smax_arr[i][j]:
                        Smax_arr[i][j].append(S_arr[i][j])
                for k in range(len(Smax_arr[i - 1][j])):
                    if Smax_arr[i - 1][j][k] >= S_arr[i][j] and Smax_arr[i - 1][j][k] not in Smax_arr[i][j]:
                        Smax_arr[i][j].append(Smax_arr[i - 1][j][k])
                    elif Smax_arr[i - 1][j][k] < S_arr[i][j] and S_arr[i][j] not in Smax_arr[i][j]:
                        Smax_arr[i][j].append(S_arr[i][j])

        # print("Stock tree:")
        # print(S_arr)
        # print("_____")
        # print("Smax tree:")
        # print(Smax_arr)

        for i in range(n + 1):
            for j in range(i + 1):
                Smax_arr[i][j].sort(reverse=True)
                # print(Smax_arr[i][j])

        # 生成每一個股價S下的每一個可能SMax的Put tree
        for j in range(n + 1):
            for k in range(len(Smax_arr[n][j])):
                payoff = Smax_arr[n][j][k] - S_arr[n][j]
                Put_arr[n][j].append(payoff)
        for i in range(n):  # 對每個時間(反序)
            # print(f"now in time {i + 1}")
            for j in range(n - i):  # 對每種可能股價
                for k in range(len(Smax_arr[n - 1 - i][j])):  # 對每個可能Smax
                    last1 = 0
                    last2 = 0
                    put_up = 0
                    put_down = 0
                    for up in range(last1, len(Smax_arr[n - i][j])):  # 對下一期往上的每個可能Smax
                        if Smax_arr[n - i - 1][j][k] == Smax_arr[n - i][j][up]:
                            last1 = up
                            put_up = Put_arr[n - i][j][up]
                            break
                    if put_up == 0:
                        for up in range(last1, len(Smax_arr[n - i][j])):
                            if Smax_arr[n - i][j][up] == S_arr[n - i][j]:
                                put_up = Put_arr[n - i][j][up]
                                break

                    for down in range(last2, len(Smax_arr[n - i][j + 1])):  # 對下一期往下的每個可能Smax
                        if Smax_arr[n - i - 1][j][k] == Smax_arr[n - i][j + 1][down]:
                            last1 = down
                            put_down = Put_arr[n - i][j + 1][down]
                            break
                    if put_down == 0:
                        for down in range(last2, len(Smax_arr[n - i][j + 1])):
                            if Smax_arr[n - i][j + 1][down] == S_arr[n - i][j + 1]:
                                put_down = Put_arr[n - i][j + 1][down]
                                break

                    intrinsic_price = (p * put_up + q * put_down) * math.exp(-r * dt)
                    early_exercise_price = Smax_arr[n - 1 - i][j][k] - S_arr[n - 1 - i][j]

                    if early_exercise_price > intrinsic_price and self.type == "NA":
                        Put_arr[n - 1 - i][j].append(early_exercise_price)
                    else:
                        Put_arr[n - 1 - i][j].append(intrinsic_price)

        # print("_____")
        # print("Put tree:")
        # for i in range(n+1):
        #     for j in range(n+1):
        #         print(Put_arr[i][j],end=",")
        #     print()
        # print("_____")

        print(f"The option price S_max = {Smax_t}, n = {n}, option type = {self.type}, by binomial tree is: {Put_arr[0][0]}")


class lookback_quick_approach:
    def __init__(self, S_t, r, q, T, t, sigma, Smax_t, n, type1):
        self.S_t = S_t
        self.r = r
        self.q = q
        self.T = T
        self.t = t
        self.sigma = sigma
        self.Smax_t = Smax_t
        self.n = n
        self.type = type1

        self.S_arr = [[] for i in range(self.n + 1)]
        self.Smax_arr = [[[] for j in range(self.n + 1)] for i in range(self.n + 1)]
        self.Put_arr = [[[] for j in range(self.n + 1)] for i in range(self.n + 1)]
        self.dt = (self.T - self.t) / self.n
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = math.exp(-sigma * math.sqrt(self.dt))
        self.p = (math.exp((r - q) * self.dt) - self.d) / (self.u - self.d)
        self.q = 1 - self.p

    def option_price(self):
        S_t = self.S_t
        r = self.r
        Smax_t = self.Smax_t
        n = self.n
        S_arr = self.S_arr
        Smax_arr = self.Smax_arr
        Put_arr = self.Put_arr
        dt = self.dt
        u = self.u
        d = self.d
        p = self.p
        q = self.q


        # 生成Stock tree
        for j in range(n + 1):
            S_arr[n].append(S_t * (u ** (n - j)) * (d ** j))

        for j in range(n):
            S_arr[n - 1].append(S_t * (u ** (n - 1 - j)) * (d ** j))

        for i in range(n - 1):
            for j in range(n - i - 1):
                S_arr[n - i - 2].append(S_arr[n - i][j + 1])

        # 如果 Smax_t = S0
        if abs(Smax_t - S_arr[0][0]) <= 10 ** -5:
            Smax_t = S_arr[0][0]

        # 生成Stock max tree
        Smax_arr[0][0].append(Smax_t)

        # bonus 1 的地方
        # 把 smax 跟比他大的Ｓ放進去
        for i in range(1, n + 1):
            for j in range(i + 1):
                Smax_arr[i][j].append(max(Smax_t, S_arr[i][j]))
        # 把前面比他大的Ｓ且不在smax 中放進去
        for i in range(2, n + 1):
            for j in range(1, i):
                for k in range(1, j + 1):
                    if S_arr[i - k][j - k] > Smax_t and S_arr[i - k][j - k] not in Smax_arr:
                        Smax_arr[i][j].append(S_arr[i - k][j - k])

        for i in range(n + 1):
            for j in range(i + 1):
                Smax_arr[i][j].sort(reverse=True)

        # 生成每一個股價S下的每一個可能SMax的Put tree
        for j in range(n + 1):
            for k in range(len(Smax_arr[n][j])):
                payoff = Smax_arr[n][j][k] - S_arr[n][j]
                Put_arr[n][j].append(payoff)
        for i in range(n):  # 對每個時間(反序)
            # print(f"now in time {i + 1}")
            for j in range(n - i):  # 對每種可能股價
                for k in range(len(Smax_arr[n - 1 - i][j])):  # 對每個可能Smax
                    last1 = 0
                    last2 = 0
                    put_up = 0
                    put_down = 0
                    for up in range(last1, len(Smax_arr[n - i][j])):  # 對下一期往上的每個可能Smax
                        if Smax_arr[n - i - 1][j][k] == Smax_arr[n - i][j][up]:
                            last1 = up
                            put_up = Put_arr[n - i][j][up]
                            break
                    if put_up == 0:
                        for up in range(last1, len(Smax_arr[n - i][j])):
                            if Smax_arr[n - i][j][up] == S_arr[n - i][j]:
                                put_up = Put_arr[n - i][j][up]
                                break

                    for down in range(last2, len(Smax_arr[n - i][j + 1])):  # 對下一期往下的每個可能Smax
                        if Smax_arr[n - i - 1][j][k] == Smax_arr[n - i][j + 1][down]:
                            last1 = down
                            put_down = Put_arr[n - i][j + 1][down]
                            break
                    if put_down == 0:
                        for down in range(last2, len(Smax_arr[n - i][j + 1])):
                            if Smax_arr[n - i][j + 1][down] == S_arr[n - i][j + 1]:
                                put_down = Put_arr[n - i][j + 1][down]
                                break

                    intrinsic_price = (p * put_up + q * put_down) * math.exp(-r * dt)
                    early_exercise_price = Smax_arr[n - 1 - i][j][k] - S_arr[n - 1 - i][j]

                    if early_exercise_price > intrinsic_price and self.type == "NA":
                        Put_arr[n - 1 - i][j].append(early_exercise_price)
                    else:
                        Put_arr[n - 1 - i][j].append(intrinsic_price)
        print(f"The option price S_max = {self.Smax_t}, n = {self.n}, option type = {self.type}, by binomial tree is: {Put_arr[0][0]}")



class lookback_Cheuk_and_Vorst:
    def __init__(self, S_t, r, q, T, t, sigma, n, type1):
        self.S_t = S_t
        self.r = r
        self.q = q
        self.T = T
        self.t = t
        self.sigma = sigma
        self.type = type1
        self.n = n
        self.dt = (T - t) / n
        self.u = math.exp(sigma * math.sqrt(self.dt))
        # self.u = Smax_t/S_t
        self.d = math.exp(-sigma * math.sqrt(self.dt))
        self.mu = math.exp((r-q)*self.dt)
        self.p = (self.mu*self.u-1)/(self.mu*(self.u-self.d))
        self.q = 1 - self.p

    def payoff(self, x):
        return max(x-1,0)

    def option_price(self):
        S_t = self.S_t
        n = self.n
        u = self.u
        p = self.p
        q = self.q

        ratio_tree = [[] for i in range(n+1)]
        for i in range(n+1):
            for j in range(i+1):
                ratio_tree[i].append(u**j)
        for j in range(n+1):
            ratio_tree[n][j] = self.payoff(ratio_tree[n][j])

        if self.type == "NA":
            for i in range(n):
                for j in range(n - i):
                    if j >= 1:
                        backward = p * ratio_tree[n - i][j - 1] + q * ratio_tree[n - i][j + 1]
                        ratio_tree[n - 1 - i][j] = max(ratio_tree[n - 1 - i][j]-1, backward)
                    else:
                        backward = p * ratio_tree[n - i][j] + q * ratio_tree[n - i][j + 1]
                        ratio_tree[n - 1 - i][j] = max(ratio_tree[n - 1 - i][j]-1, backward)
        elif self.type == "EU":
            for i in range(n):
                for j in range(n - i):
                    if j >= 1:
                        backward = p * ratio_tree[n - i][j - 1] + q * ratio_tree[n - i][j + 1]
                        ratio_tree[n - 1 - i][j] = backward
                    else:
                        backward = p * ratio_tree[n - i][j] + q * ratio_tree[n - i][j + 1]
                        ratio_tree[n - 1 - i][j] = backward

        print(f"The option price, n = {n}, option type = {self.type}, by binomial tree is: {ratio_tree[0][0] * S_t}")


print('----'*20)
print("binomial")
a1 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 50, 100, "EU").option_price()
# a2 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 50, 300, "EU").option_price()


print("")
a3 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 60, 100, "EU").option_price()
# a4 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 60, 300, "EU").option_price()
print("")
a5 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 70, 100, "EU").option_price()
# a6 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 70, 300, "EU").option_price()
print("")
a7 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 50, 100, "NA").option_price()
# a8 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 50, 300, "NA").option_price()
print("")
a9 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 60, 100, "NA").option_price()
# a10 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 60, 300, "NA").option_price()
print("")
a11 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 70, 100, "NA").option_price()
# a12 = lookback_binomial(50, 0.1, 0, 0.25, 0, 0.4, 70, 300, "NA").option_price()

print('----'*20)
print("MC")

b1 = lookback_MC(50, 0.1, 0, 0.25, 0, 0.4, 50, 10000, 20, 100)
# b2 = lookback_MC(50, 0.1, 0, 0.25, 0, 0.4, 50, 10000, 20, 300)
print("")

b3 = lookback_MC(50, 0.1, 0, 0.25, 0, 0.4, 60, 10000, 20, 100)
# b4 = lookback_MC(50, 0.1, 0, 0.25, 0, 0.4, 60, 10000, 20, 300)
print("")

b5 = lookback_MC(50, 0.1, 0, 0.25, 0, 0.4, 70, 10000, 20, 100)
# b6 = lookback_MC(50, 0.1, 0, 0.25, 0, 0.4, 70, 10000, 20, 300)
print("")

print('----'*20)
print("bonus_1")
c1 = lookback_quick_approach(50, 0.1, 0, 0.25, 0, 0.4, 50, 100, "EU").option_price()
# c2 = bonus_1(50, 0.1, 0, 0.25, 0, 0.4, 50, 300, "EU").option_price()
print("")
c3 = lookback_quick_approach(50, 0.1, 0, 0.25, 0, 0.4, 60, 100, "EU").option_price()
# c4 = bonus_1(50, 0.1, 0, 0.25, 0, 0.4, 60, 300, "EU").option_price()
print("")
c5 = lookback_quick_approach(50, 0.1, 0, 0.25, 0, 0.4, 70, 100, "EU").option_price()
# c6 = bonus_1(50, 0.1, 0, 0.25, 0, 0.4, 70, 300, "EU").option_price()
print("")
c7 = lookback_quick_approach(50, 0.1, 0, 0.25, 0, 0.4, 50, 100, "NA").option_price()
# c8 = bonus_1(50, 0.1, 0, 0.25, 0, 0.4, 50, 300, "NA").option_price()
print("")
c9 = lookback_quick_approach(50, 0.1, 0, 0.25, 0, 0.4, 60, 100, "NA").option_price()
# c10 = bonus_1(50, 0.1, 0, 0.25, 0, 0.4, 60, 300, "NA").option_price()
print("")
c11 = lookback_quick_approach(50, 0.1, 0, 0.25, 0, 0.4, 70, 100, "NA").option_price()
# c12 = bonus_1(50, 0.1, 0, 0.25, 0, 0.4, 70, 300, "NA").option_price()
print("")



print('----'*20)
print("bonus_2")
d1 = lookback_Cheuk_and_Vorst(50, 0.1, 0, 0.25, 0, 0.4, 1000, "EU").option_price()
d2 = lookback_Cheuk_and_Vorst(50, 0.1, 0, 0.25, 0, 0.4, 1000, "NA").option_price()
print("")