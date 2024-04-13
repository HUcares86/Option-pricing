import math, copy, random
import numpy as np
from scipy.stats import norm
#輸入參數
# K = float(input())
# r = float(input())
# T = float(input())
# simul = int(input())
# repet = int(input())
# n = int(input())
# S0_list = []
# q_list = []
# sigma_list = []
# rho_matrix = [[] for i in range(n)]
# for i in range(n):
#     S0_list.append(float(input()))
# for i in range(n):
#     q_list.append(float(input()))
# for i in range(n):
#     sigma_list.append(float(input()))
# for i in range(n):
#     for j in range(n):
#         rho_matrix[i].append(float(input()))


class rainbow:
    def __init__(self, K, r, T, sim, rep, n, S0_list, q_list, sigma_list, rho_matrix):
        self.K = K
        self.r = r
        self.T = T
        self.sim = sim
        self.rep = rep
        self.n = n
        self.S0_list = S0_list
        self.q_list = q_list
        self.sigma_list = sigma_list
        self.rho_matrix = rho_matrix

    def sqrt(self, x):
        return x**0.5

    def exp(self, x):
        return math.exp(x)

    def payoff(self, s):
        maxi = 0
        for i in range(n):
            if s[i] > maxi:
                maxi = s[i]
        return max(maxi - K, 0)

    def cho(self, C):
        n = self.n
        A = [[0 for u in range(n)] for v in range(n)]
        A[0][0] = self.sqrt(C[0][0])
        for j in range(1, n):
            A[0][j] = C[0][j] / A[0][0]
        for i in range(1, n - 1):
            A[i][i] = C[i][i]
            for k in range(i):
                A[i][i] -= A[k][i] ** 2
            A[i][i] = self.sqrt(A[i][i])

            for j in range(i + 1, n):
                A[i][j] = C[i][j]
                for k in range(i):
                    A[i][j] -= A[k][i] * A[k][j]
                A[i][j] = A[i][j] / A[i][i]
        A[n - 1][n - 1] = C[n - 1][n - 1]
        for k in range(n - 1):
            A[n - 1][n - 1] -= A[k][n - 1] ** 2
        A[n - 1][n - 1] = self.sqrt(A[n - 1][n - 1])
        return (A)

    def MC_option_pricing(self):
        n = self.n
        #共變異數矩陣
        C = [[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                Cij = self.rho_matrix[i][j] * self.sigma_list[i] * self.sigma_list[j] * T
                C[i].append(Cij)
        # 共變異Cholesky分解矩陣
        A = self.cho(C)
        A_array = np.array(A)


        #模擬抽樣
        rep_price = []
        dis_rep_price = []
        for a in range(self.rep):
            sim_price = [] #紀錄simul次的選擇權payoff
            for b in range(self.sim):
                z_list = np.zeros(n) #每次模擬的n個z值
                r_list = np.zeros(n) #每次模擬的n個r值
                stock_list = [] #每次模擬的n個股票的最終價格
                for i in range(n):
                    z_i = random.gauss(0, 1)
                    z_list[i] = z_i
                r_list = np.matmul(z_list, A_array)
                for i in range(n):
                    ST_i = S0_list[i] * self.exp((r - q_list[i] - (sigma_list[i]**2)/2) * T + r_list[i])
                    stock_list.append(ST_i)
                sim_price.append(self.payoff(stock_list))
            rep_price.append(np.mean(sim_price))


        for i in range(self.rep):
            dis_rep_price.append(self.exp(-r*T) * rep_price[i])

        dis_rep_array = np.array(dis_rep_price)
        std = np.std(dis_rep_array)
        mean = np.mean(dis_rep_array)

        # print(f"C matrix is:")
        # for i in range(n):
        #     for j in range(n):
        #         print("%.5f" % C[i][j], end=',')
        #     print()
        # print()
        # print(f"A matrix is:")
        # for i in range(n):
        #     for j in range(n):
        #         print("%.5f" % A[i][j], end=',')
        #     print()
        # print()


        print(f"The option price is {mean}")
        print(f"The option price's std is {std}")
        print(f"The option price 95% C.I is({mean-2*std},{mean+2*std})")

    def MC_var_option_pricing(self):
        n = self.n
        # 共變異數矩陣
        C = [[] for i in range(n)]
        for i in range(n):
            for j in range(n):
                Cij = rho_matrix[i][j] * sigma_list[i] * sigma_list[j] * T
                C[i].append(Cij)

        rep_price = []
        dis_rep_price = []
        z_array = np.zeros((self.rep, n, self.sim))  # z_array[repet][stock][simul]
        z_hat_array = np.zeros((self.rep, n, self.sim))
        z_hat_tran_array = np.zeros((self.rep, self.sim, n))
        z_prime_array = np.zeros((self.rep, self.sim, n))
        for a in range(self.rep):
            for i in range(n):
                for b in range(self.sim // 2):
                    z_array[a][i][b] = random.gauss(0, 1)
                    z_array[a][i][b + 5000] = -z_array[a][i][b]
            for i in range(n):
                z_bar = np.mean(z_array[a][i])
                for b in range(self.sim):
                    z_hat_array[a][i][b] = z_array[a][i][b] - z_bar
            C_hat = np.cov(z_array[a])
            A_hat = self.cho(C_hat)
            A_hat_inverse = np.linalg.inv(A_hat)
            z_hat_tran_array[a] = np.transpose(z_hat_array[a])
            z_prime_array[a] = np.matmul(z_hat_tran_array[a], A_hat_inverse)
            z_array[a] = np.transpose(z_prime_array[a])

        # 模擬抽樣
        rep_price = []
        dis_rep_price = []
        A = self.cho(C)
        for a in range(self.rep):
            sim_price = []  # 紀錄simul次的選擇權payoff
            for b in range(self.sim):
                r_list = []
                stock_list = []
                for i in range(n):
                    r_i = 0
                    for j in range(n):
                        r_i += z_array[a][j][b] * A[j][i]
                    r_list.append(r_i)
                for i in range(n):
                    ST_i = S0_list[i] * self.exp((r - q_list[i] - (sigma_list[i] ** 2) / 2) * T + r_list[i])
                    stock_list.append(ST_i)
                sim_price.append(self.payoff(stock_list))
            rep_price.append(np.mean(sim_price))

        for a in range(self.rep):
            dis_rep_price.append(self.exp(-r * T) * rep_price[a])

        dis_rep_array = np.array(dis_rep_price)
        std = np.std(dis_rep_array)
        mean = np.mean(dis_rep_array)

        # print(f"C matrix is:")
        # for i input1 range(n):
        #     for j input1 range(n):
        #         print("%.3f" % C[i][j], end=',')
        #     print()
        # print()
        # print(f"A matrix is:")
        # for i input1 range(n):
        #     for j input1 range(n):
        #         print("%.3f" % A[i][j], end=',')
        #     print()
        # print()

        print(f"The option price is {mean}")
        print(f"The option price's std is {std}")
        print(f"The option price 95% C.I is({mean - 2 * std},{mean + 2 * std})")

    def MC_inverse_Cholesky_option_pricing(self):
        n = self.n
        # 共變異數矩陣
        C = [[] for i in range(n)]
        for i in range(n):
            for j in range(n):
                Cij = rho_matrix[i][j] * sigma_list[i] * sigma_list[j] * T
                C[i].append(Cij)

        # 共變異Cholesky分解矩陣
        A = [[0 for j in range(n)] for i in range(n)]
        A[0][0] = self.sqrt(C[0][0])
        for j in range(1, n):
            A[0][j] = C[0][j] / A[0][0]
        for i in range(1, n - 1):
            A[i][i] = C[i][i]
            for k in range(i):
                A[i][i] -= A[k][i] ** 2
            A[i][i] = self.sqrt(A[i][i])

            for j in range(i + 1, n):
                A[i][j] = C[i][j]
                for k in range(i):
                    A[i][j] -= A[k][i] * A[k][j]
                A[i][j] = A[i][j] / A[i][i]
        A[n - 1][n - 1] = C[n - 1][n - 1]
        for k in range(n - 1):
            A[n - 1][n - 1] -= A[k][n - 1] ** 2
        A[n - 1][n - 1] = self.sqrt(A[n - 1][n - 1])

        # 模擬抽樣
        rep_price = []
        dis_rep_price = []
        z_array = np.zeros((self.rep, n, self.sim))  # z_array[repet][stock][simul]
        for a in range(self.rep):
            for i in range(n):
                for b in range(self.sim // 2):
                    z_array[a][i][b] = random.gauss(0, 1)
                for b in range(self.sim // 2):
                    z_array[a][i][self.sim // 2 + b] = -z_array[a][i][b]
                z_std = np.std(z_array[a][i])
                for b in range(self.sim):
                    z_array[a][i][b] = z_array[a][i][b] / z_std

        for a in range(self.rep):
            sim_price = []  # 紀錄simul次的選擇權payoff
            for b in range(self.sim):
                r_list = []
                stock_list = []
                for i in range(n):
                    r_i = 0
                    for j in range(n):
                        r_i += z_array[a][j][b] * A[j][i]
                    r_list.append(r_i)
                for i in range(n):
                    ST_i = S0_list[i] * self.exp((r - q_list[i] - (sigma_list[i] ** 2) / 2) * T + r_list[i])
                    stock_list.append(ST_i)
                sim_price.append(self.payoff(stock_list))
            rep_price.append(np.mean(sim_price))

        for i in range(self.rep):
            dis_rep_price.append(self.exp(-r * T) * rep_price[i])

        dis_rep_array = np.array(dis_rep_price)
        std = np.std(dis_rep_array)
        mean = np.mean(dis_rep_array)

        # print(f"C matrix is:")
        # for i input1 range(n):
        #     for j input1 range(n):
        #         print("%.3f" % C[i][j], end=',')
        #     print()
        # print()
        # print(f"A matrix is:")
        # for i input1 range(n):
        #     for j input1 range(n):
        #         print("%.3f" % A[i][j], end=',')
        #     print()
        # print()

        print(f"The option price is {mean}")
        print(f"The option price's std is {std}")
        print(f"The option price 95% C.I is({mean - 2 * std},{mean + 2 * std})")


"---------- 1 --------------------------------------------------------------------------------"
print("---------- 1 --------------------------------------------------------------------------------")
K = 100
r = 0.1
T = 0.5
sim = 10000
rep = 20
n = 2
S0_list = [95, 95]
q_list = [ 0.05, 0.05]
sigma_list = [ 0.5, 0.5]
rho_matrix = [
                [ 1.0, 1.0 ],
                [ 1.0, 1.0 ]]
a = rainbow(K, r, T, sim, rep, n, S0_list, q_list, sigma_list, rho_matrix)
print("++++++ MC ++++++")
print("")
a.MC_option_pricing()
print("++++++ MC var ++++++")
print("")
a.MC_var_option_pricing()
print("++++++ MC inverse_Cholesky ++++++")
print("")
a.MC_inverse_Cholesky_option_pricing()


"---------- 2 --------------------------------------------------------------------------------"
print("")
print("---------- 2 --------------------------------------------------------------------------------")
K = 100
r = 0.1
T = 0.5
sim = 10000
rep = 20
n = 2
S0_list = [95, 95]
q_list = [ 0.05, 0.05]
sigma_list = [ 0.5, 0.5]
rho_matrix = [
                [ 1.0, -1.0 ],
                [ -1.0, 1.0 ]]
a = rainbow(K, r, T, sim, rep, n, S0_list, q_list, sigma_list, rho_matrix)
print("++++++ MC ++++++")
print("")
a.MC_option_pricing()
print("++++++ MC var ++++++")
print("")
a.MC_var_option_pricing()
print("++++++ MC inverse_Cholesky ++++++")
print("")
a.MC_inverse_Cholesky_option_pricing()

"---------- 3 --------------------------------------------------------------------------------"
print("")
print("---------- 3 --------------------------------------------------------------------------------")
K = 100
r = 0.1
T = 0.5
sim = 10000
rep = 20
n = 5
S0_list = [ 95, 95, 95, 95, 95]
q_list = [ 0.05, 0.05, 0.05, 0.05, 0.05]
sigma_list =  [ 0.5, 0.5, 0.5, 0.5, 0.5]
rho_matrix = [
                [  1.0, 0.5, 0.5, 0.5, 0.5 ],
                  [ 0.5, 1.0, 0.5, 0.5, 0.5 ],
                  [ 0.5, 0.5, 1.0, 0.5, 0.5 ],
                  [ 0.5, 0.5, 0.5, 1.0, 0.5 ],
                  [ 0.5, 0.5, 0.5, 0.5, 1.0 ]
                 ]
a = rainbow(K, r, T, sim, rep, n, S0_list, q_list, sigma_list, rho_matrix)
print("++++++ MC ++++++")
print("")
a.MC_option_pricing()
print("++++++ MC var ++++++")
print("")
a.MC_var_option_pricing()
print("++++++ MC inverse_Cholesky ++++++")
print("")
a.MC_inverse_Cholesky_option_pricing()
