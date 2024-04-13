import numpy as np
from numpy import log as ln
from numpy.linalg import inv, det
import matplotlib.pyplot as plt


class lsm_monte_calo_data(object):
    def __init__(self, S, K, T, t, r, q, sigma, rep_num, sim_num, seg_num, S_max, Save_t):
        # parameters
        self.S = S
        self.k = K
        self.T = T
        self.t = t
        self.r = r
        self.q = q
        self.sigma = sigma
        self.rep_num = rep_num
        self.sim_num = sim_num
        self.seg_num = seg_num
        self.S_max = S_max
        self.Save_t = Save_t
        self.dt = (T - t) / seg_num

        # matrixs
        self.stock_mat = np.zeros(shape=(self.sim_num, self.seg_num + 1))     # stock matrix
        self.stock_mat[:, 0] = S
        self.ran_mat = np.random.normal(0, 1, size=(self.sim_num, self.seg_num))

        for i in range(1, self.seg_num + 1):
            self.stock_mat[:, i] = np.exp(np.log(self.stock_mat[:, i - 1]) + (r - q - sigma ** 2 / 2) * self.dt + (self.dt ** 0.5 * sigma) * self.ran_mat[:, i - 1])
        self.payoff_mat = np.zeros(shape=(self.sim_num, self.seg_num + 1))    # payoff matrix
        self.hv_mat = np.zeros(shape=(self.sim_num, self.seg_num + 1))    # holding value matrix
        self.ev_mat = np.zeros(shape=(self.sim_num, self.seg_num + 1))    # exercise value matrix
        self.Ehv_mat = np.zeros(shape=(self.sim_num, self.seg_num + 1))  # expected holding value matrix


def regress_param(x, y):
    # model
    # y = a*x**2 + b*x + c + np.random.randn(100)
    # find regression parameters
    X = np.vstack([x**2, x, np.ones(x.shape)]).T
    # print('X',X)
    # x_hat = np.linalg.inv(X.T@(X))@(X.T)@(y)
    x_hat = inv(X.T.dot(X)).dot(X.T).dot(y)
    # print('x_hat',x_hat)
    return x_hat


def regress_param1(x, smax, y):
    # model
    # y = a*x**2 + b*x + c + np.random.randn(100)
    # find regression parameters
    X = np.vstack([x**2, x, np.array(smax)**2, smax, np.ones(x.shape)]).T
    # print('X',X)
    # x_hat = np.linalg.inv(X.T@(X))@(X.T)@(y)
    x_hat = inv(X.T.dot(X)).dot(X.T).dot(y)
    # print('x_hat',x_hat)
    return x_hat


def lsm(x, option_type):
    if option_type == 'put':
        # i = 3 想
        # step 1 : determine the payoff for each path at maturity
        # print('stock_mat\n', x.stock_mat)
        x.payoff_mat[:, x.seg_num] = np.where(x.k > x.stock_mat[:, x.seg_num], x.k - x.stock_mat[:, x.seg_num], 0)  # put t = 3 (k-s)
        # print('payoff_mat_0\n',x.payoff_mat)
        for i in range(x.seg_num, 1, -1):   # 3
            # print("i：", i)
            # step 2 : determine the payoff for each path at maturity
            x.ev_mat[:, i-1] = np.where(x.k > x.stock_mat[:, i-1], x.k - x.stock_mat[:, i-1], 0)  # t = 3  s > k out of money 沒有holding value, 設為0
            # print('ev_mat\n', x.ev_mat)
            x.hv_mat[:, i-1] = np.where(x.payoff_mat[:, i] > 0, x.payoff_mat[:,i]*np.exp(-x.r*x.dt), 0)  # 折現回去算t = 2 的 holding value , t = 2  s > k out of money 沒有holding value, 設為0
            # print('hv_mat\n',x.hv_mat)
            # x.hv_mat[:,i-1][x.k < x.stock_mat[:,i-1]] = 0    # t = 2  s > k out of money 沒有holding value, 設為0
            param = regress_param(x.stock_mat[:, i-1][x.ev_mat[:, i-1] > 0 ], x.hv_mat[:, i-1][x.ev_mat[:, i-1] > 0]) # 取得 regress的參數
            pp = np.poly1d(param)
            x.Ehv_mat[:, i-1] = np.where(x.k > x.stock_mat[:, i-1], pp(x.stock_mat[:, i-1]), 0)
            # print('Ehv_mat\n', x.Ehv_mat)
            # x.Ehv_mat[:, i-1][x.k > x.stock_mat[:,i-1]] = 0
            # compare E[HV} < EV
            x.payoff_mat[:, i-1] = np.where(x.ev_mat[:, i-1] > x.Ehv_mat[:, i-1], x.ev_mat[:, i-1], x.hv_mat[:, i-1])
            # print('payoff_mat\n', x.payoff_mat)
        x.payoff_mat[:, 0] = x.payoff_mat[:, 1]*np.exp(-x.r*x.dt)
        #print(x.payoff_mat)
        # print('payoff_mat_final\n', x.payoff_mat)
        option_value = np.mean(x.payoff_mat[:, 0])
        # print('option_value',option_value)
        return option_value
    if option_type == 'call':
        # i = 3 想
        # step 1 : determine the payoff for each path at maturity
        # print('stock_mat\n', x.stock_mat)
        x.payoff_mat[:, x.seg_num] = np.where(x.k < x.stock_mat[:, x.seg_num],  x.stock_mat[:, x.seg_num] - x.k, 0)  # put t = 3 (k-s)
        # print('payoff_mat_0\n',x.payoff_mat)
        for i in range(x.seg_num, 1, -1):   # 3
            # print("i：", i)
            # step 2 : determine the payoff for each path at maturity
            x.ev_mat[:, i-1] = np.where(x.k < x.stock_mat[:, i-1],x.stock_mat[:, i-1] - x.k, 0)  # t = 3  s > k out of money 沒有holding value, 設為0
            # print('ev_mat\n', x.ev_mat)
            x.hv_mat[:, i-1] = np.where(x.payoff_mat[:, i] > 0, x.payoff_mat[:,i]*np.exp(-x.r*x.dt), 0)  # 折現回去算t = 2 的 holding value , t = 2  s > k out of money 沒有holding value, 設為0
            # print('hv_mat\n',x.hv_mat)
            # x.hv_mat[:,i-1][x.k < x.stock_mat[:,i-1]] = 0    # t = 2  s > k out of money 沒有holding value, 設為0
            param = regress_param(x.stock_mat[:, i-1][x.ev_mat[:, i-1] > 0 ], x.hv_mat[:, i-1][x.ev_mat[:, i-1] > 0]) # 取得 regress的參數
            pp = np.poly1d(param)
            x.Ehv_mat[:, i-1] = np.where(x.k < x.stock_mat[:, i-1], pp(x.stock_mat[:, i-1]), 0)
            # print('Ehv_mat\n', x.Ehv_mat)
            # x.Ehv_mat[:, i-1][x.k > x.stock_mat[:,i-1]] = 0
            # compare E[HV} < EV
            x.payoff_mat[:, i-1] = np.where(x.ev_mat[:, i-1] > x.Ehv_mat[:, i-1], x.ev_mat[:, i-1], x.hv_mat[:, i-1])
            # print('payoff_mat\n', x.payoff_mat)
        x.payoff_mat[:, 0] = x.payoff_mat[:, 1]*np.exp(-x.r*x.dt)
        # print('payoff_mat_final\n', x.payoff_mat)
        option_value = np.mean(x.payoff_mat[:, 0])
        # print('option_value',option_value)
        return option_value
    if option_type == 'max_put':
        # i = 3 想
        # step 1 : determine the payoff for each path at maturity
        # print('stock_mat\n', x.stock_mat)
        S_max_lst = [x.S_max] * x.sim_num
        max_k = np.where(S_max_lst > np.max(x.stock_mat, axis=1), S_max_lst, np.max(x.stock_mat, axis=1))
        x.payoff_mat[:, x.seg_num] = np.where(max_k > x.stock_mat[:, x.seg_num], max_k - x.stock_mat[:, x.seg_num], 0)  # put t = 3 (k-s)
        # print('payoff_mat_0\n',x.payoff_mat)
        for i in range(x.seg_num, 1, -1):   # 3
            # print("i：", i)
            # step 2 : determine the payoff for each path at maturity
            S_max_lst = [x.S_max] * x.sim_num
            max_k = np.where(S_max_lst > np.max(x.stock_mat[:, :i+1], axis=1), S_max_lst, np.max(x.stock_mat[:, :i+1],axis=1))
            # print(len(max_k))
            x.ev_mat[:, i-1] = np.where(max_k > x.stock_mat[:, i-1], max_k-x.stock_mat[:, i-1], 0)  # t = 3  s > k out of money 沒有holding value, 設為0
            # print('ev_mat\n', x.ev_mat)
            x.hv_mat[:, i-1] = np.where(x.payoff_mat[:, i] > 0, x.payoff_mat[:, i]*np.exp(-x.r*x.dt), 0)  # 折現回去算t = 2 的 holding value , t = 2  s > k out of money 沒有holding value, 設為0
            # print('hv_mat\n',x.hv_mat)
            # x.hv_mat[:,i-1][x.k < x.stock_mat[:,i-1]] = 0    # t = 2  s > k out of money 沒有holding value, 設為0

            param = regress_param1(x.stock_mat[:, i-1][x.ev_mat[:, i-1] > 0], S_max_lst, x.hv_mat[:, i-1][x.ev_mat[:, i-1] > 0]) # 取得 regress的參數
            #param = regress_param1(x.stock_mat[:, i - 1][x.ev_mat[:, i - 1] > 0], max_k[x.ev_mat[:, i - 1] > 0],x.hv_mat[:, i - 1][x.ev_mat[:, i - 1] > 0])  # 取得 regress的參數
            pp = np.poly1d(param)
            x.Ehv_mat[:, i-1] = np.where(max_k > x.stock_mat[:, i-1], pp(x.stock_mat[:, i-1]), 0)
            # print('Ehv_mat\n', x.Ehv_mat)
            # x.Ehv_mat[:, i-1][x.k > x.stock_mat[:,i-1]] = 0
            # compare E[HV} < EV
            x.payoff_mat[:, i-1] = np.where(x.ev_mat[:, i-1] > x.Ehv_mat[:, i-1], x.ev_mat[:, i-1], x.hv_mat[:, i-1])
            # print('payoff_mat\n', x.payoff_mat)
        x.payoff_mat[:, 0] = x.payoff_mat[:, 1]*np.exp(-x.r*x.dt)
        # print('payoff_mat_final\n', x.payoff_mat)
        option_value = np.mean(x.payoff_mat[:, 0])
        # print('option_value',option_value)
        return option_value
    if option_type == 'avg_call':
        # i = 3 想
        # step 1 : determine the payoff for each path at maturity
        # print('stock_mat\n', x.stock_mat)

        S_avg = (np.mean(x.stock_mat, axis=1) * x.seg_num + x.Save_t * ((x.t / (x.T - x.t) * x.seg_num) + 1)) / ((x.T / (x.T - x.t) * x.seg_num) + 1)
        # print(S_avg)


        x.payoff_mat[:, x.seg_num] = np.where(S_avg > x.k, S_avg - x.k, 0)  # put t = 3 (k-s)
        # print('payoff_mat_0\n',x.payoff_mat)
        for i in range(x.seg_num, 1, -1):   # 3
            # print("i：", i)
            # step 2 : determine the payoff for each path at maturity
            S_avg = (np.mean(x.stock_mat[:,:i+1], axis=1) * i + x.Save_t * ((x.t / (x.T - x.t) * x.seg_num) + 1))/(i+(x.t / (x.T - x.t) * x.seg_num))
            # print(S_avg)
            # Save = (S_path_df.iloc[:time + 1].mean(axis=0).values * (time + 1) + Save_t * (int(t / (T - t) * n) + 1)) / (time + 1 + int(t / (T - t) * n) + 1)
            x.ev_mat[:, i-1] = np.where(x.k < S_avg, S_avg - x.k, 0)  # t = 3  s > k out of money 沒有holding value, 設為0
            #print(x.ev_mat[:, i-1])
            # print('ev_mat\n', x.ev_mat)
            x.hv_mat[:, i-1] = np.where(x.payoff_mat[:, i] > 0, x.payoff_mat[:,i]*np.exp(-x.r*x.dt), 0)  # 折現回去算t = 2 的 holding value , t = 2  s > k out of money 沒有holding value, 設為0
            # print('hv_mat\n',x.hv_mat)
            # x.hv_mat[:,i-1][x.k < x.stock_mat[:,i-1]] = 0    # t = 2  s > k out of money 沒有holding value, 設為0
            # print(x.stock_mat[:,i])
            #param = regress_param1(x.stock_mat[:, i-1][x.ev_mat[:, i-1] > 0], S_avg[x.ev_mat[:, i-1] > 0], x.hv_mat[:, i-1][x.ev_mat[:, i-1] > 0]) # 取得 regress的參數
            param = regress_param(S_avg[x.ev_mat[:, i - 1] > 0], x.hv_mat[:, i - 1][x.ev_mat[:, i - 1] > 0])  # 取得 regress的參數

            pp = np.poly1d(param)
            x.Ehv_mat[:, i-1] = np.where(S_avg > x.k, pp(S_avg), 0)
            # print('Ehv_mat\n', x.Ehv_mat)
            # x.Ehv_mat[:, i-1][x.k > x.stock_mat[:,i-1]] = 0
            # compare E[HV} < EV
            x.payoff_mat[:, i-1] = np.where(x.ev_mat[:, i-1] > x.Ehv_mat[:, i-1], x.ev_mat[:, i-1], x.hv_mat[:, i-1])
            # print('payoff_mat\n', x.payoff_mat)
        x.payoff_mat[:, 0] = x.payoff_mat[:, 1]*np.exp(-x.r*x.dt)
        # print('payoff_mat_final\n', x.payoff_mat)
        option_value = np.mean(x.payoff_mat[:, 0])
        # print('option_value',option_value)
        return option_value


"""---------------------------------------------"""


def calculate_options(S, K, T, t, r, q, sigma, rep_num, sim_num, seg_num,  option_type, S_max, Save_t):
    if option_type == 'put':
        print(f'S: {S}, K:{ K}, T: {T}, t: {t}, r: {r}, q: {q}, sigma: {sigma}, option_type: {option_type}')
    if option_type == 'max_put':
        print(f'S: {S}, K:{ K}, T: {T}, t: {t}, r: {r}, q:{ q}, sigma: {sigma}, option_type: {option_type}, S_max: {S_max}')
    if option_type == "avg_call":
        print(f'S: {S}, K:{ K}, T:{ T}, t: {t}, r: {r}, q: {q}, sigma: {sigma},option_type: {option_type}, Save_t: {Save_t}')
    value = []
    for i in range(rep_num):
        # print(f'i: {i}')
        qqq = lsm_monte_calo_data(S, K, T, t, r, q, sigma, rep_num, sim_num, seg_num, S_max, Save_t)
        value.append(lsm(qqq, option_type))

    print(f'American {option_type} option by LSM')
    print('option value: ', np.mean(value))
    print(f'CI:[{np.mean(value) - 2 * np.std(value)},{np.mean(value) + 2 * np.std(value)}]')
    print()
    print("-------"*10)
    print()


#option_lst = ['put'], 'call','max_put', 'avg_call'

# calculate_options(S, K, T, t, r, q, sigma, rep_num, sim_num, seg_num, option_list, S_max, Save_t)
calculate_options(50, 50, 0.5, 0, 0.1, 0.05, 0.4, 20, 10000, 100, 'put', 60, 50)
print("")
calculate_options(50, 50, 0.25, 0, 0.1, 0.0, 0.4, 20, 10000, 100, 'max_put', 60, 50)
print("")
calculate_options(50, 50, 0.5, 0.25, 0.1, 0.05, 0.8, 20, 10000, 100, 'avg_call', 60, 50)






