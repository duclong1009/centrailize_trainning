import os

import numpy as np
import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.io import savemat, loadmat

from .parameters import *


def plot_location(aploc, terloc, args):
    path = os.path.join(args.data_folder, f'init_data/figures/{args.scenario_name}_M_{args.M}_K_{args.K}_D_{args.D}_'
                                          f'bs_{args.bs_dist}-{args.total_step}')

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(min(100, terloc.shape[0])):
        plt.figure(1)
        plt.scatter(aploc[:, 0], aploc[:, 1], marker='x')
        plt.scatter(terloc[i, :, 0], terloc[i, :, 1], c='red')
        saved_path = os.path.join(path, f'location-{i}.png')
        plt.savefig(saved_path)
        plt.cla()


def plot_trajectory(aploc, user_loc, args):
    """
        trajec shape (args.n_train_data, args.max_steps, args.K, 2)
    """
    path = os.path.join(args.data_folder, f'init_data/figures/{args.scenario_name}_M_{args.M}_K_{args.K}_D_{args.D}_'
                                          f'bs_{args.bs_dist}-{args.total_step}')

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(min(100, user_loc.shape[0])):
        plt.scatter(aploc[:, 0], aploc[:, 1])
        for k in range(args.K):
            if user_loc[i, 0, k, 0] == user_loc[i, -1, k, 0] and user_loc[i, 0, k, 1] == user_loc[i, -1, k, 1]:
                plt.scatter(user_loc[i, 0, k, 0], user_loc[i, 0, k, 1], color='black', marker='*')
            else:
                plt.scatter(user_loc[i, 0, k, 0], user_loc[i, 0, k, 1], color='red', marker='x')
                plt.plot(user_loc[i, :, k, 0], user_loc[i, :, k, 1], color='red')

        plt.savefig(os.path.join(path, 'trajectory_{}.png'.format(i)))
        plt.cla()


def generate_beta(aploc, args):
    v = (args.max_veloc * 1000 / 3600)
    c = 3 * (10 ** 8)
    f = 1.9 * (10 ** 9)
    Tc = c / (4 * v * f)  # coherence time

    if args.schedule_time < 0:
        scheduling_time = Tc
    else:
        scheduling_time = args.schedule_time

    max_veloc = scheduling_time * v / 1000

    BETAA = np.zeros(shape=(args.total_step, args.M, args.K))
    user_loc = np.zeros(shape=(args.total_step, args.K, 2))

    terloc = np.random.uniform(0, args.D, (args.K, 2))
    # user velocity
    velo = np.zeros((args.K, 2))
    # self.velo[np.arange(K), np.random.randint(0, 2, size=K)] = 0.0000028 * np.random.rand(K)-0.0000014
    velo[np.arange(args.K), np.random.randint(0, 2, size=args.K)] = \
        2 * max_veloc * np.random.rand(args.K) - max_veloc
    distvelo = (velo[:, 0] ** 2 + velo[:, 1] ** 2) ** 0.5

    # initial fading
    shafad_train = np.random.randn(args.K, args.total_step)
    sh_AP_train = 1 / (2 ** 0.5) * args.sigma_shd * np.random.randn(args.M, 1)
    sh_Ter_train = 1 / (2 ** 0.5) * args.sigma_shd * np.random.randn(args.K, 1)

    BETAA[0] = cellsetting(aploc, terloc, sh_AP_train, sh_Ter_train, args)
    user_loc[0] = terloc

    # change_route = np.random.choice(np.arange(int(args.total_step/3), args.total_step))

    for step in tqdm.tqdm(range(1, args.total_step, 1)):
        cor = (2 ** (-distvelo / 0.1)).reshape([args.K, 1])
        sh_Ter_train = sh_Ter_train * cor + (1 - cor ** 2) ** 0.5 * (
                1 / (2 ** 0.5) * args.sigma_shd * (shafad_train[:, step].reshape([args.K, 1])))

        terloc = terloc + velo

        BETAA[step] = cellsetting(aploc, terloc, sh_AP_train, sh_Ter_train, args)
        user_loc[step] = terloc

    return BETAA, user_loc


# calculate estimated channels
def init_V(BETAA, Phii, args):
    """
    Return the estimated channel
    @param BETAA: shape [num_scenario, num_steps, num_bs, num_user]
    @param Phii: shape [num_scenario, num_user, num_user]
    @param args: other arguments
    @return:
    """
    V = np.zeros(shape=(args.num_scenario, args.total_step, args.M, args.K))
    for i in range(args.num_scenario):
        for j in range(args.total_step):
            Var = np.zeros((args.M, args.K))
            mau = np.zeros((args.M, args.K))
            for m in range(args.M):
                for k in range(args.K):
                    mau[m, k] = (np.linalg.norm(
                        BETAA[i, j, m, :] ** 0.5 * (np.dot(Phii[i, :, k].T, Phii[i])))) ** 2

            for m in range(args.M):
                for k in range(args.K):
                    Var[m, k] = args.trtau * Pp * BETAA[i, j, m, k] ** 2 / (
                            args.trtau * Pp * mau[m, k] + 1)

            V[i, j, :, :] = Var

    return V


def load_mobile_data(args, is_test=False):
    """
    Load or create data for mobile scenario
    @param args: all arguments
    @param is_test: whether to load test_data
    @return:
    """
    if is_test:
        path = os.path.join(args.data_folder, f'test_data/{args.scenario_name}_data_M_{args.M}_K_{args.K}_D_{args.D}_'
                                              f'bs_{args.bs_dist}-{args.total_step}-'
                                              f'{args.num_scenario}-{args.max_veloc}.mat')
    else:
        path = os.path.join(args.data_folder, f'init_data/{args.scenario_name}_data_M_{args.M}_K_{args.K}_D_{args.D}_'
                                              f'bs_{args.bs_dist}-{args.total_step}-'
                                              f'{args.num_scenario}-{args.max_veloc}.mat')

    if os.path.isfile(path):
        print('---> Loading saved locations <---')

        saved_data = loadmat(path)
        beta = saved_data['beta']
        user_loc = saved_data['user_loc']
        phi = saved_data['phi']
        V = saved_data['V']
        # V = init_V(beta, phi, args)
        # savemat(path, {'beta': beta, 'user_loc': user_loc, 'phi': phi, 'V': V})

    else:
        print('---> Generating new locations <---')
        if args.bs_dist == 'random':
            aploc = np.random.uniform(0, args.D, (args.M, 2))
        elif args.bs_dist == 'grid':
            n = int(np.sqrt(args.M))
            args.M = n ** 2
            aploc = np.zeros(shape=(args.M, 2))

            x, y = np.meshgrid(np.linspace(0.001, args.D - 0.001, n), np.linspace(0.001, args.D - 0.001, n))
            aploc[:, 0] = x.flatten()
            aploc[:, 1] = y.flatten()
        else:
            raise NotImplementedError('NotImplementedError')

        beta_locaitons = Parallel(n_jobs=os.cpu_count())(
            delayed(generate_beta)(aploc, args) for _ in range(args.num_scenario))
        phi = init_Phii(args)
        beta, user_loc = [], []
        for i in range(args.num_scenario):
            beta.append(beta_locaitons[i][0])
            user_loc.append(beta_locaitons[i][1])

        beta = np.stack(beta, axis=0)  # shape [num_scenario, num_steps, num_bs, num_user]
        user_loc = np.stack(user_loc, axis=0)

        V = init_V(beta, phi, args)

        savemat(path, {'beta': beta, 'user_loc': user_loc, 'phi': phi, 'V': V})
        plot_trajectory(aploc, user_loc, args)
    return beta, user_loc, phi, V


def init_Phii(args):
    """
    Init the Phii which is used for calculate beta
    @param args:
    @return:
    """
    num_scenario = args.num_scenario

    print('---> Generating new Phii <---')
    Phii = np.zeros(shape=(num_scenario, args.K, args.K))
    for i in range(num_scenario):
        U, sigma, vt = np.linalg.svd(np.random.randn(args.trtau, args.trtau))  # u include tau orthogonal sequences
        Phii[i] = np.copy(U)  # (K, K)

    return Phii


def cellsetting(maps, kters, sh_AP, sh_Ter, args):
    AP = np.zeros((args.M, 2, 9))  # randomly locations of M APs
    AP[:, :, 0] = maps
    # wrapped around (8 neighbor cells)
    D1 = np.zeros((args.M, 2))
    D1[:, 0] = D1[:, 0] + args.D * np.ones(args.M)  # right
    AP[:, :, 1] = AP[:, :, 0] + D1

    D2 = np.zeros((args.M, 2))
    D2[:, 1] = D2[:, 1] + args.D * np.ones(args.M)  # up
    AP[:, :, 2] = AP[:, :, 0] + D2

    D3 = np.zeros((args.M, 2))
    D3[:, 0] = D3[:, 0] - args.D * np.ones(args.M)  # left
    AP[:, :, 3] = AP[:, :, 0] + D3

    D4 = np.zeros((args.M, 2))
    D4[:, 1] = D4[:, 1] - args.D * np.ones(args.M)  # down
    AP[:, :, 4] = AP[:, :, 0] + D4

    D5 = np.zeros((args.M, 2))
    D5[:, 0] = D5[:, 0] + args.D * np.ones(args.M)  # right down
    D5[:, 1] = D5[:, 1] - args.D * np.ones(args.M)
    AP[:, :, 5] = AP[:, :, 0] + D5

    D6 = np.zeros((args.M, 2))
    D6[:, 0] = D6[:, 0] - args.D * np.ones(args.M)  # left up
    D6[:, 1] = D6[:, 1] + args.D * np.ones(args.M)
    AP[:, :, 6] = AP[:, :, 0] + D6

    D7 = np.zeros((args.M, 2))
    D7 = D7 + args.D * np.ones((args.M, 2))  # right up
    AP[:, :, 7] = AP[:, :, 0] + D7

    D8 = np.zeros((args.M, 2))
    D8 = D8 - args.D * np.ones((args.M, 2))  # left down
    AP[:, :, 8] = AP[:, :, 0] + D8

    # Randomly locations of K terminals:
    Ter = np.zeros((args.K, 2, 1))
    Ter[:, :, 0] = kters

    Z_shd = np.zeros((args.M, args.K))
    for m in range(args.M):
        for k in range(args.K):
            Z_shd[m, k] = sh_AP[m] + sh_Ter[k]

    BETAA = np.zeros((args.M, args.K))
    dist = np.zeros((args.M, args.K))
    for m in range(args.M):
        for k in range(args.K):
            dist[m, k] = min(np.linalg.norm(AP[m, :, 0] - Ter[k, :, 0]), np.linalg.norm(AP[m, :, 1] - Ter[k, :, 0]),
                             np.linalg.norm(AP[m, :, 2] - Ter[k, :, 0]), np.linalg.norm(AP[m, :, 3] - Ter[k, :, 0]),
                             np.linalg.norm(AP[m, :, 4] - Ter[k, :, 0]), np.linalg.norm(AP[m, :, 5] - Ter[k, :, 0]),
                             np.linalg.norm(AP[m, :, 6] - Ter[k, :, 0]), np.linalg.norm(AP[m, :, 7] - Ter[k, :, 0]),
                             np.linalg.norm(AP[m, :, 8] - Ter[k, :, 0]))
            if dist[m, k] < d0:
                betadB = -L - 35 * math.log10(d1) + 20 * math.log10(d1) - 20 * math.log10(d0)
            elif d0 <= dist[m, k] <= d1:
                betadB = -L - 35 * math.log10(d1) + 20 * math.log10(d1) - 20 * math.log10(dist[m, k])
            else:
                betadB = -L - 35 * math.log10(dist[m, k]) + Z_shd[m, k]
            BETAA[m, k] = 10 ** (betadB / 10)
    return BETAA
