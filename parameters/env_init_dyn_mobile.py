import os
import pickle

import numpy as np
import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from .parameters import *


def plot_location(aploc, terloc, num_user, args):
    path = os.path.join(args.data_folder, f'init_data/figures/{args.scenario_name}_M_{args.M}_K_{num_user}_D_{args.D}_'
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


def plot_trajectory(aploc, user_loc, index, num_user, args):
    """
        trajec shape (args.n_train_data, args.max_steps, num_user, 2)
    """
    path = os.path.join(args.data_folder, f'init_data/figures/{args.scenario_name}_M_{args.M}_K_{args.K}_D_{args.D}_'
                                          f'bs_{args.bs_dist}-{args.total_step}')
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(min(100, user_loc.shape[0])):
        plt.scatter(aploc[:, 0], aploc[:, 1])
        for k in range(num_user):
            if user_loc[i, 0, k, 0] == user_loc[i, -1, k, 0] and user_loc[i, 0, k, 1] == user_loc[i, -1, k, 1]:
                plt.scatter(user_loc[i, 0, k, 0], user_loc[i, 0, k, 1], color='black', marker='*')
            else:
                plt.scatter(user_loc[i, 0, k, 0], user_loc[i, 0, k, 1], color='red', marker='x')
                plt.plot(user_loc[i, :, k, 0], user_loc[i, :, k, 1], color='red')

        plt.savefig(os.path.join(path, 'trajectory_{}_{}_{}.png'.format(index, num_user, i)))
        plt.cla()


def generate_beta(aploc, num_user, args):
    v = (args.max_veloc * 1000 / 3600)
    c = 3 * (10 ** 8)
    f = 1.9 * (10 ** 9)
    Tc = c / (4 * v * f)  # coherence time

    if args.schedule_time < 0:
        scheduling_time = Tc
    else:
        scheduling_time = args.schedule_time

    max_veloc = scheduling_time * v / 1000

    BETAA = np.zeros(shape=(args.total_step, args.M, num_user))
    user_loc = np.zeros(shape=(args.total_step, num_user, 2))

    terloc = np.random.uniform(0, args.D, (num_user, 2))
    # user velocity
    velo = np.zeros((num_user, 2))
    # self.velo[np.arange(K), np.random.randint(0, 2, size=K)] = 0.0000028 * np.random.rand(K)-0.0000014
    velo[np.arange(num_user), np.random.randint(0, 2, size=num_user)] = \
        2 * max_veloc * np.random.rand(num_user) - max_veloc
    distvelo = (velo[:, 0] ** 2 + velo[:, 1] ** 2) ** 0.5

    # initial fading
    shafad_train = np.random.randn(num_user, args.total_step)
    sh_AP_train = 1 / (2 ** 0.5) * args.sigma_shd * np.random.randn(args.M, 1)
    sh_Ter_train = 1 / (2 ** 0.5) * args.sigma_shd * np.random.randn(num_user, 1)

    BETAA[0] = cellsetting(aploc, terloc, sh_AP_train, sh_Ter_train, num_user, args)
    user_loc[0] = terloc

    # change_route = np.random.choice(np.arange(int(args.total_step/3), args.total_step))

    for step in tqdm.tqdm(range(1, args.total_step, 1)):
        cor = (2 ** (-distvelo / 0.1)).reshape([num_user, 1])
        sh_Ter_train = sh_Ter_train * cor + (1 - cor ** 2) ** 0.5 * (
                1 / (2 ** 0.5) * args.sigma_shd * (shafad_train[:, step].reshape([num_user, 1])))

        terloc = terloc + velo

        BETAA[step] = cellsetting(aploc, terloc, sh_AP_train, sh_Ter_train, num_user, args)
        user_loc[step] = terloc

    return BETAA, user_loc


def init_data_dyn_mobile(args):
    path = os.path.join(args.data_folder, f'init_data/{args.scenario_name}_data_M_{args.M}_K_{args.K}_D_{args.D}_'
                                          f'bs_{args.bs_dist}-{args.total_step}-'
                                          f'{args.num_scenario}-{args.max_veloc}.mat')

    if os.path.isfile(path):
        print('---> Loading saved locations <---')
        with open(path, 'rb') as pf:
            data_object = pickle.load(pf)

    else:
        print('---> Generating new locations <---')
        num_scenario = args.num_scenario
        n_train_data = int(num_scenario * args.train_size)
        n_test_data = int(num_scenario * args.test_size)

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

        data_object = []

        train_data = init_dyn_data(aploc, n_train_data, args, is_train=True)

        if args.scenario_name == 'dyn_mobile_fix_train' or args.scenario_name == 'dyn_mobile_more_data':
            # using the same test data as dyn_mobile scenario
            path_dyn_mobile = os.path.join(args.data_folder,
                                           f'init_data/dyn_mobile_data_M_{args.M}_K_{args.K}_D_{args.D}_'
                                           f'bs_{args.bs_dist}-{args.total_step}-'
                                           f'200-{args.max_veloc}.mat')
            with open(path_dyn_mobile, 'rb') as pf:
                data_dyn_mobile = pickle.load(pf)
                test_data = data_dyn_mobile[1]

        else:
            test_data = init_dyn_data(aploc, n_test_data, args, is_train=False)

        data_object.append(train_data)
        data_object.append(test_data)
        with open(path, 'wb') as pf:
            pickle.dump(data_object, pf)
            pf.close()

    return data_object


def init_dyn_data(aploc, num_scenario, args, is_train=False):
    if is_train:
        min_num_user = int(args.K / 2)
        max_num_user = int(args.K)
    else:
        min_num_user = int(args.K / 3)
        max_num_user = int(args.K * 3 / 2)

    if args.scenario_name == 'dyn_mobile_fix_train' and is_train:
        number_users = np.array([args.K] * args.num_user_case)
    else:
        number_users = np.random.choice(np.arange(min_num_user, max_num_user), size=args.num_user_case)
    num_scenario_per_num_user_case = int(num_scenario / args.num_user_case)
    data_object = []
    for data_index, num_user in enumerate(number_users):
        beta_locaitons = Parallel(n_jobs=os.cpu_count())(
            delayed(generate_beta)(aploc, num_user, args) for _ in range(num_scenario_per_num_user_case))
        phi = init_Phii(num_user, args)
        beta, user_loc = [], []
        for i in range(num_scenario_per_num_user_case):
            beta.append(beta_locaitons[i][0])
            user_loc.append(beta_locaitons[i][1])

        beta = np.stack(beta, axis=0)
        user_loc = np.stack(user_loc, axis=0)
        V = init_V(beta.shape[0], num_user, beta, phi, args)

        data_object.append({
            'data_index': data_index,
            'num_user': num_user,
            'num_scenario': num_scenario_per_num_user_case,
            'data': {'beta': beta, 'user_loc': user_loc, 'phi': phi, 'V': V}
        })
        plot_trajectory(aploc, user_loc, data_index, num_user, args)

    return data_object


def init_Phii(num_user, args):
    num_scenario = args.num_scenario

    print('---> Generating new Phii <---')
    Phii = np.zeros(shape=(num_scenario, num_user, num_user))
    for i in range(num_scenario):
        U, sigma, vt = np.linalg.svd(np.random.randn(num_user, num_user))  # u include tau orthogonal sequences
        Phii[i] = np.copy(U)  # (K, K)

    return Phii


def init_V(num_scenario, num_user, BETAA, Phii, args):
    """
    Return the estimated channel
    @param BETAA: shape [num_scenario, num_steps, num_bs, num_user]
    @param Phii: shape [num_scenario, num_user, num_user]
    @param args: other arguments
    @return:
    """
    V = np.zeros(shape=(num_scenario, args.total_step, args.M, num_user))
    for i in range(num_scenario):
        for j in range(args.total_step):
            Var = np.zeros((args.M, num_user))
            mau = np.zeros((args.M, num_user))
            for m in range(args.M):
                for k in range(num_user):
                    mau[m, k] = (np.linalg.norm(
                        BETAA[i, j, m, :] ** 0.5 * (np.dot(Phii[i, :, k].T, Phii[i])))) ** 2

            for m in range(args.M):
                for k in range(num_user):
                    Var[m, k] = args.trtau * Pp * BETAA[i, j, m, k] ** 2 / (
                            args.trtau * Pp * mau[m, k] + 1)

            V[i, j, :, :] = Var

    return V


def cellsetting(maps, kters, sh_AP, sh_Ter, num_user, args):
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
    Ter = np.zeros((num_user, 2, 1))
    Ter[:, :, 0] = kters

    Z_shd = np.zeros((args.M, num_user))
    for m in range(args.M):
        for k in range(num_user):
            Z_shd[m, k] = sh_AP[m] + sh_Ter[k]

    BETAA = np.zeros((args.M, num_user))
    dist = np.zeros((args.M, num_user))
    for m in range(args.M):
        for k in range(num_user):
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
