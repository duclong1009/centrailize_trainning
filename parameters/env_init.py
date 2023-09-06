import os

import numpy as np
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


def plot_trajectory(aploc, trajec, args):
    """
        trajec shape (args.n_train_data, args.max_steps, args.K, 2)
    """
    path = os.path.join(args.data_folder, f'init_data/figures/{args.scenario_name}_M_{args.M}_K_{args.K}_D_{args.D}_'
                                          f'bs_{args.bs_dist}-{args.max_steps}-{args.total_step}')

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(min(100, trajec.shape[0])):
        plt.scatter(aploc[:, 0], aploc[:, 1])
        for k in range(args.K):
            if trajec[i, 0, k, 0] == trajec[i, -1, k, 0] and trajec[i, 0, k, 1] == trajec[i, -1, k, 1]:
                plt.scatter(trajec[i, 0, k, 0], trajec[i, 0, k, 1], color='black', marker='*')
            else:
                plt.scatter(trajec[i, 0, k, 0], trajec[i, 0, k, 1], color='red', marker='x')
                plt.plot(trajec[i, :, k, 0], trajec[i, :, k, 1], color='red')

        plt.savefig(os.path.join(path, 'trajectory_{}.png'.format(i)))
        plt.cla()


def init_location(args):
    num_scenario = args.num_scenario
    n_train_data = int(num_scenario * args.train_size)
    n_test_data = int(num_scenario * args.test_size)

    print('---> Generating new locations <---')
    terloc = np.zeros(shape=(num_scenario, args.K, 2))

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

    terloc_based = np.random.uniform(0, args.D, (args.K, 2))

    for i in range(num_scenario):
        mask = np.random.choice([0, 1], size=(args.K,), p=[0.3, 0.7])
        mask = np.asarray([mask, mask]).T
        terloc[i] = terloc_based * mask + (1 - mask) * np.random.uniform(0, args.D, (args.K, 2))

    return terloc, aploc


def init_shadowing(args):
    num_scenario = args.num_scenario

    print('---> Generating new shadowing <---')
    sh_AP = np.zeros(shape=(num_scenario, args.M, 1))
    sh_Ter = np.zeros(shape=(num_scenario, args.K, 1))
    for i in range(num_scenario):
        sh_AP[i] = 1 / (2 ** 0.5) * args.sigma_shd * np.random.randn(args.M, 1)  # (M, 1)
        sh_Ter[i] = 1 / (2 ** 0.5) * args.sigma_shd * np.random.randn(args.K, 1)  # (K, 1)

    return sh_AP, sh_Ter


def init_betaa(aploc, terloc, sh_AP, sh_Ter, args):
    num_scenario = args.num_scenario

    print('---> Generating new beta <---')
    BETAA = np.zeros(shape=(num_scenario, args.M, args.K))
    for i in range(num_scenario):
        BETAA[i] = cellsetting(aploc, terloc[i], sh_AP[i], sh_Ter[i], args)

    return BETAA


def init_Phii(args):
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


def init_data_static(args):
    path = os.path.join(args.data_folder, f'init_data/{args.scenario_name}_data_M_{args.M}_K_{args.K}_D_{args.D}_'
                                          f'bs_{args.bs_dist}-{args.total_step}.mat')

    if os.path.isfile(path):
        print('---> Loading saved locations <---')

        loc_data = loadmat(path)
        beta = loc_data['beta']
        user_loc = loc_data['user_loc']
        phi = loc_data['phi']
    else:

        user_loc, aploc = init_location(args)

        plot_location(aploc, user_loc, args)

        sh_AP, sh_Ter = init_shadowing(args)

        beta = init_betaa(aploc, user_loc, sh_AP, sh_Ter, args)

        phi = init_Phii(args)
        savemat(path, {'beta': beta, 'user_loc': user_loc, 'phi': phi})
    return beta, user_loc, phi
