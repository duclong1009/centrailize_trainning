import os

import matplotlib.pyplot as plt
import numpy as np

data_folder = '../../data/'

DATASETS = ['abilene', 'geant', 'nobel', 'germany', 'gnnet-75', 'gnnet-100']
DAYSIZES = [288, 96, 288, 288, 288, 288]


def plot_total_traffic(dataset, day_size):
    data_path = os.path.join(data_folder, f'{dataset}.npz')

    data = np.load(data_path)
    traffic_demands = data['traffic_demands']
    traffic_demands = np.reshape(traffic_demands, newshape=(traffic_demands.shape[0], -1))
    total_traffic = np.sum(traffic_demands, axis=-1)

    total_steps = day_size * 7 * 2 if day_size * 7 * 2 < traffic_demands.shape[0] else traffic_demands.shape[0]
    total_traffic = total_traffic[:total_steps]
    x = np.arange(total_steps)

    plt.plot(x, total_traffic)
    save_folder = os.path.join(data_folder, f'data_analysis')
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f'total_traffic_{dataset}.png')
    plt.savefig(save_file, dpi=200)
    plt.close()


def plot_topk_flows(dataset, day_size):
    data_path = os.path.join(data_folder, f'{dataset}.npz')

    data = np.load(data_path)
    traffic_demands = data['traffic_demands']
    traffic_demands = np.reshape(traffic_demands, newshape=(traffic_demands.shape[0], -1))

    total_steps = day_size * 7 * 2 if day_size * 7 * 2 < traffic_demands.shape[0] else traffic_demands.shape[0]
    traffic_demands = traffic_demands[:total_steps]

    mean_traffic_flows = np.mean(traffic_demands, axis=0)

    index_topk = np.argsort(mean_traffic_flows)[::-1][:5]

    x = np.arange(total_steps)
    for idx in index_topk:
        plt.plot(x, traffic_demands[:, idx], label=f'flow_{idx}')

    save_folder = os.path.join(data_folder, f'data_analysis')
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f'topk_traffic_{dataset}.png')
    plt.savefig(save_file, dpi=200)
    plt.close()


for i, dataset in enumerate(DATASETS):
    plot_total_traffic(dataset, day_size=DAYSIZES[i])
    plot_topk_flows(dataset, day_size=DAYSIZES[i])
