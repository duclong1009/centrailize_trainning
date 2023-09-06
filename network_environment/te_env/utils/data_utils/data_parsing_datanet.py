import itertools
import os

import numpy as np

from datanetAPI import DatanetAPI

num_node = 100

data_folder = '../../data'

path_to_data_folder = os.path.join(data_folder, f'gnnet-data/{num_node}/')
reader = DatanetAPI(path_to_data_folder)
it = iter(reader)

dataset = f'gnnet-{num_node}'
adj_mx = np.zeros(shape=(num_node, num_node), dtype=int)
capacity_mx = np.zeros(shape=(num_node, num_node), dtype=float)
cost_mx = np.zeros(shape=(num_node, num_node), dtype=float)

for sample in it:
    data_file_break = str.split(sample.data_set_file, '-')
    steps = str.split(str.split(data_file_break[-1], '.')[0], '_')
    start_step = int(steps[1])
    if f'results_{num_node}' in sample.data_set_file:
        for srcID, dstID in itertools.product(np.arange(num_node), np.arange(num_node)):
            link = sample.get_link_properties(srcID, dstID)
            if link is not None:
                adj_mx[srcID, dstID] = 1
                capacity_mx[srcID, dstID] = float(link['bandwidth'])
                cost_mx[srcID, dstID] = float(link['weight'])
        break

t = 0
max_t = 24
total_steps = 300
steps_count = 0

traffic_demands = np.zeros(shape=(total_steps, num_node, num_node), dtype=float)
for sample in it:
    data_file_break = str.split(sample.data_set_file, '-')
    steps = str.split(str.split(data_file_break[-1], '.')[0], '_')
    start_step = int(steps[1])
    if start_step < total_steps - max_t:
        for srcID, dstID in itertools.product(np.arange(num_node), np.arange(num_node)):
            traffic_demands[start_step + t, srcID, dstID] = float(
                sample.get_srcdst_traffic(srcID, dstID)['AggInfo']['AvgBw'])
        t += 1
        if t > max_t:
            t = 0

        steps_count += 1
    else:
        t = 0

traffic_demands = traffic_demands[0:steps_count + 1]
# print(traffic_demands.shape)
data = {
    'adj_mx': adj_mx,
    'capacity_mx': capacity_mx,
    'cost_mx': cost_mx,
    'traffic_demands': traffic_demands
}

save_data_path = os.path.join(data_folder, f'{dataset}.npz')
with open(save_data_path, 'wb') as fp:
    np.savez_compressed(fp, **data)
