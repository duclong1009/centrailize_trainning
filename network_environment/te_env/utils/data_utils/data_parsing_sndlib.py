import os
import xml.etree.ElementTree as ET

import numpy as np


def parse_data(dataset, adjust_capacity):
    print('Dataset: ', dataset)
    data_folder = '../../data'

    path_to_data_folder = os.path.join(data_folder, f'{dataset}_raw_xml/')
    path_to_data_folder_demands = os.path.join(data_folder, f'{dataset}_raw_xml/demands/')

    all_file_demands = os.listdir(path_to_data_folder_demands)
    all_file_demands.sort()

    node_names = []

    network_structure_file = os.path.join(path_to_data_folder, f'{dataset}.xml')
    file = os.path.join(path_to_data_folder, network_structure_file)

    tree = ET.parse(file)
    root = tree.getroot()
    for child in root:
        if 'networkStructure' in child.tag:
            for network_strucure_xml in child:
                if 'nodes' in network_strucure_xml.tag:
                    for node in network_strucure_xml:
                        node_names.append(node.attrib['id'])

    # print(node_names)
    num_node = len(node_names)
    # print('Num_nodes', len(node_names))

    adj_mx = np.zeros(shape=(num_node, num_node), dtype=int)
    cost_mx = np.zeros(shape=(num_node, num_node), dtype=float)
    capacity_mx = np.zeros(shape=(num_node, num_node), dtype=float)

    file = os.path.join(path_to_data_folder, network_structure_file)

    tree = ET.parse(file)
    root = tree.getroot()
    for child in root:
        if 'networkStructure' in child.tag:
            for network_strucure_xml in child:
                if 'links' in network_strucure_xml.tag:
                    for link in network_strucure_xml:
                        source = link[0].text
                        destination = link[1].text
                        srcID = node_names.index(source)
                        dstID = node_names.index(destination)

                        capacity = -1
                        cost = -1
                        for link_atribute in link:

                            if 'preInstalledModule' in link_atribute.tag and dataset == 'abilene':
                                capacity = float(link_atribute[0].text) / adjust_capacity
                                cost = float(link_atribute[1].text)
                            elif 'additionalModules' in link_atribute.tag and dataset == 'geant':
                                capacity = float(link_atribute[0][0].text) / adjust_capacity
                                cost = float(link_atribute[0][1].text)
                            elif 'additionalModules' in link_atribute.tag and dataset == 'nobel':
                                capacity = float(link_atribute[-1][0].text) / adjust_capacity
                                cost = float(link_atribute[-1][1].text)
                            elif 'additionalModules' in link_atribute.tag and dataset == 'germany':
                                capacity = float(link_atribute[0][0].text) / adjust_capacity
                                cost = float(link_atribute[0][1].text)

                        assert capacity >= 0
                        assert cost >= 0
                        adj_mx[srcID, dstID] = 1
                        capacity_mx[srcID, dstID] = capacity
                        cost_mx[srcID, dstID] = 10

                        adj_mx[dstID, srcID] = 1
                        capacity_mx[dstID, srcID] = capacity
                        cost_mx[dstID, srcID] = 10

    n_timesteps = len(all_file_demands)
    print('Total timesteps', n_timesteps)
    traffic_demands = np.zeros(shape=(n_timesteps, num_node, num_node), dtype=float)

    for t, filename in enumerate(all_file_demands):
        file = os.path.join(path_to_data_folder_demands, filename)

        tree = ET.parse(file)
        root = tree.getroot()
        for child in root:
            if 'demands' in child.tag:
                for demands in child:
                    source = demands[0].text
                    destination = demands[1].text
                    values = demands[2].text

                    srcID = node_names.index(source)
                    dstID = node_names.index(destination)

                    traffic_demands[t, srcID, dstID] = float(values)

    # print(traffic_demands.mean())

    data = {
        'adj_mx': adj_mx,
        'capacity_mx': capacity_mx,
        'cost_mx': cost_mx,
        'traffic_demands': traffic_demands
    }

    save_data_path = os.path.join(data_folder, f'{dataset}.npz')
    with open(save_data_path, 'wb') as fp:
        np.savez_compressed(fp, **data)

    # traffic_demands_15mins = []
    # if dataset == 'geant':
    #     step = 1
    # else:
    #     step = 3
    # for i in range(0, n_timesteps - step, step):
    #     traffic_demands_15mins.append(np.mean(traffic_demands[i:i + step], axis=0))
    #
    # traffic_demands_15mins = np.array(traffic_demands_15mins)
    # print(traffic_demands_15mins.shape)
    #
    # traffic_demands_15mins_means = np.mean(traffic_demands_15mins, axis=0)
    # traffic_demands_15mins_maxs = np.max(traffic_demands_15mins, axis=0)
    # traffic_demands_15mins_mins = np.min(traffic_demands_15mins, axis=0)
    #
    # total_traffic_per_step = np.sum(np.reshape(traffic_demands_15mins, newshape=(traffic_demands_15mins.shape[0], -1)),
    #                                 axis=-1)
    # mean_total_traffic = np.mean(total_traffic_per_step)
    # std_total_traffic = np.std(total_traffic_per_step)
    #
    # total_gen_step = 96 * 5
    # if total_gen_step > traffic_demands_15mins.shape[0]:
    #     total_traffic = np.sum(traffic_demands_15mins) * (int(total_gen_step / traffic_demands_15mins.shape[0]))
    # else:
    #     total_traffic = np.sum(traffic_demands_15mins[0:total_gen_step])
    #
    # traffic_demands_15mins_gen = modulated_gravity_tm(num_nodes=num_node, num_tms=total_gen_step,
    #                                                   mean_traffic=mean_total_traffic,
    #                                                   pm_ratio=1.5,
    #                                                   t_ratio=0.5,
    #                                                   diurnal_freq=1 / 96,
    #                                                   spatial_variance=50)
    # traffic_demands_15mins_syn = traffic_demands_15mins_gen.matrix
    # traffic_demands_15mins_syn = np.transpose(traffic_demands_15mins_syn, (2, 0, 1))
    # print(traffic_demands_15mins_syn.shape)
    # print(traffic_demands_15mins_syn.min(), traffic_demands.min())
    # print(traffic_demands_15mins_syn.max(), traffic_demands.max())
    # print(traffic_demands_15mins_syn.mean(), traffic_demands.mean())
    # print(traffic_demands_15mins_syn.std(), traffic_demands.std())
    # print(traffic_demands_15mins_syn.max() / traffic_demands_15mins_syn.mean(),
    #       traffic_demands.max() / traffic_demands.mean())
    #
    # data = {
    #     'adj_mx': adj_mx,
    #     'capacity_mx': capacity_mx,
    #     'cost_mx': cost_mx,
    #     'traffic_demands': traffic_demands_15mins_syn
    # }
    #
    # save_data_path = os.path.join(data_folder, f'{dataset}_syn.npz')
    # with open(save_data_path, 'wb') as fp:
    #     np.savez_compressed(fp, **data)


datasets = ['germany']
# datasets = ['abilene', 'nobel', 'germany', 'geant']
adjust_capacity = [0.05]
# adjust_capacity = [1, 1, 0.05, 5]
for i, dataset in enumerate(datasets):
    parse_data(dataset, adjust_capacity[i])
