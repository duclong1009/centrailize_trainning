import copy
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from network_environment.te_env.utils.compressive_sensing.ksvd import KSVD


# from sklearn.preprocessing import MinMaxScaler


class MinMaxScaler:
    def __init__(self, copy=True):
        self.min = np.inf
        self.max = -np.inf
        self.copy = copy
        self.fit_data = False

    def fit(self, data):
        self.min = np.min(data) if np.min(data) < self.min else self.min
        self.max = np.max(data) if np.max(data) > self.max else self.max
        self.fit_data = True

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        if not self.fit_data:
            raise RuntimeError('Fit data first!')

        if self.copy:
            _data = np.copy(data)
        else:
            _data = data

        scaled_data = (_data - self.min) / (self.max - self.min + 1e-10)
        return scaled_data

    def inverse_transform(self, data):
        if not self.fit_data:
            raise RuntimeError('Fit data first!')

        if self.copy:
            _data = np.copy(data)
        else:
            _data = data

        inverse_data = _data * (self.max - self.min + 1e-10) + self.min
        return inverse_data


def prepare_tm_cycle(data, routing_cycle, len_tra):
    x = []
    for t in range(len_tra, data.shape[0], routing_cycle):
        tm_cycle = data[t - len_tra:t + routing_cycle]
        if tm_cycle.shape[0] == routing_cycle + len_tra:
            x.append(tm_cycle)

    x = np.stack(x, axis=0)
    return x


def get_data_size(dataset, day_size):
    if 'abilene' in dataset:
        # train_size = 1500
        # test_size = 1500
        train_size = 300
        test_size = 300
    elif 'geant' in dataset:
        train_size = 500
        test_size = 500
    else:
        raise NotImplementedError
    return train_size, test_size


def data_split(args):
    path = os.path.join(args.data_folder, f'{args.dataset}.npz')

    all_data = np.load(path)
    data = all_data['traffic_demands']

    if len(data.shape) > 2:
        data = np.reshape(data, newshape=(data.shape[0], -1))

    # calculate num node
    T, F = data.shape
    N = int(np.sqrt(F))
    args.num_node = N
    args.num_flow = F
    # print('Data shape', data.shape)

    data[data <= 0] = 1e-4
    data[data == np.nan] = 1e-4
    # Train-test split
    if 'abilene' in args.dataset or 'geant' in args.dataset:
        train_size, test_size = get_data_size(dataset=args.dataset, day_size=args.day_size)
        total_steps = train_size + test_size
        data_traffic = data[:total_steps]

    else:
        total_steps = data.shape[0]
        data_traffic = data[:total_steps]
        train_size = int(total_steps * 0.5)
        test_size = total_steps - train_size

    # total_steps = 100 if data.shape[0] > 100 else data.shape[0]
    # data_traffic = data[:total_steps]
    # train_size = int(total_steps * 0.7)
    # test_size = total_steps - train_size

    train_size = int(train_size * args.train_size)  # in case of not use all training data
    cs_data_size = train_size if train_size < 500 else 500

    if train_size % args.n_rollout_threads != 0:
        train_size = int(train_size / args.n_rollout_threads) * args.n_rollout_threads

    if test_size % args.n_eval_rollout_threads != 0:
        test_size = int(test_size / args.n_eval_rollout_threads) * args.n_eval_rollout_threads

    if int(train_size / args.n_rollout_threads) < 4:
        raise ValueError('|--- Too few matrices per threads. Reducing number of training threads')
    if int(test_size / args.n_eval_rollout_threads) < 4:
        raise ValueError('|--- Too few matrices per threads. Reducing number of eval threads')

    train_df, test_df = data_traffic[0:train_size], data_traffic[-test_size:]  # total dataset
    cs_data = data_traffic[0:cs_data_size]

    sc = MinMaxScaler(copy=True)
    sc.fit(data_traffic)

    train_scaled = sc.transform(train_df)
    test_scaled = sc.transform(test_df)
    cs_data_scaled = sc.transform(cs_data)

    # Converting the time series to samples
    n_node = args.num_node
    train_df = np.reshape(train_df, newshape=(train_df.shape[0], n_node, n_node))
    test_df = np.reshape(test_df, newshape=(test_df.shape[0], n_node, n_node))
    cs_data = np.reshape(cs_data, newshape=(cs_data.shape[0], n_node, n_node))
    train_scaled = np.reshape(train_scaled, newshape=(train_scaled.shape[0], n_node, n_node))
    test_scaled = np.reshape(test_scaled, newshape=(test_scaled.shape[0], n_node, n_node))
    cs_data_scaled = np.reshape(cs_data_scaled, newshape=(cs_data_scaled.shape[0], n_node, n_node))

    return_data = {
        'train/scaled': train_scaled,
        'test/scaled': test_scaled,
        'cs_data/scaled': cs_data_scaled,
        'scaler': sc,
        'train/gt': train_df,
        'test/gt': test_df,
        'cs_data/gt': cs_data,
    }

    print('train data', train_df.shape)
    print('test data', test_df.shape)

    return return_data


def get_psi(args, data):
    os.makedirs(os.path.join(args.data_folder, 'compressive_sensing/'), exist_ok=True)
    save_path = os.path.join(args.data_folder, f'compressive_sensing/psi_{args.dataset}_{args.train_size}.pkl')
    if not os.path.exists(save_path):

        X = copy.deepcopy(data['cs_data/gt'])
        if len(X.shape) > 2:
            X = np.reshape(X, newshape=(X.shape[0], -1))

        N_F = args.num_node * args.num_node
        D = np.zeros(shape=(N_F, N_F))

        psiT, ST = KSVD(D).fit(X)
        obj = {
            'psiT': psiT,
            'ST': ST
        }

        with open(save_path, 'wb') as fp:
            pickle.dump(obj, fp)
            fp.close()

    else:
        with open(save_path, 'rb') as fp:
            obj = pickle.load(fp)
            fp.close()
        psiT = obj['psiT']

    data.update({'psi': psiT})
    return


def get_phi(top_k_index, nseries):
    G = np.zeros((top_k_index.shape[0], nseries))

    for i, j in enumerate(G):
        j[top_k_index[i]] = 1

    return G


def load_data(args):
    # loading dataset
    data = data_split(args=args)
    if args.obs_state == 5:
        get_psi(args, data)
    return data


def load_raw_data_tm_pred(args):
    data_path = os.path.join(args.data_folder, f'tm_pred_data/{args.dataset}.npz')
    if not os.path.exists(data_path):
        n = 100
        traffic_matrix_data = []
        link_util = []
        label = []
        for i in range(n):
            data_path_i = os.path.join(args.data_folder, f'tm_pred_data/{args.dataset}_rank_{i}.npz')
            try:
                data_i = np.load(data_path_i)
            except:
                break
            traffic_matrix_data.append(data_i['x_tm'])
            link_util.append(data_i['x_link_util'])
            label.append(data_i['y'])

        traffic_matrix_data = np.concatenate(traffic_matrix_data, axis=0)
        link_util = np.concatenate(link_util, axis=0)
        label = np.concatenate(label, axis=0)
        data = {
            'x_tm': traffic_matrix_data,
            'x_link_util': link_util,
            'y': label
        }

        with open(data_path, 'wb') as fp:
            np.savez_compressed(fp, **data)

    else:
        data = np.load(data_path)

    return data


def multi_zip(X, Y, Z):
    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            yield X[i, j], Y[i], Z[i]


def load_data_tm_pred(args):
    data = load_raw_data_tm_pred(args)
    x_tm = data['x_tm']
    x_link_util = data['x_link_util']
    y = data['y']

    print(f'|--- Number of sample: {x_tm.shape[0]}')

    scaler = MinMaxScaler(copy=True)
    scaler.fit(y)
    x_tm = scaler.transform(x_tm)
    y = scaler.transform(y)

    shuffle_idx = np.arange(0, x_tm.shape[0])
    np.random.shuffle(shuffle_idx)
    x_tm = x_tm[shuffle_idx]
    x_link_util = x_link_util[shuffle_idx]
    y = y[shuffle_idx]

    train_size = int(x_tm.shape[0] * 0.6)
    val_size = int(x_tm.shape[0] * 0.1)

    x_tm_train, x_tm_val, x_tm_test = x_tm[:train_size], \
        x_tm[train_size:train_size + val_size], \
        x_tm[-(train_size + val_size):]
    x_link_util_train, x_link_util_val, x_link_util_test = x_link_util[:train_size], \
        x_link_util[train_size:train_size + val_size], \
        x_link_util[-(train_size + val_size):]
    y_train, y_val, y_test = y[:train_size], \
        y[train_size:train_size + val_size], \
        y[-(train_size + val_size):]

    x_tm_train = torch.from_numpy(x_tm_train).to(dtype=torch.float32, device=args.device)
    x_tm_val = torch.from_numpy(x_tm_val).to(dtype=torch.float32, device=args.device)
    x_tm_test = torch.from_numpy(x_tm_test).to(dtype=torch.float32, device=args.device)
    x_link_util_train = torch.from_numpy(x_link_util_train).to(dtype=torch.float32, device=args.device)
    x_link_util_val = torch.from_numpy(x_link_util_val).to(dtype=torch.float32, device=args.device)
    x_link_util_test = torch.from_numpy(x_link_util_test).to(dtype=torch.float32, device=args.device)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float32, device=args.device)
    y_val = torch.from_numpy(y_val).to(dtype=torch.float32, device=args.device)
    y_test = torch.from_numpy(y_test).to(dtype=torch.float32, device=args.device)

    train_loader = DataLoader(list(multi_zip(x_tm_train, x_link_util_train, y_train)), shuffle=False,
                              batch_size=64, drop_last=True)
    val_loader = DataLoader(list(multi_zip(x_tm_val, x_link_util_val, y_val)), shuffle=False,
                            batch_size=64, drop_last=True)
    test_loader = DataLoader(list(multi_zip(x_tm_test, x_link_util_test, y_test)), shuffle=False,
                             batch_size=64, drop_last=True)
    loader = {'train': train_loader,
              'val': val_loader,
              'test': test_loader}

    return loader


def get_scaler_tm_pred_data(args):
    data = load_raw_data_tm_pred(args)
    x_tm = data['x_tm']
    x_link_util = data['x_link_util']
    y = data['y']

    print(f'|--- Number of sample: {x_tm.shape[0]}')

    scaler = MinMaxScaler(copy=True)
    scaler.fit(y)
    return scaler
