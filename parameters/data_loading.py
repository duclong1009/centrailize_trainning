import os
import copy
from parameters.env_init import init_data_static
from parameters.env_init_dyn_mobile import init_data_dyn_mobile
from parameters.env_init_mobile import load_mobile_data
import pickle
import numpy as np
# from sklearn.preprocessing import MinMaxScaler

def get_data_size(dataset, debug=False):
    if 'abilene' in dataset:
        train_size = 1500
        test_size = 1500
    elif 'geant' in dataset:
        train_size = 500
        test_size = 500
    elif 'gnnet-75':
        train_size = 20
        test_size = 20
    elif 'gnnet-100':
        train_size = 20
        test_size = 20
    elif 'gnnet-40':
        train_size = 250
        test_size = 250
    elif 'ct-gen':
        train_size = 150
        test_size = 150
    else:
        raise NotImplementedError

    return train_size, test_size

class MinMaxScaler:
    def __init__(self, use_copy=True):
        self.min = np.inf
        self.max = -np.inf
        self.use_copy = use_copy
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

        if self.use_copy:
            _data = copy.deepcopy(data)
        else:
            _data = data

        scaled_data = (_data - self.min) / (self.max - self.min + 1e-10)
        return scaled_data

    def inverse_transform(self, data):
        if not self.fit_data:
            raise RuntimeError('Fit data first!')

        if self.use_copy:
            _data = copy.deepcopy(data)
        else:
            _data = data

        inverse_data = _data * (self.max - self.min + 1e-10) + self.min
        return inverse_data


def data_split_cs(data_traffic, train_size, scaler, n_node, return_data):
    cs_data_size = train_size if train_size < 500 else 500
    cs_data = data_traffic[0:cs_data_size]
    cs_data_scaled = scaler.transform(cs_data)

    cs_data = np.reshape(cs_data, newshape=(cs_data.shape[0], n_node, n_node))
    cs_data_scaled = np.reshape(cs_data_scaled, newshape=(cs_data_scaled.shape[0], n_node, n_node))

    return_data.update(
        {
            'cs_data/scaled': cs_data_scaled,
            'cs_data/gt': cs_data,
        }
    )
    return return_data


def load_data(args):
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

    data[data <= 0] = 1e-4
    data[data == np.nan] = 1e-4
    # Train-test split
    if 'abilene' in args.dataset or 'geant' in args.dataset:
        train_size, test_size = get_data_size(dataset=args.dataset, debug=False)
        total_steps = train_size + test_size
        data_traffic = data[:total_steps]

    else:
        total_steps = data.shape[0]
        data_traffic = data[:total_steps]
        train_size = int(total_steps * 0.5)
        test_size = total_steps - train_size

    train_size = int(train_size * args.train_size)  # in case of not use all training data

    if train_size % args.n_rollout_threads != 0:
        train_size = int(train_size / args.n_rollout_threads) * args.n_rollout_threads

    if test_size % args.n_eval_rollout_threads != 0:
        test_size = int(test_size / args.n_eval_rollout_threads) * args.n_eval_rollout_threads

    if int(train_size / args.n_rollout_threads) < 4:
        raise ValueError('|--- Too few matrices per threads. Reducing number of training threads')
    if int(test_size / args.n_eval_rollout_threads) < 4:
        raise ValueError('|--- Too few matrices per threads. Reducing number of eval threads')
    args.episode_length = train_size / args.n_rollout_threads
    train_df, test_df = data_traffic[0:train_size], data_traffic[-test_size:]  # total dataset

    sc = MinMaxScaler(use_copy=True)

    sc.fit(data_traffic)

    train_scaled = sc.transform(train_df)
    test_scaled = sc.transform(test_df)

    # Converting the time series to samples
    n_node = args.num_node
    train_df = np.reshape(train_df, newshape=(train_df.shape[0], n_node, n_node))
    test_df = np.reshape(test_df, newshape=(test_df.shape[0], n_node, n_node))
    train_scaled = np.reshape(train_scaled, newshape=(train_scaled.shape[0], n_node, n_node))
    test_scaled = np.reshape(test_scaled, newshape=(test_scaled.shape[0], n_node, n_node))

    return_data = {
        'train/scaled': train_scaled,
        'test/scaled': test_scaled,
        'scaler': sc,
        'train/gt': train_df,
        'test/gt': test_df,
    }
    if args.obs_state == 5:
        data_split_cs(data_traffic, train_size, sc, n_node, return_data)

    print('train data', train_df.shape)
    print('test data', test_df.shape)


    return return_data


def load_final_test_data(args):
    # loading dataset
    test_data_folder = os.path.join(args.data_folder, 'test_data')
    if not os.path.exists(test_data_folder):
        os.makedirs(test_data_folder)

    if 'dyn_mobile' in args.scenario_name:
        list_of_data = init_data_dyn_mobile(args)
        test_data = list_of_data

        data = {
            'test': test_data
        }

    else:
        if args.scenario_name == 'mobile':
            BETAA, user_loc, Phii, V = load_mobile_data(args, is_test=True)
            V_test = V

        else:
            BETAA, user_loc, Phii = init_data_static(args)
            V_train, V_test = None, None

        print('Scenario', args.scenario_name)
        print('BETAA', BETAA.shape)
        print('user_loc', user_loc.shape)
        print('Phii', Phii.shape)

        BETAA_test = BETAA
        Phii_test = Phii

        data = {
            'test/BETAA': BETAA_test,
            'test/Phii': Phii_test,
            'test/V': V_test,
        }

    return data


def prepare_dyn_data(data, mode='train'):
    num_user = data['num_user']
    num_scenario = data['num_scenario']
    BETAA = data['data']['beta']
    user_loc = data['data']['user_loc']
    Phii = data['data']['phi']
    V = data['data']['V']

    sub_data = {
        f'{mode}/BETAA': BETAA,
        f'{mode}/Phii': Phii,
        f'{mode}/V': V,
    }

    return sub_data, num_user, num_scenario
