import os.path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG, A2C, PPO, SAC, DQN


def plot_results(sarl_results, uniform_results,
                 figures_dir, m_rate):
    for mr in m_rate:
        Y = np.arange(0, sarl_results['sum_rate_{}'.format(mr)].shape[0])
        plt.plot(Y, sarl_results['sum_rate_{}'.format(mr)], label='DDPG-min_rate_{}'.format(mr))
        plt.plot(Y, uniform_results['sum_rate_{}'.format(mr)], label='uniform max min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Rate (Mbits/s')
    fname = 'sumrate.png',
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    for mr in m_rate:
        Y = np.arange(0, sarl_results['sum_power_{}'.format(mr)].shape[0])
        plt.plot(Y, sarl_results['sum_power_{}'.format(mr)], label='DDPG-min_rate_{}'.format(mr))
        plt.plot(Y, uniform_results['sum_power_{}'.format(mr)], label='uniform max min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Power (Mbits/s')
    fname = 'sumpower.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    for mr in m_rate:
        Y = np.arange(0, sarl_results['sum_reward_{}'.format(mr)].shape[0])
        plt.plot(Y, sarl_results['sum_reward_{}'.format(mr)], label='DDPG-min_rate_{}'.format(mr))
        plt.plot(Y, uniform_results['sum_reward_{}'.format(mr)], label='uniform max min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Reward (Mbits/s')
    fname = 'sumreward.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    for mr in m_rate:
        Y = np.arange(0, sarl_results['sum_good_user_{}'.format(mr)].shape[0])
        plt.plot(Y, sarl_results['sum_good_user_{}'.format(mr)], label='DDPG-min_rate_{}'.format(mr))
        plt.plot(Y, uniform_results['sum_good_user_{}'.format(mr)], label='uniform max min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Good user (Mbits/s')
    fname = 'sumgooduser.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    for mr in m_rate:
        Y = np.arange(0, sarl_results['min_rate_{}'.format(mr)].shape[0])
        plt.plot(Y, sarl_results['min_rate_{}'.format(mr)], label='DDPG-min_rate_{}'.format(mr))
        plt.plot(Y, uniform_results['min_rate_{}'.format(mr)], label='uniform max min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Minimum Rate (Mbits/s')
    fname = 'summinrate.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    # YY = np.linspace(1, args.max_eps, args.max_eps)
    # plt.figure(7)
    # plt.plot(YY, EPreward)
    # plt.legend(loc='best')
    # plt.xlabel('Episodes')
    # plt.ylabel('Average reward')
    # fname = 'Average reward.png'
    # fname = os.path.join(figures_dir, fname)
    # plt.savefig(fname)
    # plt.cla()


def plot_training(sarl_results, uniform_results,
                  figures_dir, m_rate):
    # plotting Reward
    Y = np.arange(0, sarl_results['training_reward_{}'.format(m_rate[0])].shape[0])

    for mr in m_rate:
        plt.plot(Y, sarl_results['training_reward_{}'.format(mr)][:, 0], label='DDPG-min_rate_{}'.format(mr))
        plt.plot(Y, uniform_results['training_reward_{}'.format(mr)][:, 0], label='uniform max min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    fname = 'training_reward.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    # plotting Sum rate
    plt.plot(Y, uniform_results['training_reward_{}'.format(m_rate[0])][:, 1],
             label='uniform max')
    for mr in m_rate:
        plt.plot(Y, sarl_results['training_reward_{}'.format(mr)][:, 1], label='DDPG-min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Sum Rate (Mbits/s)')
    fname = 'training_sum_rate.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    # plotting Power
    plt.plot(Y, uniform_results['training_reward_{}'.format(m_rate[0])][:, 2],
             label='uniform max')
    for mr in m_rate:
        plt.plot(Y, sarl_results['training_reward_{}'.format(mr)][:, 2], label='DDPG-min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Power (Watt)')
    fname = 'training_power.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()

    # plotting Good user
    for mr in m_rate:
        plt.plot(Y, sarl_results['training_reward_{}'.format(mr)][:, 3], label='DDPG-min_rate_{}'.format(mr))
        plt.plot(Y, uniform_results['training_reward_{}'.format(mr)][:, 3], label='uniform max min_rate_{}'.format(mr))
    plt.legend(loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Number of good users')
    fname = 'training_good_user.png'
    fname = os.path.join(figures_dir, fname)
    plt.savefig(fname)
    plt.cla()


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def load_model(env, log_dir, args):
    if args.alg == 'SAC':
        model = SAC.load(os.path.join(log_dir, 'best_model_{}_{}'.format(args.alg, args.K)), env=env,
                         device=args.device)
    elif args.alg == 'A2C':
        model = A2C.load(os.path.join(log_dir, 'best_model_{}_{}'.format(args.alg, args.K)), env=env,
                         device=args.device)
    elif args.alg == 'DDPG':
        model = DDPG.load(os.path.join(log_dir, 'best_model_{}_{}'.format(args.alg, args.K)), env=env,
                          device=args.device)
    elif args.alg == 'PPO':
        model = PPO.load(os.path.join(log_dir, 'best_model_{}_{}'.format(args.alg, args.K)), env=env,
                         device=args.device)
    elif args.alg == 'DQN':
        model = DQN.load(os.path.join(log_dir, 'best_model_{}_{}'.format(args.alg, args.K)), env=env,
                         device=args.device)
    else:
        raise NotImplementedError('Not supported!')

    return model


import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def make_custom_env(env, rank, seed, monitor_dir, mon_info_keywords):
    def _init(env):
        if seed is not None:
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
        # Wrap the env in a Monitor wrapper
        # to have additional training information
        # monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        # Create the monitor folder if needed
        # if monitor_path is not None:
        #     os.makedirs(monitor_dir, exist_ok=True)
        # env = Monitor(env, filename=monitor_path, info_keywords=mon_info_keywords)
        # Optionally, wrap the environment with the provided wrapper
        return env

    return _init(env)


def make_vec_customenv(
        env_id,
        n_envs,
        seed=None,
        start_index=0,
        monitor_dir=None,
        wrapper_class=None,
        env_kwargs=None,
        vec_env_cls=None,
        vec_env_kwargs=None,
        monitor_kwargs=None,
        wrapper_kwargs=None,
):
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs[rank])
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def find_closest_divisor(number, target):
    closest_divisor = None
    min_diff = float('inf')
    for divisor in range(1, number + 1):
        if number % divisor == 0:
            diff = abs(target - divisor)
            if diff < min_diff:
                min_diff = diff
                closest_divisor = divisor
    return closest_divisor
