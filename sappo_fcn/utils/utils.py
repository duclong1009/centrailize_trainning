import os.path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG, A2C, PPO, SAC, DQN



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
