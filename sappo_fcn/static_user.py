# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


"""
torch = 0.41
"""
import sys

import numpy as np
import torch
sys.path.append('../')

import os.path

import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from network_environment.te_env.env import MasrNodeEnv
sns.set_theme()
from stable_baselines3 import DDPG, A2C, PPO, SAC, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from parameters.utils import info_keywords
from sappo_fcn.utils.utils import load_model, find_closest_divisor
from stable_baselines3.common.utils import set_random_seed

from scipy.io import savemat

import pickle


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir: str, writer, save_path, alg, episode_length, verbose=0, resume_step=0, nenv=10):
        # self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = writer
        self.best_mean_reward = -np.inf
        self.save_path = save_path
        self.alg = alg

        self.resume_step = resume_step
        self.last_reward = -np.Inf
        self.episode_length = episode_length
        self.nenv = nenv

        self.episode_count = 0
        self.log_data = {}
        self.log_data_path = os.path.join(self.log_dir, 'log_data_train.pkl')

    def _on_step(self) -> bool:

        # Log additional tensor
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        data = load_results(self.log_dir)
        if len(x) > 0:
            mean_reward = np.mean(y[-self.nenv:]) / self.episode_length
            if mean_reward != self.last_reward:
                self.writer.add_scalar('Train/Global_Reward', mean_reward, self.episode_count)
                rate = np.mean(data.Sum_Rate.values[-self.nenv:])
                power = np.mean(data.Sum_Power_Usage.values[-self.nenv:])
                gu = np.mean(data.Per_Satisfied_User.values[-self.nenv:])

                self.writer.add_scalar('Train/Sum_Rate', rate, self.episode_count)
                self.writer.add_scalar('Train/Sum_Power_Usage', power, self.episode_count)
                self.writer.add_scalar('Train/Per_Satisfied_User', gu, self.episode_count)

                # # New best model, you could save the agent here
                # if mean_reward > self.best_mean_reward:
                #     self.best_mean_reward = mean_reward
                #     # Example for saving best model
                #     if self.verbose > 0:
                #         print(f"Saving new best model to {self.save_path}.zip")
                self.model.save(self.save_path)
                self.last_reward = mean_reward

                if 'Train/Global_Reward' not in self.log_data.keys():
                    self.log_data['Train/Global_Reward'] = [mean_reward]
                    self.log_data['Train/Sum_Rate'] = [rate]
                    self.log_data['Train/Sum_Power_Usage'] = [power]
                    self.log_data['Train/Per_Satisfied_User'] = [gu]
                    self.log_data['Train/step'] = [self.episode_count]
                else:

                    self.log_data['Train/Global_Reward'].append(mean_reward)
                    self.log_data['Train/Sum_Rate'].append(rate)
                    self.log_data['Train/Sum_Power_Usage'].append(power)
                    self.log_data['Train/Per_Satisfied_User'].append(gu)
                    self.log_data['Train/step'].append(self.episode_count)

                self.episode_count += 1

                with open(self.log_data_path, 'wb') as fp:
                    pickle.dump(self.log_data, fp)

        return True


class RunTestCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env, K, log_dir: str, writer, alg='SAC', check_freq=100, verbose=0, resume_step=0,
                 device='cuda:0'):
        # self.is_tb_set = False
        super(RunTestCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = writer
        self.env = env
        self.check_freq = check_freq
        self.alg = alg
        self.K = K

        self.resume_step = resume_step
        self.log_rewards = []

        self.device = device

        self.episode_count = 0
        self.log_data = {}
        self.log_data_path = os.path.join(self.log_dir, 'log_data_test.pkl')

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq != 0:
            return True

        # Log additional tensor
        model_path = os.path.join(self.log_dir, 'models.zip')
        if not os.path.isfile(model_path):
            return True

        if self.alg == 'sac':
            model = SAC.load(model_path,
                             device=self.device)
        elif self.alg == 'a2c':
            model = A2C.load(model_path,
                             device=self.device)
        elif self.alg == 'ddpg':
            model = DDPG.load(model_path,
                              device=self.device)
        elif self.alg == 'ppo':
            model = PPO.load(model_path,
                             device=self.device)
        elif self.alg == 'dqn':
            model = DQN.load(model_path,
                             device=self.device)
        else:
            raise NotImplementedError('Not supported!')

        obs = self.env.reset()
        done = False

        sum_rate, sum_power, sum_good_user, sum_reward = [], [], [], []
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            sum_reward.append(rewards)
            sum_rate.append(info['Sum_Rate'])
            sum_power.append(info['Sum_Power_Usage'])
            sum_good_user.append(info['Per_Satisfied_User'])

        sum_reward = np.asarray(sum_reward).mean()
        sum_rate = np.asarray(sum_rate).mean()
        sum_power = np.asarray(sum_power).mean()
        sum_good_user = np.asarray(sum_good_user).mean()

        self.writer.add_scalar('Test/Global_Reward', sum_reward, self.episode_count)
        self.writer.add_scalar('Test/Sum_Rate', sum_rate, self.episode_count)
        self.writer.add_scalar('Test/Sum_Power_Usage', sum_power, self.episode_count)
        self.writer.add_scalar('Test/Per_Satisfied_User', sum_good_user, self.episode_count)

        self.log_rewards.append([sum_reward, sum_rate, sum_power, sum_good_user])
        np.savetxt(os.path.join(self.log_dir, 'monitor_test.txt'), np.asarray(self.log_rewards), delimiter=',')

        if 'Train/Global_Reward' not in self.log_data.keys():
            self.log_data['Test/Global_Reward'] = [sum_reward]
            self.log_data['Test/Sum_Rate'] = [sum_rate]
            self.log_data['Test/Sum_Power_Usage'] = [sum_power]
            self.log_data['Test/Per_Satisfied_User'] = [sum_good_user]
            self.log_data['Test/step'] = [self.episode_count]
        else:

            self.log_data['Test/Global_Reward'].append(sum_reward)
            self.log_data['Test/Sum_Rate'].append(sum_rate)
            self.log_data['Test/Sum_Power_Usage'].append(sum_power)
            self.log_data['Test/Per_Satisfied_User'].append(sum_good_user)
            self.log_data['Test/step'].append(self.episode_count)
        self.episode_count += 1

        with open(self.log_data_path, 'wb') as fp:
            pickle.dump(self.log_data, fp)

        return True


def make_env(rank, args, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = MasrNodeEnv(rank=rank, args=args, is_eval=False)
        # env = Monitor(env, os.path.join(f'../logs/{args.algorithm_name}/', args.experiment_name, f'monitor_{rank}'),
        #               info_keywords=info_keywords)
        check_env(env)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def run_static_user(data, run_dir, args):
    writer = SummaryWriter(log_dir=run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    model_dir = str(run_dir / 'models')

    # ------------------------------------- init location ap/user  -------------------------------------
    args.data = data
    episode_length = args.episode_length

    if not args.test:
        eval_env = MasrNodeEnv(rank=0, args=args, is_eval=True)
        env = SubprocVecEnv([make_env(rank=i, args=args) for i in range(args.n_rollout_threads)])


        # ------------------------------------- init alg  -------------------------------------
        print('---> Training new model')

        if args.algorithm_name == 'sac':
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            model = SAC('MlpPolicy', env, verbose=1, action_noise=action_noise, gamma=args.gamma,
                        device=args.device)
        elif args.algorithm_name == 'a2c':
            model = A2C('MlpPolicy', env, verbose=1, gamma=args.gamma, device=args.device)
        elif args.algorithm_name == 'ppo':
            pi = []
            vf = []
            for _ in range(args.layer_N):
                pi.append(args.hidden_size)
                vf.append(args.hidden_size)
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=dict(pi=pi, vf=vf))

            batch_size = find_closest_divisor(episode_length * args.n_rollout_threads, target=64)
            model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                        verbose=1, gamma=args.gamma, device=args.device, n_steps=episode_length,
                        batch_size=batch_size)
        elif args.algorithm_name == 'dqn':
            model = DQN('MlpPolicy', env, verbose=1, gamma=args.gamma, device=args.device)
        elif args.algorithm_name == 'ddpg':
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, gamma=args.gamma, device=args.device,
                         train_freq=(episode_length, 'step'))
        else:
            raise NotImplementedError('Not supported!')

        model.learn(total_timesteps=args.num_env_steps,
                    callback=[TensorboardCallback(log_dir=run_dir, writer=writer, save_path=model_dir,
                                                  alg=args.algorithm_name,
                                                  episode_length=episode_length,
                                                  nenv=args.n_rollout_threads),
                              RunTestCallback(env=eval_env, K=args.K, log_dir=run_dir,
                                              alg=args.algorithm_name,
                                              writer=writer,
                                              check_freq=episode_length,
                                              device=args.device)],
                    log_interval=1
                    )

        del model

    # run test
    eval_env = CellfreeSARLStaticEnv(rank=0, args=args, is_eval=True)

    print('---> Run test')
    # Loading best model
    model = load_model(eval_env, log_dir=model_dir, args=args)
    # Run final test
    obs = eval_env.reset()
    done = False

    sum_rate, sum_power, sum_good_user, sum_reward = [], [], [], []
    user_rate, min_rate = [], []
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = eval_env.step(action)
        sum_reward.append(rewards)
        sum_rate.append(info['Rate'])
        sum_power.append(info['Power'])
        sum_good_user.append(info['Good_user'])
        user_rate.append(info['User_rate'])
        min_rate.append(info['min_rate'])

    sum_reward = np.asarray(sum_reward)
    sum_rate = np.asarray(sum_rate)
    sum_power = np.asarray(sum_power)
    sum_good_user = np.asarray(sum_good_user)
    user_rate = np.asarray(user_rate)
    min_rate = np.asarray(min_rate)

    # saving test results
    np.save(os.path.join(run_dir, 'Sum_rate'), sum_rate)
    np.save(os.path.join(run_dir, 'Sum_reward'), sum_reward.mean())
    np.save(os.path.join(run_dir, 'Sum_power'), sum_power)
    np.save(os.path.join(run_dir, 'Sum_good_user'), sum_good_user)
    np.save(os.path.join(run_dir, 'Min_rate'), min_rate)

    np.save(os.path.join(run_dir, 'User_rate'), user_rate)
    print('user rate shape:', user_rate.shape)
    savemat(os.path.join(run_dir, 'User_rate.mat'), {'user_rate': user_rate})