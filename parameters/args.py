import argparse
from math import sqrt


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser(
        description='MASR', formatter_class=argparse.RawDescriptionHelpFormatter)
    # mappo = MARL-CTDE
    # ippo = MARL-IL
    # ppo = SARL
    # MARL-CTDE-shared = [--algorithm_name=mappo] + [--runner==shared]
    parser.add_argument('--algorithm_name', type=str, help='alg', default='mappo', choices=['mappo', 'ippo', 'ppo'])

    # mobile: experiment with mobile users but fixed number of users
    # dyn_mobile: mobile users with dynamic number of users
    # dyn_mobile_fix_train: Section 4 in the report, train with fixed number of user but test on dynamic number of users
    parser.add_argument("--scenario_name", type=str, default="mobile", choices=['mobile', 'static', 'dyn_mobile',
                                                                                'dyn_mobile_fix_train'],
                        help="an identifier to distinguish different experiment.")
    # number of generated scenarios (=200 in the experiments)
    parser.add_argument("--num_scenario", type=int, default=100,
                        help="number of mobile scenarios.")
    parser.add_argument('--min_rate', type=float, help='minimum rate requirement', default=8)
    parser.add_argument('--M', type=int, help='M: number of antenna', default=100)
    parser.add_argument('--K', type=int, help='K: number of users (agents)', default=50)
    parser.add_argument('--max_K', type=int, help='K: number of users (agents)', default=100)

    parser.add_argument('--D', type=float, help='D: size of the area', default=0.1)
    parser.add_argument('--bs_dist', type=str, help='Base station location type',
                        default='random', choices=['grid', 'random'])
    parser.add_argument('--sigma_shd', type=float, help='sigma_shd in dB', default=8)
    parser.add_argument('--trtau', type=int, help='trtau', default=10)
    parser.add_argument('--Lbeta', type=float, help='top largest beta', default=0.2)

    parser.add_argument('--total_step', type=int, help='total number of generated data (location of users)',
                        default=100)
    parser.add_argument('--train_size', type=float, help='number of data sample for training',
                        default=0.7)
    parser.add_argument('--max_veloc', type=float, help='number of data sample for training',
                        default=40)
    parser.add_argument('--schedule_time', type=float, help='scheduling time, if <= 0 --> using coherence time',
                        default=0.1)

    # training parameter
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=20,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=20,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e10,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--env_name", type=str, default='CellfreeMIMO', help="specify the name of environment")
    parser.add_argument("--episode_length", type=int,
                        default=1000, help="Max length for any episode")
    parser.add_argument("--device", type=str, default='cuda:0', help="specify the name of device",
                        choices=['cuda:0', 'cuda:1', 'cpu'])

    # for dynamic number of user scenario. The number of training rounds per case.
    parser.add_argument("--num_training_round", type=int,
                        default=1, help="The number of training rounds per case (dyn_mobile scenario)")
    parser.add_argument("--num_user_case", type=int,
                        default=10, help="The number of cases of dynamic users")
    parser.add_argument("--min_scenario_per_case", type=int,
                        default=5, help="Minimum number of scenario per case of num_user (in dyn_mobile)")

    # network parameters
    parser.add_argument("--runner", type=str, choices=['shared', 'federated', 'separated', 'federated_critic'],
                        default='dynamic_user_runner', help='Whether agent share the same policy')

    parser.add_argument("--use_centralized_V", action='store_false',
                        default=False, help="Whether to use centralized V function")

    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False,
                        help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        help='use a recurrent policy')
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficient of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    parser.add_argument('--use_rnn_policy', action='store_true', default=False,
                        help='use rnn policy')
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # save parameters
    parser.add_argument("--save_interval", type=int, default=1,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=True,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=1,
                        help="number of episodes of a single evaluation.")

    # parameter for dataset
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--obs_state', type=int, default=3,
                        help='State setting, (default 0)')
    parser.add_argument('--action_state', type=int, default=0, choices=[0, 1],
                        help='State setting, (default 0)')
    parser.add_argument('--num_power_level', type=int, default=20, help='State setting, (default 0)')

    parser.add_argument('--global_state', type=int, default=3,
                        help='State setting, (default 0)')
    parser.add_argument('--reward', type=str, help='reward function',
                        default='SF', choices=['PF1', 'PF2', 'SF', 'SFP', 'SF2', 'const', 'EE'])
    parser.add_argument('--reward_type', type=str, help='reward type',
                        default='global', choices=['global', 'local', 'combine'])
    parser.add_argument('--weight', type=float, help='reward weight', default=0.1)

    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--max_explore_step', type=int, default=100)
    # reward
    parser.add_argument('--restore', action='store_true', help='Load models and buffer from the latest checkpoint')
    parser.add_argument('--continuous_training', action='store_true', help='Keep training in the final test')
    parser.add_argument('--cuda_deterministic', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--dataset', type=str, default="abilene")
    parser.add_argument('--day_size',type=int , default=1)
    parser.add_argument('--input_len', type=int, default=1)
    parser.add_argument('--n_sr', type=int, default=2, choices=[2, 3],
                        help='n-segment path, (default 2)')
    parser.add_argument('--n_path', type=int, default=3)
    args = parser.parse_args()
    checking_parameter(args)

    if args.bs_dist == 'grid':
        n = int(sqrt(args.M))
        args.M = n ** 2

    if args.obs_state <= 3:
        args.Lbeta = 1.0

    if args.train_size > 0.7:
        args.train_size = 0.7

    args.test_size = 1.0 - args.train_size
    train_size = int(args.num_scenario * args.train_size)
    test_size = int(args.num_scenario * args.test_size)  # always test with last 30%

    n_rollout_threads = find_closest_divisor(train_size, target=args.n_rollout_threads)
    args.n_rollout_threads = n_rollout_threads

    if 'dyn_mobile' in args.scenario_name:
        min_scenario_per_case = args.min_scenario_per_case
        args.num_user_case = int(test_size/min_scenario_per_case)

        num_scenario_per_case = int(args.num_scenario / args.num_user_case)
        n_rollout_threads = find_closest_divisor(num_scenario_per_case, target=args.n_rollout_threads)
        args.n_rollout_threads = n_rollout_threads

        args.n_eval_rollout_threads = args.n_rollout_threads
        n_rollout_threads = find_closest_divisor(num_scenario_per_case, target=args.n_eval_rollout_threads)
        args.n_eval_rollout_threads = n_rollout_threads

    else:
        if args.algorithm_name == 'mappo' or args.algorithm_name == 'ippo':
            n_rollout_threads = find_closest_divisor(train_size, target=args.n_rollout_threads)
            args.n_rollout_threads = n_rollout_threads

            args.n_eval_rollout_threads = args.n_rollout_threads
            n_rollout_threads = find_closest_divisor(test_size, target=args.n_eval_rollout_threads)
            args.n_eval_rollout_threads = n_rollout_threads
        else:
            n_rollout_threads = find_closest_divisor(train_size, target=args.n_rollout_threads)
            args.n_rollout_threads = n_rollout_threads

            args.n_eval_rollout_threads = 1

    print(f'nvev: {args.n_rollout_threads}')
    print(f'nvev_eval: {args.n_eval_rollout_threads}')

    if args.scenario_name == 'mobile':
        episode_length = int(
            args.num_scenario * args.train_size * args.total_step / args.n_rollout_threads)
    else:
        episode_length = int(
            args.num_scenario * args.train_size / args.n_rollout_threads)

    args.episode_length = episode_length
    print(f'episode_length: {args.episode_length}')

    return args


def checking_parameter(args):
    if args.algorithm_name != 'mappo' and args.runner == 'shared':
        raise ValueError("Shared policy runner is only available for mappo")

    if args.algorithm_name == 'ippo' and args.global_state != 0:
        raise ValueError('ippo only accepts global_state = 0')

    if 'dyn_mobile' in args.scenario_name and args.runner == 'separated':
        raise ValueError(f'{args.scenario_name} only works with shared runner or federated runner')

    if args.test:
        args.restore = True
        args.num_scenario = args.num_scenario
        args.episode_length = 100
        args.ppo_epoch = 50


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


def update_args_dyn_user(data, num_user, num_scenario, args, is_train=False):
    args.data = data
    args.K = num_user
    args.num_scenario = num_scenario
    args.trtau = num_user  # training length
    args.num_agent = num_user
    if not is_train:
        if num_scenario <= args.n_eval_rollout_threads:
            args.n_eval_rollout_threads = num_scenario
        n_eval_rollout_threads = find_closest_divisor(num_scenario, target=args.n_eval_rollout_threads)
        args.n_eval_rollout_threads = n_eval_rollout_threads

        round_length = num_scenario * args.total_step // args.n_rollout_threads
        args.episode_length = round_length
    else:
        if num_scenario <= args.n_rollout_threads:
            args.n_rollout_threads = num_scenario
        n_rollout_threads = find_closest_divisor(num_scenario, target=args.n_rollout_threads)
        args.n_rollout_threads = n_rollout_threads

        round_length = num_scenario * args.total_step // args.n_rollout_threads
        args.episode_length = round_length

    return round_length
