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
    parser.add_argument('--runner_type', type=str, default="qos_rl", choices=['qos_rl', 'topk'])
    parser.add_argument('--algorithm_name', type=str, help='alg', default='ppo', choices=['mappo', 'ippo', 'ppo'])
   
    parser.add_argument('--train_size', type=float, help='number of data sample for training',
                        default=1)

    parser.add_argument('--K', type=int, default=1)

    # training parameter
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=20,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e10,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--episode_length", type=int,
                        default=1000, help="Max length for any episode")
    parser.add_argument("--device", type=str, default='cuda:0', help="specify the name of device",
                        choices=['cuda:0', 'cuda:1', 'cpu'])

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
    parser.add_argument('--selected_ratio', type=float, default=0.1)    

    parser.add_argument('--time_out', type=int, default=1)
    args = parser.parse_args()
    # checking_parameter(args)


    

    print(f'nvev: {args.n_rollout_threads}')
    print(f'nvev_eval: {args.n_eval_rollout_threads}')

    if 'abilene' in args.dataset:
        args.num_node = 12
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
    elif 'geant' in args.dataset:
        args.num_node = 22
        args.num_flow = args.num_node * args.num_node
        args.day_size = 96
    elif 'sdn' in args.dataset:
        args.num_node = 14
        args.num_flow = args.num_node * args.num_node
        args.day_size = 1440
    elif 'nobel' in args.dataset:
        args.num_node = 17
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
        args.train_size = 1.0
    elif 'germany' in args.dataset:
        args.num_node = 50
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
        args.train_size = 1.0
    elif 'gnnet-75' in args.dataset:
        args.num_node = 75
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
        args.train_size = 1.0
    elif 'gnnet-100' in args.dataset:
        args.num_node = 100
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
        args.train_size = 1.0
    elif 'gnnet-40' in args.dataset:
        args.num_node = 40
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
        args.train_size = 1.0
    elif 'ct-gen' in args.dataset:
        args.num_node = 149
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
        args.train_size = 1.0
    else:
        raise ValueError('Dataset not found!')


    return args




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