import numpy as np
from gym import spaces

from parameters.parameters import power_u


def set_obs_space(args):
    if args.obs_state == 1:
        # only use user_rate
        observation_space = spaces.Box(low=0, high=1, shape=(args.K,), dtype=np.float64)
    elif args.obs_state == 2:
        # only use beta
        observation_space = spaces.Box(low=0, high=1, shape=(args.M * args.K,), dtype=np.float64)
    elif args.obs_state == 3:
        # user_rate + beta
        observation_space = spaces.Box(low=0, high=1, shape=(args.K + args.M * args.K,), dtype=np.float64)
    elif args.obs_state == 4:
        # user_rate + Lbeta
        observation_space = spaces.Box(low=0, high=1, shape=(args.K + args.Lbeta * args.K,), dtype=np.float64)
    else:
        raise NotImplementedError

    return observation_space


def set_action_space(args):
    if args.action_state == 0:
        action_space = spaces.MultiDiscrete([args.num_power_level] * args.K)
    elif args.action_state == 1:  # continuous actions
        action_space = spaces.Box(low=0, high=power_u, shape=(args.K,), dtype=np.float32)
    else:
        raise NotImplementedError

    return action_space

# def set_global_space(num_agent, observation_space, args):
#
#     if args.global_state == 1 or args.global_state == 3:
#         # (indicators and all betas) or (users' rate + all beta)
#         state_space = spaces.Box(low=0, high=1, shape=((args.M + 1) * args.num_agent,), dtype=np.float64)
#     elif args.global_state == 2:
#         # (indicators and all betas) or (users' rate + all beta)
#         state_space = spaces.Box(low=0, high=1, shape=(args.M * args.num_agent,), dtype=np.float64)
#     elif args.global_state == 4:
#         state_space = spaces.Box(low=0, high=1, shape=((args.Lbeta + 1) * args.num_agent,), dtype=np.float64)
#     elif args.global_state == 5:
#         # state: indicators + Bs_index
#         state_space = spaces.Box(low=0, high=1, shape=((args.Lbeta + 1) * args.num_agent,), dtype=np.float32)
#     elif args.global_state == 6:
#         # state: indicators + Lbeta + Bs_index
#         state_space = spaces.Box(low=0, high=1, shape=((args.Lbeta * 2 + 1) * args.num_agent), dtype=np.float32)
#     elif args.global_state == 7:
#         # combine all the obs of all agents
#         state_space = spaces.Box(low=0, high=1, shape=(observation_space[0].shape[0] * num_agent,),
#                                  dtype=np.float32)
#     else:
#         raise NotImplementedError
#
#     return state_space
