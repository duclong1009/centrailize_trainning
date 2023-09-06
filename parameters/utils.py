info_keywords = ("Global_Reward", "Sum_Rate", "Sum_Power_Usage", "Per_Satisfied_User", "Minimum_Rate")


def get_experiment_name(args):
    if args.algorithm_name == 'mappo' or args.algorithm_name == 'ippo':

        experiment_name = f'{args.algorithm_name}-{args.scenario_name}-' \
                          f'{args.global_state}-{args.obs_state}-{args.action_state}-{args.num_power_level}-' \
                          f'{args.reward}-{args.reward_type}-' \
                          f'{args.M}-{args.K}-{args.D}-{args.bs_dist}-{args.min_rate}-{args.Lbeta}-' \
                          f'{args.train_size}-{args.max_veloc}-{args.use_recurrent_policy}-{args.num_scenario}-' \
                          f'{args.total_step}'
        if args.runner == 'shared':
            experiment_name += f'_{args.runner}_policy'
        if args.runner == 'federated' or 'federated' in args.runner:
            experiment_name += f'_{args.runner}_policy'

    else:
        experiment_name = f'{args.algorithm_name}-{args.scenario_name}-' \
                          f'{args.obs_state}-{args.action_state}-{args.num_power_level}-' \
                          f'{args.reward}-' \
                          f'{args.M}-{args.K}-{args.D}-{args.bs_dist}-{args.min_rate}-{args.Lbeta}-' \
                          f'{args.train_size}-{args.max_veloc}-{args.num_scenario}-' \
                          f'{args.total_step}'

    experiment_name += f'_{args.weight}'
    experiment_name += f'_{args.layer_N}'

    return experiment_name
