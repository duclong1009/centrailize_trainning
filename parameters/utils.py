info_keywords = ("rewards", "mlu")


def get_experiment_name(args):
    experiment_name =   f'Qos_TE_{args.dataset}' \
                           f'_{args.n_path}_gamma_{args.gamma}' \
                           f'_global_state_{args.global_state}' \
                           f'_obs_state_{args.obs_state}_reward_{args.reward}' \
                           f'_value_loss_coef_{args.value_loss_coef}_entropy_coef_{args.entropy_coef}' \
                           f'_clip_param_{args.clip_param}' \
                           f'_n_layer_{args.layer_N}_hidden_size_{args.hidden_size}' \
                           f'_ppo_ep_{args.ppo_epoch}'


    return experiment_name
