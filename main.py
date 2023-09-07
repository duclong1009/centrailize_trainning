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
from pathlib import Path

import numpy as np


import os.path

import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

sns.set_theme()
from parameters.data_loading import load_data
from parameters.args import get_args

from parameters.utils import get_experiment_name


from sappo_fcn.fix_num_user import run_mobile_fix_num_user


def main(args):
    # ------------------------------------- log dir/tensorboard  -------------------------------------
    args.trtau = args.K  # training length
    args.Lbeta = int(args.Lbeta * args.M)  # number of Lbeta

    args.experiment_name = get_experiment_name(args)
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/logs") / args.algorithm_name / args.experiment_name
    print(run_dir)
    writer = SummaryWriter(log_dir=run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    model_dir = str(run_dir / 'models')

    # ------------------------------------- init location ap/user  -------------------------------------
    
    data = load_data(args)
    args.n_agents = args.num_node
    episode_length = args.episode_length

    run_mobile_fix_num_user(data, run_dir, args)



if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    main(args)
