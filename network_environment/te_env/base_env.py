import gym
import sys
sys.path.append('../')

from .utils import *

class BaseEnv(gym.Env):
    def __init__(self, rank, args, is_eval=False) -> None:
        super().__init__()
        self.is_eval = is_eval

        self.nenvs = args.n_eval_rollout_threads if is_eval else args.n_rollout_threads
        self.rank = rank

        self.args = args
        self.data = args.data

        self.max_episode_step = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads


        # networking environment
        if self.is_eval:
            self.tm = self.data['test/gt']  # traffic matrices (n_timestep, n_node, n_node)
            self.tm_scaled = self.data['test/scaled']  # traffic matrices (n_timestep, n_node, n_node)
        else:
            self.tm = self.data['train/gt']  # traffic matrices (n_timestep, n_node, n_node)
            self.tm_scaled = self.data['train/scaled']  # traffic matrices (n_timestep, n_node, n_node)

        total_timesteps = self.tm.shape[0]
        self.num_timesteps = int(total_timesteps / self.nenvs)
        start_step = rank * self.num_timesteps
        stop_step = start_step + self.num_timesteps
        self.tm = self.tm[start_step:stop_step]
        self.tm_scaled = self.tm_scaled[start_step:stop_step]

        self.n_timesteps = self.tm.shape[0]
        self.hist_step = self.args.input_len

        self.nx_graph = load_network_topology(self.args.dataset, self.args.data_folder)
        args.nx_graph = self.nx_graph

        self.num_node = self.nx_graph.number_of_nodes()
        assert self.num_node == args.num_node
        self.num_link = self.nx_graph.number_of_edges()
        args.num_link = self.num_link

        self.segments = compute_path(self.nx_graph, self.args.dataset, self.args.data_folder, self.rank)
        self.link2flow = None
        self.link2flow, self.flow2link = compute_lssr_path(args)
        self.ub = get_solution_bound(self.flow2link, args)
        self.flow2node_sr = calculate_flow2node_sr(args, self.flow2link)
        self.link2index = get_link2idex(self.nx_graph)
        self.observation_space = set_obs_space(args=self.args)
        self.action_space = set_action_space(args=self.args)
        self.idx2flow = get_idx2flow(args)
        self.step_count = 0
        self.episode_count = 0
        self.scaler = self.data['scaler']
        self.link_util = np.zeros(shape=(self.num_link,))
        self.path_mlu = np.zeros(shape=(self.num_node, self.num_node))
        self.tm_index = self.hist_step
        self.list_link_strated_at, self.list_link_end_at = self.get_link_started_end_at(args)
        self.set_ENH = get_set_ENH(args, self.flow2link)

    def get_link_started_end_at(self, args ):
        list_link_strated_at = {}
        list_link_end_at = {}
        for i in range(self.num_node):
            list_link_end_at[i] = []
            list_link_strated_at[i] = []

        for u,v in self.nx_graph.edges:
            
            list_link_end_at[v].append(u)
            list_link_strated_at[u].append(v)
        
        return list_link_strated_at, list_link_end_at

    def reset(self, **kwargs):
        raise("Not implement!!!")
    

    def step(self, action, use_solution=False):
        raise("Not implement!!!")
