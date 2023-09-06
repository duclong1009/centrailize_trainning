import sys

import gym
import torch
import copy
import numpy as np
import os 
import itertools
import gymnasium
from network_environment.te_env.env_utils import *
from network_environment.te_env.traffic_routing.te_util import load_network_topology
from network_environment.te_env.utils.data_utils.data_loading import get_phi
from network_environment.te_env.utils.compressive_sensing.pursuit import sparse_coding


class MasrNodeEnv(gymnasium.Env):
    def __init__(self, rank, args, is_eval=False):
        super(MasrNodeEnv, self).__init__()
        args.rank = rank
        self.rank = rank
        self.env_name = args.env_name
        self.is_eval = is_eval
        self.solution_arr = []
        self.nenvs = args.n_eval_rollout_threads if is_eval else args.n_rollout_threads

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

        self.day_size = args.day_size

        self.nx_graph = load_network_topology(self.args.dataset, self.args.data_folder)
        args.nx_graph = self.nx_graph

        self.num_node = self.nx_graph.number_of_nodes()
        assert self.num_node == args.num_node
        self.num_link = self.nx_graph.number_of_edges()
        args.num_link = self.num_link

        self.segments = compute_path(self.nx_graph, self.args.dataset, self.args.data_folder, args.rank)
        self.link2flow = None
        self.link2flow, self.flow2link = compute_lssr_path(args)
        self.ub = get_solution_bound(self.flow2link, args)

        self.flow2node_sr = calculate_flow2node_sr(args, self.flow2link)
        self.agent2link = calculate_agent2link(args, self.flow2link)
        self.link2index = get_link2idex(self.nx_graph)
        self.n_agents = self.num_node

        # define observation and actions space
        self.global_state = args.global_state
        self.obs_state = args.obs_state
        self.action_state = args.action_state
        self.observation_space = set_obs_space(n_agents=self.n_agents, args=self.args)
        self.action_space = set_action_space(n_agents=self.n_agents, args=self.args)

        # self.action_mask = set_action_mask(n_agents=self.n_agents, n_node=self.n_node, ub=self.ub)

        # define global state space
        self.share_observation_space = set_global_state_space(args=self.args, observation_space=self.observation_space)

        self.action_mask = set_action_mask(n_agents=self.n_agents, n_node=self.num_node, ub=self.ub,
                                           num_path=self.args.n_path)

        self.step_count = 0
        self.episode_count = 0

        self.scaler = self.data['scaler']

        self.routing_rule = np.zeros(shape=(self.num_node, self.num_node))
        self.prev_rules = copy.deepcopy(self.routing_rule)
        # logging
        # self.monitor = args.monitor

        # pulp solver
        self.segment2pathid = get_segment2pathid(flow2link=self.flow2link, args=args)

        self.current_base_solution = None
        self.num_routing_change = np.zeros(shape=(self.n_agents,))

        # self.base_solution = np.zeros(shape=(self.num_node, self.num_node))  # shortest path traffic_routing solution
        self.link_util = np.zeros(shape=(self.num_link,))
        self.path_mlu = np.zeros(shape=(self.num_node, self.num_node))

        self.tm_index = self.hist_step
        self.num_flow = self.num_node
        self.flowID = np.zeros(shape=(self.num_node, self.num_flow))

        self.penalty = np.zeros(shape=(self.n_agents,))  # = -10 if violating the constraints
        self.penalty_values = -100  # = -100 if violating the constraints

        if is_eval:
            print('|--- Finish initializing EVAL network_environments {}'.format(self.rank))
        else:
            print('|--- Finish initializing TRAIN network_environments {}'.format(self.rank))

        # storing data for training tm prediction models
        if not is_eval:
            self.total_samples = self.num_timesteps * 100
            self.store_tm = np.zeros((self.total_samples, self.n_agents, self.num_node * self.num_node))
            self.store_link_util = np.zeros((self.total_samples, self.num_link))
            self.store_label = np.zeros((self.total_samples, self.num_node * self.num_node))
            self.sample_count = 0

        # using dnn to estimate future tm from partial tm at each agent
        if self.obs_state == 6 and (not self.args.use_centralized_V):
            self.tm_pred_model = args.tm_pred_model
            self.tm_pred_scaler = args.tm_pred_scaler
            self.tm_pred_model.eval()
        else:
            self.tm_pred_model = None

        # link failures

        self.use_env_link_failure = False

        self.is_link_fail = False
        self.link_fail_id = None
        self.failure_duration = None
        self.failure_interval = None

    def reset(self):
        self.penalty = np.zeros(shape=(self.n_agents,))  # = -10 if violating the constraints

        self.tm_index = self.hist_step
        mlu, self.link_util = do_routing(self.tm[self.tm_index - 1],
                                         self.routing_rule, self.nx_graph, self.num_node, self.flow2link, self.ub)
        state, observation, flowID = self._prepare_state()
        action_mask = self._get_action_mask(flowID)
        self._get_current_base_solution()
        return state, observation, action_mask

    def step(self, action, use_solution=False):
        self._get_current_base_solution()

        self.routing_rule = self._convert_action(action)
        # print(self.routing_rule.shape)
        # if self.is_eval:
        #     self.solution_arr.append(np.expand_dims(self.routing_rule,0))
        #     arr = np.concatenate(self.solution_arr,0)
        #     np.save( f"solution/rank_{self.rank}.npy",arr)
        tm = self.tm[self.tm_index]
        if self.is_link_fail:
            mlu, self.link_util = do_routing_with_link_failure(tm, self.routing_rule, self.nx_graph, self.num_node,
                                                               self.flow2link,
                                                               self.ub, self.link_fail_id)
        else:
            mlu, self.link_util = do_routing(tm, self.routing_rule, self.nx_graph, self.num_node, self.flow2link,
                                             self.ub)
        self.path_mlu = get_path_mlu(self.routing_rule, self.num_node, self.flow2link, self.link_util, self.nx_graph)

        self.num_routing_change = get_rc(n_agents=self.n_agents, rules=self.routing_rule,
                                         prev_rules=self.prev_rules, steps=self.tm_index - 1)

        rewards = self._reward(mlu=mlu, path_mlu=self.path_mlu)
        mpu = np.max(self.path_mlu, axis=1)  # average paths utilization of each agent (n_agents,)

        infos = []
        for i in range(self.n_agents):
            agent_info = {'mpu': mpu[i], 'mlu': mlu,
                          'penalty': int(self.penalty[i] / self.penalty_values) if self.penalty[i] < 0 else 0,
                          'rc': self.num_routing_change[i]}

            for solver_name in self.args.solvers_list:
                agent_info.update({f'mlu_{solver_name}': self.base_mlus[solver_name][self.tm_index]})

            infos.append(agent_info)

        state, observation, flowID, dones = self._next_obs()

        self.prev_rules = copy.deepcopy(self.routing_rule)
        action_mask = self._get_action_mask(flowID)

        self.penalty = np.zeros(shape=(self.n_agents,))  # = -10 if violating the constraints reset penalty

        return state, observation, action_mask, rewards, dones, infos

    def _reward(self, mlu, path_mlu, rc=0, loss=0.0):
        """
            mlu: maximum link utilization (1,)
            mpu:  max path utilization (n_agent)
        """

        mlu_reaward = [-1.0 * mlu] * self.n_agents

        mpu = np.max(path_mlu, axis=1)  # max paths utilization of each agent (n_agents,)
        mpu_rewards = -1.0 * mpu

        spu = np.sum(path_mlu, axis=1)  # sum paths utilization of each agent (n_agents,)
        spu_rewards = -1.0 * spu

        mlu_reaward = np.array(mlu_reaward)
        mpu_rewards = np.array(mpu_rewards) + self.penalty

        rc_rewards = -1.0 * self.num_routing_change / self.num_node

        if self.args.reward == 1:  # only using mlu
            rewards = copy.deepcopy(mlu_reaward)
        elif self.args.reward == 2:  # only using mpu
            rewards = copy.deepcopy(mpu_rewards)
        elif self.args.reward == 3:  # combine mlu and mpu
            rewards = mlu_reaward + mpu_rewards
        elif self.args.reward == 4:  # consider traffic_routing changes
            local_reward = mpu_rewards + rc_rewards
            rewards = 0.5 * mlu_reaward + 0.5 * local_reward
        elif self.args.reward == 5:  # DATE
            rewards = (2.5 / self.num_node) * mlu_reaward + (0.05 / self.num_node) * spu_rewards
            # alpha = 2.5/|D_h|; beta = 0.05/|D_h|; |D_h| total number of flows per agent == num_node
        else:
            raise NotImplementedError('Reward function {} not supported'.format(self.args.reward))

        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1)
        return rewards

    def _convert_action(self, action):
        if self.args.base_solution == 'sp':  # use the shortest path traffic_routing for other flows
            routing_rule = copy.deepcopy(np.zeros(shape=(self.num_node, self.num_node)))
        else:
            routing_rule = copy.deepcopy(self.current_base_solution)  # use base solution for other flows

        if self.action_state == 0:
            routing_rule = self._update_routing_rules(action, routing_rule, self.flowID)
            return routing_rule

        elif self.action_state == 1:
            n_flows = min(int(self.args.topk * self.num_node), self.num_node)  # only consider the topk-largest flows
            flowID = action[:, :n_flows]
            action = action[:, n_flows:]
            routing_rule = self._update_routing_rules(action, routing_rule, flowID)
            return routing_rule
        else:
            raise NotImplementedError

    def _update_routing_rules(self, action, routing_rule, flowID):
        for i in range(self.n_agents):
            for j in range(len(flowID[i])):
                dstID = flowID[i, j]
                if action[i, j] >= self.ub[i, dstID]:
                    self.penalty[i] += self.penalty_values
                    routing_rule[i, dstID] = 0
                else:
                    routing_rule[i, dstID] = action[i, j]

                if self.is_link_fail:  # check if any link fail
                    k = routing_rule[i, dstID]
                    path = self.flow2link[(i, dstID)][k]
                    if self._check_invalid_path(path) and self.ub[i, dstID] > 1:
                        # if link_failure in paths and there is alternative path -> add penalty
                        self.penalty[i] += self.penalty_values

        return routing_rule

    def _prepare_state(self):

        scaled_tm = self.tm_scaled[self.tm_index - self.hist_step:self.tm_index]

        original_tm = self.tm[self.tm_index - self.hist_step:self.tm_index]
        # print(original_tm.shape)
        if self.obs_state == 3 or self.obs_state == 4 or self.obs_state == 5 or self.obs_state == 6:
            traffic_observations = self._get_traffic_observations(self.tm[self.tm_index - 1], self.routing_rule)
        else:
            traffic_observations = None

        if self.is_link_fail:
            self.link_util[self.link_fail_id] = 1.0

        flowID = []

        if self.obs_state == 1:  # observation = [dstID, link_utilization_ratio] (links not in agent path are set -1)
            observation = []
            for i in range(self.n_agents):
                mean_tm = np.mean(original_tm[:, i], axis=0)
                mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                flowID.append(mean_tm_index_sort)

                link_u = -1.0 * np.ones_like(self.link_util)
                link_u[self.agent2link[i]] = link_u[self.agent2link[i]] / np.max(self.link_util)

                obs_i = np.zeros(self.observation_space[i].shape)
                for j in range(self.num_flow):
                    dst_id = mean_tm_index_sort[j]
                    obs_i[j] = dst_id * 1.0 / self.num_node

                obs_i[self.num_flow:] = copy.deepcopy(link_u)

                obs_i = obs_i.flatten()
                observation.append(obs_i)

        elif self.obs_state == 2:  # observation = [dstID, demand, all_link_u]
            observation = []
            for i in range(self.n_agents):
                obs_i = np.zeros(self.observation_space[i].shape)
                mean_tm = np.mean(original_tm[:, i], axis=0)
                mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                flowID.append(mean_tm_index_sort)
                for j in range(self.num_flow):
                    dst_id = mean_tm_index_sort[j]
                    obs_i[j] = dst_id * 1.0 / self.num_node
                    obs_i[self.num_flow + j] = scaled_tm[:, i, dst_id]
                obs_i[self.num_flow * 2:] = copy.deepcopy(self.link_util)
                obs_i = obs_i.flatten()
                observation.append(obs_i)
        elif self.obs_state == 3:  # observation = [dstID, link_u, partial_traffic_matrix]
            traffic_observations[traffic_observations < 0] = 0
            traffic_observations = np.reshape(traffic_observations, newshape=(self.n_agents, -1))
            traffic_observations_scaled = self.data['scaler'].transform(traffic_observations)  # scaling traffic data
            observation = []
            for i in range(self.n_agents):
                obs_i = np.zeros(self.observation_space[i].shape)
                mean_tm = np.mean(original_tm[:, i], axis=0)
                mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                flowID.append(mean_tm_index_sort)
                for j in range(self.num_flow):
                    dst_id = mean_tm_index_sort[j]
                    obs_i[j] = dst_id * 1.0 / self.num_node
                obs_i[self.num_flow:self.num_flow + self.num_link] = copy.deepcopy(self.link_util)
                obs_i[self.num_flow + self.num_link:] = copy.deepcopy(traffic_observations_scaled[i])

                obs_i = obs_i.flatten()
                observation.append(obs_i)
            # np.save("flowID.npy",np.array(flowID))
            # breakpoint()
            data_file = os.path.join(self.args.data_folder, 'tm_pred_data',
                                     f'{self.args.dataset}_rank_{self.args.rank}.npz')
            if not self.is_eval and not os.path.exists(data_file):
                if self.sample_count < self.total_samples:
                    self.store_tm[self.sample_count] = np.reshape(traffic_observations,
                                                                  newshape=(
                                                                      self.n_agents, self.num_node * self.num_node))
                    self.store_link_util[self.sample_count] = self.link_util
                    self.store_label[self.sample_count] = self.tm[self.tm_index].flatten()
                    self.sample_count += 1
                else:
                    saved_path = os.path.join(self.args.data_folder, 'tm_pred_data')
                    if not os.path.exists(saved_path):
                        os.makedirs(saved_path, exist_ok=True)
                    saved_file = os.path.join(saved_path, f'{self.args.dataset}_rank_{self.args.rank}.npz')
                    if not os.path.exists(saved_file):
                        saved_data = {
                            'x_tm': self.store_tm,
                            'x_link_util': self.store_link_util,
                            'y': self.store_label
                        }
                        np.savez_compressed(saved_file, **saved_data)

                        print(f'|--- Collect all data sample for tm prediction, saved at {saved_file}')

        elif self.obs_state == 4:  # observation = [dstID, link_u, partial_traffic_matrix]
            traffic_observations[traffic_observations < 0] = 0
            traffic_observations = np.reshape(traffic_observations, newshape=(self.n_agents, -1))
            traffic_observations_scaled = self.data['scaler'].transform(traffic_observations)  # scaling traffic data
            observation = []
            for i in range(self.n_agents):
                obs_i = np.zeros(self.observation_space[i].shape)
                mean_tm = np.mean(original_tm[:, i], axis=0)
                mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                flowID.append(mean_tm_index_sort)
                for j in range(self.num_flow):
                    dst_id = mean_tm_index_sort[j]
                    obs_i[j] = dst_id * 1.0 / self.num_node

                obs_i[self.num_flow:self.num_flow + self.num_link] = copy.deepcopy(self.link_util)
                obs_i[self.num_flow + self.num_link:] = copy.deepcopy(traffic_observations_scaled[i])

                observation.append(obs_i)
        elif self.obs_state == 5:  # observation:[dstID, reconstructed_traffic_matrix]
            traffic_observations_reconstructed = self._reconstruct_tm(traffic_observations)
            observation = []
            for i in range(self.n_agents):
                obs_i = np.zeros(self.observation_space[i].shape)
                mean_tm = np.mean(original_tm[:, i], axis=0)
                mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                flowID.append(mean_tm_index_sort)
                for j in range(self.num_flow):
                    dst_id = mean_tm_index_sort[j]
                    obs_i[j] = dst_id * 1.0 / self.num_node

                obs_i[self.num_flow:] = copy.deepcopy(traffic_observations_reconstructed[i])

                observation.append(obs_i)

        elif self.obs_state == 6:  # observation = [dstID, demand, prev_action, all_link_u, partial_traffic_matrix]
            traffic_observations[traffic_observations < 0] = 0
            traffic_observations = np.reshape(traffic_observations, newshape=(self.n_agents, -1))
            traffic_observations_scaled = self.tm_pred_scaler.transform(traffic_observations)  # scaling traffic data

            with torch.no_grad():
                link_u_torch = torch.from_numpy(self.link_util).to(dtype=torch.float32, device=self.args.device)
                link_u_torch = torch.unsqueeze(link_u_torch, dim=0)

                observation = []
                for i in range(self.n_agents):
                    # for predicting tm from partial tm and link u
                    partial_tm_torch = torch.from_numpy(traffic_observations_scaled[i]).to(dtype=torch.float32,
                                                                                           device=self.args.device)
                    partial_tm_torch = torch.unsqueeze(partial_tm_torch, dim=0)
                    pred_tm = self.tm_pred_model(partial_tm_torch, link_u_torch)
                    pred_tm = pred_tm.cpu().numpy()
                    pred_tm[pred_tm < 0] = 0.0
                    pred_tm[pred_tm > 1] = 1.0

                    # preparing new observation
                    obs_i = np.zeros(self.observation_space[i].shape)
                    mean_tm = np.mean(original_tm[:, i], axis=0)
                    mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                    flowID.append(mean_tm_index_sort)
                    for j in range(self.num_flow):
                        dst_id = mean_tm_index_sort[j]
                        obs_i[j] = dst_id * 1.0 / self.num_node
                        obs_i[self.num_flow + j] = scaled_tm[:, i, dst_id]

                    obs_i[self.num_flow * 2:] = copy.deepcopy(pred_tm)

                    obs_i = obs_i.flatten()
                    observation.append(obs_i)

        elif self.obs_state == 7:  # observation = [dstID, demand]
            observation = []
            for i in range(self.n_agents):
                obs_i = np.zeros(self.observation_space[i].shape)
                mean_tm = np.mean(original_tm[:, i], axis=0)
                mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                flowID.append(mean_tm_index_sort)
                for j in range(self.num_flow):
                    dst_id = mean_tm_index_sort[j]
                    obs_i[j] = dst_id * 1.0 / self.num_node
                    obs_i[self.num_flow + j] = scaled_tm[:, i, dst_id]

                obs_i = obs_i.flatten()
                observation.append(obs_i)

        elif self.obs_state == 8:  # observation = [dstID, demand, actual_traffic_matrix]
            observation = []
            for i in range(self.n_agents):
                obs_i = np.zeros(self.observation_space[i].shape)
                mean_tm = np.mean(original_tm[:, i], axis=0)
                mean_tm_index_sort = np.argsort(mean_tm)[::-1][:self.num_flow]
                flowID.append(mean_tm_index_sort)
                for j in range(self.num_flow):
                    dst_id = mean_tm_index_sort[j]
                    obs_i[j] = dst_id * 1.0 / self.num_node
                    obs_i[self.num_flow + j] = scaled_tm[:, i, dst_id]

                obs_i[self.num_flow * 2:] = copy.deepcopy(scaled_tm.flatten())

                obs_i = obs_i.flatten()
                observation.append(obs_i)

        else:
            raise NotImplementedError()

        observation = np.array(observation, dtype=np.float64)  # shape (n_agent, num_flow*features)

        if self.args.use_centralized_V:
            global_state = np.zeros(self.share_observation_space.shape)
            global_state[:self.num_link] = copy.deepcopy(self.link_util)
            global_state[self.num_link:] = copy.deepcopy(scaled_tm.flatten())

        else:
            global_state = []
            for i in range(self.n_agents):
                global_state_i = self.share_observation_space[i].sample()
                global_state.append(global_state_i)
            global_state = np.array(global_state)  # shape (n_agent, obs_shape) = (n_agent, num_flow*features)

        flowID = np.array(flowID, dtype=int)
        self.flowID = copy.deepcopy(flowID)

        return global_state, observation, flowID

    def _next_obs(self):
        done = self._is_done()
        self._update_counter(is_done=done)

        if done:
            observation = []
            for i in range(self.n_agents):
                obs_i = self.observation_space[i].sample()
                observation.append(obs_i)

            observation = np.array(observation)  # shape (n_agent, obs_shape) = (n_agent, num_flow*features)
            if not self.args.use_centralized_V:
                state = []
                for i in range(self.n_agents):
                    state_i = self.share_observation_space[i].sample()
                    state.append(state_i)
                state = np.array(state)  # shape (n_agent, obs_shape) = (n_agent, num_flow*features)
            else:
                state = self.share_observation_space.sample()
            flowID = np.zeros(shape=(self.num_node, self.num_flow), dtype=int)
            self.flowID = copy.deepcopy(flowID)
        else:
            state, observation, flowID = self._prepare_state()

        # state = state.astype(np.float64)
        # observation = observation.astype(np.float64)
        dones = np.array([done] * self.n_agents)
        self._check_link_failure()
        return state, observation, flowID, dones

    def _is_done(self):
        if self.tm_index == self.n_timesteps - 1:
            return True
        else:
            return False

    def _update_counter(self, is_done):
        self.tm_index += 1

        if is_done:
            self.tm_index = self.hist_step
            self.episode_count += 1

    def _get_action_mask(self, flowID):
        action_mask = np.zeros((self.n_agents, self.num_flow, self.args.n_path))
        for agentID in range(self.n_agents):
            for i in range(self.num_flow):
                try:
                    action_mask[agentID, i] = self.action_mask[agentID, flowID[agentID, i]]
                except:
                    print('|--- ERROR!')
                    print(self.rank, self.tm_index, flowID[agentID, i], agentID, i)
                    raise RuntimeError

        return action_mask


    def _get_current_base_solution(self):
        solver_name = self.args.base_solution
        if solver_name == 'sp':
            self.current_base_solution = np.zeros(shape=(self.num_node, self.num_node))
        else:
            self.current_base_solution = self.base_solutions[solver_name][self.tm_index]

    def _reconstruct_tm(self, partial_tm):
        psi = self.data['psi']
        scaler = self.data['scaler']
        reconstructed_scaled_tm = []
        for agent_i in range(self.n_agents):
            X = copy.deepcopy(partial_tm[agent_i])
            X = X.flatten()
            index = np.argwhere(X >= 0).flatten()  # element < 0 (-1) is not observed
            phi = get_phi(index, self.num_node * self.num_node)
            z = X[index]
            z = np.expand_dims(z, 0)

            ShatT = sparse_coding(ZT=z, phiT=phi.T, psiT=psi)
            tm_cs = np.dot(ShatT, psi)
            tm_cs[:, index] = z[0, :]
            tm_cs[tm_cs < 0.0] = 0.0
            reconstructed_scaled_tm.append(tm_cs)

        reconstructed_scaled_tm = np.concatenate(reconstructed_scaled_tm)
        reconstructed_scaled_tm[reconstructed_scaled_tm > scaler.max] = scaler.max
        reconstructed_scaled_tm[reconstructed_scaled_tm < scaler.min] = scaler.min
        reconstructed_scaled_tm = scaler.transform(reconstructed_scaled_tm)
        return reconstructed_scaled_tm

    def _get_traffic_observations(self, tm, routing_rule):
        flow2node_sr = self.flow2node_sr

        # element < 0 (-1) is not observed
        traffic_observations = -1 + np.zeros(shape=(self.n_agents, self.num_node, self.num_node))
        for i in range(self.n_agents):
            for src, dst in itertools.product(range(self.num_node), range(self.num_node)):
                if src == i or dst == i:
                    traffic_observations[i, src, dst] = tm[src, dst]
                    continue

                if len(flow2node_sr[src, dst]) == 0:
                    nodes = []
                else:
                    pathID = int(routing_rule[src, dst])
                    nodes = flow2node_sr[src, dst][pathID]

                if i in nodes:
                    traffic_observations[i, src, dst] = tm[src, dst]

        return traffic_observations

    def _check_link_failure(self):
        if self.use_env_link_failure:
            if self.is_link_fail:
                self.failure_duration += 1
                resume_link = []
                for i in range(self.link_fail_id.shape[0]):
                    if self.failure_duration[i] > self.failure_interval[i]:
                        resume_link.append(i)

                if len(resume_link) > 0:
                    self.link_fail_id = np.delete(self.link_fail_id, resume_link)
                    self.failure_duration = np.delete(self.failure_duration, resume_link)
                    self.failure_interval = np.delete(self.failure_interval, resume_link)

                if len(self.link_fail_id) == 0:
                    self.is_link_fail = False

            if self.tm_index < int(0.5 * self.n_timesteps) and not self.is_link_fail:
                if np.random.rand() < 0.1:
                    self.is_link_fail = True
                    n_link_fail = np.random.randint(1, 4)
                    self.link_fail_id = np.random.choice(np.arange(self.num_link), size=(n_link_fail,))
                    self.failure_interval = np.random.randint(5, 50, size=(n_link_fail,))
                    self.failure_duration = np.zeros((n_link_fail,))
        else:
            self.is_link_fail = False

    def _check_invalid_path(self, path):
        for u, v in path:
            link_id = self.link2index[u, v]
            if link_id in self.link_fail_id:
                return True

        return False

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
