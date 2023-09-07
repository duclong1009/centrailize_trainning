import sys
sys.path.append('../')
from .base_env import BaseEnv
from .utils import *
import numpy as np
import copy 
from .utils import do_routing
class TE_Env(BaseEnv):
    def __init__(self, rank, args, is_eval=False) -> None:
        super().__init__(rank, args, is_eval)
        self.routing_rule = np.zeros(shape=(self.num_node, self.num_node))
        self.current_base_solution = np.zeros(shape=(self.num_node, self.num_node))
        self.penalty = 0
        self.penalty_values = -100

    def reset(self, **kwargs):
        self.tm_index = self.hist_step
        mlu, self.link_util = do_routing(self.tm[self.tm_index - 1],
                                         self.routing_rule, self.nx_graph, self.num_node, self.flow2link, self.ub)
        observation = self._prepare_state()
        self.tm_index = 1
        # self._get_current_base_solution()
        infos = {"mlu": mlu}
        self.penalty = 0
        return observation, infos
    
    def _prepare_state(self):
        max_link_utils = 5
        link_util = copy.deepcopy(self.link_util)
        link_util = link_util/ max_link_utils
        link_util = np.where(link_util > 1 , 1 , link_util)

        scaled_tm = self.tm_scaled[self.tm_index - self.hist_step:self.tm_index]
        original_tm = self.tm[self.tm_index - self.hist_step:self.tm_index]

        observation = np.zeros(self.observation_space.shape) 
        observation[: self.num_node * self.num_node] = scaled_tm.flatten()
        observation[self.num_node * self.num_node:] = link_util

        return observation
    
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

    def _get_next_state(self, is_reset=False):
        if is_reset:
            observation = self.observation_space.sample()

            done = False
            self._get_current_data()

        else:
            self._update_counter()
            done = self._is_done()
            self._get_current_data()

            observation = self._prepare_state()

        return observation, done
    
    def _next_obs(self):
        done = self._is_done()
        self._update_counter(is_done=done)

        if done:
            observation = self.observation_space.sample()
        else:
            observation = self._prepare_state()
        return  observation, done

    def _convert_action(self, action):
        routing_rule = np.array(action).reshape(self.num_node, self.num_node)
        for node_id in range(self.num_node):
            for des_id in range(self.num_node):
                k = routing_rule[node_id, des_id]
                if k >= self.ub[node_id, des_id]:
                    routing_rule[node_id, des_id] = 0
                    self.penalty += self.penalty_values
                
        return routing_rule
    
    def _reward(self, mlu, path_mlu, rc=0, loss=0.0):
        """
            mlu: maximum link utilization (1,)
            mpu:  max path utilization (n_agent)
        """
        mlu_reaward = -1.0 * mlu
        mpu = np.max(path_mlu, axis=0)  # max paths utilization of each agent (n_agents,)
        mpu_rewards = -1.0 * mpu
        mpu_rewards = mpu_rewards + self.penalty
        rewards = mlu_reaward + mpu_rewards
       
        return mlu_reaward

    
    def step(self, action, use_solution=False):
        self.routing_rule = self._convert_action(action)
        tm = self.tm[self.tm_index]
        mlu, self.link_util = do_routing(tm, self.routing_rule, self.nx_graph, self.num_node, self.flow2link,
                                             self.ub)
        self.path_mlu = get_path_mlu(self.routing_rule, self.num_node, self.flow2link, self.link_util, self.nx_graph)

        rewards = self._reward(mlu=mlu, path_mlu=self.path_mlu)

        observation, dones = self._next_obs()
        self.penalty = 0
        info = {"rewards": np.mean(rewards),
                 "mlu": np.mean(mlu) }
        return observation, rewards,False, dones, info
    
    def _get_current_base_solution(self):
        solver_name = self.args.base_solution
        if solver_name == 'sp':
            self.current_base_solution = np.zeros(shape=(self.num_node, self.num_node))
        else:
            self.current_base_solution = self.base_solutions[solver_name][self.tm_index]

 

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