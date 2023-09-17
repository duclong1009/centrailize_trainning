import sys
sys.path.append('../')
from .base_env import BaseEnv
from .utils import *
import numpy as np
import copy 
from .utils import do_routing, softmax
from solver import OneStepSRTopKSolver
import time

class TE_Env(BaseEnv):
    def __init__(self, rank, args, is_eval=False) -> None:
        super().__init__(rank, args, is_eval)
        self.current_base_solution = np.zeros(shape=(self.num_node, self.num_node))
        self.penalty = 0
        self.penalty_values = -10
        self.args.data_length = self.tm.shape[0]
        self.routing_rule = init_routing_rule(self.num_node, self.args.n_path)


    def reset(self, **kwargs):
        self.tm_index = self.hist_step
        mlu, self.link_util = do_routing(self.tm[self.tm_index - 1], self.routing_rule, self.nx_graph, self.flow2link, self.ub, self.link2idx, self.idx2flow)
        observation = self._prepare_state()
        self.tm_index = 1
        # self._get_current_base_solution()
        infos = {"mlu": mlu,
                    "mlu_opt":mlu}
        self.penalty = 0
        return observation, infos
    
    def _prepare_state(self):
        scaled_tm = self.tm_scaled[self.tm_index - self.hist_step:self.tm_index]
        original_tm = self.tm[self.tm_index - self.hist_step:self.tm_index]

        observation = np.zeros(self.observation_space.shape) 
        observation[: self.num_node * self.num_node] = scaled_tm.flatten()
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

        else:
            self._update_counter()
            done = self._is_done()

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

    
    def _reward(self, mlu):
        """
            mlu: maximum link utilization (1,)
            mpu:  max path utilization (n_agent)
        """

        mlu_reaward = - mlu
        rewards = mlu_reaward + self.penalty
    
        return rewards 

    
    def step(self, action, use_solution=False):
        tm = self.tm[self.tm_index]
        self.routing_rule = self._convert_action(action)
        import copy
        mlu, link_mlu = do_routing(self.tm[self.tm_index - 1], self.routing_rule, self.nx_graph, self.flow2link, self.ub, self.link2idx, self.idx2flow)
        rewards  = self._reward(mlu)
        observation, dones = self._next_obs()
        self.penalty = 0
        info = {"rewards": np.mean(rewards),
                "mlu": np.mean(mlu),
                 "penalty":self.penalty }

        return observation, rewards, dones, False, info
    
    

    def _get_current_base_solution(self):
        solver_name = self.args.base_solution
        if solver_name == 'sp':
            self.current_base_solution = np.zeros(shape=(self.num_node, self.num_node))
        else:
            self.current_base_solution = self.base_solutions[solver_name][self.tm_index]


    def _convert_action(self, action):
        ub = self.ub
        routing_rule = np.ones((self.num_node, self.num_node,self.args.n_path)) * np.inf

        for i, value in enumerate(action): 
            u,v,k = self.idx2flow[i]
            k_bound = ub[u,v]
            if k < k_bound:
                routing_rule[u,v,k] = value
        
        routing_rule = softmax(routing_rule, axis=2)
        # routing_rule = routing_rule.reshape(-1, routing_rule.shape[2])
        return routing_rule.flatten()