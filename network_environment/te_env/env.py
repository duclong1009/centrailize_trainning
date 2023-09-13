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

    
    def _reward(self, mlu, action):
        """
            mlu: maximum link utilization (1,)
            mpu:  max path utilization (n_agent)
        """
        if sum(action) > (self.args.selected_ratio * self.num_node * (self.num_node - 1)):
            self.penalty += self.penalty_values

        mlu_reaward = -1.0 * mlu
        rewards = mlu_reaward + self.penalty
       
        return rewards

    def lp_solve(self,action, tm):
        return 0.5
    
    def step(self, action, use_solution=False):
        tm = self.tm[self.tm_index]
        mlu = self.lp_solve(action, tm)
        self.critical_link = self._convert_action(action)
        rewards = self._reward(mlu=mlu, action=action)
        observation, dones = self._next_obs()
        self.penalty = 0
        info = {"rewards": np.mean(rewards),
                "mlu": np.mean(mlu) }
        print(action)
        print(sum(action))


        # self.routing_rule = self._convert_action(action)
        
        # tm = self.tm[self.tm_index]
        # mlu, self.link_util = do_routing(tm, self.routing_rule, self.nx_graph, self.num_node, self.flow2link,
        #                                      self.ub)
        # self.path_mlu = get_path_mlu(self.routing_rule, self.num_node, self.flow2link, self.link_util, self.nx_graph)

        # rewards = self._reward(mlu=mlu, path_mlu=self.path_mlu)

        # observation, dones = self._next_obs()
        # self.penalty = 0
        # info = {"rewards": np.mean(rewards),
        #          "mlu": np.mean(mlu) }
        return observation, rewards,False, dones, info
    
    

    def _get_current_base_solution(self):
        solver_name = self.args.base_solution
        if solver_name == 'sp':
            self.current_base_solution = np.zeros(shape=(self.num_node, self.num_node))
        else:
            self.current_base_solution = self.base_solutions[solver_name][self.tm_index]


    def _convert_action(self, action):
        critical_flow = []
        for i,act in enumerate(action):
            if act == 1:
                critical_flow += self.idx2flow[i]
        return critical_flow