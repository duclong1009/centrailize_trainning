import sys
sys.path.append('../')
from .base_env import BaseEnv
from .utils import *
import numpy as np
import copy 
from .utils import do_routing
from solver import OneStepSRTopKSolver
import time

class TE_Env(BaseEnv):
    def __init__(self, rank, args, is_eval=False) -> None:
        super().__init__(rank, args, is_eval)
        self.routing_rule = np.zeros(shape=(self.num_node, self.num_node))
        self.current_base_solution = np.zeros(shape=(self.num_node, self.num_node))
        self.penalty = 0
        self.penalty_values = -10
        # print(self.data.shape)
        # print(self.tm.shape)
        self.args.data_length = self.tm.shape[0]
        self.mlu_opt = np.ones(self.tm.shape[0]) * -1
        all_flow_idx = []
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            if i !=j:
                all_flow_idx.append([i,j])
        self.all_flow_idx = all_flow_idx

        self.solver = OneStepSRTopKSolver(args,self.nx_graph, args.time_out ,False, self.list_link_strated_at, self.list_link_end_at, self.idx2flow, self.set_ENH, self.all_flow_idx)
        
        

    def reset(self, **kwargs):
        self.tm_index = self.hist_step
        mlu, self.link_util = do_routing(self.tm[self.tm_index - 1],
                                         self.routing_rule, self.nx_graph, self.num_node, self.flow2link, self.ub)
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

    
    def _reward(self, mlu, action):
        """
            mlu: maximum link utilization (1,)
            mpu:  max path utilization (n_agent)
        """
        if sum(action) > (self.args.selected_ratio * self.num_node * (self.num_node - 1)):
            self.penalty = self.penalty_values

        mlu_reaward = mlu
        rewards = mlu_reaward + self.penalty
    
        return rewards

    def lp_solve(self,critical_flow, tm, index):
        mlu, var_dict = self.solver.solve(tm, critical_flow)
        if self.mlu_opt[index] == -1:
            mlu_opt, _ = self.solver.solve(tm, self.all_flow_idx)
            self.mlu_opt[index] = mlu_opt
        else:
            mlu_opt = self.mlu_opt[index]

        if self.is_eval:
            with open(f"output/var_dict_{self.tm_index}.json", "w") as f:
                import json
                json.dump(var_dict, f)
                
        if mlu == 0:
            mlu = 1
        return mlu, mlu_opt
    
    def step(self, action, use_solution=False):
        
        tm = self.tm[self.tm_index]
        self.critical_flow = self._convert_action(action)
        time1 = time.time() 
        mlu, mlu_opt = self.lp_solve(self.critical_flow, tm, self.tm_index)
        lp_time = time.time() - time1
        rewards = self._reward(mlu_opt/mlu, action=action)
        observation, dones = self._next_obs()
        self.penalty = 0
        info = {"rewards": np.mean(rewards),
                "mlu": np.mean(mlu),
                "mlu_opt": np.mean(mlu_opt),
                "lp_time": lp_time }

        return observation, rewards, dones, False, info
    
    

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
                critical_flow.append(self.idx2flow[i])
        return critical_flow