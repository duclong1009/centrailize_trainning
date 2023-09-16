

import sys

import numpy as np
import torch
sys.path.append('../')

import os.path
from solver import OneStepSRTopKSolver
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from network_environment.te_env.utils import *
class RunTopK:
    def __init__(self, args, is_eval=True) -> None:

        self.args = args
        self.data = args.data

        self.tm = self.data['test/gt']
        self.nx_graph = load_network_topology(self.args.dataset, self.args.data_folder)
        args.nx_graph = self.nx_graph

        self.num_node = self.nx_graph.number_of_nodes()
        assert self.num_node == args.num_node
        self.num_link = self.nx_graph.number_of_edges()
        args.num_link = self.num_link
        self.num_topflow = int(self.num_node * self.args.selected_ratio)

        self.segments = compute_path(self.nx_graph, self.args.dataset, self.args.data_folder, 0)
        self.link2flow = None
        self.link2flow, self.flow2link = compute_lssr_path(args)
        self.ub = get_solution_bound(self.flow2link, args)
        self.flow2node_sr = calculate_flow2node_sr(args, self.flow2link)
        self.link2index = get_link2idex(self.nx_graph)
        self.idx2flow = get_idx2flow(args)
        self.step_count = 0
        self.episode_count = 0
        self.scaler = self.data['scaler']
        self.link_util = np.zeros(shape=(self.num_link,))
        self.path_mlu = np.zeros(shape=(self.num_node, self.num_node))
        self.list_link_strated_at, self.list_link_end_at = self.get_link_started_end_at(args)
        self.set_ENH = get_set_ENH(args, self.flow2link)

        all_flow_idx = []
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            if i !=j:
                all_flow_idx.append((i,j))
        self.all_flow_idx = all_flow_idx
        print(self.all_flow_idx)
        self.solver = OneStepSRTopKSolver(args,self.nx_graph, args.time_out ,False, self.list_link_strated_at, self.list_link_end_at, self.idx2flow, self.set_ENH, self.all_flow_idx)
    
    def get_topk_flow_all(self, tm):
        tm_idx_sort = np.argsort(tm.flatten())[::-1][:int(self.args.selected_ratio * self.num_node * self.num_node)]
        critical_flow = []
        for index in tm_idx_sort:
            i = index // self.num_node
            j = index % self.num_node
            critical_flow.append((i,j))
        return critical_flow
    def get_topk_flow(self, tm):
        tm_idx_sort = np.argsort(tm)[:,::-1][:,:self.num_topflow]
        critical_flow = []
        for i in range(tm_idx_sort.shape[0]):
            for j in tm_idx_sort[i]:
                critical_flow.append((i,j))
        return critical_flow

    def lp_solve(self,critical_flow, tm):
        mlu, var_dict = self.solver.solve(tm, critical_flow)
      

        return mlu
    def solve(self,):
        list_mlu = []
        tm_lenght = self.tm.shape[0]
        for index in range(tm_lenght):
            critical_flow = self.get_topk_flow(self.tm[index])
            mlu = self.lp_solve(critical_flow, self.tm[index])
            list_mlu.append(mlu)
        print(np.mean(np.array(list_mlu)[1:]))

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
    
def run_topk(data, run_dir, args):
    writer = SummaryWriter(log_dir=run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    model_dir = str(run_dir / 'models')

    # ------------------------------------- init location ap/user  -------------------------------------
    args.data = data
    # breakpoint()
    episode_length = int(args.episode_length)
    print("episode length", episode_length)
    tm = data['test/gt']
    solver = RunTopK(args)
    solver.solve()
    # for data in 