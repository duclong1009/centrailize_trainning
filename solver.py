import itertools

import numpy as np
import pulp as pl
from pulp.apis.coin_api import PULP_CBC_CMD

class Solver:
    def __init__(self, graph, timeout, verbose):
        self.G = graph
        self.num_node = graph.number_of_nodes()
        self.num_edge = graph.number_of_edges()
        self.indices_edge = np.arange(self.num_edge)
        self.list_edges = list(self.G.edges)
        self.timeout = timeout
        self.verbose = verbose

    def solve(self, tm, solution=None, eps=1e-12):
        pass

    def evaluate(self, solution, tm):
        pass

class OneStepSRTopKSolver(Solver):

    def __init__(self,args, graph, timeout, verbose, list_link_strated_at, list_link_end_at,idx2flow, set_ENH, all_flow):
        super(OneStepSRTopKSolver, self).__init__(graph, timeout, verbose)
        """
        G: networkx Digraph, a network topology
        """
        
        self.problem = None
        self.var_dict = None
        self.solution = None
        self.solver = PULP_CBC_CMD(timeLimit=timeout, msg=False)
        self.num_node = args.num_node
        self.n_critical_flows = 0
        self.list_link_end_at = list_link_end_at
        self.list_link_strated_at = list_link_strated_at
        self.idx2flow = idx2flow
        self.set_ENH = set_ENH
        self.idx2edge, self.n_y = self.get_idx2edge()
        self.all_flow = all_flow
        self.capacity = {}
        for u, v in self.G.edges:
            self.capacity[(u,v)] = self.G.get_edge_data(u, v)['capacity']

    def get_idx2edge(self,):
        idx2edge = {}
        num_node = self.num_node
        count = 0
        for d in range(num_node):
            for u, v in self.G.edges:
                idx2edge[(u,v,d)] = count
                count += 1
                
        return idx2edge, count

    def create_problem(self, tm, critical_flow_idx):
        num_node = self.num_node
        self.n_critical_flows = len(critical_flow_idx)
        # 1) create optimization model
        lp_problem = pl.LpProblem("LP_Problem", pl.LpMinimize)
        uti = pl.LpVariable(name='utilization', lowBound=0.0, cat='Continuous')

        y = pl.LpVariable.dicts(name='y',
                                    indices=np.arange(self.n_y),
                                    cat='Continuous')

        # 2) objective function
        # minimize maximum link utilization
        lp_problem += uti

        #8bc 
        for i, j in self.G.edges:
            link_capacity = self.capacity[(i,j)]
            lp_problem += pl.lpSum([y[self.idx2edge[(i,j,d)]] for d in range(num_node)]) <= link_capacity * uti
        #8d 
        for flow_id in critical_flow_idx:
            i,d = flow_id
            sum_end = pl.lpSum([y[self.idx2edge[(k,i,d)]] for k in self.list_link_end_at[i]])
            sum_in = pl.lpSum([y[self.idx2edge[(i,k,d)]] for k in self.list_link_strated_at[i]])

            eq = sum_end - sum_in
            lp_problem += eq == -tm[i,d]
        
        
        #8e
        for flow in self.all_flow:
            if flow not in critical_flow_idx:
                i,d = flow
                len_set_ENH = len(self.set_ENH[(i,d)])
                for k in self.list_link_strated_at[i]:
                    if k in self.set_ENH[(i,d)]:
                        lp_problem += y[self.idx2edge[(i,k,d)]] == 0
                    else:
                        lp_problem += y[self.idx2edge[(i,k,d)]] == (pl.lpSum([y[self.idx2edge[(n,i,d)]] for n in self.list_link_end_at[i]]) + tm[i,d])/len_set_ENH
        
        #8f
        for d in range(num_node):
            sum_end = pl.lpSum([y[self.idx2edge[(k,d,d)]] for k in self.list_link_end_at[d]])
            sum_in = pl.lpSum([y[self.idx2edge[(d,k,d)]] for k in self.list_link_strated_at[d]])
            eq = sum_end - sum_in
            total_t = pl.lpSum([tm[s,d] for s in range(num_node)])
            lp_problem += eq == total_t

        #8g
        for d in range(num_node):
            for i,j in self.G.edges:
                lp_problem += y[self.idx2edge[(i,j,d)]] >=0

        return lp_problem, uti

    def init_solution(self):
        solution = np.zeros([self.num_node, self.num_node, self.num_node])
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            solution[i, j, i] = 1
        return solution

    def extract_solution(self, problem):
        # extract solution
        self.var_dict = {}
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue
        return self.var_dict['utilization'], self.var_dict

    def solve(self, tm, critical_flow_idx):
        self.critical_flow_idx = critical_flow_idx
        problem, x = self.create_problem(tm, critical_flow_idx)
        problem.solve(solver=self.solver)
        self.problem = problem
        self.solution, self.var_dict = self.extract_solution(problem)
        return self.solution, self.var_dict


if __name__ == "__main__":

    import json
    with open("critical_flow_idx.json", "r") as f:
        critical_flow_idx = json.load(f)
    # critical_flow_idx_ = {}
    # for u in critical_flow_idx.keys():
    #     critical_flow_idx_[int(u)] = critical_flow_idx[u]

    with open("idx2flow.json", "r") as f:
        idx2flow = json.load(f)
    idx2flow_ = {}
    for u in idx2flow.keys():
        idx2flow_[int(u)] = idx2flow[u]

    with open("list_link_end_at.json", "r") as f:
        list_link_end_at = json.load(f)

    list_link_end_at_ = {}
    for u in list_link_end_at.keys():
        list_link_end_at_[int(u)] = list_link_end_at[u]

    with open("list_link_stated_at.json", "r") as f:
        list_link_stated_at = json.load(f)
    list_link_stated_at_ = {}
    for u in list_link_stated_at.keys():
        list_link_stated_at_[int(u)] = list_link_stated_at[u]
    from network_environment.te_env.utils import *
    import numpy as np
    tm = np.load("tm.npy")

    args = {"num_node":12}
    # args.num_node = 12
    # breakpoint()
    nx_graph = load_network_topology("abilene", "data")
    solver = OneStepSRTopKSolver(args, nx_graph, 600,1, list_link_stated_at_, list_link_end_at_, idx2flow_, list_link_stated_at_ )
    # solver.create_problem(tm, critical_flow_idx)
    # breakpoint()
    u = solver.solve(tm, critical_flow_idx)
    
    num_node = 12
    all_flow_idx = []
    for i, j in itertools.product(range(num_node), range(num_node)):
        if i !=j:
            all_flow_idx.append([i,j])
    u_opt = solver.solve(tm, all_flow_idx)
    print(u, u_opt)