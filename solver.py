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

    def __init__(self, graph, timeout, verbose, list_link_strated_at, list_link_end_at):
        super(OneStepSRTopKSolver, self).__init__(graph, timeout, verbose)
        """
        G: networkx Digraph, a network topology
        """
        self.problem = None
        self.var_dict = None
        self.solution = None
        self.status = None
        self.solver = PULP_CBC_CMD(timeLimit=timeout, msg=False)

        self.critical_flow_idx = None
        self.n_critical_flows = 0
        self.list_link_end_at = list_link_end_at
        self.list_link_strated_at = list_link_strated_at

    def create_problem(self, tm, allocated_link_capacity):
        """
        Create LP problem for critical flows:
        tm: traffic matrix
        critical_flow_idx: list of critical flows [(src, dst)]
        allocated_link_capacity: remaining link capacity after routing non-critical flows as SP
        """
        self.n_critical_flows = len(self.critical_flow_idx)

        # 1) create optimization model
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')

        x = pl.LpVariable.dicts(name='x',
                                indexs=np.arange(self.n_critical_flows * self.num_node),
                                cat='Binary')

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function
        for u, v in self.G.edges:
            allocated_capacity = allocated_link_capacity[u, v]
            link_capacity = self.G.get_edge_data(u, v)['capacity']
            load = pl.lpSum(
                x[f * self.num_node + k] * tm[self.critical_flow_idx[f]] * self.edge_in_segment(
                    f, k, u, v) for f, k in itertools.product(range(self.n_critical_flows), range(self.num_node))
            )
            load = pl.lpSum([load, allocated_capacity])
            problem += load <= theta * link_capacity

        # 3) constraint function
        # ensure all traffic are routed
        for f in range(self.n_critical_flows):
            problem += pl.lpSum(x[f * self.num_node + k] for k in range(self.num_node)) == 1

        return problem, x

    def extract_solution(self, problem):

        solution = self.init_solution()
        # extract solution
        self.var_dict = {}
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue

        # self.solution = np.empty([self.num_node, self.num_node, self.num_node])
        for f, k in itertools.product(range(self.n_critical_flows), range(self.num_node)):
            src, dst = self.critical_flow_idx[f]
            solution[src, dst, k] = self.var_dict['x_{}'.format(f * self.num_node + k)]

        return solution


    def init_solution(self):
        solution = np.zeros([self.num_node, self.num_node, self.num_node])
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            solution[i, j, i] = 1
        return solution

    def init_routing(self, tm):
        """
        Calculate the remaining link capacity after routing non-critical flows as shortest path
        tm: the traffic matrix shape(num_node, num_node)
        critical_flow_idx: list of critical flows [(i, j),..]
        """
        allocated_link_capacity = {}
        solution = self.init_solution()

        for u, v in self.G.edges:
            load = 0
            for i, j in itertools.product(range(self.num_node), range(self.num_node)):
                if (i, j) not in self.critical_flow_idx:
                    load += sum(
                        [solution[i, j, k] * tm[i, j] * edge_in_segment(self.segments, i, j, k, u, v)
                         for k in range(self.num_node)])

            allocated_link_capacity[u, v] = load
        return allocated_link_capacity

    def solve(self, tm, critical_flow_idx, solution=None, eps=1e-12):
        self.critical_flow_idx = critical_flow_idx
        allocated_link_capacity = self.init_routing(tm=tm)
        problem, x = self.create_problem(tm, allocated_link_capacity)
        problem.solve(solver=self.solver)
        self.problem = problem
        self.solution = self.extract_solution(problem)
        return self.solution

    def get_paths(self, i, j):
        if i == j:
            list_k = [i]
        else:
            list_k = np.where(self.solution[i, j] > 0)[0]
        paths = []
        for k in list_k:
            path = []
            path += shortest_path(self.G, i, k)[:-1]
            path += shortest_path(self.G, k, j)
            paths.append((k, path))
        return paths