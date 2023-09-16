import os 
import numpy as np
import networkx as nx
import itertools
import gymnasium
from joblib import delayed, Parallel
import pickle 
import time

def get_idx2flow(args):
    num_node = args.num_node
    idx2flow = {}
    k = 0
    for i,j in itertools.product(range(num_node), range(num_node)):
        if i != j:
            idx2flow[k] = (i,j)
            k += 1
    return idx2flow


def set_obs_space(args):
    observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(args.num_node * args.num_node,), dtype=np.float64)
    return observation_space

def set_action_space(args):
    num_node = args.num_node  # total number of flows per agent
    num_path = args.n_path
    num_flow = num_node * num_node
    n_candidate = num_node * (num_node - 1)
    action_space = gymnasium.spaces.MultiBinary(n_candidate)
    return action_space

def load_network_topology(dataset, data_folder):
    path = os.path.join(data_folder, f'{dataset}.npz')
    data = np.load(path)

    adj = data['adj_mx']
    capacity_mx = data['capacity_mx']
    cost_mx = data['cost_mx']

    # print(adj.shape)
    num_node = adj.shape[0]
    # initialize graph
    G = nx.DiGraph()
    for i in range(num_node):
        G.add_node(i, label=str(i))
    # add weight, capacity, delay to edge attributes

    for src in range(num_node):
        for dst in range(num_node):
            if adj[src, dst] == 1:
                G.add_edge(src, dst, weight=cost_mx[src, dst],
                           capacity=capacity_mx[src, dst])
    return G

def link_in_path(i, j, u, v, k, flow2link):
    if (u, v) in flow2link[(i, j)][k]:
        return 1
    else:
        return 0

def get_set_ENH(args, flow2link):
    set_ENH = {}

    num_node = args.num_node
    for i in range(num_node):
        for j in range(num_node):
            set_ENH[(i, j)] = []
            if i != j:
                sp = flow2link[(i, j)][0]
                for path in flow2link[(i, j)]:
                    if len(path) == len(sp):
                        next_hop = path[0][1]
                        if next_hop not in set_ENH[(i, j)]:
                            set_ENH[(i, j)].append(next_hop)
    return set_ENH

def has_path(i, j, flow2link):
    if flow2link[(i, j)]:
        return True
    return False

def save(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

def get_segments(graph):
    n = graph.number_of_nodes()
    segments = {}
    segments_edges = Parallel(n_jobs=os.cpu_count() * 2)(delayed(get_paths)(graph, i, j)
                                                         for i, j in itertools.product(range(n), range(n)))
    for i, j in itertools.product(range(n), range(n)):
        segments[i, j] = segments_edges[i * n + j]

    return segments

def compute_path(graph, dataset, datapath, rank):
    path = os.path.join(datapath, 'topo/2sr_segment_{}.pkl'.format(dataset))

    if not os.path.exists(path) and rank == 0:
        segments = get_segments(graph)
        data = {
            'segments': segments,
        }
        save(path, data)
    else:
        while not os.path.exists(path):
            time.sleep(5)

        data = load(path)
        segments = data['segments']

    return segments

def do_routing(tm, routing_rule, nx_graph, num_node, flow2link, ub):
    """
        routing rule: (node, node)
    """
    # extract utilization
    link_utils = []

    mlu = 0
    for link_id, (u, v) in enumerate(nx_graph.edges):
        traffic_load = 0
        for i, j in itertools.product(range(num_node), range(num_node)):
            if has_path(i, j, flow2link):
                k = int(routing_rule[i, j])
                if k >= ub[i, j]:
                    raise RuntimeError(f' {i} {j} {k} {ub[i, j]}')
                else:
                    traffic_load += link_in_path(i, j, u, v, k, flow2link) * tm[i, j]

        capacity = nx_graph.get_edge_data(u, v)['capacity']
        utilization = traffic_load / capacity
        link_utils.append(utilization)
        if utilization >= mlu:
            mlu = utilization

    link_utils = np.asarray(link_utils)
    return mlu, link_utils

def get_path_mlu(routing_rule, num_node, flow2link, link_util, nx_graph):
    path_mlu = np.zeros(shape=(num_node, num_node))

    for i, j in itertools.product(range(num_node), range(num_node)):
        if i == j:
            path_mlu[i, j] = 0.0
        else:
            k = int(routing_rule[i, j])
            path = flow2link[(i, j)][k]
            p_mlu = 0.0
            for link_id, (u, v) in enumerate(nx_graph.edges):
                if (u, v) in path and link_util[link_id] > p_mlu:
                    p_mlu = link_util[link_id]

            path_mlu[i, j] = p_mlu
    return path_mlu

def initialize_link2flow(args):
    """
    link2flow is a dictionary:
        - key: link id (u, v)
        - value: list of flows id (i, j)
    """
    nx_graph = args.nx_graph
    link2flow = {}
    for u, v in nx_graph.edges:
        link2flow[(u, v)] = []
    return link2flow

def get_2sr_paths(i, j, args):
    """
    get all simple path for flow (i, j) on graph G
    return:
        - flows: list of paths
        - path: list of links on path (u, v)
    """
    nx_graph = args.nx_graph

    n_node = args.num_node
    if i != j:
        path_edges = []
        paths = []
        for k in range(n_node):
            try:
                edges, path = get_2sr_path(i, j, k, paths, nx_graph)
                if edges is not None:
                    path_edges.append(edges)
                    paths.append(path)
            except nx.NetworkXNoPath:
                pass
        # sort paths by their total link weights for heuristic
        path_edges = sort_paths(nx_graph, path_edges)
        return path_edges
    else:
        return []

def get_3sr_paths(i, j, args):
    """
    get all simple path for flow (i, j) on graph G
    return:
        - flows: list of paths
        - path: list of links on path (u, v)
    """
    nx_graph = args.nx_graph

    n_node = args.num_node
    if i != j:
        path_edges = []
        paths = []
        for k1, k2 in itertools.product(range(n_node), range(n_node)):
            try:
                edges, path = get_3sr_path(i, j, k1, k2, paths, nx_graph)
                if edges is not None:
                    path_edges.append(edges)
                    paths.append(path)
            except nx.NetworkXNoPath:
                pass
        # sort paths by their total link weights for heuristic
        path_edges = sort_paths(nx_graph, path_edges)
        return path_edges
    else:
        return []
    
def initialize_flow2link(args):
    """
    flow2link is a dictionary:
        - key: flow id (i, j)
        - value: list of paths
        - path: list of links on path (u, v)
    """
    n_node = args.num_node
    flow2link = {}
    if args.n_sr == 2:
        list_paths = Parallel(n_jobs=10)(delayed(get_2sr_paths)(i, j, args)
                                         for i, j in itertools.product(range(n_node), range(n_node)))
    else:
        list_paths = Parallel(n_jobs=10)(delayed(get_3sr_paths)(i, j, args)
                                         for i, j in itertools.product(range(n_node), range(n_node)))

    # for i, j in itertools.product(range(self.n_node), range(self.n_node)):
    #     flow2link[i, j] = self._get_paths(i, j)

    for i, j in itertools.product(range(n_node), range(n_node)):
        flow2link[i, j] = list_paths[i * n_node + j]

    return flow2link

def compute_lssr_path(args):
    path = os.path.join(args.data_folder, f'topo/ls{args.n_sr}sr_segment_{args.dataset}.pkl')

    if not os.path.exists(path) and args.rank == 0:

        # print('|--- Compute segment and save to {}'.format(path))
        link2flow = initialize_link2flow(args)
        flow2link = initialize_flow2link(args)
        data = {
            'link2flow': link2flow,
            'flow2link': flow2link,
        }
        save(path, data)

    else:

        while not os.path.exists(path):
            time.sleep(5)

        # print('|--- Load precomputed segment from {}'.format(path))
        data = load(path)
        link2flow = data['link2flow']
        flow2link = data['flow2link']

    return link2flow, flow2link

def get_solution_bound(flow2link, args):
    n_node = args.num_node

    ub = np.empty([n_node, n_node], dtype=int)
    for i, j in itertools.product(range(n_node), range(n_node)):
        ub[i, j] = len(flow2link[(i, j)])
    ub[ub == 0] = 1
    return ub

def load(path):
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def calculate_flow2node_sr(args, flow2link):
    saved_path = os.path.join(args.data_folder, f'topo/{args.n_sr}sr_flow2node_{args.dataset}.pkl')
    if os.path.exists(saved_path):
        data = load(saved_path)
        flow2node = data['flow2node']
    else:

        flow2node = {}
        n_node = args.num_node

        for i, j in itertools.product(range(n_node), range(n_node)):
            paths = flow2link[i, j]
            flow2node[i, j] = []
            for path in paths:
                nodes = []
                for u, v in path:
                    nodes.append(u)
                    nodes.append(v)
                nodes = np.array(nodes)
                nodes = np.unique(nodes)
                flow2node[i, j].append(nodes)
        data = {
            'flow2node': flow2node
        }
        save(saved_path, data)

    return flow2node

def get_link2idex(graph):
    link2index = {}
    for link_id, (u, v) in enumerate(graph.edges):
        link2index[u, v] = link_id

    return link2index