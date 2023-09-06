import copy
import itertools
import os
import pickle
import time

import networkx as nx
import numpy as np
from gym import spaces
from joblib import delayed, Parallel
from tqdm import tqdm
from tqdm import trange


def load(path):
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')


def get_path(graph, i, j, k):
    """
    get a path for flow (i, j) with middle point k
    return:
        - list of edges on path, list of nodes in path or (None, None) in case of duplicated path or non-simple path
    """
    p_ik = shortest_path(graph, i, k)
    p_kj = shortest_path(graph, k, j)

    edges_ik, edges_kj = [], []
    # compute edges from path p_ik, p_kj (which is 2 lists of nodes)
    for u, v in zip(p_ik[:-1], p_ik[1:]):
        edges_ik.append((u, v))
    for u, v in zip(p_kj[:-1], p_kj[1:]):
        edges_kj.append((u, v))
    return edges_ik, edges_kj


def sort_paths(graph, paths):
    weights = [[sum(graph.get_edge_data(u, v)['weight'] for u, v in path)] for path in paths]
    paths = [path for weights, path in sorted(zip(weights, paths), key=lambda x: x[0])]
    return paths


def get_paths(graph, i, j):
    """
    get all simple path for flow (i, j) on graph G
    return:
        - flows: list of paths
        - path: list of links on path (u, v)
    """
    if i != j:
        N = graph.number_of_nodes()

        path_edges = []
        for k in range(N):
            try:
                edges = get_path(graph, i, j, k)
                path_edges.append(edges)
            except nx.NetworkXNoPath:
                pass
        return path_edges
    else:
        return []


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

    if not os.path.exists(os.path.join(datapath, 'topo')):
        os.makedirs(os.path.join(datapath, 'topo'))

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


def get_2sr_path(i, j, k, paths, nx_graph):
    """
    get a path for flow (i, j) with middle point k
    return:
        - list of edges on path, list of nodes in path or (None, None) in case of duplicated path or non-simple path
    """
    if i == k:
        return None, None

    p_ik = shortest_path(nx_graph, i, k)
    p_kj = shortest_path(nx_graph, k, j)
    p = p_ik[:-1] + p_kj

    # remove redundant paths and non-simple path and i == k
    if len(p) != len(set(p)) or p in paths:
        return None, None

    edges = []
    # compute edges from path p_ik, p_kj (which are 2 lists of nodes)
    for u, v in zip(p_ik[:-1], p_ik[1:]):
        edges.append((u, v))
    for u, v in zip(p_kj[:-1], p_kj[1:]):
        edges.append((u, v))
    return edges, p


def get_3sr_path(i, j, k1, k2, paths, nx_graph):
    """
    get a path for flow (i, j) with middle point k
    return:
        - list of edges on path, list of nodes in path or (None, None) in case of duplicated path or non-simple path
    """

    if k1 == k2:
        return get_2sr_path(i, j, k1, paths, nx_graph)
    elif i == k1 or j == k1:
        return get_2sr_path(i, j, k2, paths, nx_graph)
    elif j == k2 or i == k2:
        return get_2sr_path(i, j, k1, paths, nx_graph)
    else:

        p_ik1 = shortest_path(nx_graph, i, k1)
        p_k1k2 = shortest_path(nx_graph, k1, k2)
        p_k2j = shortest_path(nx_graph, k2, j)
        p = p_ik1[:-1] + p_k1k2[:-1] + p_k2j

        # remove redundant paths and non-simple path and i == k
        if len(p) != len(set(p)) or p in paths:
            return None, None

        edges = []
        # compute edges from path p_ik, p_kj (which are 2 lists of nodes)
        for u, v in zip(p_ik1[:-1], p_ik1[1:]):
            edges.append((u, v))
        for u, v in zip(p_k1k2[:-1], p_k1k2[1:]):
            edges.append((u, v))
        for u, v in zip(p_k2j[:-1], p_k2j[1:]):
            edges.append((u, v))
        return edges, p


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


def calculate_agent2link(args, flow2link):
    graph = args.nx_graph
    save_path = os.path.join(args.data_folder, f'topo/{args.n_sr}sr_agent2link_{args.dataset}.pkl')

    if args.rank == 0:
        if os.path.exists(save_path):
            data = load(save_path)
            agent2link = data['agent2link']
        else:
            link2index = get_link2idex(graph)
            agent2link = []
            for i in range(args.num_node):
                agent2link_i = []
                for j in range(args.num_node):
                    paths = flow2link[i, j]
                    for path in paths:
                        for u, v in path:
                            agent2link_i.append(link2index[u, v])

                agent2link_i = np.array(agent2link_i)
                agent2link_i = np.unique(agent2link_i)
                agent2link.append(agent2link_i)

            data = {
                'agent2link': agent2link
            }
            save(save_path, data)
    else:
        while not os.path.exists(save_path):
            time.sleep(5)

        data = load(save_path)
        agent2link = data['agent2link']

    return agent2link


def calculate_flow2node_sp(args):  # flow2node for shortest path traffic_routing

    saved_path = os.path.join(args.data_folder, f'topo/sp_flow2node_{args.dataset}.pkl')
    if os.path.exists(saved_path):
        data = load(saved_path)
        flow2node = data['flow2node']
    else:

        flow2node = {}
        graph = args.nx_graph
        n_node = args.num_node

        for i, j in itertools.product(range(n_node), range(n_node)):
            nodes = shortest_path(graph, i, j)
            nodes = np.unique(nodes)
            flow2node[i, j] = nodes
        data = {
            'flow2node': flow2node
        }
        save(saved_path, data)

    return flow2node


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

import gymnasium

def set_obs_space(n_agents, args):
    """
        n_agents: number of agents
        ub: maximum number of action of each agent
    """
    hist_step = args.input_len  # number of steps of historical traffic volumes (scaled traffic from 0->1)
    num_node = args.num_node  # total number of flows per agent
    num_link = args.num_link

    if args.action_state == 0:
        num_flow = num_node  # only consider the topk-largest flows
    elif args.action_state == 1:
        num_flow = num_node
    else:
        raise NotImplementedError

    if args.obs_state == 1:
        observation_space = []
        for i in range(n_agents):
            # observation of each agent: link utilization ratio
            #             (num_link)
            observation_space.append(gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(num_flow + num_link,), dtype=np.float64))

    elif args.obs_state == 2:
        observation_space, action_space = [], []
        for i in range(n_agents):
            # observation of each agent:
            #             (num_flow, [dstID] + [demand]*hist_step + [prev_action] + num_link)
            observation_space.append(gymnasium.spaces.Box(low=0.0, high=1.0,
                                                shape=(num_flow * 2 + num_link,), dtype=np.float64))

    elif args.obs_state == 3:
        observation_space, action_space = [], []
        for i in range(n_agents):
            # observation of each agent:
            #         (num_flow, [dstID] + [demand]*hist_step + [prev_action] + num_link + [partial tm: n_node*n_node])
            observation_space.append(gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=(num_flow + num_link + num_node * num_node,),
                dtype=np.float64))

    elif args.obs_state == 4:  # similar to 3 but no link u
        observation_space, action_space = [], []
        for i in range(n_agents):
            # observation of each agent:
            #         (num_flow, [dstID] + [pred tm: n_node*n_node])
            observation_space.append(gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=((num_flow + num_node * num_node + num_link),),
                dtype=np.float64))

    elif args.obs_state == 5:  # similar to 6 but using cs
        observation_space, action_space = [], []
        for i in range(n_agents):
            # observation of each agent:
            #         (num_flow, [dstID] + [pred tm: n_node*n_node])
            observation_space.append(gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=((num_flow + num_node * num_node),),
                dtype=np.float64))

    elif args.obs_state == 6:
        observation_space, action_space = [], []
        for i in range(n_agents):
            # observation of each agent:
            #         (num_flow, [dstID] + [demand]*hist_step + [prev_action] + num_link + [pred tm: n_node*n_node])
            observation_space.append(gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=((num_flow * 2 + num_node * num_node),),
                dtype=np.float64))
    elif args.obs_state == 7:
        observation_space, action_space = [], []
        for i in range(n_agents):
            # observation of each agent:
            #         (num_flow, [dstID] + [demand]*hist_step + [prev_action] + num_link + [pred tm: n_node*n_node])
            observation_space.append(gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=((num_flow * 2),),
                dtype=np.float64))
    elif args.obs_state == 8:
        observation_space, action_space = [], []
        for i in range(n_agents):
            # observation of each agent: [dstID, demand, all_TMs]
            observation_space.append(gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=((num_flow * 2 + num_node * num_node),),
                dtype=np.float64))
    else:
        raise NotImplementedError('state {} is not supported'.format(args.obs_state))

    return observation_space


def set_action_space(n_agents, args):
    """
        n_agents: number of agents
        ub: maximum number of action of each agent
    """
    num_node = args.num_node  # total number of flows per agent
    num_flow = num_node  # only consider the topk-largest flows
    num_path = args.n_path

    if args.action_state == 0:
        action_space = []
        action_nvec = np.zeros(shape=(num_flow,))
        action_nvec[:] = num_path

        for i in range(n_agents):
            action_space.append(gymnasium.spaces.MultiDiscrete(nvec=[num_path] * num_flow))
    elif args.action_state == 1:
        action_space = []
        action_nvec = np.zeros(shape=(num_flow + num_flow,))
        action_nvec[:num_flow] = num_node
        action_nvec[num_flow:] = num_path
        for i in range(n_agents):
            action_space.append(gymnasium.spaces.MultiDiscrete(nvec=action_nvec))

    else:
        raise NotImplementedError('state {} is not supported'.format(args.obs_state))

    return action_space


def set_global_state_space(args, observation_space):
    num_node = args.num_node  # total number of flows per agent
    num_link = args.num_link
    return_state_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(num_link + (num_node * num_node),),
                                        dtype=np.float64)


    return return_state_space


def set_action_mask(n_agents, n_node, ub, num_path):
    action_mask = []
    for i in range(n_agents):
        mask_i = np.zeros(shape=(n_node, num_path))
        for j in range(n_node):
            mask_i[j, 0:ub[i, j]] = 1

        action_mask.append(mask_i)

    action_mask = np.array(action_mask)
    return action_mask


def get_segment2pathid(flow2link, args):
    num_node = args.num_node
    nx_graph = args.nx_graph

    os.makedirs(os.path.join(args.data_folder, 'topo'), exist_ok=True)
    save_path = os.path.join(args.data_folder, 'topo/segment2pathid_{}.pk'.format(args.dataset))
    if not os.path.isfile(save_path) and args.rank == 0:

        segment2pathid = {}
        for i, j, k in itertools.product(range(num_node), range(num_node), range(num_node)):
            # print(i, j, k)
            if i == j or k == i or k == j:
                segment2pathid[i, j, k] = 0
                continue

            p_ik = shortest_path(nx_graph, i, k)
            p_kj = shortest_path(nx_graph, k, j)
            path = p_ik[:-1] + p_kj

            if len(path) == 0:
                segment2pathid[i, j, k] = 0
            else:
                while len(path) != len(set(path)):
                    list_node = []
                    for index in range(len(path)):
                        if path[index] not in list_node:
                            list_node.append(path[index])
                        else:
                            idx = list_node.index(path[index])
                            path = path[:idx + 1] + path[index + 1:]
                            break

                optimal_path = []
                for u, v in zip(path[:-1], path[1:]):
                    optimal_path.append((u, v))

                segment2pathid[i, j, k] = 0
                path_list = flow2link[i, j]
                for path_id, ls_path in enumerate(path_list):
                    if optimal_path == ls_path:
                        segment2pathid[i, j, k] = path_id
                        break

        with open(save_path, 'wb') as f:
            pickle.dump(segment2pathid, f)
    else:
        while not os.path.exists(save_path):
            time.sleep(5)

        with open(save_path, "rb") as f:
            segment2pathid = pickle.load(f)

    return segment2pathid


def count_routing_change(solution1, solution2):
    return np.sum(solution1 != solution2)


def get_rc(n_agents, rules, prev_rules, steps):
    if steps == 0:
        return np.zeros(shape=(n_agents,))
    route_changes = []
    for i in range(n_agents):
        rc = count_routing_change(rules[i], prev_rules[i])
        route_changes.append(rc)
    route_changes = np.array(route_changes)
    return route_changes


def get_optimal_solution(solver, tm, step, routing_cycle, n_node, n_timesteps, len_tra):
    """
    obtain near-optimal (60s timeout) solution for the next traffic_routing cycle
    tm: the maximum traffic matrix of the next traffic_routing cycle shape [node, node]
    solution: the traffic_routing rule shape [ node * node, ]
    """
    if step >= n_timesteps:
        solution = solver.init_solution()
    else:
        tm = tm[step, len_tra:]
        tm = np.reshape(tm, newshape=(routing_cycle, -1))
        tm = np.max(tm, axis=0)
        tm = np.reshape(tm, newshape=(n_node, n_node))
        solution = solver.solve(tm)

    solution = np.argmax(solution, axis=1)
    return solution


def extract_results(results):
    mlus, solutions = [], []
    for _mlu, _solution in results:
        mlus.append(_mlu)
        solutions.append(_solution)

    mlus = np.stack(mlus, axis=0)
    solutions = np.stack(solutions, axis=0)

    return mlus, solutions


def p0_optimal_solver(solver_name, solver, gt_tms, num_node, args):
    gt_tms = gt_tms.reshape((-1, num_node, num_node))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))

    num_flow_topk = int(args.topk * num_node * num_node)

    u = []
    solutions = []
    iter = trange(gt_tms.shape[0])
    for i in iter:

        if solver_name == 'solver_all':
            solver.solve(gt_tms[i])
        elif solver_name == 'solver_topk':
            tm = copy.deepcopy(gt_tms[i])
            tm = tm.flatten()
            topk_idx = np.argsort(tm)[::-1][:num_flow_topk]
            topk_flows_idx = [(int(idx / num_node), int(idx % num_node)) for idx in topk_idx]
            solver.solve(gt_tms[i], topk_flows_idx)

        else:
            raise NotImplementedError

        solution = solver.solution
        solutions.append(solution)
        util = solver.evaluate(gt_tms[i], solution)
        print(util, solver.var_dict['theta'])
        u.append(util)
        iter.set_description(f'[{solver_name}] Rank {args.rank} step={i} u={util}')

    solutions = np.stack(solutions, axis=0)
    return u, solutions


def optimal_solver(solver_name, solver, tms, args):
    print(f'|--- Start getting optimal solution {args.rank}')

    num_node = args.num_node

    mlus, solution = p0_optimal_solver(solver_name=solver_name, solver=solver, gt_tms=tms, num_node=num_node, args=args)

    solution = np.reshape(solution, newshape=(-1, num_node, num_node, num_node))

    solution = np.argmax(solution, axis=-1)  # (time, num_node, num_node) [i,j]=index_intermediate_node
    print(f'|--- Finish getting optimal solution {args.rank}')

    return mlus, solution


def heuristic_solver(solver, gt_tms, num_node, rank):
    gt_tms = gt_tms.reshape((-1, num_node, num_node))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))

    mlus = []
    solutions = []

    solution = solver.initialize_solution()
    # routing the first step using the initial solution
    mlus.append(solver.evaluate(gt_tms[0], solution))
    solutions.append(solution)
    # using traffic from the previous step to solve the problem. the solution is used to evaluate the traffic of the
    # next step. (collect data -> solve problem -> apply rules) --> rules are applied for the next step
    solution = solver.solve(tm=gt_tms[0], solution=None)  # solve backtrack solution (line 131)
    iter = trange(1, gt_tms.shape[0], 1)
    for i in iter:
        mlus.append(solver.evaluate(gt_tms[i], solution))
        p_solution = copy.deepcopy(solution)
        solutions.append(solution)
        solution = solver.solve(tm=gt_tms[i], solution=p_solution)  # solve backtrack solution (line 131)
        iter.set_description(f'[Heuristic_solver] Rank {rank} step={i}')

    mlus = np.array(mlus)
    solutions = np.array(solutions)
    return mlus, solutions


def get_base_solutions(solver_name, solver, tms, segment2pathid, flow2link, ub, args, is_eval=False):
    data_folder = args.data_folder
    os.makedirs(os.path.join(data_folder, f'core/base_solution'), exist_ok=True)

    dataset = args.dataset
    rank = args.rank
    nx_graph = args.nx_graph
    num_node = args.num_node
    num_timesteps = tms.shape[0]

    if solver_name == 'solver_topk':
        _solver_name = solver_name + f'_{args.topk}'
    else:
        _solver_name = solver_name

    if is_eval:
        save_solution_path = os.path.join(data_folder, f'core/base_solution/{dataset}_{_solver_name}-{rank}_eval.npz')
    else:
        save_solution_path = os.path.join(data_folder, f'core/base_solution/{dataset}_{_solver_name}-{rank}.npz')

    if not os.path.exists(save_solution_path):

        if solver_name == 'ls2sr':
            _, solutions = heuristic_solver(solver, tms, num_node, rank)

            mpus = np.zeros((num_timesteps, num_node))
            mlus = np.zeros((num_timesteps,))
            for t in tqdm(range(num_timesteps)):
                tm = tms[t]
                routing_rule = solutions[t]
                mlu, link_util = do_routing(tm, routing_rule, nx_graph, num_node, flow2link, ub)
                path_util = get_path_mlu(routing_rule, num_node, flow2link, link_util, nx_graph)
                mpus[t] = np.max(path_util, axis=1)
                mlus[t] = mlu
                # assert mlu == mlus_[t]
        elif solver_name == 'srls':

            solutions = np.zeros(shape=(num_timesteps, num_node, num_node))
            _, srls_solutions = heuristic_solver(solver, tms, num_node, rank)
            # breakpoint()
            srls_solutions = np.argmax(srls_solutions, axis=-1)  # (time, num_node, num_node)
            # [i,j] = index_intermediate_node

            mpus = np.zeros((num_timesteps, num_node))
            mlus = np.zeros((num_timesteps,))

            for t in range(num_timesteps):
                for i, j in itertools.product(range(num_node), range(num_node)):
                    m = srls_solutions[t, i, j]
                    try:
                        solutions[t, i, j] = segment2pathid[i, j, m]
                    except:
                        # print(i, j, m)
                        raise RuntimeError

                tm = tms[t]
                routing_rule = solutions[t]
                mlu, link_util = do_routing(tm, routing_rule, nx_graph, num_node, flow2link, ub)
                path_util = get_path_mlu(routing_rule, num_node, flow2link, link_util, nx_graph)
                mpus[t] = np.max(path_util, axis=1)
                mlus[t] = mlu
                # assert mlu == mlus_[t]

        elif 'solver' in solver_name:
            solutions = np.zeros(shape=(num_timesteps, num_node, num_node))
            _, optimal_solutions = optimal_solver(solver_name, solver, tms, args)
            mpus = np.zeros((num_timesteps, num_node))
            mlus = np.zeros((num_timesteps,))

            for t in range(num_timesteps):
                for i, j in itertools.product(range(num_node), range(num_node)):
                    m = optimal_solutions[t, i, j]
                    try:
                        solutions[t, i, j] = segment2pathid[i, j, m]
                    except:
                        # print(i, j, m)
                        raise RuntimeError

                tm = tms[t]
                routing_rule = solutions[t]
                mlu, link_util = do_routing(tm, routing_rule, nx_graph, num_node, flow2link, ub)
                path_util = get_path_mlu(routing_rule, num_node, flow2link, link_util, nx_graph)
                mpus[t] = np.max(path_util, axis=1)
                mlus[t] = mlu
                # assert mlu == mlus_[t]
        else:
            raise NotImplementedError

        # print(mlus.shape, solutions.shape, mpus.shape)
        data = {
            'solutions': solutions,
            'mlus': mlus,
            'mpus': mpus
        }
        with open(save_solution_path, 'wb') as fp:
            np.savez_compressed(fp, **data)
    else:
        data = np.load(save_solution_path)
        solutions = data['solutions']
        mlus = data['mlus']
        mpus = data['mpus']

    if args.rank == 0:
        concate_base_solution(solver_name, args, is_eval)

    return solutions, mlus, mpus


def concate_base_solution(solver_name, args, is_eval=False):
    n = args.n_eval_rollout_threads if is_eval else args.n_rollout_threads

    solutions = []
    mlus = []
    mpus = []

    if solver_name == 'solver_topk':
        _solver_name = solver_name + f'_{args.topk}'
    else:
        _solver_name = solver_name

    data_folder = os.path.join(args.data_folder, 'core/base_solution')
    dataset = args.dataset
    for i in range(n):

        if is_eval:
            path_file = f'{data_folder}/{dataset}_{_solver_name}-{i}_eval.npz'
        else:
            path_file = f'{data_folder}/{dataset}_{_solver_name}-{i}.npz'
        while not os.path.exists(path_file):
            time.sleep(5)

        print(f'|--- Agent {i} eval={is_eval} finished!')
        data = np.load(path_file)

        solution = data['solutions']
        mlu = data['mlus']
        mpu = data['mpus']

        solutions.append(solution)
        mlus.append(mlu)
        mpus.append(mpu)

    solutions = np.concatenate(solutions, axis=0)
    mlus = np.concatenate(mlus, axis=0)
    mpus = np.concatenate(mpus, axis=0)

    # print(solutions.shape, mlus.shape, mpus.shape)
    data = {
        'solutions': solutions,
        'mlus': mlus,
        'mpus': mpus
    }
    if is_eval:
        save_path = f'{data_folder}/{dataset}_{_solver_name}_eval.npz'
    else:
        save_path = f'{data_folder}/{dataset}_{_solver_name}.npz'

    with open(save_path, 'wb') as fp:
        np.savez_compressed(fp, **data)


def has_path(i, j, flow2link):
    if flow2link[(i, j)]:
        return True
    return False


def link_in_path(i, j, u, v, k, flow2link):
    if (u, v) in flow2link[(i, j)][k]:
        return 1
    else:
        return 0


def do_routing(tm, routing_rule, nx_graph, num_node, flow2link, ub):
    """
        traffic_routing rule: (node, node)
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


def do_routing_with_link_failure(tm, routing_rule, nx_graph, num_node, flow2link, ub, link_fail_id):
    """
        traffic_routing rule: (node, node)
    """
    # extract utilization
    link_utils = []

    mlu = 0
    for link_id, (u, v) in enumerate(nx_graph.edges):
        if link_id in link_fail_id:
            traffic_load = 0.0
        else:
            traffic_load = 0.0
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
