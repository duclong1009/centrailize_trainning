import os

import numpy as np
import tqdm
from joblib import delayed, Parallel

from .ls2sr import LS2SRSolver
from .max_step_sr import MaxStepSRSolver
from .multi_step_sr import MultiStepSRSolver
from .oblivious_routing import ObliviousRoutingSolver
from .one_step_sr import OneStepSRSolver
from .shortest_path_routing import SPSolver
from .te_util import load_network_topology, compute_path, createGraph_srls, extract_results, \
    get_route_changes_heuristic, get_route_changes_optimal


def p0_optimal_solver(solver, tms, gt_tms, num_node):
    gt_tms = gt_tms.reshape((-1, num_node, num_node))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))

    u = []
    solutions = []
    for i in range(gt_tms.shape[0]):
        try:
            solver.solve(gt_tms[i])
        except:
            pass

        solution = solver.solution
        solutions.append(solution)
        u.append(solver.evaluate(gt_tms[i], solution))

    solutions = np.stack(solutions, axis=0)
    return u, solutions


def p2_heuristic_solver(solver, tms, gt_tms, num_node, p_solution=None):
    u = []
    tms = tms.reshape((-1, num_node, num_node))
    gt_tms = gt_tms.reshape((-1, num_node, num_node))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(num_node))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))
    tm = tms.reshape((num_node, num_node))

    solution = solver.solve(tm=tm, solution=p_solution)  # solve backtrack solution (line 131)

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i], solution))
    return u, solution


def p1_optimal_solver(solver, tms, gt_tms, num_node):
    tms, gt_tms = prepare_traffic_data(tms, gt_tms, num_node)

    return run_optimal_mssr(solver, tms, gt_tms)


def p3_optimal_solver(solver, tms, gt_tms, num_node):
    tms, gt_tms = prepare_traffic_data(tms, gt_tms, num_node)

    return run_optimal_mssr(solver, tms, gt_tms)


def p2_optimal_solver(solver, tms, gt_tms, num_node):
    tms, gt_tms = prepare_traffic_data(tms, gt_tms, num_node)
    tms = tms.reshape((num_node, num_node))

    return run_optimal_mssr(solver, tms, gt_tms)


def prepare_traffic_data(tms, gt_tms, num_node):
    tms = tms.reshape((-1, num_node, num_node))
    gt_tms = gt_tms.reshape((-1, num_node, num_node))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(num_node))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))
    return tms, gt_tms


def run_optimal_mssr(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        solver.solution = solver.init_solution()
    solution = solver.solution
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i], solution=solution))

    return u, solution

