#!/usr/bin/env python3
"""
Gurobi implementation of the Uncapacitated Hub Location Problem with Multiple Assignments (UHLPMA).

Based on the path-based (compact) formulation (Equations 6-10) from:
Contreras, Cordeau, and Laporte (2011), Operations Research 59(6), pp. 1477-1490.

The model minimizes total fixed hub costs plus transportation costs, where commodities
are routed through one or two hubs with economies of scale on inter-hub links.
"""

import argparse
import json
import math
import time
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
_GUROBI_CODE_START_TIME = time.time()
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass


def load_instance(instance_path):
    """Load problem instance from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def compute_distances(nodes, TC):
    """Compute scaled Euclidean distances between all node pairs."""
    n = len(nodes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = nodes[i]['x'] - nodes[j]['x']
            dy = nodes[i]['y'] - nodes[j]['y']
            d = math.sqrt(dx * dx + dy * dy) * TC
            dist[i][j] = d
            dist[j][i] = d
    return dist


def solve_uhlpma(data, time_limit):
    """
    Solve UHLPMA using the path-based formulation (Equations 6-10).

    Uses hub edges e ∈ E (unordered pairs or singletons of hubs) and
    preprocesses to eliminate dominated hub edges per commodity.
    """
    nodes = data['nodes']
    commodities = data['commodities']
    params = data['parameters']

    n = len(nodes)
    tau = params['tau']
    TC = params['TC']
    chi = params['chi']
    delta = params['delta']

    H = list(range(n))  # All nodes are potential hubs (H = N)

    # Use distances from the instance if available, otherwise compute
    if 'distances' in data:
        raw_dist = data['distances']
        # Distances in the instance may already be scaled; check by comparing
        # Instance-schema contract: stored distances are already scaled by TC.
        dist = [[raw_dist[i][j] for j in range(n)] for i in range(n)]
    else:
        dist = compute_distances(nodes, TC)

    # Fixed costs
    f = [nodes[i]['fixed_cost'] for i in range(n)]

    # Build commodities list
    K = []
    for idx, comm in enumerate(commodities):
        K.append({
            'id': idx,
            'origin': comm['origin'],
            'destination': comm['destination'],
            'flow': comm['flow']
        })

    # Merge opposite commodity pairs when chi == delta (Property 3)
    if abs(chi - delta) < 1e-9:
        merged = {}
        for k in K:
            o, d_node = k['origin'], k['destination']
            key = (min(o, d_node), max(o, d_node))
            if key in merged:
                merged[key]['flow'] += k['flow']
            else:
                merged[key] = {
                    'id': len(merged),
                    'origin': key[0],
                    'destination': key[1],
                    'flow': k['flow']
                }
        K = list(merged.values())
        for idx, k in enumerate(K):
            k['id'] = idx

    # Compute hub edges E: all singletons {i} and pairs {i,j} for i,j in H
    # For each commodity k, compute F_{ek} and build E_k

    def compute_F_hat(o_k, d_k, W_k, i, j):
        """Compute directed transportation cost F_hat_{ijk}."""
        return W_k * (chi * dist[o_k][i] + tau * dist[i][j] + delta * dist[j][d_k])

    def compute_F_edge(o_k, d_k, W_k, i, j=None):
        """Compute undirected transportation cost for hub edge e and commodity k."""
        if j is None:
            # Singleton edge {i}
            return compute_F_hat(o_k, d_k, W_k, i, i)
        else:
            # Edge {i,j}, take minimum of both directions
            return min(compute_F_hat(o_k, d_k, W_k, i, j),
                       compute_F_hat(o_k, d_k, W_k, j, i))

    # Build E_k for each commodity k
    # E_k includes singleton edges and non-dominated pair edges
    E_k_list = []  # For each commodity k: list of (edge_tuple, F_ek)

    for k in K:
        o_k = k['origin']
        d_k = k['destination']
        W_k = k['flow']

        # Compute singleton costs
        singleton_costs = {}
        for i in H:
            singleton_costs[i] = compute_F_edge(o_k, d_k, W_k, i)

        edges_k = []

        # Add all singleton edges
        for i in H:
            edges_k.append(((i,), singleton_costs[i]))

        # Property 2: if o(k) == d(k) and distances symmetric, only singleton edges
        if o_k != d_k:
            # Add non-dominated pair edges (Property 1)
            for i, j in combinations(H, 2):
                F_ej = compute_F_edge(o_k, d_k, W_k, i, j)
                # Property 1: eliminate if F_ek > min(F_{e1,k}, F_{e2,k})
                if F_ej < min(singleton_costs[i], singleton_costs[j]):
                    edges_k.append(((i, j), F_ej))

        E_k_list.append(edges_k)

    # Build Gurobi model
    model = gp.Model("UHLPMA")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # z_i: binary, 1 if hub opened at node i
    z = model.addVars(H, vtype=GRB.BINARY, name="z")

    # x_{ek}: continuous, routing variable for commodity k on hub edge e
    x = {}
    for k_idx, k in enumerate(K):
        for e_idx, (edge, F_ek) in enumerate(E_k_list[k_idx]):
            x[k_idx, e_idx] = model.addVar(
                lb=0.0, ub=1.0, obj=F_ek, vtype=GRB.CONTINUOUS,
                name=f"x_{k_idx}_{e_idx}"
            )

    # Objective: minimize sum f_i z_i + sum F_ek x_ek
    model.setObjective(
        gp.quicksum(f[i] * z[i] for i in H) +
        gp.quicksum(F_ek * x[k_idx, e_idx]
                     for k_idx in range(len(K))
                     for e_idx, (edge, F_ek) in enumerate(E_k_list[k_idx])),
        GRB.MINIMIZE
    )

    # Constraint (7): sum_{e in E_k} x_{ek} = 1 for all k
    for k_idx in range(len(K)):
        model.addConstr(
            gp.quicksum(x[k_idx, e_idx]
                        for e_idx in range(len(E_k_list[k_idx]))) == 1,
            name=f"assign_{k_idx}"
        )

    # Constraint (8): sum_{e in E_k: i in e} x_{ek} <= z_i for all i in H, k in K
    for i in H:
        for k_idx in range(len(K)):
            relevant = []
            for e_idx, (edge, F_ek) in enumerate(E_k_list[k_idx]):
                if i in edge:
                    relevant.append(x[k_idx, e_idx])
            if relevant:
                model.addConstr(
                    gp.quicksum(relevant) <= z[i],
                    name=f"hub_{i}_{k_idx}"
                )

    # Constraint: at least one hub must be open (constraint 17)
    model.addConstr(gp.quicksum(z[i] for i in H) >= 1, name="at_least_one_hub")

    # Optimize
    model.optimize()

    # Extract solution
    result = {}

    if model.SolCount > 0:
        result['objective_value'] = model.ObjVal
        result['status'] = model.Status
        result['gap'] = model.MIPGap if hasattr(model, 'MIPGap') else 0.0

        # Open hubs
        open_hubs = [i for i in H if z[i].X > 0.5]
        result['open_hubs'] = open_hubs
        result['num_open_hubs'] = len(open_hubs)

        # Hub fixed cost
        result['total_fixed_cost'] = sum(f[i] for i in open_hubs)
        result['total_transport_cost'] = result['objective_value'] - result['total_fixed_cost']

        # Commodity assignments
        assignments = []
        for k_idx, k in enumerate(K):
            for e_idx, (edge, F_ek) in enumerate(E_k_list[k_idx]):
                if x[k_idx, e_idx].X > 0.5:
                    assignments.append({
                        'commodity': k_idx,
                        'origin': k['origin'],
                        'destination': k['destination'],
                        'hub_edge': list(edge),
                        'cost': F_ek
                    })
                    break
        result['assignments'] = assignments
    else:
        result['objective_value'] = None
        result['status'] = model.Status
        result['message'] = 'No feasible solution found within time limit.'

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Solve UHLPMA using Gurobi (path-based formulation).'
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path for the output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds.')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = solve_uhlpma(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    if result['objective_value'] is not None:
        print(f"Objective value: {result['objective_value']:.6f}")
        print(f"Open hubs: {result.get('open_hubs', [])}")
    else:
        print("No feasible solution found.")


if __name__ == '__main__':
    main()
