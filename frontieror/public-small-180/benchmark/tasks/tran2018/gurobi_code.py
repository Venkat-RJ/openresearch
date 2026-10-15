#!/usr/bin/env python3
"""
Gurobi implementation of the Alternative-Fuel Station Location Problem (ASLP)
using the [P3] arc-cover/path-cover formulation from Capar et al. (2013),
as adopted in:

Tran, Nagy, Nguyen, Wassan (2018) "An Efficient Heuristic Algorithm for the
Alternative-Fuel Station Location Problem", EJOR.

Formulation [ASLP]:
  max  z_0 = sum_{q in Q} f_q * y_q
  s.t. sum_{i in K^q_{jk}} x_i >= y_q,  for all q in Q, arc (j,k) in A_q
       sum_{i in N} x_i = p
       x_i in {0,1}, y_q in {0,1}  (y_q relaxed to [0,1] per paper Table 1)

K^q_{jk} computation uses the round-trip path: the vehicle starts at the
origin with a full tank (range R). For each directed arc on the round trip,
K^q_{jk} is the set of nodes that could have refueled the vehicle such that
it has enough fuel to traverse the arc.
"""

import argparse
import json
import sys

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
import time
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


def load_instance(path):
    """Load problem instance from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def build_arc_distance_lookup(arcs):
    """Build bidirectional distance lookup from arc list."""
    dist = {}
    for arc in arcs:
        i, j, d = arc['from'], arc['to'], arc['distance']
        dist[(i, j)] = d
        dist[(j, i)] = d
    return dist


def compute_K_sets(od_pairs, arc_dist, R):
    """
    For each OD pair, compute K^q_{jk} for each directed arc on the round trip.

    The round-trip path visits nodes in order:
      Forward: n_0, n_1, ..., n_{L-1}
      Return:  n_{L-2}, n_{L-3}, ..., n_0

    The vehicle starts at the origin (pos 0) with a full tank of range R.
    Any arc whose cumulative distance from the origin is <= R is automatically
    covered by the initial fuel and needs no constraint.

    For other arcs, K^q_{jk} = set of nodes at positions c <= p on the round
    trip such that rt_cum[p+1] - rt_cum[c] <= R.

    Returns a list of dicts (one per OD pair), each with:
      'arc_constraints': list of {'K': set of node IDs, 'covered': bool}
      'path_nodes': set of node IDs on the path
      'flow': traffic flow volume
    """
    all_od_info = []

    for q_idx, od in enumerate(od_pairs):
        path = od['shortest_path']
        L = len(path)
        flow = od['flow']

        if L < 2:
            all_od_info.append({
                'arc_constraints': [],
                'path_nodes': set(path),
                'flow': flow
            })
            continue

        # Edge distances along the path
        edge_dists = []
        for i in range(L - 1):
            d = arc_dist.get((path[i], path[i + 1]))
            if d is None:
                raise ValueError(
                    f"No arc distance for ({path[i]}, {path[i+1]}) "
                    f"on OD pair {q_idx}"
                )
            edge_dists.append(d)

        # Round-trip: 2L-1 positions, 2(L-1) arcs
        num_positions = 2 * L - 1
        num_arcs = 2 * (L - 1)

        # Node at each round-trip position
        rt_nodes = list(path) + list(reversed(path[:-1]))

        # Cumulative distances along the round trip
        rt_cum = [0.0] * num_positions
        for p in range(1, num_positions):
            if p < L:
                rt_cum[p] = rt_cum[p - 1] + edge_dists[p - 1]
            else:
                # Return arc at position p-1 corresponds to forward edge
                # at index (2*(L-1) - p)
                ret_edge_idx = 2 * (L - 1) - p
                rt_cum[p] = rt_cum[p - 1] + edge_dists[ret_edge_idx]

        # For each arc on the round trip, compute K set
        arc_constraints = []
        for p in range(num_arcs):
            arc_end_cum = rt_cum[p + 1]

            # Check if the origin's initial full tank covers this arc
            if arc_end_cum <= R + 1e-9:
                arc_constraints.append({'K': set(), 'covered': True})
                continue

            # Find all positions c <= p where refueling at c covers this arc
            K = set()
            for c in range(p + 1):
                if arc_end_cum - rt_cum[c] <= R + 1e-9:
                    K.add(rt_nodes[c])

            # Position 0 (origin) cannot be in K here since arc_end_cum > R,
            # meaning rt_cum[p+1] - rt_cum[0] = rt_cum[p+1] > R.
            # So origin's initial fuel is already accounted for.

            arc_constraints.append({'K': K, 'covered': False})

        all_od_info.append({
            'arc_constraints': arc_constraints,
            'path_nodes': set(path),
            'flow': flow
        })

    return all_od_info


def main():
    parser = argparse.ArgumentParser(
        description="ASLP Solver using Gurobi (P3 formulation)"
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument('--solution_path', type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument('--time_limit', type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    instance = load_instance(args.instance_path)

    n = instance['parameters']['n']
    p = instance['parameters']['p']
    R = instance['parameters']['R']
    nodes = instance['nodes']
    arcs = instance['arcs']
    od_pairs = instance['od_pairs']

    node_ids = [node['id'] for node in nodes]

    # Build arc distance lookup
    arc_dist = build_arc_distance_lookup(arcs)

    # Compute K sets for all OD pairs
    all_od_info = compute_K_sets(od_pairs, arc_dist, R)

    # ========================================================================
    # Build Gurobi model for [ASLP]
    # ========================================================================
    model = gp.Model("ASLP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x_i: binary, 1 if station at node i
    x = {}
    for i in node_ids:
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    # y_q: continuous [0,1] (relaxed from binary per paper Table 1 note:
    # "y_q can be relaxed to continuous without affecting optimal integer solution")
    y = {}
    for q_idx in range(len(od_pairs)):
        y[q_idx] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"y_{q_idx}"
        )

    model.update()

    # Objective (14): maximize total refueled traffic flow
    model.setObjective(
        gp.quicksum(all_od_info[q]['flow'] * y[q]
                     for q in range(len(od_pairs))),
        GRB.MAXIMIZE
    )

    # Constraint (16): exactly p stations
    model.addConstr(
        gp.quicksum(x[i] for i in node_ids) == p,
        "station_count"
    )

    # Constraint (15): arc coverage for each OD pair and each arc
    for q_idx, od_info in enumerate(all_od_info):
        for arc_idx, arc_con in enumerate(od_info['arc_constraints']):
            if arc_con['covered']:
                # Arc covered by initial fuel at origin - no constraint needed
                continue

            K = arc_con['K']
            if len(K) == 0:
                # No node can cover this arc -> flow cannot be refueled
                model.addConstr(
                    y[q_idx] == 0,
                    f"infeasible_q{q_idx}_a{arc_idx}"
                )
            else:
                model.addConstr(
                    gp.quicksum(x[i] for i in K) >= y[q_idx],
                    f"arc_cover_q{q_idx}_a{arc_idx}"
                )

    # Solve
    model.optimize()

    # ========================================================================
    # Extract and output solution
    # ========================================================================
    solution = {}
    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        solution['stations'] = [i for i in node_ids if x[i].X > 0.5]
        solution['refueled_od_pairs'] = [
            q_idx for q_idx in range(len(od_pairs)) if y[q_idx].X > 0.5
        ]
        if model.Status == GRB.OPTIMAL:
            solution['status'] = 'optimal'
        elif model.Status == GRB.TIME_LIMIT:
            solution['status'] = 'time_limit'
        else:
            solution['status'] = 'feasible'
        solution['gap'] = model.MIPGap
    else:
        solution['objective_value'] = 0.0
        solution['stations'] = []
        solution['refueled_od_pairs'] = []
        solution['status'] = 'no_solution'
        solution['gap'] = float('inf')

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Objective value: {solution['objective_value']}")
    print(f"Status: {solution['status']}")


if __name__ == '__main__':
    main()
