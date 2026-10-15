#!/usr/bin/env python3
"""
Gurobi ILP implementation of Min-Power Symmetric Connectivity (MinPSC).

Paper: Bentert et al., "Parameterized Algorithms for Power-Efficiently
       Connecting Wireless Sensor Networks: Theory and Experiments" (2020)

Problem 1.1 (MinPSC):
  Given a connected undirected graph G=(V,E) with edge weights w: E -> N,
  find a connected spanning subgraph T=(V,F), F subset of E, that minimizes
    sum_{v in V} max_{{u,v} in F} w({u,v})

The paper does NOT provide its own ILP formulation. It references ILP models
EX1 and EX2 from Montemanni and Gambardella (2005), which are
**NOT EXPLICITLY SPECIFIED IN THIS PAPER**.

INFERRED ASSUMPTION: We implement a standard ILP for MinPSC using:
  - Binary edge selection variables x_e
  - Continuous variables p_v for vertex power cost
  - Linearization of max via p_v >= w(e) * x_e
  - Single-commodity flow for connectivity enforcement
    (pick an arbitrary root, send n-1 units of flow, each vertex absorbs 1)
"""

import argparse
import json
import math
import sys
from itertools import combinations

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
def load_instance(instance_path):
    """Load a MinPSC instance from JSON."""
    with open(instance_path, 'r') as f:
        data = json.load(f)

    vertices = data["vertices"]
    n = data["n"]
    m = data["m"]

    # Build vertex list
    V = []
    coords = {}
    lower_bounds = {}
    for v_info in vertices:
        vid = v_info["id"]
        V.append(vid)
        coords[vid] = (v_info["x"], v_info["y"])
        lower_bounds[vid] = v_info["lower_bound"]

    # Build complete graph edges with squared Euclidean distance weights
    # The instance specifies "complete" graph with "squared_euclidean_distance"
    edges = []
    edge_weights = {}
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            u, v = V[i], V[j]
            dx = coords[u][0] - coords[v][0]
            dy = coords[u][1] - coords[v][1]
            w = dx * dx + dy * dy
            edges.append((u, v))
            edge_weights[(u, v)] = w
            edge_weights[(v, u)] = w

    return V, edges, edge_weights, coords, lower_bounds, data


def solve_minpsc_gurobi(V, edges, edge_weights, time_limit):
    """
    Solve MinPSC using Gurobi ILP.

    Formulation:
      min sum_{v in V} p_v
      s.t.
        p_v >= w({u,v}) * x_{u,v}   for all edges {u,v} and both endpoints
        T = (V, F) is connected (enforced via single-commodity flow)
        x_e in {0,1}, p_v >= 0

    Connectivity via single-commodity flow:
      - Pick root r (vertex 0). Root sends n-1 units of flow.
      - Each non-root vertex absorbs exactly 1 unit.
      - Flow on directed arc (u,v) is bounded by (n-1)*x_{u,v}.

    **NOT EXPLICITLY SPECIFIED IN PAPER**: The exact connectivity formulation.
    INFERRED ASSUMPTION: We use single-commodity flow which is a standard
    approach for enforcing connectivity in ILP formulations.
    """
    n = len(V)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, []

    model = gp.Model("MinPSC")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Binary edge selection variables
    x = {}
    for (u, v) in edges:
        x[(u, v)] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

    # Vertex power cost variables
    p = {}
    for v in V:
        p[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_{v}")

    # Flow variables: directed arcs for each undirected edge
    # Pick root as V[0]
    root = V[0]
    f_var = {}
    for (u, v) in edges:
        f_var[(u, v)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                      ub=n - 1, name=f"f_{u}_{v}")
        f_var[(v, u)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                      ub=n - 1, name=f"f_{v}_{u}")

    model.update()

    # Objective: minimize sum of vertex power costs
    model.setObjective(gp.quicksum(p[v] for v in V), GRB.MINIMIZE)

    # Constraint: p_v >= w(e) * x_e for each edge e incident to v
    for (u, v) in edges:
        w = edge_weights[(u, v)]
        model.addConstr(p[u] >= w * x[(u, v)], name=f"pwr_{u}_{u}_{v}")
        model.addConstr(p[v] >= w * x[(u, v)], name=f"pwr_{v}_{u}_{v}")

    # Flow conservation: each non-root absorbs 1 unit
    # Build adjacency for flow
    # For root: outflow - inflow = n - 1
    # For others: outflow - inflow = -1
    for v in V:
        outflow = gp.LinExpr()
        inflow = gp.LinExpr()
        for (u, w_v) in edges:
            if u == v:
                outflow.add(f_var[(u, w_v)])
                inflow.add(f_var[(w_v, u)])
            elif w_v == v:
                outflow.add(f_var[(w_v, u)])
                inflow.add(f_var[(u, w_v)])

        if v == root:
            model.addConstr(outflow - inflow == n - 1,
                            name=f"flow_root_{v}")
        else:
            model.addConstr(outflow - inflow == -1,
                            name=f"flow_{v}")

    # Flow capacity: flow on arc (u,v) <= (n-1) * x_{u,v}
    for (u, v) in edges:
        model.addConstr(f_var[(u, v)] <= (n - 1) * x[(u, v)],
                        name=f"cap_{u}_{v}")
        model.addConstr(f_var[(v, u)] <= (n - 1) * x[(u, v)],
                        name=f"cap_{v}_{u}")

    # Solve
    model.optimize()

    # Extract solution
    obj_val = None
    selected_edges = []

    if model.SolCount > 0:
        obj_val = model.ObjVal
        for (u, v) in edges:
            if x[(u, v)].X > 0.5:
                selected_edges.append((u, v))
    else:
        obj_val = None

    return obj_val, selected_edges


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi ILP solver for MinPSC (Min-Power Symmetric Connectivity)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    V, edges, edge_weights, coords, lower_bounds, instance_data = \
        load_instance(args.instance_path)

    # Solve
    obj_val, selected_edges = solve_minpsc_gurobi(
        V, edges, edge_weights, args.time_limit
    )

    # Build solution output
    solution = {
        "problem": "MinPSC",
        "instance_path": args.instance_path,
        "solver": "Gurobi",
        "objective_value": obj_val if obj_val is not None else None,
        "selected_edges": [list(e) for e in selected_edges],
        "num_selected_edges": len(selected_edges),
    }

    # Compute per-vertex costs if solution found
    if obj_val is not None:
        vertex_costs = {v: 0.0 for v in V}
        for (u, v) in selected_edges:
            w = edge_weights[(u, v)]
            if w > vertex_costs[u]:
                vertex_costs[u] = w
            if w > vertex_costs[v]:
                vertex_costs[v] = w
        solution["vertex_costs"] = {str(v): c for v, c in vertex_costs.items()}

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    if obj_val is not None:
        print(f"Optimal objective value: {obj_val}")
    else:
        print("No feasible solution found within time limit.")


if __name__ == "__main__":
    main()
