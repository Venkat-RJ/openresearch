#!/usr/bin/env python3
"""
Gurobi ILP formulation for the Maximum Clique Problem (MCP).

Paper: "A new exact maximum clique algorithm for large and massive sparse graphs"
Authors: Pablo San Segundo, Alvaro Lopez, Panos M. Pardalos (2016)

The paper does NOT present an ILP formulation; it uses a combinatorial branch-and-bound
algorithm (BBMCSP). The standard ILP formulation below is the canonical textbook
formulation of MCP:

    max  sum_{i in V} x_i
    s.t. x_i + x_j <= 1,  for all (i,j) not in E, i < j
         x_i in {0, 1},   for all i in V
"""

import argparse
import json
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
    with open(path, 'r') as f:
        data = json.load(f)
    graph = data['graph']
    n = graph['num_vertices']
    edges = set()
    # Instance uses adjacency_list (dict: vertex_str -> [neighbor, ...])
    adj_list = graph['adjacency_list']
    for u_str, neighbors in adj_list.items():
        u = int(u_str)
        for v in neighbors:
            if u != v:
                edges.add((min(u, v), max(u, v)))
    return n, edges


def solve_max_clique(instance_path, solution_path, time_limit):
    n, edges = load_instance(instance_path)

    # Build adjacency set for fast lookup
    adj = set(edges)

    model = gp.Model("MaxClique")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x_i = 1 if vertex i is in the clique
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective: maximize clique size
    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)

    # Constraint (C1): for every pair of non-adjacent vertices, at most one can be in clique
    # x_i + x_j <= 1 for all (i,j) not in E, i < j
    # For efficiency, build neighbor sets first
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    for i in range(n):
        for j in range(i + 1, n):
            if j not in neighbors[i]:
                model.addConstr(x[i] + x[j] <= 1)

    model.optimize()

    # Extract solution
    objective_value = 0
    clique_vertices = []

    if model.SolCount > 0:
        objective_value = round(model.ObjVal)
        clique_vertices = [i for i in range(n) if x[i].X > 0.5]

    solution = {
        "objective_value": objective_value,
        "clique_size": objective_value,
        "clique_vertices": clique_vertices,
        "status": model.Status,
        "solver": "gurobi",
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
    }

    with open(solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Objective value: {objective_value}")
    print(f"Clique vertices: {clique_vertices}")
    print(f"Solution written to {solution_path}")


def main():
    parser = argparse.ArgumentParser(description="Gurobi solver for Maximum Clique Problem")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_max_clique(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
