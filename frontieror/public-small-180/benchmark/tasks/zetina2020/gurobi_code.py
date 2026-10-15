#!/usr/bin/env python3
"""
Gurobi implementation of the mathematical optimization models from:
"Efficient Internal Strategies in Quantum Relaxation based Branch-and-Bound"
Matsuyama, Huang, Nishimura, Yamashiro (2024)

Implements:
  - TSP: Constrained quadratic binary formulation (Eq. 8)
  - MaxCut: Quadratic binary formulation (Eq. 7, negated for minimization)

The general formulation is:
  S: z* = min{ x^T Q x : x in X }
  where X = { x in {0,1}^n : Ax <= b }
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
def solve_tsp(instance, time_limit):
    """Solve TSP using constrained quadratic formulation (Eq. 8).

    min sum_t sum_{i,j} d_{ij} * x_{i,t} * x_{j,(t+1) mod N}
    s.t.
      sum_{i=0}^{N-1} x_{i,t} = 1   for all t   (C1: time step assignment)
      sum_{t=0}^{N-1} x_{i,t} = 1   for all i   (C2: city visit)
      x_{i,t} in {0,1}
    """
    N = instance["parameters"]["n_cities"]
    d = instance["data"]["distance_matrix"]

    model = gp.Model("TSP_QR_BnB")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Binary variables x[i,t]: 1 if salesman is in city i at time t
    x = {}
    for i in range(N):
        for t in range(N):
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    model.update()

    # Objective (Eq. 8): min sum_t sum_{i,j} d_{ij} x_{i,t} x_{j,(t+1) mod N}
    obj = gp.QuadExpr()
    for t in range(N):
        t_next = (t + 1) % N
        for i in range(N):
            for j in range(N):
                if d[i][j] != 0:
                    obj += d[i][j] * x[i, t] * x[j, t_next]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint C1: each time step has exactly one city
    for t in range(N):
        model.addConstr(
            gp.quicksum(x[i, t] for i in range(N)) == 1,
            name=f"time_step_{t}"
        )

    # Constraint C2: each city is visited exactly once
    for i in range(N):
        model.addConstr(
            gp.quicksum(x[i, t] for t in range(N)) == 1,
            name=f"city_{i}"
        )

    model.optimize()

    result = {
        "objective_value": None,
        "status": model.status,
        "status_description": None,
        "solution": None,
        "tour": None,
    }

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    result["status_description"] = status_map.get(model.status, f"OTHER({model.status})")

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        sol = {}
        tour = [None] * N
        for i in range(N):
            for t in range(N):
                val = int(round(x[i, t].X))
                sol[f"x_{i}_{t}"] = val
                if val == 1:
                    tour[t] = i
        result["solution"] = sol
        result["tour"] = tour

    return result


def solve_maxcut(instance, time_limit):
    """Solve MaxCut using spin variables (Eq. 7).

    max (1/2) sum_{(i,j) in E} (1 - s_i s_j),  s_i in {-1, 1}
    Spin variable s_i encoded as s_i = 1 - 2*x_i with x_i in {0, 1}.
    No constraints (unconstrained binary quadratic optimization).
    """
    n_nodes = instance["parameters"]["n_nodes"]
    edges = instance["data"]["edges"]

    model = gp.Model("MaxCut_QR_BnB")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    x = {}
    for i in range(n_nodes):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    model.update()

    # Spin variables s_i in {-1, 1} (math_model Eq. 7), encoded via x_i.
    s = {i: 1 - 2 * x[i] for i in range(n_nodes)}

    # Objective (Eq. 7): max (1/2) sum_{(i,j) in E} (1 - s_i s_j)
    obj = gp.QuadExpr()
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        # **NOT SPECIFIED IN PAPER**: Whether MaxCut edges are weighted.
        # Assumed unweighted (w=1) from 3-regular graph context.
        w = float(edge[2]) if len(edge) > 2 else 1.0
        obj += 0.5 * w * (1 - s[i] * s[j])

    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()

    result = {
        "objective_value": None,
        "status": model.status,
        "status_description": None,
        "solution": None,
    }

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
    }
    result["status_description"] = status_map.get(model.status, f"OTHER({model.status})")

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        sol = {}
        for i in range(n_nodes):
            sol[f"x_{i}"] = int(round(x[i].X))
        result["solution"] = sol

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for BQP problems (TSP and MaxCut) from QR-BnB paper"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for output solution JSON (e.g., gurobi_solution_1.json)")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    problem_type = instance["problem_type"]

    if problem_type == "TSP":
        result = solve_tsp(instance, args.time_limit)
    elif problem_type == "MaxCut":
        result = solve_maxcut(instance, args.time_limit)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    print(f"Status: {result['status_description']}")


if __name__ == "__main__":
    main()
