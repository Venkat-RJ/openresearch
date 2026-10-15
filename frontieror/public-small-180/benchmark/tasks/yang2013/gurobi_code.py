#!/usr/bin/env python3
"""Solve the 0-1 Quadratic Knapsack Problem (QKP) using Gurobi.

Problem formulation (Yang et al., 2013):
    max  sum_{i=1}^{n} sum_{j=1}^{i} p_{ij} * x_i * x_j
    s.t. sum_{j=1}^{n} w_j * x_j <= c
         x_j in {0, 1}, j = 1, ..., n
"""

import argparse
import json
import time

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


def solve_qkp(instance_path: str, solution_path: str, time_limit: int) -> None:
    # Load instance
    with open(instance_path, "r") as f:
        instance = json.load(f)

    instance_id = instance["instance_id"]
    n = instance["n"]
    capacity = instance["capacity"]
    weights = instance["weights"]
    profit_matrix = instance["profit_matrix"]

    # Default solution path
    if solution_path is None:
        solution_path = f"gurobi_solution_{instance_id}.json"

    # Build model
    model = gp.Model("QKP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # Binary decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Knapsack constraint: sum w_j * x_j <= c
    model.addConstr(
        gp.quicksum(weights[j] * x[j] for j in range(n)) <= capacity,
        name="knapsack",
    )

    # Objective: max sum_{i=1}^{n} sum_{j=1}^{i} p_{ij} * x_i * x_j
    # Diagonal terms (i == j): p_{jj} * x_j * x_j = p_{jj} * x_j (since binary)
    # Off-diagonal terms (i != j): p_{ij} * x_i * x_j
    obj = gp.QuadExpr()

    # Linear (diagonal) terms
    for j in range(n):
        if profit_matrix[j][j] != 0:
            obj += profit_matrix[j][j] * x[j]

    # Quadratic (off-diagonal) terms: sum over i > j of p_{ij} * x_i * x_j
    # The formulation sums i from 1..n, j from 1..i, so for i != j we get
    # each unordered pair (i, j) with i > j exactly once.
    for i in range(1, n):
        for j in range(i):
            if profit_matrix[i][j] != 0:
                obj += profit_matrix[i][j] * x[i] * x[j]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # Extract solution
    result = {
        "instance_id": instance_id,
        "n": n,
        "solver": "gurobi",
        "time_limit": time_limit,
        "solve_time": solve_time,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap
        result["solution"] = [int(x[j].X + 0.5) for j in range(n)]
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
    else:
        result["objective_value"] = None
        result["solution"] = None
        result["status"] = "infeasible_or_no_solution"

    result["gurobi_status"] = model.Status

    # Write solution
    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Status: {result['status']}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    print(f"Solve time: {solve_time:.2f}s")
    print(f"Solution written to: {solution_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve the 0-1 Quadratic Knapsack Problem using Gurobi"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON instance file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default=None,
        help="Path for the output solution JSON (default: gurobi_solution_{instance_id}.json)",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="Maximum solver time in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_qkp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
