#!/usr/bin/env python3
"""
Solve the p-median problem using the CR (Canonical Representation) formulation
from Garcia-Quiles et al. (2011) with Gurobi.

CR Formulation:
    min  sum_i sum_{k=2}^{G_i} (D_{ik} - D_{i,k-1}) * z_{ik}
    s.t. sum_i y_i = p
         z_{ik} + sum_{j: c_{ij} < D_{ik}} y_j >= 1,  for all i, k=2..G_i
         y_i in {0,1}, z_{ik} >= 0
"""

import argparse
import json
import math
import time
_GUROBI_CODE_START_TIME = time.time()

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
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


def compute_cost_matrix(coords):
    """Compute rounded Euclidean distance matrix."""
    n = len(coords)
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            dist = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
            c[i][j] = dist
            c[j][i] = dist
    return c


def solve_pmedian_cr(instance_path, solution_path, time_limit):
    # Load instance
    with open(instance_path, "r") as f:
        instance = json.load(f)

    n = instance["n"]
    p = instance["p"]
    coords = instance["node_coordinates"]

    # Compute cost matrix
    c = compute_cost_matrix(coords)

    # For each customer i, compute sorted distinct cost values D_i
    # D_i[0] = 0, D_i[1] < D_i[2] < ... < D_i[G_i - 1]
    D = []
    for i in range(n):
        distinct = sorted(set(c[i][j] for j in range(n)))
        # Ensure 0 is first (c[i][i] = 0 guarantees this)
        D.append(distinct)

    # Precompute for constraint (5): for each i, k, the set of j with c_{ij} < D_{ik}
    # Store as list of lists: S[i][k] = [j indices where c_{ij} < D[i][k]]
    S = []
    for i in range(n):
        Si = [None]  # placeholder for k=0 (unused, k starts at 1 in 1-indexed, but we use 0-indexed with k starting at 1)
        for k in range(1, len(D[i])):
            threshold = D[i][k]
            Si.append([j for j in range(n) if c[i][j] < threshold])
        S.append(Si)

    # Build Gurobi model
    model = gp.Model("pmedian_CR")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LogToConsole", 1)

    # Decision variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    z = {}
    for i in range(n):
        for k in range(1, len(D[i])):  # k=1..G_i-1 in 0-indexed (corresponds to k=2..G_i in 1-indexed)
            z[i, k] = model.addVar(lb=0.0, name=f"z_{i}_{k}")

    model.update()

    # Objective: min sum_i sum_{k=1}^{G_i-1} (D[i][k] - D[i][k-1]) * z[i,k]
    obj = gp.LinExpr()
    for i in range(n):
        for k in range(1, len(D[i])):
            coeff = D[i][k] - D[i][k - 1]
            obj.addTerms(coeff, z[i, k])
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (4): sum_i y_i = p
    model.addConstr(gp.quicksum(y[i] for i in range(n)) == p, name="num_medians")

    # Constraint (5): z_{ik} + sum_{j: c_{ij} < D_{ik}} y_j >= 1
    for i in range(n):
        for k in range(1, len(D[i])):
            lhs = z[i, k] + gp.quicksum(y[j] for j in S[i][k])
            model.addConstr(lhs >= 1, name=f"cover_{i}_{k}")

    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # Extract solution
    result = {
        "instance_path": instance_path,
        "formulation": "CR",
        "n": n,
        "p": p,
        "time_limit": time_limit,
        "solve_time": round(solve_time, 4),
        "status": model.Status,
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
        }.get(model.Status, str(model.Status)),
    }

    if model.SolCount > 0:
        result["objective_value"] = round(model.ObjVal)
        result["best_bound"] = round(model.ObjBound, 4)
        result["mip_gap"] = round(model.MIPGap, 6) if model.MIPGap < GRB.INFINITY else None

        # Extract open facilities
        opened = [i for i in range(n) if y[i].X > 0.5]
        result["opened_facilities"] = opened

        # Compute assignment: each customer assigned to nearest open facility
        assignments = []
        for i in range(n):
            best_j = min(opened, key=lambda j: c[i][j])
            assignments.append({"customer": i, "facility": best_j, "cost": c[i][best_j]})
        result["assignments"] = assignments
    else:
        result["objective_value"] = None
        result["note"] = "No feasible solution found."

    # Write solution
    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status_name']}")
        print(f"Solve time: {result['solve_time']}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve p-median using CR formulation with Gurobi"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to instance JSON file"
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path to output solution JSON file",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Gurobi time limit in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_pmedian_cr(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
