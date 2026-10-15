"""
Gurobi implementation of the Quadratic Multiple Knapsack Problem (QMKP).
Model (1)-(4) from Fleszar (2022), "A branch-and-bound algorithm for the
quadratic multiple knapsack problem", EJOR.

The QMKP model is:
  max  sum_k sum_i v_i * x_{ik} + sum_k sum_{i<j} v_{ij} * x_{ik} * x_{jk}
  s.t. sum_i w_i * x_{ik} <= c_k   for all k
       sum_k x_{ik} <= 1            for all i
       x_{ik} in {0,1}

Per the paper (Section 3): Gurobi parameter PreQLinearize is left at default
(automatic linearization). Number of threads set to 1.
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
def solve_qmkp(instance_path, solution_path, time_limit):
    # Load instance
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    m = data['m']
    weights = data['weights']
    values = data['values']
    pairwise = data['pairwise_values']
    capacity = data['capacity']

    # Items indexed 0..n-1, knapsacks indexed 0..m-1
    items = range(n)
    knapsacks = range(m)

    # Create model
    model = gp.Model("QMKP")

    # Parameters per paper: single thread, default PreQLinearize, time limit
    model.setParam("Threads", 1)
    # PreQLinearize left at default (automatic) as specified in the paper
    model.setParam("TimeLimit", time_limit)

    # Decision variables: x[i,k] = 1 if item i is packed in knapsack k
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # Objective (1): max sum_k (sum_i v_i * x_{ik} + sum_{i<j} v_{ij} * x_{ik} * x_{jk})
    obj = gp.QuadExpr()

    # Linear part: individual values
    for k in knapsacks:
        for i in items:
            obj += values[i] * x[i, k]

    # Quadratic part: pairwise values
    for k in knapsacks:
        for i in items:
            for j in range(i + 1, n):
                vij = pairwise[i][j]
                if vij != 0:
                    obj += vij * x[i, k] * x[j, k]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints (2): capacity constraints
    for k in knapsacks:
        model.addConstr(
            gp.quicksum(weights[i] * x[i, k] for i in items) <= capacity,
            name=f"cap_{k}"
        )

    # Constraints (3): each item in at most one knapsack
    for i in items:
        model.addConstr(
            gp.quicksum(x[i, k] for k in knapsacks) <= 1,
            name=f"assign_{i}"
        )

    # Optimize
    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result['objective_value'] = model.ObjVal

        # Extract assignment
        assignment = {}
        for i in items:
            for k in knapsacks:
                if x[i, k].X > 0.5:
                    assignment[str(i)] = k
                    break
        result['assignment'] = assignment

        # Additional info
        result['status'] = model.Status
        result['mip_gap'] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0
        if model.Status == GRB.OPTIMAL:
            result['optimal'] = True
        else:
            result['optimal'] = False
    else:
        # No feasible solution found
        result['objective_value'] = None
        result['status'] = model.Status
        result['optimal'] = False

    # Write solution
    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Solve QMKP using Gurobi (model 1-4)")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path for the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_qmkp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == '__main__':
    main()
