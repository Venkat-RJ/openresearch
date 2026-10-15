"""
Gurobi implementation of the 0-1 Knapsack Problem (KP).
Based on: Martello, Pisinger, and Toth (1999)
"Dynamic Programming and Strong Bounds for the 0-1 Knapsack Problem"
Management Science 45(3): 414-424

Formulation (1):
  maximize   z = sum_{j=1}^{n} p_j x_j
  subject to sum_{j=1}^{n} w_j x_j <= c
             x_j in {0, 1}, j = 1, ..., n
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


def solve_knapsack(instance_path, solution_path, time_limit):
    # Load instance
    with open(instance_path, "r") as f:
        data = json.load(f)

    n = data["n"]
    capacity = data["capacity"]
    items = data["items"]

    profits = [item["profit"] for item in items]
    weights = [item["weight"] for item in items]

    # Build model
    model = gp.Model("knapsack_01")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x_j in {0, 1}
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective: maximize sum p_j * x_j
    model.setObjective(
        gp.quicksum(profits[j] * x[j] for j in range(n)),
        GRB.MAXIMIZE
    )

    # Constraint: sum w_j * x_j <= c
    model.addConstr(
        gp.quicksum(weights[j] * x[j] for j in range(n)) <= capacity,
        name="capacity"
    )

    # Solve
    model.optimize()

    # Extract solution
    result = {"objective_value": None, "selected_items": []}

    if model.SolCount > 0:
        result["objective_value"] = round(model.ObjVal)
        result["selected_items"] = [
            items[j]["id"] for j in range(n) if x[j].X > 0.5
        ]
    else:
        result["objective_value"] = 0
        result["selected_items"] = []

    # Write solution
    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Objective value: {result['objective_value']}")
    print(f"Solution written to: {solution_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve 0-1 Knapsack Problem using Gurobi"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_knapsack(args.instance_path, args.solution_path, args.time_limit)
