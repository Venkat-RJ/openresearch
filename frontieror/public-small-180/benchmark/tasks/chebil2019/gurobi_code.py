"""
Gurobi implementation of KPS_1: 0-1 Knapsack Problem with Setups.

Source: Della Croce, Salassa, Scatamacchia (2016) - "An exact approach for
        the 0-1 Knapsack Problem with Setups"

Formulation KPS_1 (Standard ILP):
  maximize   sum_i sum_j p_{ij} x_{ij} - sum_i f_i y_i
  subject to sum_i sum_j w_{ij} x_{ij} + sum_i d_i y_i <= b
             x_{ij} <= y_i   for all i, j
             x_{ij} in {0,1}, y_i in {0,1}
"""

import json
import argparse
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
    with open(path, "r") as f:
        return json.load(f)


def solve_kps(instance, time_limit):
    N = instance["N"]
    b = instance["knapsack_capacity"]
    families = instance["families"]

    model = gp.Model("KPS_1")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = {}  # x[i][j]: 1 if item j of family i is selected
    y = {}  # y[i]: 1 if family i is activated

    for fam in families:
        i = fam["family_id"]
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        for item in fam["items"]:
            j = item["item_id"]
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.update()

    # Objective: maximize total profit minus setup costs
    obj = gp.LinExpr()
    for fam in families:
        i = fam["family_id"]
        f_i = fam["setup_cost"]
        obj -= f_i * y[i]
        for item in fam["items"]:
            j = item["item_id"]
            p_ij = item["profit"]
            obj += p_ij * x[i, j]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint (2): capacity
    cap_expr = gp.LinExpr()
    for fam in families:
        i = fam["family_id"]
        d_i = fam["setup_capacity"]
        cap_expr += d_i * y[i]
        for item in fam["items"]:
            j = item["item_id"]
            w_ij = item["weight"]
            cap_expr += w_ij * x[i, j]
    model.addConstr(cap_expr <= b, "capacity")

    # Constraint (3): linking constraints x_{ij} <= y_i
    for fam in families:
        i = fam["family_id"]
        for item in fam["items"]:
            j = item["item_id"]
            model.addConstr(x[i, j] <= y[i], f"link_{i}_{j}")

    # Solve
    model.optimize()

    # Extract solution
    solution = {
        "objective_value": None,
        "status": None,
        "families_selected": [],
        "items_selected": [],
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        for fam in families:
            i = fam["family_id"]
            if y[i].X > 0.5:
                solution["families_selected"].append(i)
                for item in fam["items"]:
                    j = item["item_id"]
                    if x[i, j].X > 0.5:
                        solution["items_selected"].append({"family": i, "item": j})
    else:
        solution["objective_value"] = 0
        solution["status"] = "no_solution_found"

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve KPS using Gurobi (KPS_1 formulation)"
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

    instance = load_instance(args.instance_path)
    solution = solve_kps(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Objective value: {solution['objective_value']}")


if __name__ == "__main__":
    main()
