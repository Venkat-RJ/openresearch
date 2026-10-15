"""
Gurobi implementation of the Knapsack Problem with Setup (KPS).
Based on: Chebil & Khemakhem (2015), Computers & Operations Research.

Maximize z = sum_i sum_j c_{ij} x_{ij} + sum_i f_i y_i
subject to:
  sum_i sum_j a_{ij} x_{ij} + sum_i d_i y_i <= b
  x_{ij} <= y_i  for all i, j
  x_{ij}, y_i in {0, 1}
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


def solve_kps(instance_path, solution_path, time_limit):
    with open(instance_path, "r") as f:
        data = json.load(f)

    N = data["N"]
    b = data["b"]
    classes = data["classes"]

    model = gp.Model("KPS")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = {}  # x[i][j]: binary, select item j of class i
    y = {}  # y[i]: binary, setup class i

    for cls in classes:
        i = cls["class_id"]
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        for item in cls["items"]:
            j = item["item_id"]
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.update()

    # Objective: maximize total profit (f_i are negative, so +f_i*y_i subtracts setup cost)
    obj = gp.LinExpr()
    for cls in classes:
        i = cls["class_id"]
        obj += cls["f_i"] * y[i]
        for item in cls["items"]:
            j = item["item_id"]
            obj += item["c_ij"] * x[i, j]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraint
    cap = gp.LinExpr()
    for cls in classes:
        i = cls["class_id"]
        cap += cls["d_i"] * y[i]
        for item in cls["items"]:
            j = item["item_id"]
            cap += item["a_ij"] * x[i, j]
    model.addConstr(cap <= b, "capacity")

    # Linking constraints: x_{ij} <= y_i
    for cls in classes:
        i = cls["class_id"]
        for item in cls["items"]:
            j = item["item_id"]
            model.addConstr(x[i, j] <= y[i], f"link_{i}_{j}")

    # Solve
    model.optimize()

    # Extract solution
    objective_value = None
    solution = {"classes": []}

    if model.SolCount > 0:
        objective_value = model.ObjVal
        for cls in classes:
            i = cls["class_id"]
            cls_sol = {
                "class_id": i,
                "y_i": int(round(y[i].X)),
                "items": [],
            }
            for item in cls["items"]:
                j = item["item_id"]
                cls_sol["items"].append(
                    {"item_id": j, "x_ij": int(round(x[i, j].X))}
                )
            solution["classes"].append(cls_sol)

    solution["objective_value"] = objective_value

    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Objective value: {objective_value}")
    print(f"Solution written to: {solution_path}")


def main():
    parser = argparse.ArgumentParser(description="Solve KPS using Gurobi")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve_kps(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
