"""
Gurobi MIP formulation for the Multidimensional Multiple Choice Knapsack Problem (MMKP).

Reference: Mansini and Zanotti (2020), "A Core-Based Exact Algorithm for the
           Multidimensional Multiple Choice Knapsack Problem",
           INFORMS Journal on Computing.

Model:
  Maximize  sum_{i,j} p_{ij} x_{ij}                         (1)
  s.t.      sum_{i,j} w_{ij}^k x_{ij} <= c_k,  k=1..m      (2)
            sum_{j} x_{ij} = 1,                  i=1..n      (3)
            x_{ij} in {0,1}                                   (4)
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
    return data


def solve_mmkp(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    n = data["parameters"]["n_groups"]
    m = data["parameters"]["m_resources"]
    capacities = data["capacities"]
    groups = data["groups"]

    model = gp.Model("MMKP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    # Suppress output to keep solution clean; remove if verbose output desired
    model.setParam("OutputFlag", 1)

    # Decision variables: x[i][j] binary
    x = {}
    for group in groups:
        i = group["group_id"]
        for item in group["items"]:
            j = item["item_id"]
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    model.update()

    # Objective (1): maximize total profit
    model.setObjective(
        gp.quicksum(
            item["profit"] * x[group["group_id"], item["item_id"]]
            for group in groups
            for item in group["items"]
        ),
        GRB.MAXIMIZE
    )

    # Constraints (2): resource capacity constraints
    for k in range(m):
        model.addConstr(
            gp.quicksum(
                item["weights"][k] * x[group["group_id"], item["item_id"]]
                for group in groups
                for item in group["items"]
            ) <= capacities[k],
            name=f"resource_{k}"
        )

    # Constraints (3): exactly one item per group
    for group in groups:
        i = group["group_id"]
        model.addConstr(
            gp.quicksum(x[i, item["item_id"]] for item in group["items"]) == 1,
            name=f"choice_{i}"
        )

    # Solve
    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = model.Status
        result["mip_gap"] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0
        result["selected_items"] = []
        for group in groups:
            i = group["group_id"]
            for item in group["items"]:
                j = item["item_id"]
                if x[i, j].X > 0.5:
                    result["selected_items"].append({
                        "group_id": i,
                        "item_id": j,
                        "profit": item["profit"],
                        "weights": item["weights"]
                    })
                    break
    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["mip_gap"] = None
        result["selected_items"] = []

    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found.")


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIP solver for the MMKP (Mansini & Zanotti, 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_mmkp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
