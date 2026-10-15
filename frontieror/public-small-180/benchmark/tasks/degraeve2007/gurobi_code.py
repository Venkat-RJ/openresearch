"""
CLST (Capacitated Lot Sizing with Setup Times) model using Gurobi.

Based on the formulation from Degraeve et al. (2007).
"""

import argparse
import json
import sys

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
def solve_clst(instance_path: str, solution_path: str, time_limit: float) -> None:
    # Load instance data
    with open(instance_path, "r") as f:
        data = json.load(f)

    n = data["dimensions"]["num_products"]
    m = data["dimensions"]["num_periods"]
    params = data["parameters"]

    d = params["demand"]                        # n x m
    hc = params["holding_cost"]                 # n x m
    sc = params["setup_cost"]                   # n x m
    vc = params["variable_production_cost"]     # n x m
    fc = params["initial_inventory_cost"]       # n
    st = params["setup_time"]                   # n x m
    vt = params["variable_production_time"]     # n x m
    cap = params["capacity"]                    # m

    # Precompute cumulative demand sd[i][t] = sum of d[i][l] for l=t..m-1
    # (using 0-indexed arrays)
    sd = [[0] * m for _ in range(n)]
    for i in range(n):
        sd[i][m - 1] = d[i][m - 1]
        for t in range(m - 2, -1, -1):
            sd[i][t] = sd[i][t + 1] + d[i][t]

    # Build model
    model = gp.Model("CLST")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # Decision variables
    x = {}  # production quantity
    y = {}  # setup indicator (binary)
    s = {}  # ending inventory
    si = {} # initial inventory

    for i in range(n):
        si[i] = model.addVar(lb=0.0, name=f"si_{i}")
        for t in range(m):
            x[i, t] = model.addVar(lb=0.0, name=f"x_{i}_{t}")
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
            s[i, t] = model.addVar(lb=0.0, name=f"s_{i}_{t}")

    model.update()

    # Objective
    obj = gp.LinExpr()
    for i in range(n):
        obj += fc[i] * si[i]
        for t in range(m):
            obj += sc[i][t] * y[i, t] + vc[i][t] * x[i, t] + hc[i][t] * s[i, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (2.1): inventory balance for t=0
    for i in range(n):
        model.addConstr(
            si[i] + x[i, 0] == d[i][0] + s[i, 0],
            name=f"balance_{i}_0"
        )

    # Constraint (2.2): inventory balance for t >= 1
    for i in range(n):
        for t in range(1, m):
            model.addConstr(
                s[i, t - 1] + x[i, t] == d[i][t] + s[i, t],
                name=f"balance_{i}_{t}"
            )

    # Constraint (3): production upper bound linked to setup
    for i in range(n):
        for t in range(m):
            ub = min(
                (cap[t] - st[i][t]) / vt[i][t],
                sd[i][t]
            )
            # Ensure non-negative upper bound
            ub = max(ub, 0.0)
            model.addConstr(
                x[i, t] <= ub * y[i, t],
                name=f"setup_link_{i}_{t}"
            )

    # Constraint (4): capacity constraint per period
    for t in range(m):
        model.addConstr(
            gp.quicksum(st[i][t] * y[i, t] + vt[i][t] * x[i, t] for i in range(n)) <= cap[t],
            name=f"capacity_{t}"
        )

    # Optimize
    model.optimize()

    # Build solution output
    solution = {
        "instance_path": instance_path,
        "solver": "gurobi",
        "status": model.Status,
        "status_name": _status_name(model.Status),
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["best_bound"] = model.ObjBound
        solution["mip_gap"] = model.MIPGap
        solution["solve_time"] = model.Runtime
        solution["node_count"] = int(model.NodeCount)

        # Extract variable values
        x_val = [[x[i, t].X for t in range(m)] for i in range(n)]
        y_val = [[round(y[i, t].X) for t in range(m)] for i in range(n)]
        s_val = [[s[i, t].X for t in range(m)] for i in range(n)]
        si_val = [si[i].X for i in range(n)]

        solution["variables"] = {
            "production": x_val,
            "setup": y_val,
            "inventory": s_val,
            "initial_inventory": si_val,
        }
    else:
        solution["objective_value"] = None
        solution["solve_time"] = model.Runtime

    # Write solution
    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Status: {solution['status_name']}")
    if solution["objective_value"] is not None:
        print(f"Objective: {solution['objective_value']:.4f}")
    print(f"Solution written to {solution_path}")


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"UNKNOWN_{status}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve CLST (Capacitated Lot Sizing with Setup Times) using Gurobi"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the instance JSON file"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to write the solution JSON file"
    )
    parser.add_argument(
        "--time_limit", type=float, default=3600.0,
        help="Time limit in seconds (default: 3600)"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_clst(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
