"""
Gurobi implementation of the Capacitated Lot-Sizing Problem with Setup Times (CLST).
Reference: Fragkos, Degraeve, and De Reyck (2016),
  "A Horizon Decomposition Approach for the Capacitated Lot-Sizing Problem with Setup Times",
  INFORMS Journal on Computing 28(3): 465-482.

Mathematical model: Section 3.1, pp. 467 (equations 1-5).
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
def load_instance(path):
    with open(path) as f:
        data = json.load(f)
    return data


def solve_clst(data, time_limit):
    n = data["n_items"]       # number of items
    m = data["n_periods"]     # number of periods

    # Parameters (0-indexed internally)
    cap = data["capacity"]          # cap[t], t in 0..m-1
    d = data["demands"]             # d[i][t]
    sc = data["setup_costs"]        # sc[i][t]
    vc = data["variable_costs"]     # vc[i][t]
    hc = data["holding_costs"]      # hc[i][t]
    st = data["setup_times"]        # st[i][t]
    vt = data["variable_times"]     # vt[i][t]
    big_M = data["big_M"]           # big_M[i][t]
    init_inv = data["initial_inventory"]  # s_{i,1} for each item i

    # Compute sd_{itk} = sum_{j=t}^{k} d[i][j] (0-indexed)
    # sd[i][t] = sum from period t to m-1
    sd_to_end = [[0.0] * m for _ in range(n)]
    for i in range(n):
        cumsum = 0.0
        for t in range(m - 1, -1, -1):
            cumsum += d[i][t]
            sd_to_end[i][t] = cumsum

    model = gp.Model("CLST")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i][t]: production quantity, i in 0..n-1, t in 0..m-1
    x = [[model.addVar(lb=0.0, name=f"x_{i}_{t}") for t in range(m)] for i in range(n)]
    # s[i][t]: inventory at beginning of period t, t in 0..m (index m = s_{i,m+1})
    s = [[model.addVar(lb=0.0, name=f"s_{i}_{t}") for t in range(m + 1)] for i in range(n)]
    # y[i][t]: binary setup variable
    y = [[model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}") for t in range(m)] for i in range(n)]

    # Objective (1): minimize setup + production + holding costs
    # NOTE: hc_{it} in the paper is the cost of holding inventory from period t-1 to t,
    # indexed over periods t in T. The inventory variable s_{it} is the inventory at the
    # beginning of period t. The paper sums hc_{it} * s_{it} for t in T (1..m).
    # In 0-indexed terms, s[i][t] for t in 0..m-1, multiplied by hc[i][t].
    obj = gp.quicksum(
        sc[i][t] * y[i][t] + vc[i][t] * x[i][t] + hc[i][t] * s[i][t]
        for i in range(n) for t in range(m)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (2): flow balance
    # s_{it} + x_{it} = d_{it} + s_{i,t+1}  for all i, t in T
    for i in range(n):
        for t in range(m):
            model.addConstr(
                s[i][t] + x[i][t] == d[i][t] + s[i][t + 1],
                name=f"flow_{i}_{t}"
            )

    # Constraint (3): production-setup link
    # x_{it} <= M_{it} * y_{it}
    for i in range(n):
        for t in range(m):
            model.addConstr(
                x[i][t] <= big_M[i][t] * y[i][t],
                name=f"setup_link_{i}_{t}"
            )

    # Constraint (4): capacity per period
    # sum_i st_{it}*y_{it} + sum_i vt_{it}*x_{it} <= cap_t
    for t in range(m):
        model.addConstr(
            gp.quicksum(st[i][t] * y[i][t] + vt[i][t] * x[i][t] for i in range(n)) <= cap[t],
            name=f"capacity_{t}"
        )

    # Constraint (5): boundary conditions
    # s_{i,m+1} = 0 (ending inventory is zero)
    for i in range(n):
        model.addConstr(s[i][m] == 0.0, name=f"end_inv_{i}")

    # Initial inventory: s[i][0] represents s_{i,1} (inventory at the start).
    # The math model (constraints 1-5) only requires s_{it} >= 0 (nonnegativity)
    # and s_{i,m+1} = 0 (ending inventory).  There is no equality constraint
    # fixing initial inventory to zero.  Section 3.1 (p. 467) notes that
    # initial inventory is allowed at a high cost (Vanderbeck 1998).
    # We set a lower bound so that s[i][0] >= init_inv[i] (the provided
    # starting stock), letting the solver acquire additional initial inventory
    # if needed.  The holding-cost term hc[i][0]*s[i][0] already in the
    # objective penalises any excess.
    for i in range(n):
        s[i][0].lb = init_inv[i]

    model.optimize()

    # Extract solution
    status = model.Status
    result = {}

    if model.SolCount > 0:
        obj_val = model.ObjVal
        x_sol = [[x[i][t].X for t in range(m)] for i in range(n)]
        s_sol = [[s[i][t].X for t in range(m + 1)] for i in range(n)]
        y_sol = [[round(y[i][t].X) for t in range(m)] for i in range(n)]

        result = {
            "objective_value": obj_val,
            "status": model.Status,
            "x": x_sol,
            "s": s_sol,
            "y": y_sol,
        }
    else:
        # No feasible solution found within time limit
        result = {
            "objective_value": None,
            "status": model.Status,
            "x": None,
            "s": None,
            "y": None,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Gurobi solver for CLST (Fragkos et al. 2016)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = solve_clst(data, args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found within time limit.")


if __name__ == "__main__":
    main()
