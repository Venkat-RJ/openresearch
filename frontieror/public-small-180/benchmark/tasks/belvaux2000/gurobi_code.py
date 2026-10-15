"""
Gurobi implementation of the Con-NT Small Bucket Lot-Sizing Model (SB-1).

From: Belvaux & Wolsey (2000), "bc-prod: A Specialized Branch-and-Cut System
for Lot-Sizing Problems", Management Science 46(5):724-738.

This implements the Con-NT model (Example 2 in the paper, Equations 9-14):
  min  sum_{i,k,t} (f * y_{ikt} + g * z_{ikt})
       + sum_{i,t} (h_i * s_{it} + e_i * r_{it})
  s.t.
    (9)  s_{i,t-1} - r_{i,t-1} + sum_k x_{ikt} = d_{it} + s_{it} - r_{it}   for all i, t
    (10) x_{ikt} + sigma_k * z_{ikt} <= C_k * y_{ikt}                        for all i, k, t
    (11) z_{ikt} >= y_{ikt} - y_{ik,t-1}                                      for all i, k, t
    (12) sum_i y_{ikt} <= 1                                                    for all k, t
    (13) z_{ikt} <= y_{ikt}                                                    for all i, k, t
    (14) y_{ik,t-1} + z_{ikt} + sum_{j!=i} (y_{jkt} - z_{jkt}) <= 1           for all i, k, t
    s, r, x >= 0;  y, z in {0, 1}
    s_0 = r_0 = 0
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
    with open(path, 'r') as f:
        return json.load(f)


def solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    NI = data["parameters"]["NI"]
    NK = data["parameters"]["NK"]
    NT = data["parameters"]["NT"]

    f = data["costs"]["setup_cost_f"]       # scalar setup cost
    g = data["costs"]["startup_cost_g"]     # scalar startup cost
    h = data["costs"]["holding_cost_h"]     # list length NI
    e = data["costs"]["backlogging_cost_e"] # list length NI

    C = data["machine_data"]["capacity_C"]         # list length NK
    sigma = data["machine_data"]["startup_time_sigma"]  # list length NK

    # demand[i][t] -- NI x NT
    demand = data["demand"]

    s0 = data["initial_conditions"]["initial_stock_s0"]    # list length NI
    r0 = data["initial_conditions"]["initial_backlog_r0"]  # list length NI

    # Index ranges (0-based internally)
    items = range(NI)
    machines = range(NK)
    periods = range(NT)  # 0..NT-1 represents periods 1..NT

    # Create model
    model = gp.Model("ConNT_LotSizing")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i,k,t]: production of item i on machine k in period t
    x = {}
    for i in items:
        for k in machines:
            for t in periods:
                x[i, k, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                           name=f"x_{i}_{k}_{t}")

    # s[i,t]: stock of item i at end of period t
    s = {}
    for i in items:
        for t in periods:
            s[i, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                   name=f"s_{i}_{t}")

    # r[i,t]: backlog of item i at end of period t
    r = {}
    for i in items:
        for t in periods:
            r[i, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                   name=f"r_{i}_{t}")

    # y[i,k,t]: setup variable (binary)
    y = {}
    for i in items:
        for k in machines:
            for t in periods:
                y[i, k, t] = model.addVar(vtype=GRB.BINARY,
                                           name=f"y_{i}_{k}_{t}")

    # z[i,k,t]: startup variable (binary)
    z = {}
    for i in items:
        for k in machines:
            for t in periods:
                z[i, k, t] = model.addVar(vtype=GRB.BINARY,
                                           name=f"z_{i}_{k}_{t}")

    model.update()

    # Objective: min sum_{i,k,t} (f * y + g * z) + sum_{i,t} (h_i * s + e_i * r)
    obj = gp.LinExpr()
    for i in items:
        for k in machines:
            for t in periods:
                obj += f * y[i, k, t] + g * z[i, k, t]
        for t in periods:
            obj += h[i] * s[i, t] + e[i] * r[i, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (9): Flow balance
    # s_{i,t-1} - r_{i,t-1} + sum_k x_{ikt} = d_{it} + s_{it} - r_{it}
    for i in items:
        for t in periods:
            lhs = gp.LinExpr()
            # s_{i,t-1}: if t=0, use s0[i]
            if t == 0:
                lhs += s0[i]
            else:
                lhs += s[i, t - 1]
            # -r_{i,t-1}: if t=0, use r0[i]
            if t == 0:
                lhs += -r0[i]
            else:
                lhs += -r[i, t - 1]
            # + sum_k x_{ikt}
            for k in machines:
                lhs += x[i, k, t]

            rhs = demand[i][t] + s[i, t] - r[i, t]
            model.addConstr(lhs == rhs, name=f"flow_{i}_{t}")

    # Constraint (10): Capacity with startup time
    # x_{ikt} + sigma_k * z_{ikt} <= C_k * y_{ikt}
    for i in items:
        for k in machines:
            for t in periods:
                model.addConstr(
                    x[i, k, t] + sigma[k] * z[i, k, t] <= C[k] * y[i, k, t],
                    name=f"cap_{i}_{k}_{t}"
                )

    # Constraint (11): Startup definition
    # z_{ikt} >= y_{ikt} - y_{ik,t-1}
    # For t=0: y_{ik,-1} is not defined.
    # ASSUMPTION (inferred): y_{ik,0} (period before the horizon) = 0,
    # meaning no machine is set up before the planning horizon starts.
    # This is consistent with the initial conditions s_0 = r_0 = 0.
    for i in items:
        for k in machines:
            for t in periods:
                if t == 0:
                    y_prev = 0
                else:
                    y_prev = y[i, k, t - 1]
                model.addConstr(
                    z[i, k, t] >= y[i, k, t] - y_prev,
                    name=f"startup_{i}_{k}_{t}"
                )

    # Constraint (12): At most one item per machine per period
    # sum_i y_{ikt} <= 1
    for k in machines:
        for t in periods:
            model.addConstr(
                gp.quicksum(y[i, k, t] for i in items) <= 1,
                name=f"onesetup_{k}_{t}"
            )

    # Constraint (13): z_{ikt} <= y_{ikt}
    for i in items:
        for k in machines:
            for t in periods:
                model.addConstr(
                    z[i, k, t] <= y[i, k, t],
                    name=f"zbound_{i}_{k}_{t}"
                )

    # Constraint (14): Linking constraint
    # y_{ik,t-1} + z_{ikt} + sum_{j!=i} (y_{jkt} - z_{jkt}) <= 1
    # For t=0: y_{ik,-1} = 0 (same assumption as constraint 11)
    for i in items:
        for k in machines:
            for t in periods:
                if t == 0:
                    y_prev_i = 0
                else:
                    y_prev_i = y[i, k, t - 1]
                lhs = y_prev_i + z[i, k, t]
                for j in items:
                    if j != i:
                        lhs += y[j, k, t] - z[j, k, t]
                model.addConstr(lhs <= 1, name=f"link_{i}_{k}_{t}")

    # Solve
    model.optimize()

    # Extract solution
    result = {
        "instance_id": data.get("instance_id", 1),
        "solver": "gurobi",
        "status": model.Status,
        "objective_value": None,
        "production": {},
        "stock": {},
        "backlog": {},
        "setup": {},
        "startup": {}
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        for i in items:
            for k in machines:
                for t in periods:
                    val = x[i, k, t].X
                    if val > 1e-6:
                        result["production"][f"x_{i}_{k}_{t}"] = round(val, 4)
            for t in periods:
                s_val = s[i, t].X
                r_val = r[i, t].X
                if s_val > 1e-6:
                    result["stock"][f"s_{i}_{t}"] = round(s_val, 4)
                if r_val > 1e-6:
                    result["backlog"][f"r_{i}_{t}"] = round(r_val, 4)
            for k in machines:
                for t in periods:
                    if y[i, k, t].X > 0.5:
                        result["setup"][f"y_{i}_{k}_{t}"] = 1
                    if z[i, k, t].X > 0.5:
                        result["startup"][f"z_{i}_{k}_{t}"] = 1

        if model.Status == GRB.OPTIMAL:
            result["status_text"] = "OPTIMAL"
        elif model.Status == GRB.TIME_LIMIT:
            result["status_text"] = "TIME_LIMIT"
            result["best_bound"] = model.ObjBound
            result["gap"] = model.MIPGap
        else:
            result["status_text"] = f"STATUS_{model.Status}"
    else:
        result["status_text"] = "NO_SOLUTION_FOUND"
        result["objective_value"] = None

    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    print(f"Status: {result['status_text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gurobi solver for the Con-NT Small Bucket Lot-Sizing Problem"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)
