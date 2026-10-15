"""
STD formulation of the 3LSPD-U (Three-Level Lot-Sizing Problem with
Distribution and Uncapacitated) using Gurobi.

Reference: Cunha et al. (2021)
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


def solve(instance_path: str, solution_path: str, time_limit: float):
    with open(instance_path) as f:
        data = json.load(f)

    num_retailers = data["num_retailers"]
    num_warehouses = data["num_warehouses"]
    NT = data["num_periods"]

    # Index sets
    retailers = list(range(num_retailers))
    warehouses = list(range(num_warehouses))
    periods = list(range(NT))  # 0-indexed: period index t corresponds to period t+1

    # Retailer-warehouse mapping
    retailer_warehouse = data["retailer_warehouse_assignment"]  # retailer r -> warehouse index
    warehouse_retailers = {}
    for w_key, r_list in data["warehouse_retailer_sets"].items():
        warehouse_retailers[int(w_key)] = r_list

    # Demands: demands[r][t]
    demands = data["demands"]

    # Aggregated demand for warehouse w in period t = sum of its retailers' demands
    # Aggregated demand for plant in period t = total demand
    warehouse_demand = [[0.0] * NT for _ in warehouses]
    plant_demand = [0.0] * NT
    for w in warehouses:
        for t in periods:
            warehouse_demand[w][t] = sum(demands[r][t] for r in warehouse_retailers[w])
    for t in periods:
        plant_demand[t] = sum(demands[r][t] for r in retailers)

    # Cumulative demand from period k to NT-1 (inclusive, 0-indexed)
    # For retailers
    retailer_cum = [[0.0] * NT for _ in retailers]
    for r in retailers:
        retailer_cum[r][NT - 1] = demands[r][NT - 1]
        for t in range(NT - 2, -1, -1):
            retailer_cum[r][t] = retailer_cum[r][t + 1] + demands[r][t]

    # For warehouses
    warehouse_cum = [[0.0] * NT for _ in warehouses]
    for w in warehouses:
        warehouse_cum[w][NT - 1] = warehouse_demand[w][NT - 1]
        for t in range(NT - 2, -1, -1):
            warehouse_cum[w][t] = warehouse_cum[w][t + 1] + warehouse_demand[w][t]

    # For plant
    plant_cum = [0.0] * NT
    plant_cum[NT - 1] = plant_demand[NT - 1]
    for t in range(NT - 2, -1, -1):
        plant_cum[t] = plant_cum[t + 1] + plant_demand[t]

    # Costs
    sc_plant = data["fixed_costs"]["plant"]          # length NT
    sc_warehouses = data["fixed_costs"]["warehouses"]  # [w][t]
    sc_retailers = data["fixed_costs"]["retailers"]    # [r][t]

    hc_plant_val = data["holding_costs"]["plant"]        # scalar
    hc_warehouse_val = data["holding_costs"]["warehouses"]  # scalar
    hc_retailers_val = data["holding_costs"]["retailers"]   # list per retailer

    # Build model
    model = gp.Model("3LSPD_U_STD")
    model.setParam("Threads", 1)
    model.setParam("MIPGap", 1e-4)
    model.setParam("TimeLimit", time_limit)

    # Decision variables
    # Plant variables (single plant, index 'p')
    x_p = model.addVars(NT, lb=0.0, name="x_p")
    s_p = model.addVars(NT, lb=0.0, name="s_p")
    y_p = model.addVars(NT, vtype=GRB.BINARY, name="y_p")

    # Warehouse variables
    x_w = model.addVars(num_warehouses, NT, lb=0.0, name="x_w")
    s_w = model.addVars(num_warehouses, NT, lb=0.0, name="s_w")
    y_w = model.addVars(num_warehouses, NT, vtype=GRB.BINARY, name="y_w")

    # Retailer variables
    x_r = model.addVars(num_retailers, NT, lb=0.0, name="x_r")
    s_r = model.addVars(num_retailers, NT, lb=0.0, name="s_r")
    y_r = model.addVars(num_retailers, NT, vtype=GRB.BINARY, name="y_r")

    # Objective
    obj = gp.LinExpr()
    for t in periods:
        # Setup costs
        obj += sc_plant[t] * y_p[t]
        for w in warehouses:
            obj += sc_warehouses[w][t] * y_w[w, t]
        for r in retailers:
            obj += sc_retailers[r][t] * y_r[r, t]
        # Holding costs
        obj += hc_plant_val * s_p[t]
        for w in warehouses:
            obj += hc_warehouse_val * s_w[w, t]
        for r in retailers:
            obj += hc_retailers_val[r] * s_r[r, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints

    # (2) Flow balance for plant: s_{t-1} + x_t = sum_w x_w[w,t] + s_t
    for t in periods:
        s_prev = 0.0 if t == 0 else s_p[t - 1]
        model.addConstr(
            s_prev + x_p[t] == gp.quicksum(x_w[w, t] for w in warehouses) + s_p[t],
            name=f"balance_plant_{t}"
        )

    # (2) Flow balance for warehouses: s_{t-1} + x_w[w,t] = sum_r x_r[r,t] + s_w[w,t]
    for w in warehouses:
        for t in periods:
            s_prev = 0.0 if t == 0 else s_w[w, t - 1]
            model.addConstr(
                s_prev + x_w[w, t] == gp.quicksum(x_r[r, t] for r in warehouse_retailers[w]) + s_w[w, t],
                name=f"balance_warehouse_{w}_{t}"
            )

    # (3) Flow balance for retailers: s_{t-1} + x_r[r,t] = d_r_t + s_r[r,t]
    for r in retailers:
        for t in periods:
            s_prev = 0.0 if t == 0 else s_r[r, t - 1]
            model.addConstr(
                s_prev + x_r[r, t] == demands[r][t] + s_r[r, t],
                name=f"balance_retailer_{r}_{t}"
            )

    # (4) Setup forcing constraints
    # Plant
    for t in periods:
        model.addConstr(x_p[t] <= plant_cum[t] * y_p[t], name=f"setup_plant_{t}")

    # Warehouses
    for w in warehouses:
        for t in periods:
            model.addConstr(x_w[w, t] <= warehouse_cum[w][t] * y_w[w, t],
                            name=f"setup_warehouse_{w}_{t}")

    # Retailers
    for r in retailers:
        for t in periods:
            model.addConstr(x_r[r, t] <= retailer_cum[r][t] * y_r[r, t],
                            name=f"setup_retailer_{r}_{t}")

    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        # Reviewer note Fix_7: the feasibility checker requires every
        # decision variable from the STD formulation; export them alongside
        # the objective so checks (2)-(6) are no longer vacuous.
        result["x_plant"] = [float(x_p[t].X) for t in periods]
        result["s_plant"] = [float(s_p[t].X) for t in periods]
        result["y_plant"] = [float(y_p[t].X) for t in periods]
        result["x_warehouses"] = [[float(x_w[w, t].X) for t in periods] for w in warehouses]
        result["s_warehouses"] = [[float(s_w[w, t].X) for t in periods] for w in warehouses]
        result["y_warehouses"] = [[float(y_w[w, t].X) for t in periods] for w in warehouses]
        result["x_retailers"] = [[float(x_r[r, t].X) for t in periods] for r in retailers]
        result["s_retailers"] = [[float(s_r[r, t].X) for t in periods] for r in retailers]
        result["y_retailers"] = [[float(y_r[r, t].X) for t in periods] for r in retailers]
    else:
        result["objective_value"] = None

    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve 3LSPD-U using STD formulation with Gurobi")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json", help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=float, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)
