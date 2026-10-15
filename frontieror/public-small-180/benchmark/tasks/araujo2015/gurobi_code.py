"""
Gurobi implementation of the Capacitated Lot Sizing Problem with Setup Times (CLST).
Regular formulation (CL) from de Araujo et al. (2015),
"Period Decompositions for the Capacitated Lot Sizing Problem with Setup Times",
INFORMS Journal on Computing 27(3):431-448.

Formulation (CL): equations (1)-(5) from the paper.
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


def solve_clst(instance, time_limit):
    num_items = instance["num_items"]
    num_periods = instance["num_periods"]
    I = range(num_items)
    T = range(num_periods)  # 0-indexed: periods 0..num_periods-1

    d = instance["demand"]  # d[i][t]

    # Costs: may be per-item (time-invariant) or per-item-per-period
    # The paper notes data sets have time-invariant costs.
    # Instance JSON has scalar per-item costs; expand to per-item-per-period.
    sc_raw = instance["setup_cost"]
    hc_raw = instance["holding_cost"]
    vc_raw = instance["variable_cost"]
    st_raw = instance["setup_time"]
    vt_raw = instance["variable_time"]
    cap_raw = instance["capacity"]
    fc_raw = instance["initial_inventory_cost"]

    # Build per-item-per-period parameters
    # setup_cost, holding_cost, variable_cost may be scalar per item or list per item per period
    def expand_param(raw, num_items, num_periods):
        """If raw is a flat list of length num_items, expand to [i][t]. Otherwise assume already [i][t]."""
        if isinstance(raw[0], list):
            return raw
        return [[raw[i]] * num_periods for i in range(num_items)]

    sc = expand_param(sc_raw, num_items, num_periods)
    hc = expand_param(hc_raw, num_items, num_periods)
    vc = expand_param(vc_raw, num_items, num_periods)
    st = expand_param(st_raw, num_items, num_periods)
    vt = expand_param(vt_raw, num_items, num_periods)

    # capacity may be a single list of length num_periods
    cap = cap_raw

    fc = fc_raw  # per-item initial inventory cost

    # Compute sd_{i,t,|T|} = sum of demand from period t to end
    # sd[i][t] = sum(d[i][k] for k in range(t, num_periods))
    sd = [[0] * num_periods for _ in I]
    for i in I:
        sd[i][num_periods - 1] = d[i][num_periods - 1]
        for t in range(num_periods - 2, -1, -1):
            sd[i][t] = sd[i][t + 1] + d[i][t]

    # Create model
    model = gp.Model("CLST_CL")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i,t]: production quantity
    x = {}
    for i in I:
        for t in T:
            x[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{t}")

    # y[i,t]: setup binary
    y = {}
    for i in I:
        for t in T:
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")

    # s[i,t]: inventory at end of period t, t=0..num_periods-1
    # s[i,-1] = s_{i0}: initial inventory
    s = {}
    for i in I:
        s[i, -1] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{i}_0")
        for t in T:
            s[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{i}_{t+1}")

    model.update()

    # Objective (1): min sum_i fc_i * s_{i0} + sum_i sum_t (sc_{it}*y_{it} + vc_{it}*x_{it} + hc_{it}*s_{it})
    obj = gp.LinExpr()
    for i in I:
        obj += fc[i] * s[i, -1]
        for t in T:
            obj += sc[i][t] * y[i, t] + vc[i][t] * x[i, t] + hc[i][t] * s[i, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (2): s_{i,t-1} + x_{it} = d_{it} + s_{it}
    for i in I:
        for t in T:
            model.addConstr(
                s[i, t - 1] + x[i, t] == d[i][t] + s[i, t],
                name=f"demand_balance_{i}_{t}"
            )

    # Constraint (3): sum_i (st_{it}*y_{it} + vt_{it}*x_{it}) <= cap_t
    for t in T:
        model.addConstr(
            gp.quicksum(st[i][t] * y[i, t] + vt[i][t] * x[i, t] for i in I) <= cap[t],
            name=f"capacity_{t}"
        )

    # Constraint (4): x_{it} <= min((cap_t - st_{it}) / vt_{it}, sd_{it|T|}) * y_{it}
    for i in I:
        for t in T:
            if vt[i][t] > 0:
                ub = min((cap[t] - st[i][t]) / vt[i][t], sd[i][t])
            else:
                ub = sd[i][t]
            # Ensure non-negative upper bound
            ub = max(ub, 0)
            model.addConstr(
                x[i, t] <= ub * y[i, t],
                name=f"setup_forcing_{i}_{t}"
            )

    # Constraint (5): s_{i,|T|} = 0 (ending inventory is zero)
    for i in I:
        model.addConstr(s[i, num_periods - 1] == 0, name=f"ending_inv_{i}")

    # Optimize
    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = model.Status
        result["mip_gap"] = model.MIPGap if hasattr(model, 'MIPGap') else None

        # Extract variable values
        production = [[0.0] * num_periods for _ in I]
        setup = [[0] * num_periods for _ in I]
        inventory = [[0.0] * (num_periods + 1) for _ in I]  # includes initial

        for i in I:
            inventory[i][0] = s[i, -1].X
            for t in T:
                production[i][t] = x[i, t].X
                setup[i][t] = round(y[i, t].X)
                inventory[i][t + 1] = s[i, t].X

        result["production"] = production
        result["setup"] = setup
        result["inventory"] = inventory
    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["mip_gap"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve CLST using CL formulation with Gurobi"
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
    result = solve_clst(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
