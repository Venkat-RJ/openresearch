"""
Gurobi implementation of the LSNM (Lot Sizing with Nonidentical Machines) formulation
for the Capacitated Multi-Item Lot Sizing Problem with Nonidentical Machines (CMLS-NM).

Reference: Wu, Liang, and Zhang (2018), Section 3.1, Equations (1)-(7).
"""

import argparse
import json
import math
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
    """Load a CMLS-NM instance from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def parse_instance(data):
    """Parse the JSON instance into structured data for the model."""
    num_items = data["num_items"]
    num_periods = data["num_periods"]
    num_machines = data["num_machines"]

    J = list(range(num_items))          # items 0..J-1
    T = list(range(num_periods))        # periods 0..T-1
    M = list(range(num_machines))       # machines 0..M-1

    # Machine capabilities: M_j[j] = list of machine indices (0-based)
    # JSON has 1-based machine indices
    M_j = {}
    for j in J:
        M_j[j] = [m - 1 for m in data["machine_capabilities"][j]]

    # F_m[m] = items that can be produced on machine m
    F_m = {m: [] for m in M}
    for j in J:
        for m in M_j[j]:
            F_m[m].append(j)

    # mm_j: max machines per item per period
    mm_j = {j: data["max_machines_per_item_per_period"][j] for j in J}

    # Demand d[j][t]
    d = {}
    for j in J:
        for t in T:
            d[j, t] = data["demand"][j][t]

    # Production time pt[j][m]
    pt = {}
    for j in J:
        for m in M_j[j]:
            key = f"item_{j+1}_machine_{m+1}"
            pt[j, m] = data["production_time"][key]

    # Holding cost hc[j]
    hc = {j: data["holding_cost"][j] for j in J}

    # Setup cost sc[j][m]
    sc = {}
    for j in J:
        for m in M_j[j]:
            key = f"item_{j+1}_machine_{m+1}"
            sc[j, m] = data["setup_cost"][key]

    # Setup time st[j][m]
    st = {}
    for j in J:
        for m in M_j[j]:
            key = f"item_{j+1}_machine_{m+1}"
            st[j, m] = data["setup_time"][key]

    # Production cost pc[j][m]
    pc = {}
    for j in J:
        for m in M_j[j]:
            key = f"item_{j+1}_machine_{m+1}"
            pc[j, m] = data["production_cost"][key]

    # Capacity C[m][t]
    C = {}
    for m in M:
        for t in T:
            C[m, t] = data["capacity"][m][t]

    # Initial inventory I_{j,0}: given parameter (default 0 if absent).
    init_list = data.get("initial_inventory", [0.0] * num_items)
    I0 = {j: init_list[j] for j in J}

    return J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, C, I0


def build_model(J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, C, I0):
    """Build the LSNM MIP model using Gurobi."""
    model = gp.Model("CMLS_NM_LSNM")
    model.setParam("Threads", 1)

    num_T = len(T)

    # --- Decision variables ---
    # X[j,m,t]: production quantity
    X = {}
    for j in J:
        for m in M_j[j]:
            for t in T:
                X[j, m, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                           name=f"X_{j}_{m}_{t}")

    # I[j,t]: inventory at end of period t
    I = {}
    for j in J:
        for t in T:
            I[j, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                    name=f"I_{j}_{t}")

    # Y[j,m,t]: setup binary variable
    Y = {}
    for j in J:
        for m in M_j[j]:
            for t in T:
                Y[j, m, t] = model.addVar(vtype=GRB.BINARY,
                                           name=f"Y_{j}_{m}_{t}")

    model.update()

    # --- Objective function (1) ---
    obj = gp.LinExpr()
    for j in J:
        for t in T:
            for m in M_j[j]:
                obj += sc[j, m] * Y[j, m, t]
                obj += pc[j, m] * X[j, m, t]
            obj += hc[j] * I[j, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---

    # (2) Demand satisfaction / flow balance
    # sum_{m in M_j} X[j,m,t] + I[j,t-1] - I[j,t] = d[j,t]
    # Initial inventory I_{j,0} is a given parameter (I0), read from instance.
    for j in J:
        for t in T:
            lhs = gp.LinExpr()
            for m in M_j[j]:
                lhs += X[j, m, t]
            if t > 0:
                lhs += I[j, t - 1]
            else:
                lhs += I0[j]
            lhs -= I[j, t]
            model.addConstr(lhs == d[j, t],
                            name=f"demand_{j}_{t}")

    # (3) Capacity constraints
    # sum_{j in F_m} pt[j,m]*X[j,m,t] + sum_{j in F_m} st[j,m]*Y[j,m,t] <= C[m,t]
    for m in M:
        for t in T:
            lhs = gp.LinExpr()
            for j in F_m[m]:
                lhs += pt[j, m] * X[j, m, t]
                lhs += st[j, m] * Y[j, m, t]
            model.addConstr(lhs <= C[m, t],
                            name=f"capacity_{m}_{t}")

    # (4) Setup forcing / Big-M constraints
    # X[j,m,t] <= BM[j,m,t] * Y[j,m,t]
    # BM[j,m,t] = min(sum_{q=t}^{|T|-1} d[j,q], (C[m,t] - st[j,m]) / pt[j,m])
    for j in J:
        for m in M_j[j]:
            for t in T:
                # Cumulative demand from t to end
                cum_demand = sum(d[j, q] for q in range(t, num_T))
                # Capacity-based bound
                cap_bound = (C[m, t] - st[j, m]) / pt[j, m] if pt[j, m] > 0 else cum_demand
                cap_bound = max(cap_bound, 0.0)
                BM = min(cum_demand, cap_bound)
                # If BM <= 0, force X and Y to 0
                if BM <= 1e-10:
                    model.addConstr(X[j, m, t] == 0,
                                    name=f"bigM_zero_X_{j}_{m}_{t}")
                    model.addConstr(Y[j, m, t] == 0,
                                    name=f"bigM_zero_Y_{j}_{m}_{t}")
                else:
                    model.addConstr(X[j, m, t] <= BM * Y[j, m, t],
                                    name=f"bigM_{j}_{m}_{t}")

    # (5) Maximum number of machines per item per period
    # sum_{m in M_j} Y[j,m,t] <= mm_j
    for j in J:
        for t in T:
            lhs = gp.LinExpr()
            for m in M_j[j]:
                lhs += Y[j, m, t]
            model.addConstr(lhs <= mm_j[j],
                            name=f"max_machines_{j}_{t}")

    return model, X, I, Y


def solve_and_output(model, X, I, Y, J, T, M_j, args):
    """Set parameters, solve, and write solution JSON."""
    model.setParam("TimeLimit", args.time_limit)
    # Suppress output to keep things clean
    model.setParam("OutputFlag", 1)

    model.optimize()

    # Extract solution
    solution = {}
    obj_val = None

    if model.SolCount > 0:
        obj_val = model.ObjVal

        # Extract production quantities (snap to 0 when setup Y=0 to avoid solver artifacts)
        production = {}
        for j in J:
            for m in M_j[j]:
                for t in T:
                    val = X[j, m, t].X
                    y_val = Y[j, m, t].X
                    if y_val < 0.5:
                        val = 0.0
                    if val > 1e-6:
                        production[f"X_item{j+1}_machine{m+1}_period{t+1}"] = round(val, 6)

        # Extract inventory levels
        inventory = {}
        for j in J:
            for t in T:
                val = I[j, t].X
                if val > 1e-6:
                    inventory[f"I_item{j+1}_period{t+1}"] = round(val, 6)

        # Extract setup decisions
        setups = {}
        for j in J:
            for m in M_j[j]:
                for t in T:
                    val = Y[j, m, t].X
                    if val > 0.5:
                        setups[f"Y_item{j+1}_machine{m+1}_period{t+1}"] = 1

        solution["objective_value"] = obj_val
        solution["status"] = model.Status
        solution["mip_gap"] = model.MIPGap if hasattr(model, 'MIPGap') and model.Status != GRB.OPTIMAL else None
        solution["production"] = production
        solution["inventory"] = inventory
        solution["setups"] = setups
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["mip_gap"] = None
        solution["production"] = {}
        solution["inventory"] = {}
        solution["setups"] = {}

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if obj_val is not None:
        print(f"Objective value: {obj_val}")
        print(f"Solver status: {model.Status}")
    else:
        print("No feasible solution found.")


def main():
    parser = argparse.ArgumentParser(
        description="Solve CMLS-NM using LSNM formulation with Gurobi."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load and parse instance
    data = load_instance(args.instance_path)
    J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, C, I0 = parse_instance(data)

    # Build and solve model
    model, X, I, Y = build_model(J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, C, I0)
    solve_and_output(model, X, I, Y, J, T, M_j, args)


if __name__ == "__main__":
    main()
