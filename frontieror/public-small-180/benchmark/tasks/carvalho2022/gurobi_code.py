#!/usr/bin/env python3
"""
Gurobi implementation of the ILSSP-NT (Integrated Lot Sizing and Scheduling Problem
with Non-Triangular setup) on parallel machines.

Based on: Carvalho & Nascimento (2022) - "Hybrid matheuristics to solve the integrated
lot sizing and scheduling problem on parallel machines with sequence-dependent and
non-triangular setup"

The MIP formulation (1)-(20) uses a facility location problem reformulation with
network flow for feasible setup sequences.
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
def load_instance(instance_path):
    """Load instance data from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """Build and solve the ILSSP-NT MIP model using Gurobi."""

    # =========================================================================
    # EXTRACT DATA
    # =========================================================================
    n = data["dimensions"]["n"]  # number of items
    m = data["dimensions"]["m"]  # number of machines
    p = data["dimensions"]["p"]  # number of periods

    # d[i][t]: demand of item i in period t (0-indexed)
    d = data["demands"]  # n x p

    # h[i]: unitary inventory cost of item i
    h = data["inventory_costs"]  # length n

    # processing_time: f_{ikt} = processing time of item i on machine k in period t
    # In the instance this is a scalar (=1 for all i,k,t)
    proc_time_val = data["processing_time"]

    # setup_times[i][j][k]: setup time to change machine k from item i to item j
    b_raw = data["setup_times"]  # n x n x m

    # setup_costs[i][j][k]: setup cost to change machine k from item i to item j
    c_raw = data["setup_costs"]  # n x n x m

    # machine_capacities[k][t]: production capacity of machine k in period t
    T_cap = data["machine_capacities"]  # m x p

    # max_production[i][t]: M_{it} maximum production of item i in period t
    # **NOT SPECIFIED IN PAPER**: Exact formula not given; typically M_{it} = sum_{u=t}^{p} d_{iu}.
    # The instance provides this directly.
    M = data["max_production"]  # n x p

    # max_setups_per_item[i][k][t]: q_{ikt}
    # **NOT SPECIFIED IN PAPER**: Exact computation not given.
    # The instance provides this directly.
    q_raw = data["max_setups_per_item"]  # n x m x p

    # minimum_lot_sizes[i]: m_i
    min_lot = data["minimum_lot_sizes"]  # length n

    # shortcut_items: list of shortcut item indices (0-indexed)
    shortcut_items = data.get("shortcut_items", [])

    # machine_eligibility[i][k]: 1 if item i can be produced on machine k
    eligibility = data.get("machine_eligibility", [[1] * m for _ in range(n)])

    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    model = gp.Model("ILSSP_NT")
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)  # Paper uses 1 thread

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================

    # x[i,k,t,u]: amount of item i produced at machine k in period t for demand of period u
    # continuous >= 0, only defined for u >= t
    x = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                for u in range(t, p):
                    x[i, k, t, u] = model.addVar(
                        lb=0.0, vtype=GRB.CONTINUOUS,
                        name=f"x_{i}_{k}_{t}_{u}")

    # x_b[i,k,t]: production before the first setup in period t
    x_b = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                x_b[i, k, t] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS,
                    name=f"xb_{i}_{k}_{t}")

    # x_a[i,k,t]: production after the first setup in period t
    x_a = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                x_a[i, k, t] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS,
                    name=f"xa_{i}_{k}_{t}")

    # z[i,k,t]: binary, 1 if machine k is ready to produce item i at beginning of period t
    # We define t = 0..p (extra period p for boundary condition in constraint 6)
    # **NOT SPECIFIED IN PAPER**: Boundary for z_{ik,p+1}. We add z for t=p as a free
    # binary variable constrained by sum_i z_{ikt}=1 and flow conservation.
    z = {}
    for i in range(n):
        for k in range(m):
            for t in range(p + 1):
                z[i, k, t] = model.addVar(
                    vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")

    # y[i,j,k,t]: number of times production changes from item i to item j
    # on machine k in period t. Integer in {0, ..., q_{jkt}}
    y = {}
    for i in range(n):
        for j in range(n):
            for k in range(m):
                for t in range(p):
                    ub_val = q_raw[j][k][t]
                    y[i, j, k, t] = model.addVar(
                        lb=0, ub=ub_val, vtype=GRB.INTEGER,
                        name=f"y_{i}_{j}_{k}_{t}")

    # R[k,t]: binary, 1 if at least one setup is performed at machine k in period t
    R = {}
    for k in range(m):
        for t in range(p):
            R[k, t] = model.addVar(vtype=GRB.BINARY, name=f"R_{k}_{t}")

    # G[i,k,t]: binary, 1 if machine k is ready at least once to produce item i in period t
    G = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                G[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"G_{i}_{k}_{t}")

    # F0[j,k,t]: commodity flow from dummy origin (node 0) to item j
    F0 = {}
    for j in range(n):
        for k in range(m):
            for t in range(p):
                F0[j, k, t] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS,
                    name=f"F0_{j}_{k}_{t}")

    # F[i,j,k,t]: commodity flow from item i to item j
    F = {}
    for i in range(n):
        for j in range(n):
            for k in range(m):
                for t in range(p):
                    F[i, j, k, t] = model.addVar(
                        lb=0.0, vtype=GRB.CONTINUOUS,
                        name=f"F_{i}_{j}_{k}_{t}")

    model.update()

    # =========================================================================
    # OBJECTIVE FUNCTION (1)
    # min sum_{i,k,t,u>=t} (u-t)*h_i*x_{iktu} + sum_{i,j,k,t} c_{ijk}*y_{ijkt}
    # =========================================================================
    obj = gp.LinExpr()
    for i in range(n):
        for k in range(m):
            for t in range(p):
                for u in range(t, p):
                    obj += (u - t) * h[i] * x[i, k, t, u]

    for i in range(n):
        for j in range(n):
            for k in range(m):
                for t in range(p):
                    obj += c_raw[i][j][k] * y[i, j, k, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- Constraint (2): Demand satisfaction ---
    # sum_k sum_{t=1}^{u} x_{iktu} = d_{iu}  forall (i, u)
    for i in range(n):
        for u in range(p):
            model.addConstr(
                gp.quicksum(x[i, k, t, u] for k in range(m) for t in range(u + 1))
                == d[i][u],
                name=f"demand_{i}_{u}")

    # --- Constraint (3): Capacity ---
    # sum_i (sum_{u>=t} f_{ikt}*x_{iktu} + sum_j b_{jik}*y_{jikt}) <= T_{kt}
    for k in range(m):
        for t in range(p):
            model.addConstr(
                gp.quicksum(
                    proc_time_val * x[i, k, t, u]
                    for i in range(n) for u in range(t, p)
                ) + gp.quicksum(
                    b_raw[j][i][k] * y[j, i, k, t]
                    for i in range(n) for j in range(n)
                ) <= T_cap[k][t],
                name=f"capacity_{k}_{t}")

    # --- Constraint (4): Production only if setup ready ---
    # x_{iktu} <= M_{it} * G_{ikt}  forall (i, k, t, u)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                for u in range(t, p):
                    model.addConstr(
                        x[i, k, t, u] <= M[i][t] * G[i, k, t],
                        name=f"prod_setup_{i}_{k}_{t}_{u}")

    # --- Constraint (5): One setup state per machine per period ---
    # sum_i z_{ikt} = 1  forall (k, t)
    for k in range(m):
        for t in range(p + 1):  # includes boundary period p
            model.addConstr(
                gp.quicksum(z[i, k, t] for i in range(n)) == 1,
                name=f"one_state_{k}_{t}")

    # --- Constraint (6): Flow conservation of setup states ---
    # z_{ikt} + sum_j y_{jikt} = sum_j y_{ijkt} + z_{ik,t+1}  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    z[i, k, t] + gp.quicksum(y[j, i, k, t] for j in range(n))
                    == gp.quicksum(y[i, j, k, t] for j in range(n)) + z[i, k, t + 1],
                    name=f"flow_cons_{i}_{k}_{t}")

    # --- Constraint (7): G link ---
    # z_{ikt} + sum_j y_{jikt} >= G_{ikt}  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    z[i, k, t] + gp.quicksum(y[j, i, k, t] for j in range(n))
                    >= G[i, k, t],
                    name=f"G_link_{i}_{k}_{t}")

    # --- Constraint (8): Max setups limit ---
    # z_{ikt} + sum_j y_{jikt} <= q_{ikt} * G_{ikt}  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    z[i, k, t] + gp.quicksum(y[j, i, k, t] for j in range(n))
                    <= q_raw[i][k][t] * G[i, k, t],
                    name=f"max_setup_{i}_{k}_{t}")

    # --- Constraint (9): Split production ---
    # sum_{u>=t} x_{iktu} = x^a_{ikt} + x^b_{ikt}  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    gp.quicksum(x[i, k, t, u] for u in range(t, p))
                    == x_a[i, k, t] + x_b[i, k, t],
                    name=f"split_{i}_{k}_{t}")

    # --- Constraint (10): Before-setup production requires carry-over ---
    # x^b_{ikt} <= M_{it} * z_{ikt}  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    x_b[i, k, t] <= M[i][t] * z[i, k, t],
                    name=f"before_setup_{i}_{k}_{t}")

    # --- Constraint (11): Minimum lot size (not carried over) ---
    # x^a_{ikt} >= m_i * (sum_j y_{jikt} - z_{ik,t+1})  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    x_a[i, k, t] >= min_lot[i] * (
                        gp.quicksum(y[j, i, k, t] for j in range(n))
                        - z[i, k, t + 1]),
                    name=f"min_lot_{i}_{k}_{t}")

    # --- Constraint (12): Minimum lot size with carry-over ---
    # x^a_{ikt} + sum_{lambda=t+1}^{u} x^b_{ik,lambda}
    #   >= m_i * sum_j y_{jikt} - M_{it} * (sum_{lambda=t+1}^{u-1} R_{k,lambda} + 1 - R_{ku})
    # forall (i, k, t, u), u != t
    for i in range(n):
        for k in range(m):
            for t in range(p):
                for u in range(t + 1, p):
                    lhs = x_a[i, k, t] + gp.quicksum(
                        x_b[i, k, lam] for lam in range(t + 1, u + 1))
                    rhs_setup = gp.quicksum(y[j, i, k, t] for j in range(n))
                    rhs_big_m = (
                        gp.quicksum(R[k, lam] for lam in range(t + 1, u))
                        + 1 - R[k, u])
                    model.addConstr(
                        lhs >= min_lot[i] * rhs_setup - M[i][t] * rhs_big_m,
                        name=f"min_lot_co_{i}_{k}_{t}_{u}")

    # --- R linking constraints ---
    # R_{kt} is 1 iff at least one setup is performed at machine k in period t
    # We link R to y variables:
    #   sum_{i,j} y_{ijkt} >= R_{kt}  (if any setup, R=1)
    #   sum_{i,j} y_{ijkt} <= BigM * R_{kt}  (if no setup, R=0)
    for k in range(m):
        for t in range(p):
            total_y = gp.quicksum(
                y[i, j, k, t] for i in range(n) for j in range(n))
            model.addConstr(total_y >= R[k, t], name=f"R_lb_{k}_{t}")
            big_M_val = sum(q_raw[j][k][t] for j in range(n)) * n
            if big_M_val > 0:
                model.addConstr(
                    total_y <= big_M_val * R[k, t], name=f"R_ub_{k}_{t}")

    # --- Constraint (13): Sub-tour flow origin ---
    # sum_j F_{0jkt} = sum_i G_{ikt}  forall (k, t)
    for k in range(m):
        for t in range(p):
            model.addConstr(
                gp.quicksum(F0[j, k, t] for j in range(n))
                == gp.quicksum(G[i, k, t] for i in range(n)),
                name=f"flow_origin_{k}_{t}")

    # --- Constraint (14): Sub-tour flow balance ---
    # F_{0ikt} + sum_j F_{jikt} = G_{ikt} + sum_j F_{ijkt}  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    F0[i, k, t] + gp.quicksum(F[j, i, k, t] for j in range(n))
                    == G[i, k, t] + gp.quicksum(F[i, j, k, t] for j in range(n)),
                    name=f"flow_bal_{i}_{k}_{t}")

    # --- Constraint (15): Flow capacity from origin ---
    # F_{0ikt} <= n * z_{ikt}  forall (i, k, t)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    F0[i, k, t] <= n * z[i, k, t],
                    name=f"flow_cap_origin_{i}_{k}_{t}")

    # --- Constraint (16): Flow capacity on arcs ---
    # F_{ijkt} <= n * y_{ijkt}  forall (i, j, k, t)
    for i in range(n):
        for j in range(n):
            for k in range(m):
                for t in range(p):
                    model.addConstr(
                        F[i, j, k, t] <= n * y[i, j, k, t],
                        name=f"flow_cap_{i}_{j}_{k}_{t}")

    # --- Machine eligibility constraints ---
    # **INFERRED ASSUMPTION**: If machine_eligibility[i][k] == 0, item i cannot be
    # produced on machine k. We enforce G[i,k,t]=0 for ineligible pairs.
    # The q_{ikt}=0 in data already partially handles this, but we add explicit constraints.
    for i in range(n):
        for k in range(m):
            if eligibility[i][k] == 0:
                for t in range(p):
                    model.addConstr(G[i, k, t] == 0, name=f"elig_{i}_{k}_{t}")

    # =========================================================================
    # OPTIMIZE
    # =========================================================================
    model.optimize()

    # =========================================================================
    # EXTRACT SOLUTION
    # =========================================================================
    result = {}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["mip_gap"] = model.MIPGap

        # Extract production quantities
        production = {}
        for key, var in x.items():
            val = var.X
            if val > 1e-6:
                i, k, t, u = key
                production[f"x_{i}_{k}_{t}_{u}"] = round(val, 4)
        result["production"] = production

        # Extract setup decisions
        setups = {}
        for key, var in y.items():
            val = var.X
            if val > 0.5:
                i, j, k, t = key
                setups[f"y_{i}_{j}_{k}_{t}"] = round(val)
        result["setups"] = setups

        # Extract setup carry-over
        carryover = {}
        for key, var in z.items():
            val = var.X
            if val > 0.5:
                i, k, t = key
                carryover[f"z_{i}_{k}_{t}"] = 1
        result["carryover"] = carryover

    else:
        result["objective_value"] = None
        result["status"] = (
            "infeasible" if model.Status == GRB.INFEASIBLE else "no_solution")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve the ILSSP-NT using Gurobi (MIP formulation)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = build_and_solve(data, args.time_limit)

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
