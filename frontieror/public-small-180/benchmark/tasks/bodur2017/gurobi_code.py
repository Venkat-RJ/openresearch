"""
Gurobi implementation of the Stochastic Capacitated Facility Location Problem (CAP)
from Bodur, Dash, Günlük, Luedtke (2017), "Strengthened Benders Cuts for Stochastic
Integer Programs with Continuous Recourse", INFORMS Journal on Computing, 29(1):77-91.

Implements formulation (15) directly as an extensive MIP.
"""

import argparse
import json
import math
import sys

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
import time
_GUROBI_CODE_START_TIME = time.time()
# Walk up from this file's directory to find repo root (containing scripts/).
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
    with open(instance_path) as f:
        return json.load(f)


def solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    I = data["num_facilities"]   # number of facilities
    J = data["num_customers"]    # number of customers
    K = data["num_scenarios"]    # number of scenarios

    facilities = data["facilities"]
    customers = data["customers"]
    transport = data["transportation_costs"]   # transport[i][j] = cost q_{ij}
    scenarios = data["scenarios"]

    f_cost = [facilities[i]["opening_cost"] for i in range(I)]
    s_cap  = [facilities[i]["capacity"]     for i in range(I)]
    q_cost = [[transport[i][j] for j in range(J)] for i in range(I)]

    p_k     = [scenarios[k]["probability"] for k in range(K)]
    lam     = [[scenarios[k]["demands"][j] for j in range(J)] for k in range(K)]

    # max total demand across scenarios (for constraint 15d)
    max_total_demand = max(sum(lam[k][j] for j in range(J)) for k in range(K))

    model = gp.Model("CAP_SIP")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 1e-4)
    model.setParam("Threads", 1)

    # First-stage binary variables: x[i] = 1 if facility i is opened
    x = model.addVars(I, vtype=GRB.BINARY, name="x")

    # Second-stage continuous variables: y[k,i,j] = flow from facility i to customer j under scenario k
    y = model.addVars(K, I, J, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

    # Objective (15a): minimize opening costs + expected transportation costs
    obj = gp.quicksum(f_cost[i] * x[i] for i in range(I)) + \
          gp.quicksum(p_k[k] * q_cost[i][j] * y[k, i, j]
                      for k in range(K) for i in range(I) for j in range(J))
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (15b): demand satisfaction for each customer j, scenario k
    for k in range(K):
        for j in range(J):
            model.addConstr(
                gp.quicksum(y[k, i, j] for i in range(I)) >= lam[k][j],
                name=f"demand_{k}_{j}"
            )

    # Constraint (15c): capacity constraint for each facility i, scenario k
    for k in range(K):
        for i in range(I):
            model.addConstr(
                gp.quicksum(y[k, i, j] for j in range(J)) <= s_cap[i] * x[i],
                name=f"capacity_{k}_{i}"
            )

    # Constraint (15d): relatively complete recourse — total open capacity >= max total demand
    model.addConstr(
        gp.quicksum(s_cap[i] * x[i] for i in range(I)) >= max_total_demand,
        name="recourse_feasibility"
    )

    model.optimize()

    # Extract best solution found
    status = model.Status
    obj_val = None

    if status == GRB.OPTIMAL or (status == GRB.TIME_LIMIT and model.SolCount > 0):
        obj_val = model.ObjVal

        # Build solution dict
        x_sol = {i: round(x[i].X) for i in range(I)}
        y_sol = {}
        for k in range(K):
            y_sol[k] = {}
            for i in range(I):
                y_sol[k][i] = {}
                for j in range(J):
                    val = y[k, i, j].X
                    if val > 1e-9:
                        y_sol[k][i][j] = val

        solution = {
            "instance_id": data.get("instance_id", 1),
            "objective_value": obj_val,
            "status": model.Status,
            "mip_gap": model.MIPGap,
            "open_facilities": [i for i in range(I) if x_sol[i] > 0.5],
            "x": x_sol,
            "y": {str(k): {str(i): {str(j): v for j, v in row.items()}
                           for i, row in scen.items()}
                  for k, scen in y_sol.items()},
        }
    else:
        # No feasible solution found
        solution = {
            "instance_id": data.get("instance_id", 1),
            "objective_value": None,
            "status": model.Status,
            "message": "No feasible solution found within time limit."
        }

    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Objective value: {obj_val}")
    print(f"Solution written to {solution_path}")

    return obj_val


def main():
    parser = argparse.ArgumentParser(
        description="Solve Stochastic CAP using Gurobi (extensive formulation)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the JSON solution file.")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
