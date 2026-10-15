"""
Gurobi MIP implementation of the Capacitated Facility Location Problem (CFLP).

Reference: Klose & Drexl (2005), "Lower Bounds for the Capacitated Facility
Location Problem Based on Column Generation", Management Science 51(11):1689-1705.

The model:
    min  sum_{i,j} c_{ij} x_{ij} + sum_j f_j y_j
    s.t. sum_j x_{ij} = 1                for all i       (D) demand
         sum_i d_i x_{ij} <= s_j y_j     for all j       (C) capacity
         0 <= x_{ij} <= 1                for all i,j     (N)
         y_j in {0,1}                    for all j       (I)

Optionally, valid inequalities (B) x_{ij} <= y_j and (T) sum_j s_j y_j >= d^I
are added as cuts to tighten the LP relaxation.
"""

import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
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
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_model(data, time_limit):
    n_cust = data["num_customers"]
    n_plant = data["num_plants"]
    d = data["customers"]["demands"]
    s = data["plants"]["capacities"]
    f = data["plants"]["fixed_costs"]
    c = data["shipping_costs"]  # c[i][j]
    total_demand = sum(d)

    I = range(n_cust)
    J = range(n_plant)

    model = gp.Model("CFLP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = {}
    for i in I:
        for j in J:
            x[i, j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                   name=f"x_{i}_{j}")
    y = {}
    for j in J:
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

    model.update()

    # Objective
    model.setObjective(
        gp.quicksum(c[i][j] * x[i, j] for i in I for j in J)
        + gp.quicksum(f[j] * y[j] for j in J),
        GRB.MINIMIZE
    )

    # (D) Demand constraints
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, j] for j in J) == 1,
            name=f"demand_{i}"
        )

    # (C) Capacity constraints
    for j in J:
        model.addConstr(
            gp.quicksum(d[i] * x[i, j] for i in I) <= s[j] * y[j],
            name=f"capacity_{j}"
        )

    # (B) Variable upper bound constraints (valid inequality to tighten LP)
    for i in I:
        for j in J:
            model.addConstr(x[i, j] <= y[j], name=f"vub_{i}_{j}")

    # (T) Aggregate capacity constraint (valid inequality)
    model.addConstr(
        gp.quicksum(s[j] * y[j] for j in J) >= total_demand,
        name="agg_capacity"
    )

    return model, x, y


def solve_and_export(model, x, y, data, solution_path):
    n_cust = data["num_customers"]
    n_plant = data["num_plants"]

    model.optimize()

    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["gap"] = model.MIPGap
        result["status"] = model.Status
        result["solve_time"] = model.Runtime

        # Extract solution
        y_sol = {}
        for j in range(n_plant):
            val = y[j].X
            if val > 0.5:
                y_sol[j] = 1
        result["open_plants"] = sorted(y_sol.keys())

        x_sol = {}
        for i in range(n_cust):
            assignments = {}
            for j in range(n_plant):
                val = x[i, j].X
                if val > 1e-6:
                    assignments[j] = val
            x_sol[i] = assignments
        result["assignments"] = {str(i): {str(j): v for j, v in assigns.items()}
                                 for i, assigns in x_sol.items()}
    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["solve_time"] = model.Runtime

    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve CFLP using Gurobi MIP (Klose & Drexl 2005)")
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
    print(f"Loaded instance: {data['num_customers']} customers x "
          f"{data['num_plants']} plants, r={data['capacity_tightness_ratio']}")

    model, x, y = build_model(data, args.time_limit)
    result = solve_and_export(model, x, y, data, args.solution_path)

    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']:.4f}")
        print(f"Best bound: {result['best_bound']:.4f}")
        print(f"Gap: {result['gap']:.6f}")
        print(f"Open plants: {len(result['open_plants'])}")
    else:
        print("No feasible solution found.")
    print(f"Solve time: {result['solve_time']:.2f}s")


if __name__ == "__main__":
    main()
