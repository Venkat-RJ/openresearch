#!/usr/bin/env python3
"""
Gurobi implementation of PSCLP and MCLP models from:
Cordeau, Furini, and Ljubic (2019) - "Benders decomposition for very large
scale partial set covering and maximal covering location problems."

Usage:
    python gurobi_code.py --instance_path instance_1.json --solution_path gurobi_solution_1.json --time_limit 3600
"""

import argparse
import json
import os
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
        return json.load(f)


def solve_psclp(instance, time_limit):
    """
    Partial Set Covering Location Problem (PSCLP).

    min  sum_{i in I} f_i * y_i
    s.t. sum_{i in I(j)} y_i >= z_j,   for all j in J
         sum_{j in J} d_j * z_j >= D
         y_i in {0,1}, i in I
         z_j in [0,1],  j in J   (relaxed per Property 1)
    """
    n_facilities = instance["num_facilities"]
    n_customers = instance["num_customers"]
    facility_cost = instance["facility_cost"]
    customer_demands = instance["customer_demands"]
    D = instance["covering_demand_D"]
    cust_to_fac = instance["customer_to_facilities_coverage"]

    model = gp.Model("PSCLP")
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 1

    # Decision variables
    y = model.addVars(n_facilities, vtype=GRB.BINARY, name="y")
    z = model.addVars(n_customers, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")

    # Objective: minimize total facility cost
    model.setObjective(
        gp.quicksum(facility_cost[i] * y[i] for i in range(n_facilities)),
        GRB.MINIMIZE,
    )

    # Coverage constraints: z_j <= sum_{i in I(j)} y_i
    for j in range(n_customers):
        I_j = cust_to_fac.get(str(j), [])
        model.addConstr(
            gp.quicksum(y[i] for i in I_j) >= z[j],
            name=f"cover_{j}",
        )

    # Demand constraint: sum_j d_j * z_j >= D
    model.addConstr(
        gp.quicksum(customer_demands[j] * z[j] for j in range(n_customers)) >= D,
        name="demand",
    )

    start = time.time()
    model.optimize()
    wall_time = time.time() - start

    result = {
        "problem_type": "PSCLP",
        "status": model.Status,
        "status_name": _status_name(model.Status),
        "wall_time": wall_time,
        "time_limit": time_limit,
        "num_facilities": n_facilities,
        "num_customers": n_customers,
        "covering_demand_D": D,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap
        result["open_facilities"] = [i for i in range(n_facilities) if y[i].X > 0.5]
        result["num_open_facilities"] = len(result["open_facilities"])
        covered_demand = sum(
            customer_demands[j] * z[j].X for j in range(n_customers)
        )
        result["covered_demand"] = covered_demand
    else:
        result["objective_value"] = None
        result["best_bound"] = None
        result["mip_gap"] = None
        result["open_facilities"] = []
        result["num_open_facilities"] = 0
        result["covered_demand"] = 0.0

    model.dispose()
    return result


def solve_mclp(instance, time_limit):
    """
    Maximal Covering Location Problem (MCLP).

    max  sum_{j in J} d_j * z_j
    s.t. sum_{i in I} f_i * y_i <= B
         sum_{i in I(j)} y_i >= z_j,   for all j in J
         y_i in {0,1}, i in I
         z_j in [0,1],  j in J   (relaxed per Property 1)
    """
    n_facilities = instance["num_facilities"]
    n_customers = instance["num_customers"]
    facility_cost = instance["facility_cost"]
    customer_demands = instance["customer_demands"]
    B = instance["budget_B"]
    cust_to_fac = instance["customer_to_facilities_coverage"]

    model = gp.Model("MCLP")
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 1

    # Decision variables
    y = model.addVars(n_facilities, vtype=GRB.BINARY, name="y")
    z = model.addVars(n_customers, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")

    # Objective: maximize covered demand
    model.setObjective(
        gp.quicksum(customer_demands[j] * z[j] for j in range(n_customers)),
        GRB.MAXIMIZE,
    )

    # Budget constraint: sum_i f_i * y_i <= B
    model.addConstr(
        gp.quicksum(facility_cost[i] * y[i] for i in range(n_facilities)) <= B,
        name="budget",
    )

    # Coverage constraints: z_j <= sum_{i in I(j)} y_i
    for j in range(n_customers):
        I_j = cust_to_fac.get(str(j), [])
        model.addConstr(
            gp.quicksum(y[i] for i in I_j) >= z[j],
            name=f"cover_{j}",
        )

    start = time.time()
    model.optimize()
    wall_time = time.time() - start

    result = {
        "problem_type": "MCLP",
        "status": model.Status,
        "status_name": _status_name(model.Status),
        "wall_time": wall_time,
        "time_limit": time_limit,
        "num_facilities": n_facilities,
        "num_customers": n_customers,
        "budget_B": B,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap
        result["open_facilities"] = [i for i in range(n_facilities) if y[i].X > 0.5]
        result["num_open_facilities"] = len(result["open_facilities"])
        result["covered_demand"] = model.ObjVal
    else:
        result["objective_value"] = None
        result["best_bound"] = None
        result["mip_gap"] = None
        result["open_facilities"] = []
        result["num_open_facilities"] = 0
        result["covered_demand"] = 0.0

    model.dispose()
    return result


def _status_name(status):
    mapping = {
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
    return mapping.get(status, f"UNKNOWN_{status}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve PSCLP/MCLP using Gurobi (Cordeau et al. 2019)"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to instance JSON file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default=None,
        help="Path for solution JSON output. Defaults to gurobi_solution_{i}.json",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Time limit in seconds (default: 3600)",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    instance_id = instance["instance_id"]
    problem_types = instance["problem_types"]

    # Determine output directory (same as instance file)
    out_dir = os.path.dirname(os.path.abspath(args.instance_path))

    results = {}

    # Solve PSCLP if listed
    if "PSCLP" in problem_types:
        print(f"{'='*60}")
        print(f"Solving PSCLP for instance {instance_id}")
        print(f"{'='*60}")
        results["PSCLP"] = solve_psclp(instance, args.time_limit)

    # Solve MCLP if listed
    if "MCLP" in problem_types:
        print(f"{'='*60}")
        print(f"Solving MCLP for instance {instance_id}")
        print(f"{'='*60}")
        results["MCLP"] = solve_mclp(instance, args.time_limit)

    # Determine primary objective_value (first problem type solved)
    primary_type = problem_types[0]
    primary_result = results.get(primary_type, {})
    primary_obj = primary_result.get("objective_value", None)

    sol_path = args.solution_path if args.solution_path else os.path.join(
        out_dir, f"gurobi_solution_{instance_id}.json"
    )

    solution = {
        "instance_id": instance_id,
        "objective_value": primary_obj,
        "solver": "gurobi",
        "primary_problem_type": primary_type,
        "results": results,
    }
    with open(sol_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)
    print(f"\nSolution written to {sol_path}")

    # Print summary
    for ptype, res in results.items():
        print(f"\n--- {ptype} Summary ---")
        print(f"  Status: {res['status_name']}")
        print(f"  Objective: {res['objective_value']}")
        if res.get("best_bound") is not None:
            print(f"  Best bound: {res['best_bound']}")
        if res.get("mip_gap") is not None:
            print(f"  MIP gap: {res['mip_gap']:.6f}")
        print(f"  Open facilities: {res['num_open_facilities']}")
        print(f"  Wall time: {res['wall_time']:.2f}s")


if __name__ == "__main__":
    main()
