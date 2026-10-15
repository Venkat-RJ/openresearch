"""
Gurobi MIP solver for the Capacitated Hub Location Problem with Single Assignment (CHLPSA).

Implements the four-index formulation (MP) from:
  Contreras, Diaz, and Fernandez (2011),
  "Branch and Price for Large-Scale Capacitated Hub Location Problems with Single Assignment",
  INFORMS Journal on Computing, 23(1), pp. 41-55.

Formulation:
  min  sum_k f_k * z_{kk} + sum_{i,j,k,m} F_{ijkm} * x_{ijkm}   (1)
  s.t.
    sum_{k,m} x_{ijkm} = 1                  for all i,j           (2)
    z_{ik} <= z_{kk}                         for all i,k           (3)
    sum_m x_{ijkm} = z_{ik}                  for all i,j,k         (4)
    sum_k x_{ijkm} = z_{jm}                  for all i,j,m         (5)
    sum_i O_i * z_{ik} <= b_k * z_{kk}       for all k             (6)
    sum_k b_k * z_{kk} >= D                                        (7)
    x_{ijkm} >= 0                            for all i,j,k,m       (8)
    z_{ik} in {0,1}                           for all i,k           (9)
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
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def solve_chlpsa(instance, time_limit):
    n = instance["n"]
    N = range(n)

    W = instance["flow_matrix"]
    d = instance["distance_matrix"]
    f_cost = instance["setup_costs"]
    b = instance["capacities"]
    O = instance["outgoing_flow"]
    D = instance["total_flow_D"]

    chi = instance["cost_parameters"]["collection_cost_chi"]
    alpha = instance["cost_parameters"]["transfer_cost_alpha"]
    delta = instance["cost_parameters"]["distribution_cost_delta"]

    # Precompute routing costs F_{ijkm}
    F = {}
    for i in N:
        for j in N:
            for k in N:
                for m in N:
                    F[i, j, k, m] = W[i][j] * (chi * d[i][k] + alpha * d[k][m] + delta * d[m][j])

    # Create model
    model = gp.Model("CHLPSA")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # z[i,k] = 1 if node i is assigned to hub k
    z = {}
    for i in N:
        for k in N:
            z[i, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")

    # x[i,j,k,m] >= 0: flow from i to j routed via hubs k and m
    x = {}
    for i in N:
        for j in N:
            for k in N:
                for m in N:
                    x[i, j, k, m] = model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0.0, name=f"x_{i}_{j}_{k}_{m}"
                    )

    model.update()

    # Objective (1): minimize setup costs + routing costs
    obj = gp.quicksum(f_cost[k] * z[k, k] for k in N) + \
          gp.quicksum(F[i, j, k, m] * x[i, j, k, m] for i in N for j in N for k in N for m in N)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (2): exactly one path for each (i,j) pair
    for i in N:
        for j in N:
            model.addConstr(
                gp.quicksum(x[i, j, k, m] for k in N for m in N) == 1,
                name=f"path_{i}_{j}"
            )

    # Constraint (3): assignment only to open hubs
    for i in N:
        for k in N:
            model.addConstr(z[i, k] <= z[k, k], name=f"assign_{i}_{k}")

    # Constraint (4): linking x and z (origin side)
    for i in N:
        for j in N:
            for k in N:
                model.addConstr(
                    gp.quicksum(x[i, j, k, m] for m in N) == z[i, k],
                    name=f"link_orig_{i}_{j}_{k}"
                )

    # Constraint (5): linking x and z (destination side)
    for i in N:
        for j in N:
            for m in N:
                model.addConstr(
                    gp.quicksum(x[i, j, k, m] for k in N) == z[j, m],
                    name=f"link_dest_{i}_{j}_{m}"
                )

    # Constraint (6): hub capacity
    for k in N:
        model.addConstr(
            gp.quicksum(O[i] * z[i, k] for i in N) <= b[k] * z[k, k],
            name=f"capacity_{k}"
        )

    # Constraint (7): total capacity must cover total demand
    model.addConstr(
        gp.quicksum(b[k] * z[k, k] for k in N) >= D,
        name="total_capacity"
    )

    # Optimize
    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        # Extract hub locations
        hubs = [k for k in N if z[k, k].X > 0.5]
        result["hubs"] = hubs

        # Extract assignments
        assignment = []
        for i in N:
            for k in N:
                if z[i, k].X > 0.5:
                    assignment.append(k)
                    break
        result["assignment"] = assignment

        result["status"] = model.Status
        if model.Status == GRB.OPTIMAL:
            result["status_text"] = "OPTIMAL"
        elif model.Status == GRB.TIME_LIMIT:
            result["status_text"] = "TIME_LIMIT"
        else:
            result["status_text"] = f"STATUS_{model.Status}"

        result["mip_gap"] = model.MIPGap
        result["runtime"] = model.Runtime
    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["status_text"] = "NO_SOLUTION"
        result["hubs"] = []
        result["assignment"] = []

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIP solver for CHLPSA (Contreras et al. 2011)"
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
    result = solve_chlpsa(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']:.2f}")
        print(f"Hubs: {result['hubs']}")
        print(f"Assignment: {result['assignment']}")
        print(f"Status: {result['status_text']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
