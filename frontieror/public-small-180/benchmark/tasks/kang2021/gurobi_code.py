"""
Gurobi MIP formulation (P) for the Heterogeneous Drone-Truck Routing Problem (HDTRP).

Based on: Kang & Lee (2021), "An Exact Algorithm for Heterogeneous Drone-Truck
Routing Problem", Transportation Science, 55(5):1088-1112.

This implements the compact MIP formulation (P) with MTZ subtour elimination constraints.
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
    """Load and parse an HDTRP instance from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def solve_hdtrp(instance, time_limit):
    """
    Solve HDTRP using the compact MIP formulation (P) from the paper.

    Formulation (P):
        min  sum_{(i,j) in A} t^v_{ij} x_{ij} + sum_{i in N_s} w_i
        s.t. constraints (2)-(12) as described in the paper.
    """
    num_nodes = instance["num_nodes"]
    depot = instance["depot_index"]
    customers = instance["customer_indices"]
    N = customers  # customer set
    n_customers = len(N)

    # Depot duplication: s = depot (start), t = num_nodes (end/sink)
    s = depot
    t = num_nodes  # virtual sink node

    N_s = N + [s]       # N union {s}
    N_t = N + [t]       # N union {t}
    N_st = N + [s, t]   # N union {s, t}

    # Arcs: A = {(i,j) | i in N_s, j in N_t, i != j}
    A = [(i, j) for i in N_s for j in N_t if i != j]

    # Parameters
    num_drones = instance["parameters"]["num_drones"]
    L = list(range(num_drones))  # drone set

    # Truck travel times (from instance, includes service time)
    tv = instance["truck_travel_times"]

    # Drone battery consumption and delivery time (from instance)
    b = instance["drone_battery_consumption"]  # b[l][i][j]
    tau = instance["drone_delivery_time"]       # tau[l][i][j]

    # Drone battery capacities
    B = instance["parameters"]["drone_battery_capacities"]

    # Helper: get truck travel time t^v_{ij}
    # For arcs to t (sink), use travel time to depot
    def get_tv(i, j):
        ri = i if i != t else depot
        rj = j if j != t else depot
        return tv[ri][rj]

    # Helper: get drone battery consumption b^l_{ij}
    def get_b(l, i, j):
        ri = i if i != t else depot
        rj = j if j != t else depot
        return b[l][ri][rj]

    # Helper: get drone delivery time tau^l_{ij}
    def get_tau(l, i, j):
        ri = i if i != t else depot
        rj = j if j != t else depot
        return tau[l][ri][rj]

    # Big-M values
    # NOT SPECIFIED IN PAPER: Using standard choices
    # M_mtz = |N| + 1 for MTZ constraint (5)
    # M_dispatch = |N| for constraint (7)
    M_mtz = n_customers + 1
    M_dispatch = n_customers

    # Create model
    model = gp.Model("HDTRP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x_{ij} in {0,1}: truck moves from i to j
    x = {}
    for (i, j) in A:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # h^l_{ij} in {0,1}: drone l dispatched at i for delivery of j
    h = {}
    for l in L:
        for (i, j) in A:
            # h^l_{ij} only meaningful for i in N_s, j in N
            if j in N:
                h[l, i, j] = model.addVar(vtype=GRB.BINARY, name=f"h_{l}_{i}_{j}")

    # v_i: visiting order (continuous)
    v = {}
    for i in N_st:
        v[i] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"v_{i}")

    # w_i >= 0: waiting time at node i
    w = {}
    for i in N_s:
        w[i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"w_{i}")

    model.update()

    # Objective (1): min sum t^v_{ij} * x_{ij} + sum w_i
    obj = gp.quicksum(get_tv(i, j) * x[i, j] for (i, j) in A) + \
          gp.quicksum(w[i] for i in N_s)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (2): sum_{j in N} x_{sj} = 1
    model.addConstr(
        gp.quicksum(x[s, j] for j in N) == 1, name="depart_depot"
    )

    # Constraint (3): sum_{i in N} x_{it} = 1
    model.addConstr(
        gp.quicksum(x[i, t] for i in N) == 1, name="return_depot"
    )

    # Constraint (4): flow balance for each i in N
    for i in N:
        outgoing = gp.quicksum(x[i, j] for j in N_t if j != i)
        incoming = gp.quicksum(x[j, i] for j in N_s if j != i)
        model.addConstr(outgoing == incoming, name=f"flow_{i}")

    # Constraint (5): MTZ subtour elimination
    # v_i - v_j <= M(1 - x_{ij}) - 1, for all (i,j) in A
    for (i, j) in A:
        model.addConstr(
            v[i] - v[j] <= M_mtz * (1 - x[i, j]) - 1,
            name=f"mtz_{i}_{j}"
        )

    # Constraint (6): each customer served by exactly one mode
    # sum_{i in N_s: i!=j} x_{ij} + sum_{i in N_s: i!=j} sum_{l in L} h^l_{ij} = 1
    for j in N:
        truck_visit = gp.quicksum(x[i, j] for i in N_s if i != j)
        drone_visit = gp.quicksum(
            h[l, i, j] for l in L for i in N_s if i != j
        )
        model.addConstr(truck_visit + drone_visit == 1, name=f"serve_{j}")

    # Constraint (7): drones dispatched only from truck-visited nodes
    # M * sum_{j in N_t: j!=i} x_{ij} >= sum_{j in N: j!=i} sum_{l in L} h^l_{ij}
    for i in N_s:
        lhs = M_dispatch * gp.quicksum(x[i, j] for j in N_t if j != i)
        rhs = gp.quicksum(h[l, i, j] for l in L for j in N if j != i)
        model.addConstr(lhs >= rhs, name=f"dispatch_{i}")

    # Constraint (8): drone battery capacity
    # sum_{i in N_s} sum_{j in N: j!=i} b^l_{ij} h^l_{ij} <= B^l
    for l in L:
        model.addConstr(
            gp.quicksum(
                get_b(l, i, j) * h[l, i, j]
                for i in N_s for j in N if j != i
            ) <= B[l],
            name=f"battery_{l}"
        )

    # Constraint (9): waiting time >= drone delivery time
    # w_i >= sum_{j in N: j!=i} tau^l_{ij} h^l_{ij}
    for i in N_s:
        for l in L:
            model.addConstr(
                w[i] >= gp.quicksum(
                    get_tau(l, i, j) * h[l, i, j]
                    for j in N if j != i
                ),
                name=f"wait_{i}_{l}"
            )

    # Constraint (10): v_s = 0
    model.addConstr(v[s] == 0, name="vs_zero")

    # Solve
    model.optimize()

    # Extract solution
    result = {"objective_value": None}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        # Extract truck route
        truck_route = []
        for (i, j) in A:
            if x[i, j].X > 0.5:
                truck_route.append((i, j))
        result["truck_route"] = truck_route

        # Extract drone assignments
        drone_assignments = []
        for l in L:
            for i in N_s:
                for j in N:
                    if j != i and (l, i, j) in h and h[l, i, j].X > 0.5:
                        drone_assignments.append({
                            "drone": l,
                            "dispatch_node": i,
                            "delivery_node": j
                        })
        result["drone_assignments"] = drone_assignments

        # Waiting times
        waiting_times = {}
        for i in N_s:
            wt = w[i].X
            if wt > 1e-6:
                waiting_times[str(i)] = wt
        result["waiting_times"] = waiting_times

        # Solver stats
        result["solver_status"] = model.Status
        result["mip_gap"] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0
        result["solve_time"] = model.Runtime
        result["num_nodes"] = int(model.NodeCount)
    else:
        result["solver_status"] = model.Status
        result["solve_time"] = model.Runtime

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve HDTRP using Gurobi MIP formulation (P)."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path for the output solution JSON file."
    )
    parser.add_argument(
        "--time_limit", type=int, required=True,
        help="Maximum solver runtime in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    instance = load_instance(args.instance_path)

    # Solve
    result = solve_hdtrp(instance, args.time_limit)

    # Save solution
    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution saved to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
