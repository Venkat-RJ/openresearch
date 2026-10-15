"""
Compact MILP formulation for the Traveling Salesman Problem with Drone (TSP-D).

Implements formulation (1a)-(1o) plus valid inequalities from Section 3.1.1 of:
    Roberti & Ruthmair (2021), "Exact Methods for the Traveling Salesman Problem
    with Drone", Transportation Science, 55(2):315-335.

Solver: Gurobi (matching CPLEX settings from paper: single thread, IntFeasTol=1e-10,
         time limit 3600s).

Usage:
    python gurobi_code.py --instance_path instance_1.json --solution_path solution_1.json
"""

import argparse
import json
import sys

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


def load_instance(instance_path: str) -> dict:
    """Load a TSP-D instance from a JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(data: dict, time_limit: int = 3600) -> dict:
    """
    Build and solve the compact MILP formulation for TSP-D.

    Node indexing convention:
        - Node 0       : depot origin
        - Nodes 1..n   : customers (set N)
        - Node n+1     : depot destination (0' in the paper)

    Arc set A:
        {(0, j) | j in N} U {(i, j) | i, j in N, i != j} U {(i, 0') | i in N}

    Travel time lookups for arcs involving 0' use row/column 0 of the matrices,
    since 0' is the same physical location as the depot.
    """
    n = data["num_customers"]
    customers = list(range(1, n + 1))          # N = {1, ..., n}
    depot_origin = 0
    depot_dest = n + 1                          # virtual node 0'

    truck_times = data["truck_travel_times"]    # (n+1) x (n+1), indexed 0..n
    drone_times = data["drone_travel_times"]    # (n+1) x (n+1), indexed 0..n

    # Helper: map model node to matrix index (0' -> 0, all others unchanged)
    def mx(node):
        return 0 if node == depot_dest else node

    # ------------------------------------------------------------------
    # Build arc set A
    # ------------------------------------------------------------------
    arcs = []
    # Arcs from depot origin to each customer
    for j in customers:
        arcs.append((depot_origin, j))
    # Arcs between customers
    for i in customers:
        for j in customers:
            if i != j:
                arcs.append((i, j))
    # Arcs from each customer to depot destination
    for i in customers:
        arcs.append((i, depot_dest))

    # Precompute travel times on arcs
    t_T = {(i, j): truck_times[mx(i)][mx(j)] for (i, j) in arcs}
    t_D = {(i, j): drone_times[mx(i)][mx(j)] for (i, j) in arcs}

    # Big-M: sum of all truck travel times (safe upper bound on makespan).
    # ASSUMPTION (not specified in paper): M = sum of all entries in truck
    # travel time matrix.
    big_M = sum(truck_times[i][j] for i in range(n + 1) for j in range(n + 1))

    # All nodes in V = {0, 1, ..., n, n+1}
    all_nodes = [depot_origin] + customers + [depot_dest]

    # ------------------------------------------------------------------
    # Build Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model("TSP-D")

    # Solver settings (matching paper's CPLEX configuration)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)
    model.setParam("IntFeasTol", 1e-9)
    # Suppress Gurobi output for cleaner console; remove for debugging
    model.setParam("OutputFlag", 1)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # x^T_{ij}: truck traverses arc (i,j)
    x_T = model.addVars(arcs, vtype=GRB.BINARY, name="xT")
    # x^D_{ij}: drone traverses arc (i,j)
    x_D = model.addVars(arcs, vtype=GRB.BINARY, name="xD")
    # y^T_i: customer i is a truck customer
    y_T = model.addVars(customers, vtype=GRB.BINARY, name="yT")
    # y^D_i: customer i is a drone customer
    y_D = model.addVars(customers, vtype=GRB.BINARY, name="yD")
    # y^C_i: customer i is a combined customer
    y_C = model.addVars(customers, vtype=GRB.BINARY, name="yC")
    # a_i: arrival time at node i
    a = model.addVars(all_nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="a")

    # ------------------------------------------------------------------
    # Objective (1a): minimize a_{0'} (completion time)
    # ------------------------------------------------------------------
    model.setObjective(a[depot_dest], GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Constraint (1b): Truck flow conservation for each customer
    #   sum_{(i,j) in A} x^T_{ij} = sum_{(j,i) in A} x^T_{ji}  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        out_arcs = [(i, j) for (ii, j) in arcs if ii == i]
        in_arcs = [(j, i) for (j, ii) in arcs if ii == i]
        model.addConstr(
            gp.quicksum(x_T[a_] for a_ in out_arcs) ==
            gp.quicksum(x_T[a_] for a_ in in_arcs),
            name=f"truck_flow_{i}"
        )

    # ------------------------------------------------------------------
    # Constraint (1c): Truck arc-customer linking
    #   sum_{(i,j) in A} x^T_{ij} = y^T_i + y^C_i  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        out_arcs = [(i, j) for (ii, j) in arcs if ii == i]
        model.addConstr(
            gp.quicksum(x_T[a_] for a_ in out_arcs) == y_T[i] + y_C[i],
            name=f"truck_link_{i}"
        )

    # ------------------------------------------------------------------
    # Constraint (1d): Truck leaves and returns to depot exactly once
    #   sum_{(0,j) in A} x^T_{0j} = 1
    #   sum_{(i,0') in A} x^T_{i0'} = 1
    # ------------------------------------------------------------------
    depot_out_arcs = [(depot_origin, j) for j in customers]
    depot_in_arcs = [(i, depot_dest) for i in customers]
    model.addConstr(
        gp.quicksum(x_T[a_] for a_ in depot_out_arcs) == 1,
        name="truck_leave_depot"
    )
    model.addConstr(
        gp.quicksum(x_T[a_] for a_ in depot_in_arcs) == 1,
        name="truck_return_depot"
    )

    # ------------------------------------------------------------------
    # Constraint (1e): Drone flow conservation for each customer
    #   sum_{(i,j) in A} x^D_{ij} = sum_{(j,i) in A} x^D_{ji}  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        out_arcs = [(i, j) for (ii, j) in arcs if ii == i]
        in_arcs = [(j, i) for (j, ii) in arcs if ii == i]
        model.addConstr(
            gp.quicksum(x_D[a_] for a_ in out_arcs) ==
            gp.quicksum(x_D[a_] for a_ in in_arcs),
            name=f"drone_flow_{i}"
        )

    # ------------------------------------------------------------------
    # Constraint (1f): Drone arc-customer linking
    #   sum_{(i,j) in A} x^D_{ij} = y^D_i + y^C_i  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        out_arcs = [(i, j) for (ii, j) in arcs if ii == i]
        model.addConstr(
            gp.quicksum(x_D[a_] for a_ in out_arcs) == y_D[i] + y_C[i],
            name=f"drone_link_{i}"
        )

    # ------------------------------------------------------------------
    # Constraint (1g): Drone leaves and returns to depot exactly once
    #   sum_{(0,j) in A} x^D_{0j} = 1
    #   sum_{(i,0') in A} x^D_{i0'} = 1
    # ------------------------------------------------------------------
    model.addConstr(
        gp.quicksum(x_D[a_] for a_ in depot_out_arcs) == 1,
        name="drone_leave_depot"
    )
    model.addConstr(
        gp.quicksum(x_D[a_] for a_ in depot_in_arcs) == 1,
        name="drone_return_depot"
    )

    # ------------------------------------------------------------------
    # Constraint (1h): Each customer visited exactly once
    #   y^T_i + y^D_i + y^C_i = 1  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        model.addConstr(
            y_T[i] + y_D[i] + y_C[i] == 1,
            name=f"visit_once_{i}"
        )

    # ------------------------------------------------------------------
    # Constraint (1i): Truck arrival time / subtour elimination
    #   a_i + t^T_{ij} <= a_j + M(1 - x^T_{ij})  for all (i,j) in A
    # ------------------------------------------------------------------
    for (i, j) in arcs:
        model.addConstr(
            a[i] + t_T[i, j] <= a[j] + big_M * (1 - x_T[i, j]),
            name=f"truck_time_{i}_{j}"
        )

    # ------------------------------------------------------------------
    # Constraint (1j): Drone arrival time / subtour elimination
    #   a_i + t^D_{ij} <= a_j + M(1 - x^D_{ij})  for all (i,j) in A
    # ------------------------------------------------------------------
    for (i, j) in arcs:
        model.addConstr(
            a[i] + t_D[i, j] <= a[j] + big_M * (1 - x_D[i, j]),
            name=f"drone_time_{i}_{j}"
        )

    # ------------------------------------------------------------------
    # Constraint (1k): Drone leg = single drone customer
    #   x^D_{ij} + x^D_{ji} <= y^C_i + y^C_j  for all i,j in N: i < j
    # ------------------------------------------------------------------
    for i in customers:
        for j in customers:
            if i < j:
                model.addConstr(
                    x_D[i, j] + x_D[j, i] <= y_C[i] + y_C[j],
                    name=f"drone_leg_{i}_{j}"
                )

    # ------------------------------------------------------------------
    # Constraint (1l): No depot loops for drone
    #   x^D_{0,i} + x^D_{i,0'} <= 1  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        model.addConstr(
            x_D[depot_origin, i] + x_D[i, depot_dest] <= 1,
            name=f"no_depot_loop_{i}"
        )

    # ------------------------------------------------------------------
    # Fix depot origin arrival time: a_0 = 0
    # ------------------------------------------------------------------
    model.addConstr(a[depot_origin] == 0, name="depot_start")

    # ------------------------------------------------------------------
    # Valid Inequalities (Section 3.1.1)
    # ------------------------------------------------------------------
    # sum_{(i,j) in A} t^T_{ij} * x^T_{ij} <= a_{0'}
    model.addConstr(
        gp.quicksum(t_T[i, j] * x_T[i, j] for (i, j) in arcs) <= a[depot_dest],
        name="valid_ineq_truck"
    )
    # sum_{(i,j) in A} t^D_{ij} * x^D_{ij} <= a_{0'}
    model.addConstr(
        gp.quicksum(t_D[i, j] * x_D[i, j] for (i, j) in arcs) <= a[depot_dest],
        name="valid_ineq_drone"
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    result = {
        "solver": "Gurobi",
        "status": model.Status,
        "status_name": _status_name(model.Status),
        "objective_value": None,
        "best_bound": None,
        "mip_gap": None,
        "solve_time_seconds": model.Runtime,
        "truck_route": [],
        "drone_route": [],
        "customer_types": {},
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap

        # Extract truck route
        truck_route = _extract_route(x_T, arcs, depot_origin, depot_dest, n)
        result["truck_route"] = truck_route

        # Extract drone route
        drone_route = _extract_route(x_D, arcs, depot_origin, depot_dest, n)
        result["drone_route"] = drone_route

        # Extract customer types
        for i in customers:
            if y_T[i].X > 0.5:
                result["customer_types"][str(i)] = "truck"
            elif y_D[i].X > 0.5:
                result["customer_types"][str(i)] = "drone"
            elif y_C[i].X > 0.5:
                result["customer_types"][str(i)] = "combined"

        # Report arrival times for debugging / verification
        arrival_times = {}
        for node in all_nodes:
            arrival_times[str(node)] = a[node].X
        result["arrival_times"] = arrival_times

    return result


def _extract_route(x_vars, arcs, depot_origin, depot_dest, n):
    """
    Extract an ordered route from binary arc variables.

    Returns a list of node ids from depot_origin to depot_dest.
    The depot destination (n+1) is mapped back to 0 in the output
    to represent the return to the physical depot.
    """
    # Build adjacency from selected arcs
    successors = {}
    for (i, j) in arcs:
        if x_vars[i, j].X > 0.5:
            successors[i] = j

    # Trace route from depot_origin
    route = [0]  # start at depot (physical node 0)
    current = depot_origin
    visited = set()
    visited.add(current)
    max_steps = n + 2  # safety bound

    for _ in range(max_steps):
        if current not in successors:
            break
        nxt = successors[current]
        if nxt == depot_dest:
            route.append(0)  # return to depot (physical node 0)
            break
        if nxt in visited:
            # Should not happen in a valid solution; break to avoid loop
            break
        visited.add(nxt)
        route.append(nxt)
        current = nxt

    return route


def _status_name(status_code):
    """Map Gurobi status code to human-readable name."""
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.LOADED: "LOADED",
    }
    return status_map.get(status_code, f"UNKNOWN_{status_code}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve TSP-D using compact MILP formulation with Gurobi."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the input instance JSON file.",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path to write the output solution JSON file.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Maximum solver runtime in seconds.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)
    print(f"Loaded instance: {data.get('problem_name', 'TSP-D')} with "
          f"{data['num_customers']} customers")

    # Solve
    result = build_and_solve(data, args.time_limit)

    # Print summary
    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    print(f"Status:          {result['status_name']}")
    print(f"Objective value: {result['objective_value']}")
    print(f"Best bound:      {result['best_bound']}")
    print(f"MIP gap:         {result['mip_gap']}")
    print(f"Solve time (s):  {result['solve_time_seconds']:.2f}")
    if result["truck_route"]:
        print(f"Truck route:     {result['truck_route']}")
    if result["drone_route"]:
        print(f"Drone route:     {result['drone_route']}")
    if result["customer_types"]:
        print(f"Customer types:  {result['customer_types']}")
    print("=" * 60)

    # Write solution to file
    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)
    print(f"\nSolution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
