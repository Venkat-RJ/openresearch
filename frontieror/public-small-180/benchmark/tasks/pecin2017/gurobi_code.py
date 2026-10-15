#!/usr/bin/env python3
"""
gurobi_code.py

Compact arc-flow MIP formulation for the Vehicle Routing Problem with Time Windows (VRPTW).

Paper: Pecin, Contardo, Desaulniers, Uchoa (2017)
       "New Enhancements for the Exact Solution of the Vehicle Routing Problem
        with Time Windows"
       INFORMS Journal on Computing, Vol. 29, No. 3, pp. 489-502.

Mathematical model source: math_model.txt

IMPLEMENTATION NOTE:
The paper's master problem (1)-(3) is a Set Partitioning Formulation (SPF) with an
exponential number of variables (one per feasible route). It cannot be solved directly
by a MIP solver without column generation. Since column generation requires a specialized
BPC framework, we implement the equivalent compact arc-flow MIP formulation here.

INFERRED ASSUMPTION: We use the standard compact arc-flow MIP with:
  - Binary arc variables x_{ij} in {0,1}
  - Continuous service-start-time variables T_i
  - Continuous cumulative-load variables u_i (MTZ-style subtour/capacity elimination)
This is mathematically equivalent to the pricing subproblem (4)-(10) extended to a
full multi-vehicle setting, and yields the same optimal solution as the SPF.

The big-M value for time propagation is set to H (the planning horizon), which is a
valid upper bound since T_i <= l_i <= H for all nodes.
"""

import argparse
import json
import math
from collections import defaultdict

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
def euclidean(xi, yi, xj, yj):
    return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def solve(instance, time_limit):
    """
    Build and solve the compact VRPTW MIP with Gurobi.
    Returns a solution dict with 'objective_value' and route information.
    """
    nodes = instance["nodes"]
    n_nodes = instance["problem_parameters"]["n_nodes"]
    n_customers = instance["problem_parameters"]["n_customers"]
    Q = instance["problem_parameters"]["vehicle_capacity"]
    H = instance["problem_parameters"]["planning_horizon"]

    # Node indexing: 0 = depot source, 1..n = customers, n+1 = depot sink
    depot_source = 0
    depot_sink = n_nodes - 1  # = n+1 in paper notation
    customers = list(range(1, n_customers + 1))
    customers_set = set(customers)

    # Extract node attributes
    node_x = [0.0] * n_nodes
    node_y = [0.0] * n_nodes
    demand = [0.0] * n_nodes
    e = [0.0] * n_nodes       # time window open (earliest service start)
    l = [float(H)] * n_nodes  # time window close (latest service start)
    svc = [0.0] * n_nodes     # service time

    for node in nodes:
        i = node["id"]
        node_x[i] = node["x"]
        node_y[i] = node["y"]
        demand[i] = node["demand"]
        e[i] = node["time_window_open"]
        l[i] = node["time_window_close"]
        svc[i] = node["service_time"]

    # Travel cost c_{ij} = travel time t_{ij} = Euclidean distance (per paper Section 1)
    dist = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist[i, j] = euclidean(node_x[i], node_y[i], node_x[j], node_y[j])

    # Feasible arc set A (Section 2 of paper, math_model.txt):
    # A = {(i,j): i in V\{n+1}, j in V\{0}, i != j,
    #             d_i + d_j <= Q, e_i + s_i + t_{ij} <= l_j}
    A = []
    out_nbrs = defaultdict(list)
    in_nbrs = defaultdict(list)
    for i in range(n_nodes - 1):       # i in V\{n+1}
        for j in range(1, n_nodes):    # j in V\{0}
            if i == j:
                continue
            if demand[i] + demand[j] <= Q:
                if e[i] + svc[i] + dist[i, j] <= l[j]:
                    A.append((i, j))
                    out_nbrs[i].append(j)
                    in_nbrs[j].append(i)

    # ---------------------------------------------------------------------------
    # Build Gurobi model
    # ---------------------------------------------------------------------------
    model = gp.Model("VRPTW_compact_arcflow")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision variables ---

    # x[i,j] in {0,1}: 1 if arc (i,j) is used (eq. 10)
    x = model.addVars(A, vtype=GRB.BINARY, name="x")

    # T[i] >= 0: service start time at node i (eq. 9)
    T = model.addVars(range(n_nodes), lb=0.0, ub=H, vtype=GRB.CONTINUOUS, name="T")

    # u[i] >= 0: cumulative load when vehicle leaves node i
    # INFERRED ASSUMPTION: MTZ-style load variable for capacity/subtour elimination.
    # u[depot_source] = 0; for customers: demand[i] <= u[i] <= Q.
    u = model.addVars(range(n_nodes), lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="u")

    # --- Objective: minimize total travel cost (eq. 1 restated as arc costs) ---
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for (i, j) in A),
        GRB.MINIMIZE
    )

    # --- Constraints ---

    # Each customer visited exactly once: in-flow = 1 (equivalent to eq. 2)
    for c in customers:
        if in_nbrs[c]:
            model.addConstr(
                gp.quicksum(x[i, c] for i in in_nbrs[c]) == 1,
                name=f"visit_in_{c}"
            )

    # Each customer visited exactly once: out-flow = 1
    for c in customers:
        if out_nbrs[c]:
            model.addConstr(
                gp.quicksum(x[c, j] for j in out_nbrs[c]) == 1,
                name=f"visit_out_{c}"
            )

    # Depot balance: number of vehicles departing = number arriving (eq. 5)
    model.addConstr(
        gp.quicksum(x[depot_source, j] for j in out_nbrs[depot_source]) ==
        gp.quicksum(x[i, depot_sink] for i in in_nbrs[depot_sink]),
        name="depot_balance"
    )

    # Time propagation: if x[i,j]=1 then T[j] >= T[i] + svc[i] + dist[i,j] (eq. 8)
    # Linearized via big-M = H (valid upper bound on time variables)
    M_time = H
    for (i, j) in A:
        model.addConstr(
            T[i] + svc[i] + dist[i, j] <= T[j] + M_time * (1 - x[i, j]),
            name=f"tw_prop_{i}_{j}"
        )

    # Time window bounds at all nodes (eq. 9)
    for i in range(n_nodes):
        model.addConstr(T[i] >= e[i], name=f"tw_lb_{i}")
        model.addConstr(T[i] <= l[i], name=f"tw_ub_{i}")

    # Capacity via MTZ-style constraints (equiv. to eq. 7):
    # Depot source has zero initial load
    model.addConstr(u[depot_source] == 0.0, name="load_depot_source")

    # Customer load bounds
    for c in customers:
        model.addConstr(u[c] >= demand[c], name=f"load_lb_{c}")
        # Upper bound u[c] <= Q already set in variable definition

    # MTZ capacity propagation: if x[i,j]=1 then u[j] >= u[i] + demand[j]
    # Linearized: u[j] >= u[i] + demand[j] - Q*(1 - x[i,j])
    for (i, j) in A:
        if j in customers_set:
            model.addConstr(
                u[j] >= u[i] + demand[j] - Q * (1 - x[i, j]),
                name=f"cap_mtz_{i}_{j}"
            )

    # ---------------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------------
    model.optimize()

    # ---------------------------------------------------------------------------
    # Extract and return solution
    # ---------------------------------------------------------------------------
    status = model.Status
    has_solution = (status == GRB.OPTIMAL) or (
        status == GRB.TIME_LIMIT and model.SolCount > 0
    )

    if not has_solution:
        return {
            "objective_value": None,
            "status": "no_feasible_solution",
            "routes": [],
            "n_routes": 0
        }

    obj_val = model.ObjVal

    # Extract active arcs (x[i,j] > 0.5 in best incumbent)
    active_arcs = [(i, j) for (i, j) in A if x[i, j].X > 0.5]

    # Build adjacency list for route extraction
    adj = defaultdict(list)
    for (i, j) in active_arcs:
        adj[i].append(j)

    # Extract individual routes (paths from depot_source to depot_sink)
    routes = []
    while adj[depot_source]:
        j = adj[depot_source].pop(0)
        route = [depot_source, j]
        cur = j
        while cur != depot_sink:
            if not adj[cur]:
                break  # should not happen in a feasible solution
            nxt = adj[cur].pop(0)
            route.append(nxt)
            cur = nxt
        routes.append(route)

    status_str = "optimal" if status == GRB.OPTIMAL else "time_limit"

    return {
        "objective_value": obj_val,
        "status": status_str,
        "routes": routes,
        "n_routes": len(routes)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi compact arc-flow MIP solver for VRPTW "
                    "(Pecin et al. 2017)"
    )
    parser.add_argument(
        "--instance_path", required=True,
        help="Path to the JSON file containing the problem instance."
    )
    parser.add_argument(
        "--solution_path", required=True,
        help="Path where the final solution JSON file will be written "
             "(e.g., gurobi_solution_1.json)."
    )
    parser.add_argument(
        "--time_limit", type=int, required=True,
        help="Maximum solver runtime in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    solution = solve(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.4f}")
        print(f"Status: {solution['status']}")
        print(f"Number of routes: {solution['n_routes']}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == "__main__":
    main()
