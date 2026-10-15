#!/usr/bin/env python3
"""
Gurobi implementation of the Robust Capacitated Vehicle Routing Problem (RCVRP)
with Knapsack Uncertainty, based on Pessoa et al. (2021).

Since the set-partitioning formulation (14)-(17) has an exponential number of
variables (one per feasible route per vehicle type), it cannot be solved directly.
Instead, we use a compact two-index vehicle flow formulation with robust capacity
constraints, which can be handled directly by Gurobi.

The two-index formulation uses binary arc variables x_{ij} and enforces robust
capacity via subtour elimination / capacity inequalities. For D^{card}, the
worst-case demand for a subset S is computed via a sorting-based approach.
"""

import argparse
import json
import math
import itertools
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


def compute_worst_case_demand_card(customers_in_S, Gamma):
    """
    For D^{card}: max_{d in D} sum_{i in S} d_i
    = sum_{i in S} bar_d_i + hat_d_Gamma(S)
    where hat_d_Gamma(S) = sum of the Gamma largest deviations in S.
    """
    nominal_sum = sum(c['nominal_demand'] for c in customers_in_S)
    deviations = sorted([c['demand_deviation'] for c in customers_in_S], reverse=True)
    dev_sum = sum(deviations[:min(Gamma, len(deviations))])
    return nominal_sum + dev_sum


def solve_rcvrp(instance, time_limit):
    n = instance['num_customers']
    m_vehicles = instance['num_vehicles']
    C = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    unc = instance['uncertainty_set']
    Gamma = unc['Gamma']

    # Nodes: 0 = depot, 1..n = customers
    V = list(range(n + 1))
    V0 = list(range(1, n + 1))  # customer nodes

    # Customer data indexed by node id (1..n)
    bar_d = {c['id']: c['nominal_demand'] for c in customers}
    hat_d = {c['id']: c['demand_deviation'] for c in customers}
    cust_by_id = {c['id']: c for c in customers}

    # Create model
    model = gp.Model("RCVRP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x[i,j] = 1 if arc (i,j) is used
    x = {}
    for i in V:
        for j in V:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, obj=dist[i][j],
                                        name=f"x_{i}_{j}")

    model.update()

    # Constraint: each customer is visited exactly once (in-degree = 1)
    for j in V0:
        model.addConstr(
            gp.quicksum(x[i, j] for i in V if i != j) == 1,
            name=f"in_{j}"
        )

    # Constraint: each customer is left exactly once (out-degree = 1)
    for i in V0:
        model.addConstr(
            gp.quicksum(x[i, j] for j in V if j != i) == 1,
            name=f"out_{i}"
        )

    # Constraint: exactly m vehicles leave the depot
    model.addConstr(
        gp.quicksum(x[0, j] for j in V0) == m_vehicles,
        name="depot_out"
    )

    # Constraint: exactly m vehicles return to the depot
    model.addConstr(
        gp.quicksum(x[i, 0] for i in V0) == m_vehicles,
        name="depot_in"
    )

    # Subtour elimination and robust capacity constraints via lazy constraints.
    # We use a callback approach: whenever an integer solution is found,
    # check for subtours and add robust capacity inequalities.

    # We also add MTZ-like constraints as a starting point to help with subtours.
    # u[i] = cumulative demand up to customer i in its route (under nominal demand).
    # This is a simplification; the robust capacity is enforced via callbacks.
    u = {}
    for i in V0:
        u[i] = model.addVar(lb=bar_d[i], ub=C, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    model.update()

    # MTZ constraints with nominal demands (provides a lower bound on capacity)
    for i in V0:
        for j in V0:
            if i != j:
                model.addConstr(
                    u[j] >= u[i] + bar_d[j] - C * (1 - x[i, j]),
                    name=f"mtz_{i}_{j}"
                )

    # Use lazy constraints callback for robust capacity cuts
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_val = model.cbGetSolution(x)

            # Build adjacency from solution
            adj = {i: [] for i in V}
            for (i, j), var in x.items():
                if x_val[i, j] > 0.5:
                    adj[i].append(j)

            # Find routes starting from depot
            visited = set()
            routes = []
            for j in adj[0]:
                if j == 0:
                    continue
                route = []
                cur = j
                while cur != 0 and cur not in visited:
                    visited.add(cur)
                    route.append(cur)
                    nexts = adj[cur]
                    if nexts:
                        cur = nexts[0]
                    else:
                        break
                if route:
                    routes.append(route)

            # Check for subtours (nodes not connected to depot)
            all_visited = set()
            for r in routes:
                all_visited.update(r)

            # Find any customer nodes forming subtours (not connected to depot)
            unvisited = set(V0) - all_visited
            if unvisited:
                # Find connected components among unvisited nodes
                remaining = set(unvisited)
                while remaining:
                    # BFS to find a component
                    start = next(iter(remaining))
                    component = set()
                    queue = [start]
                    while queue:
                        node = queue.pop(0)
                        if node in component:
                            continue
                        component.add(node)
                        for j in V:
                            if j != node and (node, j) in x and x_val[node, j] > 0.5:
                                if j in remaining and j not in component:
                                    queue.append(j)
                    remaining -= component
                    S = list(component)
                    # Add subtour elimination constraint
                    model.cbLazy(
                        gp.quicksum(x[i, j] for i in V if i not in component
                                    for j in S if i != j and (i, j) in x) >= 1
                    )

            # Check robust capacity for each route
            for route in routes:
                route_customers = [cust_by_id[i] for i in route]
                wc_demand = compute_worst_case_demand_card(route_customers, Gamma)
                if wc_demand > C + 1e-6:
                    S = set(route)
                    # Compute the minimum number of vehicles needed for this subset
                    # under robust capacity: ceil(wc_demand / C)
                    min_vehicles = math.ceil(wc_demand / C)
                    model.cbLazy(
                        gp.quicksum(x[i, j] for i in V if i not in S
                                    for j in S if (i, j) in x) >= min_vehicles
                    )

    # Also add some initial robust rounded capacity inequalities (Eq. 18)
    # for small subsets to strengthen the formulation
    if n <= 20:
        for size in range(2, min(n + 1, 6)):
            for S_tuple in itertools.combinations(V0, size):
                S = list(S_tuple)
                S_set = set(S)
                route_customers = [cust_by_id[i] for i in S]
                wc_demand = compute_worst_case_demand_card(route_customers, Gamma)
                min_vehicles = math.ceil(wc_demand / C)
                if min_vehicles >= 2:
                    model.addConstr(
                        gp.quicksum(x[i, j] for i in V if i not in S_set
                                    for j in S if (i, j) in x) >= min_vehicles,
                        name=f"rcc_{'_'.join(str(s) for s in S)}"
                    )

    model.setParam("LazyConstraints", 1)
    model.optimize(callback)

    # Extract solution
    solution = {
        "objective_value": None,
        "status": None,
        "routes": []
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract routes
        adj = {i: [] for i in V}
        for (i, j) in x:
            if x[i, j].X > 0.5:
                adj[i].append(j)

        visited = set()
        for j in sorted(adj[0]):
            if j == 0:
                continue
            route = []
            cur = j
            while cur != 0 and cur not in visited:
                visited.add(cur)
                route.append(cur)
                nexts = adj[cur]
                if nexts:
                    cur = nexts[0]
                else:
                    break
            if route:
                route_customers = [cust_by_id[i] for i in route]
                route_cost = dist[0][route[0]]
                for idx in range(len(route) - 1):
                    route_cost += dist[route[idx]][route[idx + 1]]
                route_cost += dist[route[-1]][0]
                wc_demand = compute_worst_case_demand_card(route_customers, Gamma)
                solution["routes"].append({
                    "customers": route,
                    "cost": route_cost,
                    "worst_case_demand": wc_demand
                })
    else:
        solution["objective_value"] = None
        solution["status"] = "infeasible"

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for Robust CVRP with Knapsack Uncertainty (Pessoa et al. 2021)"
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
    solution = solve_rcvrp(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
