#!/usr/bin/env python3
"""
Gurobi implementation of the Multi-Depot Periodic Vehicle Routing Problem (MDPVRP).

Five-index vehicle flow formulation from:
Vidal et al. (2011) - "A Hybrid Genetic Algorithm for Multi-Depot and Periodic
Vehicle Routing Problems", CIRRELT-2011-05, Annex 7.1.

Subtour elimination constraints (13) are handled via lazy constraint callbacks.
"""

import argparse
import json
import math
import itertools
from gurobipy import Model, GRB, quicksum
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


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_and_solve(instance, time_limit):
    n = instance["n"]
    d = instance["d"]
    t = instance["t"]
    m = instance["m"]
    Q = instance["vehicle_capacity"]
    T = instance["route_duration_limit"]
    depots = instance["depots"]
    customers = instance["customers"]

    # Build vertex sets
    # Depot vertices: indices 0..d-1
    # Customer vertices: indices d..d+n-1
    num_vertices = d + n

    # Coordinates
    coords = {}
    for dep in depots:
        coords[dep["id"]] = (dep["x"], dep["y"])  # depot ids: 0..d-1
    for cust in customers:
        coords[d + cust["id"]] = (cust["x"], cust["y"])  # customer ids shifted by d

    depot_indices = list(range(d))
    customer_indices = list(range(d, d + n))
    all_indices = list(range(num_vertices))

    # Cost matrix
    c = {}
    for i in all_indices:
        for j in all_indices:
            xi, yi = coords[i]
            xj, yj = coords[j]
            c[i, j] = euclidean_distance(xi, yi, xj, yj)

    # Customer data (indexed by vertex index d+cust_id)
    demand = {}
    service = {}
    freq = {}
    patterns = {}
    for cust in customers:
        vi = d + cust["id"]
        demand[vi] = cust["demand"]
        service[vi] = cust["service_duration"]
        freq[vi] = cust["frequency"]
        # patterns: list of lists of period indices (1-based in data)
        patterns[vi] = cust["pattern_list"]

    # Depots have zero demand and service time
    for dep_idx in depot_indices:
        demand[dep_idx] = 0
        service[dep_idx] = 0

    # Pattern binary constants a_{pl}: for pattern p (index in L_i), period l
    # We store pattern_contains[vi] = list of sets, each set contains the periods
    pattern_contains = {}
    for vi in customer_indices:
        pattern_contains[vi] = []
        for pat in patterns[vi]:
            pattern_contains[vi].append(set(pat))

    periods = list(range(1, t + 1))
    vehicles = list(range(1, m + 1))

    # =========================================================================
    # Build Gurobi Model
    # =========================================================================
    model = Model("MDPVRP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LazyConstraints", 1)
    # Reduce output verbosity
    model.setParam("OutputFlag", 1)

    # Decision variables
    # y[i, p_idx, o]: customer i assigned to pattern index p_idx and depot o
    y = {}
    for vi in customer_indices:
        for p_idx in range(len(patterns[vi])):
            for o in depot_indices:
                y[vi, p_idx, o] = model.addVar(
                    vtype=GRB.BINARY, name=f"y_{vi}_{p_idx}_{o}"
                )

    # x[i, j, k, l, o]: vehicle k from depot o on day l travels from i to j
    # To manage size, only create variables where they make sense
    x = {}
    for o in depot_indices:
        for l in periods:
            for k in vehicles:
                for i in all_indices:
                    for j in all_indices:
                        if i == j:
                            continue
                        x[i, j, k, l, o] = model.addVar(
                            vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{l}_{o}"
                        )

    model.update()

    # Objective (5): minimize total routing cost
    model.setObjective(
        quicksum(
            c[i, j] * x[i, j, k, l, o]
            for (i, j, k, l, o) in x
        ),
        GRB.MINIMIZE,
    )

    # Constraint (6): each customer assigned exactly one pattern and one depot
    for vi in customer_indices:
        model.addConstr(
            quicksum(
                y[vi, p_idx, o]
                for p_idx in range(len(patterns[vi]))
                for o in depot_indices
            )
            == 1,
            name=f"assign_{vi}",
        )

    # Constraint (7): customer visits match assigned pattern/depot
    # sum_j sum_k x_{ijklo} - sum_p a_{pl} * y_{ipo} = 0
    # for each customer i, depot o, period l
    for vi in customer_indices:
        for o in depot_indices:
            for l in periods:
                lhs_x = quicksum(
                    x[vi, j, k, l, o]
                    for k in vehicles
                    for j in all_indices
                    if j != vi and (vi, j, k, l, o) in x
                )
                lhs_y = quicksum(
                    (1 if l in pattern_contains[vi][p_idx] else 0) * y[vi, p_idx, o]
                    for p_idx in range(len(patterns[vi]))
                )
                model.addConstr(lhs_x - lhs_y == 0, name=f"visit_{vi}_{o}_{l}")

    # Constraint (8): each vehicle leaves each depot at most once per period
    for o in depot_indices:
        for k in vehicles:
            for l in periods:
                model.addConstr(
                    quicksum(
                        x[o, j, k, l, o]
                        for j in all_indices
                        if j != o and (o, j, k, l, o) in x
                    )
                    <= 1,
                    name=f"depart_{o}_{k}_{l}",
                )

    # Constraint (9): vehicles from depot o cannot leave from other depots
    for o in depot_indices:
        for i in depot_indices:
            if i == o:
                continue
            for k in vehicles:
                for l in periods:
                    model.addConstr(
                        quicksum(
                            x[i, j, k, l, o]
                            for j in all_indices
                            if j != i and (i, j, k, l, o) in x
                        )
                        == 0,
                        name=f"nodep_{i}_{o}_{k}_{l}",
                    )

    # Constraint (10): flow conservation
    for vi in all_indices:
        for o in depot_indices:
            for k in vehicles:
                for l in periods:
                    inflow = quicksum(
                        x[j, vi, k, l, o]
                        for j in all_indices
                        if j != vi and (j, vi, k, l, o) in x
                    )
                    outflow = quicksum(
                        x[vi, j, k, l, o]
                        for j in all_indices
                        if j != vi and (vi, j, k, l, o) in x
                    )
                    model.addConstr(
                        inflow - outflow == 0,
                        name=f"flow_{vi}_{o}_{k}_{l}",
                    )

    # Constraint (11): vehicle capacity
    for o in depot_indices:
        for k in vehicles:
            for l in periods:
                model.addConstr(
                    quicksum(
                        demand[i] * x[i, j, k, l, o]
                        for i in all_indices
                        for j in all_indices
                        if i != j and (i, j, k, l, o) in x
                    )
                    <= Q,
                    name=f"cap_{o}_{k}_{l}",
                )

    # Constraint (12): route duration
    for o in depot_indices:
        for k in vehicles:
            for l in periods:
                model.addConstr(
                    quicksum(
                        (c[i, j] + service[i]) * x[i, j, k, l, o]
                        for i in all_indices
                        for j in all_indices
                        if i != j and (i, j, k, l, o) in x
                    )
                    <= T,
                    name=f"dur_{o}_{k}_{l}",
                )

    # Constraint (13): subtour elimination via lazy callbacks
    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            for o in depot_indices:
                for k in vehicles:
                    for l in periods:
                        # Get edges used by this vehicle
                        edges = []
                        for i in customer_indices:
                            for j in customer_indices:
                                if i != j and (i, j, k, l, o) in x:
                                    val = model.cbGetSolution(x[i, j, k, l, o])
                                    if val > 0.5:
                                        edges.append((i, j))
                        if not edges:
                            continue
                        # Find connected components among customers
                        visited_nodes = set()
                        for i, j in edges:
                            visited_nodes.add(i)
                            visited_nodes.add(j)
                        # Check if depot o is reachable
                        # Build adjacency from all edges (including depot)
                        adj_full = {node: set() for node in visited_nodes | {o}}
                        for i in all_indices:
                            for j in all_indices:
                                if i != j and (i, j, k, l, o) in x:
                                    val = model.cbGetSolution(x[i, j, k, l, o])
                                    if val > 0.5:
                                        adj_full.setdefault(i, set()).add(j)
                                        adj_full.setdefault(j, set()).add(i)
                        # BFS from depot
                        reached = set()
                        queue = [o]
                        while queue:
                            node = queue.pop()
                            if node in reached:
                                continue
                            reached.add(node)
                            for nb in adj_full.get(node, []):
                                if nb not in reached:
                                    queue.append(nb)
                        # Any customer nodes not reached form subtours
                        unreached = visited_nodes - reached
                        if unreached:
                            # Find connected components in unreached
                            adj_sub = {node: set() for node in unreached}
                            for i, j in edges:
                                if i in unreached and j in unreached:
                                    adj_sub[i].add(j)
                                    adj_sub[j].add(i)
                            visited_comp = set()
                            for start in unreached:
                                if start in visited_comp:
                                    continue
                                comp = set()
                                stack = [start]
                                while stack:
                                    node = stack.pop()
                                    if node in comp:
                                        continue
                                    comp.add(node)
                                    for nb in adj_sub[node]:
                                        if nb not in comp:
                                            stack.append(nb)
                                visited_comp |= comp
                                S = list(comp)
                                if len(S) >= 2:
                                    model.cbLazy(
                                        quicksum(
                                            x[i, j, k, l, o]
                                            for i in S
                                            for j in S
                                            if i != j and (i, j, k, l, o) in x
                                        )
                                        <= len(S) - 1
                                    )

    model.optimize(subtour_callback)

    # Extract solution.  When SolCount == 0 (TLE-no-incumbent, INFEAS, etc.)
    # leave objective_value as None — emitting float("inf") was a bug that
    # made downstream CSV parsers treat the run as a feasible-but-infinite
    # solution rather than "no solution found".
    obj_val = model.ObjVal if model.SolCount > 0 else None

    # Build solution details
    solution = {"objective_value": obj_val}

    if model.SolCount > 0:
        routes = []
        for o in depot_indices:
            for l in periods:
                for k in vehicles:
                    route_edges = []
                    for i in all_indices:
                        for j in all_indices:
                            if i != j and (i, j, k, l, o) in x:
                                if x[i, j, k, l, o].X > 0.5:
                                    route_edges.append((i, j))
                    if route_edges:
                        # Reconstruct route from depot
                        succ = {i: j for i, j in route_edges}
                        route_seq = []
                        current = o
                        visited_set = set()
                        while current in succ and current not in visited_set:
                            visited_set.add(current)
                            nxt = succ[current]
                            route_seq.append(nxt)
                            current = nxt
                        # Convert customer indices back to original ids
                        cust_seq = [v - d for v in route_seq if v in customer_indices]
                        if cust_seq:
                            routes.append({
                                "depot": o,
                                "period": l,
                                "vehicle": k,
                                "customers": cust_seq,
                            })
        solution["routes"] = routes

        # Pattern/depot assignments
        assignments = {}
        for vi in customer_indices:
            cust_id = vi - d
            for p_idx in range(len(patterns[vi])):
                for o in depot_indices:
                    if y[vi, p_idx, o].X > 0.5:
                        assignments[cust_id] = {
                            "depot": o,
                            "pattern": patterns[vi][p_idx],
                        }
        solution["assignments"] = assignments

    return solution


def main():
    parser = argparse.ArgumentParser(description="MDPVRP Gurobi solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    solution = build_and_solve(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2, default=str)
    print(f"Solution written to {args.solution_path}")
    print(f"Objective value: {solution['objective_value']}")


if __name__ == "__main__":
    main()
