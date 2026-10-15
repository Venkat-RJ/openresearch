#!/usr/bin/env python3
"""
Gurobi implementation of the Direct Path Model (DPM) for the
Edge Set Cost Vehicle Routing Problem with Time Windows (ESVRPTW).

Based on: Reinhardt, Jepsen, & Pisinger (2016),
"The Edge Set Cost of the Vehicle Routing Problem with Time Windows",
Transportation Science, 50(2), 694-707.

This implements the DPM formulation (Section 4) with the tightened
edge set constraints (16)-(17) and Big-M linearization (27).
"""

import argparse
import json
import math
import os
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


def load_instance(instance_path):
    """Load the ESVRPTW instance from a JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """
    Build and solve the DPM (Direct Path Model) for the ESVRPTW using Gurobi.

    DPM formulation from Section 4 of the paper:
      min  sum_{v in K} sum_{(i,j) in A} c_{ij} x_{ij}^v  +  sum_{r in R} c_r y_r   (15)
    s.t.
      sum_{v in K} sum_{j:(j,i) in A_r} x_{ji}^v <= y_r      for all r in R, i in C_r   (16)
      sum_{v in K} sum_{j:(i,j) in A_r} x_{ij}^v <= y_r      for all r in R, i in C_r   (17)
      sum_{v in K} sum_{j:(i,j) in A} x_{ij}^v = 1            for all i in C             (18)
      sum_{i in C} x_{i0}^v = sum_{i in C} x_{0i}^v           for all v in K             (19)
      sum_{j:(j,i) in A} x_{ji}^v - sum_{j:(i,j) in A} x_{ij}^v = 0
                                                                for all i in C, v in K    (20)
      sum_{(i,j) in A} d_i x_{ij}^v <= D                      for all v in K             (21)
      a_i <= t_i^v <= b_i                                      for all i in V, v in K    (22)
      t_i^v + theta_{ij} - t_j^v <= M(1 - x_{ij}^v)          for all v in K, (i,j) in A (27)
      x_{ij}^v in {0,1}                                        for all (i,j) in A, v in K (24)
      y_r in {0,1}                                              for all r in R             (25)
      t_i^v >= 0                                                for all i in V, v in K    (26)
    """
    nodes = data["nodes"]
    edge_sets = data["edge_sets"]
    free_edges = data["free_edges"]
    dist = data["distance_matrix"]
    num_customers = data["num_customers"]
    num_vehicles = data["num_vehicles"]
    vehicle_capacity = data["vehicle_capacity"]
    schedule_horizon = data["schedule_horizon"]

    # Node indices: 0 = depot, 1..num_customers = customers
    depot = 0
    customers = list(range(1, num_customers + 1))
    all_nodes = list(range(num_customers + 1))  # V = {0} union C

    # Build node data lookup
    node_data = {}
    for nd in nodes:
        node_data[nd["id"]] = nd

    # Vehicles
    K = list(range(num_vehicles))

    # Build arc set A: all directed arcs (i,j) for i,j in V, i != j
    # In the direct path model, there is exactly one edge between each pair of nodes.
    # Each undirected edge {i,j} gives two directed arcs (i,j) and (j,i).
    arcs = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                arcs.append((i, j))
    arc_set = set(arcs)

    # Cost and travel time for each arc (i,j)
    # The distance_matrix provides distances. In DPM, c_{ij} = dist[i][j] and theta_{ij} = dist[i][j].
    # **INFERRED ASSUMPTION**: The paper does not explicitly state separate cost and travel time
    # matrices. Following Solomon benchmark convention, we use Euclidean distance for both
    # arc cost and travel time. Service time is added to travel time for time window propagation.
    c = {}
    theta = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                c[(i, j)] = dist[i][j]
                # Travel time = distance + service time at node i
                # **INFERRED ASSUMPTION**: Following standard VRPTW convention (Solomon benchmarks),
                # travel time on arc (i,j) includes service time at node i.
                theta[(i, j)] = dist[i][j] + node_data[i]["service_time"]

    # Build edge set membership: for each edge set r, which arcs belong to A_r
    # and which customers are covered (C_r).
    # An undirected edge [i,j] in edge set r means arcs (i,j) and (j,i) are in A_r.
    edge_set_arcs = {}  # r -> set of arcs
    edge_set_customers = {}  # r -> set of customers covered
    arc_to_edge_sets = {}  # (i,j) -> list of edge set ids containing this arc

    for es in edge_sets:
        r = es["id"]
        edge_set_arcs[r] = set()
        edge_set_customers[r] = set()
        for edge in es["edges"]:
            u, v = edge[0], edge[1]
            edge_set_arcs[r].add((u, v))
            edge_set_arcs[r].add((v, u))
            # Customers covered: non-depot endpoints
            if u != depot:
                edge_set_customers[r].add(u)
            if v != depot:
                edge_set_customers[r].add(v)
            # Track arc->edge_set mapping
            for arc in [(u, v), (v, u)]:
                if arc not in arc_to_edge_sets:
                    arc_to_edge_sets[arc] = []
                arc_to_edge_sets[arc].append(r)

    R = [es["id"] for es in edge_sets]
    edge_set_cost = {es["id"]: es["access_cost"] for es in edge_sets}

    # Big-M value: must be greater than the duration of any route.
    # **INFERRED ASSUMPTION**: M = schedule_horizon + max service time is a safe upper bound,
    # since no route can exceed the schedule horizon. The paper states M should be
    # "greater than the duration of any route" but does not specify the exact value.
    M = schedule_horizon + max(nd["service_time"] for nd in nodes)

    # ---- Build Gurobi Model ----
    model = gp.Model("ESVRPTW_DPM")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i,j,v]: binary, 1 if vehicle v uses arc (i,j)
    x = {}
    for (i, j) in arcs:
        for v in K:
            x[i, j, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{v}")

    # y[r]: binary, 1 if edge set r is accessed
    y = {}
    for r in R:
        y[r] = model.addVar(vtype=GRB.BINARY, name=f"y_{r}")

    # t[i,v]: continuous, time vehicle v visits node i
    t = {}
    for i in all_nodes:
        for v in K:
            a_i = node_data[i]["time_window_start"]
            b_i = node_data[i]["time_window_end"]
            t[i, v] = model.addVar(
                lb=a_i, ub=b_i, vtype=GRB.CONTINUOUS, name=f"t_{i}_{v}"
            )

    model.update()

    # ---- Objective (15) ----
    # min sum_{v in K} sum_{(i,j) in A} c_{ij} x_{ij}^v + sum_{r in R} c_r y_r
    obj = gp.LinExpr()
    for (i, j) in arcs:
        for v in K:
            obj += c[(i, j)] * x[i, j, v]
    for r in R:
        obj += edge_set_cost[r] * y[r]
    model.setObjective(obj, GRB.MINIMIZE)

    # ---- Constraints ----

    # (16) sum_{v in K} sum_{j:(j,i) in A_r} x_{ji}^v <= y_r   for all r in R, i in C_r
    for r in R:
        for i in edge_set_customers[r]:
            lhs = gp.LinExpr()
            for (jj, ii) in edge_set_arcs[r]:
                if ii == i:
                    for v in K:
                        lhs += x[jj, ii, v]
            model.addConstr(lhs <= y[r], name=f"edge_set_in_{r}_{i}")

    # (17) sum_{v in K} sum_{j:(i,j) in A_r} x_{ij}^v <= y_r   for all r in R, i in C_r
    for r in R:
        for i in edge_set_customers[r]:
            lhs = gp.LinExpr()
            for (ii, jj) in edge_set_arcs[r]:
                if ii == i:
                    for v in K:
                        lhs += x[ii, jj, v]
            model.addConstr(lhs <= y[r], name=f"edge_set_out_{r}_{i}")

    # (18) sum_{v in K} sum_{j:(i,j) in A} x_{ij}^v = 1   for all i in C
    for i in customers:
        lhs = gp.LinExpr()
        for j in all_nodes:
            if j != i:
                for v in K:
                    lhs += x[i, j, v]
        model.addConstr(lhs == 1, name=f"visit_{i}")

    # (19) sum_{i in C} x_{i0}^v = sum_{i in C} x_{0i}^v   for all v in K
    for v in K:
        lhs = gp.LinExpr()
        rhs = gp.LinExpr()
        for i in customers:
            lhs += x[i, depot, v]
            rhs += x[depot, i, v]
        model.addConstr(lhs == rhs, name=f"depot_balance_{v}")

    # (20) sum_{j:(j,i) in A} x_{ji}^v - sum_{j:(i,j) in A} x_{ij}^v = 0
    #      for all i in C, v in K
    for i in customers:
        for v in K:
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()
            for j in all_nodes:
                if j != i:
                    inflow += x[j, i, v]
                    outflow += x[i, j, v]
            model.addConstr(inflow - outflow == 0, name=f"flow_{i}_{v}")

    # (21) sum_{(i,j) in A} d_i x_{ij}^v <= D   for all v in K
    for v in K:
        lhs = gp.LinExpr()
        for (i, j) in arcs:
            lhs += node_data[i]["demand"] * x[i, j, v]
        model.addConstr(lhs <= vehicle_capacity, name=f"capacity_{v}")

    # (22) a_i <= t_i^v <= b_i  -- already handled by variable bounds

    # (27) t_i^v + theta_{ij} - t_j^v <= M(1 - x_{ij}^v)   for all v in K, (i,j) in A
    # NOTE: We skip arcs returning to the depot (j == depot) because the single
    # depot time variable t[0,v] represents the departure time.  Applying the
    # constraint on return arcs would force t[0,v] >= arrival-back-at-depot,
    # conflicting with the departure-time semantics and making the model
    # infeasible whenever a route finishes later than it starts.
    for (i, j) in arcs:
        if j == depot:
            continue
        for v in K:
            model.addConstr(
                t[i, v] + theta[(i, j)] - t[j, v] <= M * (1 - x[i, j, v]),
                name=f"time_{i}_{j}_{v}",
            )

    # ---- Single-tour-per-vehicle ----
    # Without this, the formulation lets a vehicle have multiple disjoint
    # depot->...->depot sub-tours (the MTZ-style time constraint (27) is
    # explicitly skipped on return-to-depot arcs, so the time variable doesn't
    # tie sub-tours together). Force each vehicle to leave the depot at most
    # once. Combined with depot balance (19), each vehicle is then a single
    # depot->customers->depot chain.
    for v in K:
        model.addConstr(
            gp.quicksum(x[depot, j, v] for j in customers) <= 1,
            name=f"single_departure_{v}",
        )

    # ---- Symmetry breaking (optional but helpful) ----
    # **INFERRED ASSUMPTION**: The paper does not discuss symmetry breaking among
    # identical vehicles, but it is standard practice to break symmetry by ordering
    # vehicle usage. We order vehicles by the smallest customer index served.
    # We add: if vehicle v does not leave the depot, then vehicle v+1 doesn't either.
    for v in range(len(K) - 1):
        leaves_v = gp.LinExpr()
        leaves_v1 = gp.LinExpr()
        for i in customers:
            leaves_v += x[depot, i, K[v]]
            leaves_v1 += x[depot, i, K[v + 1]]
        model.addConstr(leaves_v >= leaves_v1, name=f"sym_break_{v}")

    # ---- Solve ----
    model.optimize()

    # ---- Extract solution ----
    result = {
        "objective_value": None,
        "status": None,
        "routes": [],
        "edge_sets_used": [],
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = (
            "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        )

        # Extract routes
        for v in K:
            route_arcs = []
            for (i, j) in arcs:
                if x[i, j, v].X > 0.5:
                    route_arcs.append((i, j))
            if route_arcs:
                visit_times = {}
                for i in all_nodes:
                    if t[i, v].X is not None:
                        visit_times[i] = t[i, v].X
                route_obj = {
                    "vehicle": v,
                    "arcs": route_arcs,
                    "visit_times": visit_times,
                }
                try:
                    route_obj["sequence"] = reconstruct_route(route_arcs, depot)
                except RouteReconstructionError as e:
                    route_obj["sequence"] = None
                    route_obj["warning"] = f"reconstruction_failed: {e}"
                result["routes"].append(route_obj)

        # Extract which edge sets are used
        for r in R:
            if y[r].X > 0.5:
                result["edge_sets_used"].append(r)
    else:
        result["status"] = "infeasible_or_no_solution"
        result["objective_value"] = None

    return result


class RouteReconstructionError(ValueError):
    """Raised when arcs assigned to one vehicle do not form a single
    depot->customers->depot tour. Under the corrected DPM (with the
    single_departure_per_vehicle constraint) this should never happen;
    if it does, surface it instead of silently dropping arcs."""


def reconstruct_route(route_arcs, depot):
    """Walk the arcs assigned to a single vehicle into one ordered route.

    Asserts a single depot-departure and that every arc is consumed; raises
    RouteReconstructionError otherwise so model regressions cannot hide as
    "missing customers" downstream.
    """
    if not route_arcs:
        return []

    from collections import defaultdict
    out = defaultdict(list)
    for (i, j) in route_arcs:
        out[i].append(j)

    depot_starts = list(out[depot])
    if len(depot_starts) > 1:
        raise RouteReconstructionError(
            f"vehicle has {len(depot_starts)} depot-departures: {depot_starts}; "
            f"single_departure_per_vehicle constraint missing or violated"
        )
    if len(depot_starts) == 0:
        raise RouteReconstructionError(
            f"vehicle has {len(route_arcs)} arcs but no depot-departure: "
            f"{route_arcs} — likely customer-only cycle (MTZ inactive)"
        )

    used = set()
    route = [depot]
    current = depot
    while True:
        succs = out[current]
        if not succs:
            break
        if len(succs) > 1:
            raise RouteReconstructionError(
                f"node {current} has multiple successors {succs} on one vehicle"
            )
        nxt = succs[0]
        if (current, nxt) in used:
            break
        used.add((current, nxt))
        route.append(nxt)
        if nxt == depot:
            break
        current = nxt

    if len(used) != len(route_arcs):
        unreached = [a for a in route_arcs if a not in used]
        raise RouteReconstructionError(
            f"walked {len(used)}/{len(route_arcs)} arcs; unreached={unreached} "
            f"(disconnected component in vehicle's arc set)"
        )
    return route


def main():
    parser = argparse.ArgumentParser(
        description="Solve ESVRPTW using Gurobi (DPM formulation)"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON instance file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path to write the solution JSON file",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="Maximum solver runtime in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)

    # Build and solve model
    result = build_and_solve(data, args.time_limit)

    # Convert any non-serializable types for JSON output
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return obj
        else:
            return obj

    output = make_serializable(result)

    # Write solution
    with open(args.solution_path, "w") as f:
        output["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(output, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status']}")
        print(f"Number of routes: {len(result['routes'])}")
        print(f"Edge sets used: {result['edge_sets_used']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
