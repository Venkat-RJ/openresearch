#!/usr/bin/env python3
"""
Orienteering Problem (OP) solver using Gurobi.

Implements the MIP formulation from Kobeaga et al. (2024):
  max  sum_v s_v * y_v                                        (2a)
  s.t. sum_e d_e * x_e <= d_0                                 (2b)
       x(delta(v)) - 2*y_v = 0       for all v                (2c)
       x(delta(H)) - 2*y_l - 2*y_r >= -2   (SECs, lazy)      (2d)
       y_v - x_e >= 0                 for v in V, e in delta(v)(2e)
       y_o = 1                                                 (2h)
       x_e in {0,1}                                            (2i)

Subtour elimination constraints (2d) are added dynamically via
Gurobi lazy constraint callbacks using connected component detection.
"""

import argparse
import json
import sys
from itertools import combinations

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
def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve the Orienteering Problem using Gurobi."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the instance JSON file.",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path to write the solution JSON file.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="Time limit in seconds for Gurobi.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    return parser.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_edges(n):
    """Return list of undirected edges (i, j) with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def delta(v, edges):
    """Return indices of edges incident to vertex v."""
    return [idx for idx, (i, j) in enumerate(edges) if i == v or j == v]


def find_connected_components(n, selected_edges):
    """Find connected components among vertices that have at least one
    selected edge, using union-find."""
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    active_vertices = set()
    for i, j in selected_edges:
        union(i, j)
        active_vertices.add(i)
        active_vertices.add(j)

    components = {}
    for v in active_vertices:
        root = find(v)
        if root not in components:
            components[root] = set()
        components[root].add(v)

    return list(components.values())


def solve_op(instance, time_limit=None):
    n = instance["num_nodes"]
    depot = instance["depot_vertex"]
    d0 = instance["distance_limitation_d0"]
    scores = instance["scores"]
    dist = instance["distance_matrix"]

    edges = get_edges(n)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    num_edges = len(edges)

    # Precompute delta sets for each vertex
    delta_v = [delta(v, edges) for v in range(n)]

    # Build model
    model = gp.Model("OrienteeringProblem")
    model.setParam("Threads", 1)

    # Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(num_edges, vtype=GRB.BINARY, name="x")

    # Objective (2a): maximize total score
    model.setObjective(
        gp.quicksum(scores[v] * y[v] for v in range(n)), GRB.MAXIMIZE
    )

    # Constraint (2b): distance limit
    model.addConstr(
        gp.quicksum(dist[edges[e][0]][edges[e][1]] * x[e] for e in range(num_edges))
        <= d0,
        name="distance_limit",
    )

    # Constraints (2c): degree constraints
    for v in range(n):
        model.addConstr(
            gp.quicksum(x[e] for e in delta_v[v]) - 2 * y[v] == 0,
            name=f"degree_{v}",
        )

    # Constraints (2e): logical constraints y_v - x_e >= 0 for e in delta(v)
    for v in range(n):
        for e in delta_v[v]:
            model.addConstr(
                y[v] - x[e] >= 0,
                name=f"logical_{v}_{e}",
            )

    # Constraint (2h): depot must be visited
    model.addConstr(y[depot] == 1, name="depot_visited")

    # Gurobi parameters
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LazyConstraints = 1

    def subtour_callback(model, where):
        """Lazy constraint callback: detect subtours and add SECs (2d)."""
        if where != GRB.Callback.MIPSOL:
            return

        x_vals = model.cbGetSolution(x)
        y_vals = model.cbGetSolution(y)

        # Collect selected edges (those with x_e ~ 1)
        selected_edges = [edges[e] for e in range(num_edges) if x_vals[e] > 0.5]

        if not selected_edges:
            return

        components = find_connected_components(n, selected_edges)

        # If there is only one component containing the depot, no subtour
        if len(components) <= 1:
            return

        # Find which component contains the depot
        depot_comp = None
        for comp in components:
            if depot in comp:
                depot_comp = comp
                break

        # For each component that does NOT contain the depot, add SEC (2d):
        #   x(delta(H)) - 2*y_l - 2*y_r >= -2   for l in H, r in V\H
 
        for comp in components:
            if depot in comp:
                continue

            # Compute delta(H): edges with exactly one endpoint in H
            comp_set = comp
            delta_H_indices = []
            for e_idx, (i, j) in enumerate(edges):
                if (i in comp_set) != (j in comp_set):
                    delta_H_indices.append(e_idx)

            # V \ H
            outside_H = [v for v in range(n) if v not in comp_set]

            # Add violated SEC (2d): x(delta(H)) - 2*y_l - 2*y_r >= -2
            delta_H_expr = gp.quicksum(x[e] for e in delta_H_indices)
            for l in comp:
                if y_vals[l] < 0.5:
                    continue
                for r in outside_H:
                    if y_vals[r] < 0.5:
                        continue
                    model.cbLazy(
                        delta_H_expr - 2 * y[l] - 2 * y[r] >= -2
                    )

    # Optimize
    model.optimize(subtour_callback)

    # Extract solution
    result = {
        "objective_value": None,
        "status": model.Status,
        "status_name": None,
        "runtime": model.Runtime,
        "mip_gap": None,
        "visited_vertices": [],
        "tour_edges": [],
        "tour_distance": None,
    }

    status_map = {
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
    result["status_name"] = status_map.get(model.Status, f"UNKNOWN_{model.Status}")

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["mip_gap"] = model.MIPGap

        visited = [v for v in range(n) if y[v].X > 0.5]
        result["visited_vertices"] = visited

        tour_edges = [
            list(edges[e]) for e in range(num_edges) if x[e].X > 0.5
        ]
        result["tour_edges"] = tour_edges

        tour_dist = sum(
            dist[edges[e][0]][edges[e][1]]
            for e in range(num_edges)
            if x[e].X > 0.5
        )
        result["tour_distance"] = tour_dist

    return result


def main():
    args = parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)

    result = solve_op(instance, time_limit=args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Visited vertices ({len(result['visited_vertices'])}): "
              f"{result['visited_vertices']}")
        print(f"Tour distance: {result['tour_distance']}")
        print(f"Status: {result['status_name']}")
        print(f"Runtime: {result['runtime']:.2f}s")
    else:
        print(f"No solution found. Status: {result['status_name']}")

    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
