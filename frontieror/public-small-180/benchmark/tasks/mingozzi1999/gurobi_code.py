"""
Gurobi implementation of the VRPB set-partitioning IP formulation.

Reference: Mingozzi, Giorgi, Baldacci (1999)
"An Exact Method for the Vehicle Routing Problem with Backhauls"
Transportation Science 33(3):315-329

The model (IP) is defined by Equations 3-9 in the paper:
  min sum_{l in L} c_l x_l + sum_{l in B} c_l y_l + sum_{(i,j) in A0} d_ij xi_ij
  s.t.
    (4) sum_{l in L_i} x_l = 1,  for all i in L  (visit each linehaul customer once)
    (5) sum_{l in B_j} y_l = 1,  for all j in B  (visit each backhaul customer once)
    (6) sum_{l in L_i^E} x_l - sum_{j in B0} xi_ij = 0,  for all i in L  (link linehaul path end to A0 arc)
    (7) sum_{l in B_j^S} y_l - sum_{i in L} xi_ij = 0,  for all j in B  (link backhaul path start to A0 arc)
    (8) sum_{(i,j) in A0} xi_ij = M   (exactly M routes)
    (9) x, y, xi binary
"""

import json
import argparse
import itertools
import sys
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


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def enumerate_feasible_paths_linehaul(linehaul_indices, depot, dist, demands, Q, Q_min_L):
    """
    Enumerate all feasible elementary paths in G_L starting at the depot.
    A feasible path P: depot -> i1 -> i2 -> ... -> ik  where
      Q_min_L <= sum of demands <= Q
    Returns list of (path_as_tuple_of_customer_indices, cost, terminal_vertex).
    """
    paths = []
    n_linehaul = len(linehaul_indices)

    # DFS enumeration
    def dfs(current, visited, load, cost, path):
        # Check if current path (excluding depot start) is feasible
        if len(path) > 0 and Q_min_L <= load <= Q:
            paths.append((tuple(path), cost, path[-1]))
        # Try extending
        for nxt in linehaul_indices:
            if nxt not in visited:
                new_load = load + demands[nxt]
                if new_load <= Q:
                    new_cost = cost + dist[current][nxt]
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, visited, new_load, new_cost, path)
                    path.pop()
                    visited.remove(nxt)

    dfs(depot, set(), 0, 0, [])
    return paths


def enumerate_feasible_paths_backhaul(backhaul_indices, depot, dist, demands, Q, Q_min_B):
    """
    Enumerate all feasible elementary paths in G_B ending at the depot.
    A feasible path P: j1 -> j2 -> ... -> jk -> depot  where
      Q_min_B <= sum of demands <= Q
    Returns list of (path_as_tuple_of_customer_indices, cost, starting_vertex).
    The cost includes all arcs including the final arc to depot.

    NOTE: In the paper's formulation, backhaul path cost does NOT include
    the arc to/from depot (that's handled by the A0 arcs and the depot return).
    Actually, re-reading the paper: the backhaul path in G_B ends at vertex 0.
    So the cost of a backhaul path includes the arc from the last backhaul
    customer to the depot. But the starting arc from a linehaul customer to
    the first backhaul customer is in A0 (not in the backhaul path).

    So backhaul path: j1 -> j2 -> ... -> jk -> 0
    Cost = d(j1,j2) + ... + d(j_{k-1},jk) + d(jk, 0)
    Starting vertex = j1
    """
    paths = []

    def dfs(current, visited, load, cost, path):
        # Current path is path so far (list of backhaul customers)
        if len(path) > 0 and Q_min_B <= load <= Q:
            # Complete path by adding arc to depot
            total_cost = cost + dist[current][depot]
            paths.append((tuple(path), total_cost, path[0]))
        # Try extending
        for nxt in backhaul_indices:
            if nxt not in visited:
                new_load = load + demands[nxt]
                if new_load <= Q:
                    new_cost = cost + dist[current][nxt]
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, visited, new_load, new_cost, path)
                    path.pop()
                    visited.remove(nxt)

    # Start DFS from each backhaul customer as the first node
    for start in backhaul_indices:
        visited = {start}
        dfs(start, visited, demands[start], 0, [start])

    return paths


def solve_vrpb(instance, time_limit):
    depot_idx = instance['depot']['index']
    n = instance['num_linehaul_customers']
    m = instance['num_backhaul_customers']
    M = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    dist = instance['distance_matrix']

    linehaul_indices = [c['index'] for c in instance['linehaul_customers']]
    backhaul_indices = [c['index'] for c in instance['backhaul_customers']]

    demands = {}
    demands[depot_idx] = 0
    for c in instance['linehaul_customers']:
        demands[c['index']] = c['demand']
    for c in instance['backhaul_customers']:
        demands[c['index']] = c['demand']

    # Compute Q_min
    sum_L = sum(demands[i] for i in linehaul_indices)
    sum_B = sum(demands[j] for j in backhaul_indices)
    Q_min_L = max(0, sum_L - (M - 1) * Q)
    Q_min_B = max(0, sum_B - (M - 1) * Q)

    print(f"Problem: n={n}, m={m}, M={M}, Q={Q}")
    print(f"Q_min_L={Q_min_L}, Q_min_B={Q_min_B}")
    print(f"Sum linehaul demands={sum_L}, Sum backhaul demands={sum_B}")

    # Enumerate feasible linehaul paths (starting at depot)
    print("Enumerating linehaul paths...")
    linehaul_paths = enumerate_feasible_paths_linehaul(
        linehaul_indices, depot_idx, dist, demands, Q, Q_min_L
    )
    print(f"  Found {len(linehaul_paths)} feasible linehaul paths")

    # Enumerate feasible backhaul paths (ending at depot)
    print("Enumerating backhaul paths...")
    backhaul_paths = enumerate_feasible_paths_backhaul(
        backhaul_indices, depot_idx, dist, demands, Q, Q_min_B
    )
    print(f"  Found {len(backhaul_paths)} feasible backhaul paths")

    # Build index sets
    # L_i: linehaul paths passing through customer i
    # L_i^E: linehaul paths ending at customer i
    L_i = {i: [] for i in linehaul_indices}
    L_i_E = {i: [] for i in linehaul_indices}
    for idx, (path, cost, terminal) in enumerate(linehaul_paths):
        for i in path:
            L_i[i].append(idx)
        L_i_E[terminal].append(idx)

    # B_j: backhaul paths passing through customer j
    # B_j^S: backhaul paths starting at customer j
    B_j = {j: [] for j in backhaul_indices}
    B_j_S = {j: [] for j in backhaul_indices}
    for idx, (path, cost, start) in enumerate(backhaul_paths):
        for j in path:
            B_j[j].append(idx)
        B_j_S[start].append(idx)

    # A0 = {(i,j) : i in L, j in B_0}  where B_0 = B union {0}
    B_0 = backhaul_indices + [depot_idx]
    A0_arcs = [(i, j) for i in linehaul_indices for j in B_0]

    # Build Gurobi model
    model = Model("VRPB")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = {}  # x[l] for linehaul path l
    for idx, (path, cost, terminal) in enumerate(linehaul_paths):
        x[idx] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"x_{idx}")

    y = {}  # y[l] for backhaul path l
    for idx, (path, cost, start) in enumerate(backhaul_paths):
        y[idx] = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"y_{idx}")

    xi = {}  # xi[i,j] for arc (i,j) in A0
    for (i, j) in A0_arcs:
        xi[i, j] = model.addVar(vtype=GRB.BINARY, obj=dist[i][j], name=f"xi_{i}_{j}")

    model.update()
    model.ModelSense = GRB.MINIMIZE

    # Constraint (4): each linehaul customer visited exactly once
    for i in linehaul_indices:
        model.addConstr(
            quicksum(x[idx] for idx in L_i[i]) == 1,
            name=f"visit_linehaul_{i}"
        )

    # Constraint (5): each backhaul customer visited exactly once
    for j in backhaul_indices:
        model.addConstr(
            quicksum(y[idx] for idx in B_j[j]) == 1,
            name=f"visit_backhaul_{j}"
        )

    # Constraint (6): link linehaul path ending to A0 arc
    for i in linehaul_indices:
        model.addConstr(
            quicksum(x[idx] for idx in L_i_E[i])
            - quicksum(xi[i, j] for j in B_0) == 0,
            name=f"link_linehaul_{i}"
        )

    # Constraint (7): link backhaul path starting to A0 arc
    for j in backhaul_indices:
        model.addConstr(
            quicksum(y[idx] for idx in B_j_S[j])
            - quicksum(xi[i, j] for i in linehaul_indices) == 0,
            name=f"link_backhaul_{j}"
        )

    # Constraint (8): exactly M arcs of A0
    model.addConstr(
        quicksum(xi[i, j] for (i, j) in A0_arcs) == M,
        name="num_routes"
    )

    # Optimize
    model.optimize()

    # Extract solution
    result = {"objective_value": None, "routes": [], "status": ""}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Reconstruct routes
        selected_linehaul = []
        for idx in x:
            if x[idx].X > 0.5:
                path, cost, terminal = linehaul_paths[idx]
                selected_linehaul.append((idx, path, cost, terminal))

        selected_backhaul = []
        for idx in y:
            if y[idx].X > 0.5:
                path, cost, start = backhaul_paths[idx]
                selected_backhaul.append((idx, path, cost, start))

        selected_arcs = []
        for (i, j) in A0_arcs:
            if xi[i, j].X > 0.5:
                selected_arcs.append((i, j))

        # Build routes by linking linehaul paths -> A0 arcs -> backhaul paths
        routes = []
        for arc_i, arc_j in selected_arcs:
            route = {"linehaul_path": None, "connecting_arc": (arc_i, arc_j),
                     "backhaul_path": None}
            # Find linehaul path ending at arc_i
            for idx, path, cost, terminal in selected_linehaul:
                if terminal == arc_i:
                    route["linehaul_path"] = list(path)
                    route["linehaul_cost"] = cost
                    break
            # If arc_j != depot, find backhaul path starting at arc_j
            if arc_j != depot_idx:
                for idx, path, cost, start in selected_backhaul:
                    if start == arc_j:
                        route["backhaul_path"] = list(path)
                        route["backhaul_cost"] = cost
                        break

            # Build full route sequence
            full_route = [depot_idx]
            if route["linehaul_path"]:
                full_route.extend(route["linehaul_path"])
            if route["backhaul_path"]:
                full_route.extend(route["backhaul_path"])
            full_route.append(depot_idx)
            route["full_sequence"] = full_route

            routes.append(route)

        result["routes"] = [r["full_sequence"] for r in routes]
        result["num_routes"] = len(routes)

        print(f"\nObjective value: {result['objective_value']}")
        print(f"Number of routes: {result['num_routes']}")
        for i, r in enumerate(routes):
            print(f"  Route {i+1}: {r['full_sequence']}")
    else:
        result["status"] = "infeasible"
        result["objective_value"] = None
        print("No feasible solution found.")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="VRPB solver using Gurobi (set-partitioning IP formulation)"
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
    result = solve_vrpb(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)
    print(f"\nSolution saved to {args.solution_path}")


if __name__ == "__main__":
    main()
