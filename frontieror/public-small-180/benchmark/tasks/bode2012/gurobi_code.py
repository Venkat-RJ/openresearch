r"""
Gurobi implementation of the Two-Index CARP Formulation (Equations 1-6)
from Bode & Irnich (2012), "Cut-First Branch-and-Price-Second for the CARP",
Operations Research 60(5):1167-1182.

The CARP is modeled as:
  min  sum_k c^serv' x^k + sum_k c' y^k                     (1)
  s.t. sum_k x^k_e = 1                   for all e in E_R   (2)
       x^k(delta_R(S)) + y^k(delta(S)) >= 2 x^k_f
                  for all S <= V\{d}, f in E_R(S), k in K    (3)
       x^k(delta_R(i)) + y^k(delta(i)) = 2 p^k_i
                  for all i in V, k in K                     (4)
       q' x^k <= Q                       for all k in K      (5)
       p^k in Z_+^|V|, x^k in {0,1}^|E_R|, y^k in Z_+^|E|  (6)

Since constraint (3) has exponentially many subtour elimination constraints (SEC),
we use a callback-based lazy constraint approach:
  - We solve the model without (3), then add violated SECs as lazy constraints.

**INFERRED ASSUMPTION**: The paper's SEC (3) is separated via connected-component
analysis on the support graph. For each vehicle k, we check if the edges used by k
form a connected subgraph that includes the depot. If not, for each connected
component S not containing the depot, we add the violated SEC for all required
edges f in E_R(S).
"""

import json
import argparse
import time
_GUROBI_CODE_START_TIME = time.time()
import math
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
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
        data = json.load(f)
    return data


def build_adjacency(edges, num_nodes):
    """Build adjacency list from edge list."""
    adj = {i: [] for i in range(num_nodes)}
    for e in edges:
        u, v = e["endpoints"]
        adj[u].append((v, e["edge_id"]))
        adj[v].append((u, e["edge_id"]))
    return adj


def get_delta(node_set, edges):
    """Get edges with exactly one endpoint in node_set."""
    s = set(node_set)
    result = []
    for e in edges:
        u, v = e["endpoints"]
        if (u in s) != (v in s):
            result.append(e["edge_id"])
    return result


def get_delta_R(node_set, edges):
    """Get required edges with exactly one endpoint in node_set."""
    s = set(node_set)
    result = []
    for e in edges:
        if not e["is_required"]:
            continue
        u, v = e["endpoints"]
        if (u in s) != (v in s):
            result.append(e["edge_id"])
    return result


def get_E_R_S(node_set, edges):
    """Get required edges with both endpoints in node_set."""
    s = set(node_set)
    result = []
    for e in edges:
        if not e["is_required"]:
            continue
        u, v = e["endpoints"]
        if u in s and v in s:
            result.append(e["edge_id"])
    return result


def find_connected_components(num_nodes, active_edges, edges_data):
    """Find connected components given a set of active edge ids."""
    adj = {i: set() for i in range(num_nodes)}
    active_nodes = set()
    for eid in active_edges:
        u, v = edges_data[eid]["endpoints"]
        adj[u].add(v)
        adj[v].add(u)
        active_nodes.add(u)
        active_nodes.add(v)

    visited = set()
    components = []
    for start in active_nodes:
        if start in visited:
            continue
        comp = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    stack.append(nb)
        components.append(comp)
    return components


def solve_carp(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    num_nodes = data["graph"]["num_nodes"]
    num_edges = data["graph"]["num_edges"]
    depot = data["depot"]
    num_vehicles = data["fleet"]["num_vehicles"]
    capacity = data["fleet"]["vehicle_capacity"]
    edges = data["edges"]

    # Index sets
    all_edge_ids = list(range(num_edges))
    required_edge_ids = [e["edge_id"] for e in edges if e["is_required"]]
    K = list(range(num_vehicles))

    # Edge properties
    cost = {e["edge_id"]: e["cost"] for e in edges}
    serv_cost = {e["edge_id"]: e["service_cost"] for e in edges}
    demand = {e["edge_id"]: e["demand"] for e in edges}
    is_required = {e["edge_id"]: e["is_required"] for e in edges}
    endpoints = {e["edge_id"]: tuple(e["endpoints"]) for e in edges}

    # delta(i): edges incident to node i
    delta = {i: [] for i in range(num_nodes)}
    delta_R = {i: [] for i in range(num_nodes)}
    for e in edges:
        u, v = e["endpoints"]
        delta[u].append(e["edge_id"])
        delta[v].append(e["edge_id"])
        if e["is_required"]:
            delta_R[u].append(e["edge_id"])
            delta_R[v].append(e["edge_id"])

    # Create model
    model = gp.Model("CARP_TwoIndex")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LazyConstraints", 1)
    # Reduce output verbosity slightly
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[k,e] in {0,1}: vehicle k services required edge e
    x = {}
    for k in K:
        for e_id in required_edge_ids:
            x[k, e_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{e_id}")

    # y[k,e] in Z_+: number of times vehicle k deadheads edge e
    y = {}
    for k in K:
        for e_id in all_edge_ids:
            y[k, e_id] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{k}_{e_id}")

    # p[k,i] in Z_+: parity auxiliary variable
    p = {}
    for k in K:
        for i in range(num_nodes):
            p[k, i] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"p_{k}_{i}")

    model.update()

    # Objective (1): min sum_k c^serv' x^k + sum_k c' y^k
    obj = gp.LinExpr()
    for k in K:
        for e_id in required_edge_ids:
            obj += serv_cost[e_id] * x[k, e_id]
        for e_id in all_edge_ids:
            obj += cost[e_id] * y[k, e_id]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (2): sum_k x^k_e = 1 for all e in E_R
    for e_id in required_edge_ids:
        model.addConstr(
            gp.quicksum(x[k, e_id] for k in K) == 1,
            name=f"partition_{e_id}"
        )

    # Constraint (4): x^k(delta_R(i)) + y^k(delta(i)) = 2 p^k_i for all i, k
    for k in K:
        for i in range(num_nodes):
            lhs = gp.LinExpr()
            for e_id in delta_R[i]:
                lhs += x[k, e_id]
            for e_id in delta[i]:
                lhs += y[k, e_id]
            model.addConstr(lhs == 2 * p[k, i], name=f"parity_{k}_{i}")

    # Constraint (5): q' x^k <= Q for all k
    for k in K:
        model.addConstr(
            gp.quicksum(demand[e_id] * x[k, e_id] for e_id in required_edge_ids) <= capacity,
            name=f"capacity_{k}"
        )

    # Constraint (3): Subtour Elimination Constraints (SEC) via lazy constraints
    # We add these dynamically via a callback.

    def sec_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get current solution
            x_val = {}
            y_val = {}
            for k in K:
                for e_id in required_edge_ids:
                    x_val[k, e_id] = model.cbGetSolution(x[k, e_id])
                for e_id in all_edge_ids:
                    y_val[k, e_id] = model.cbGetSolution(y[k, e_id])

            for k in K:
                # Find edges used by vehicle k (serviced or deadheaded)
                active_edges = set()
                for e_id in required_edge_ids:
                    if x_val[k, e_id] > 0.5:
                        active_edges.add(e_id)
                for e_id in all_edge_ids:
                    if y_val[k, e_id] > 0.5:
                        active_edges.add(e_id)

                if not active_edges:
                    continue

                # Find connected components
                components = find_connected_components(num_nodes, active_edges, edges)

                # For each component not containing the depot, add SEC
                for comp in components:
                    if depot in comp:
                        continue

                    # S = comp (subset of V \ {d})
                    S = comp

                    # Get required edges with both endpoints in S
                    er_s = []
                    for e_id in required_edge_ids:
                        u, v = endpoints[e_id]
                        if u in S and v in S:
                            er_s.append(e_id)

                    if not er_s:
                        continue

                    # Get delta_R(S) and delta(S)
                    delta_r_s = []
                    delta_s = []
                    for e_id in all_edge_ids:
                        u, v = endpoints[e_id]
                        if (u in S) != (v in S):
                            delta_s.append(e_id)
                            if is_required[e_id]:
                                delta_r_s.append(e_id)

                    # Add SEC: x^k(delta_R(S)) + y^k(delta(S)) >= 2 x^k_f
                    # for all f in E_R(S)
                    for f in er_s:
                        if x_val[k, f] > 0.5:
                            lhs = gp.LinExpr()
                            for e_id in delta_r_s:
                                lhs += x[k, e_id]
                            for e_id in delta_s:
                                lhs += y[k, e_id]
                            model.cbLazy(lhs >= 2 * x[k, f])

    model.optimize(sec_callback)

    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution = {
            "objective_value": obj_val,
            "status": model.Status,
            "status_str": {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
            }.get(model.Status, f"STATUS_{model.Status}"),
            "num_vehicles": num_vehicles,
            "vehicle_capacity": capacity,
            "routes": []
        }

        for k in K:
            route_info = {
                "vehicle": k,
                "serviced_edges": [],
                "deadheaded_edges": [],
                "total_demand": 0,
                "route_cost": 0.0
            }
            for e_id in required_edge_ids:
                if x[k, e_id].X > 0.5:
                    route_info["serviced_edges"].append(e_id)
                    route_info["total_demand"] += demand[e_id]
                    route_info["route_cost"] += serv_cost[e_id]
            for e_id in all_edge_ids:
                yv = round(y[k, e_id].X)
                if yv > 0:
                    route_info["deadheaded_edges"].append({
                        "edge_id": e_id,
                        "times": yv
                    })
                    route_info["route_cost"] += cost[e_id] * yv
            solution["routes"].append(route_info)

        with open(solution_path, "w") as f:
            solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(solution, f, indent=2)
        print(f"Solution written to {solution_path}")
        print(f"Objective value: {obj_val}")
    else:
        solution = {
            "objective_value": None,
            "status": model.Status,
            "status_str": "NO_SOLUTION_FOUND"
        }
        with open(solution_path, "w") as f:
            solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(solution, f, indent=2)
        print("No feasible solution found.")


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for the CARP (Two-Index Formulation)"
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

    solve_carp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
