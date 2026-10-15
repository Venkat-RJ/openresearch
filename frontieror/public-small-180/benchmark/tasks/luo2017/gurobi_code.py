"""
Gurobi MIP implementation for the Split-Delivery Vehicle Routing Problem
with Time Windows and Linear Weight-Related Cost (SDVRPTWL).

Arc-flow formulation from Luo, Qin, Zhu, and Lim (2017),
"Branch and Price and Cut for the Split-Delivery Vehicle Routing Problem
with Time Windows and Linear Weight-Related Cost,"
Transportation Science, 51(2):668-687.

Formulation: Equations (1)-(9) from the paper (Section 3).
"""

import argparse
import json
import math
import sys
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


def load_instance(instance_path):
    """Load the JSON instance file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """
    Build and solve the SDVRPTWL arc-flow MIP model using Gurobi.

    The model follows the formulation in Section 3 of Luo et al. (2017):
      min  sum_{k} sum_{i} sum_{j in V+(i)} c_{i,j} * (a * w_{i,j,k} + b * x_{i,j,k})   (1)
      s.t. Constraints (2)-(9)
    """
    # ---- Extract data ----
    n = data["num_customers"]
    Q = data["vehicle_capacity"]
    a_cost = data["cost_function"]["a"]
    b_cost = data["cost_function"]["b"]

    depot = data["depot"]
    customers = data["customers"]
    dist = data["distance_matrix"]  # (n+1) x (n+1), indices 0..n (0 = depot, 1..n = customers)

    # The distance matrix in the JSON is (n+1) x (n+1) for vertices 0..n.
    # In the model, vertex 0 = depot exit, vertex n+1 = depot entrance.
    # We create an extended distance matrix for n+2 vertices (0..n+1).
    num_vertices = n + 2  # 0, 1, ..., n, n+1

    # Build extended distance/time matrix
    # dist[0] is depot, dist[1..n] are customers
    # Vertex n+1 (depot entrance) has same location as vertex 0
    c = {}
    t_travel = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                c[i, j] = dist[i][j]
                t_travel[i, j] = dist[i][j]  # travel time = distance per instance spec

    # Add edges to/from vertex n+1 (same distances as vertex 0)
    for j in range(1, n + 1):
        c[j, n + 1] = dist[j][0]
        t_travel[j, n + 1] = dist[j][0]
    c[0, n + 1] = 0.0  # depot to depot
    t_travel[0, n + 1] = 0.0
    # Edges from n+1 are not needed (n+1 is only an entrance)

    # ---- Vertex parameters ----
    demand = [0] * num_vertices
    service = [0] * num_vertices
    early = [0] * num_vertices
    late = [0] * num_vertices

    # Depot exit (vertex 0)
    demand[0] = 0  # d_0 = +inf conceptually, but not used in constraints
    service[0] = depot["service_time"]
    early[0] = depot["time_window"][0]
    late[0] = depot["time_window"][1]

    # Customers (vertices 1..n)
    for cust in customers:
        idx = cust["id"]
        demand[idx] = cust["demand"]
        service[idx] = cust["service_time"]
        early[idx] = cust["time_window"][0]
        late[idx] = cust["time_window"][1]

    # Depot entrance (vertex n+1)
    demand[n + 1] = 0
    service[n + 1] = 0
    early[n + 1] = depot["time_window"][0]
    late[n + 1] = depot["time_window"][1]

    # ---- Compute V+(i) and V-(i) ----
    # V+(i) = {j in V : e_i + s_i + t_{i,j} <= l_j, (i,j) in E}
    # E = {(i,j) | i,j in V, i != j, i != n+1, j != 0}
    V_plus = {i: [] for i in range(num_vertices)}
    V_minus = {i: [] for i in range(num_vertices)}

    for i in range(num_vertices):
        if i == n + 1:
            continue  # no outgoing edges from n+1
        for j in range(num_vertices):
            if j == 0:
                continue  # no incoming edges to 0
            if i == j:
                continue
            if (i, j) not in t_travel:
                continue
            if early[i] + service[i] + t_travel[i, j] <= late[j]:
                V_plus[i].append(j)
                V_minus[j].append(i)

    # ---- Number of vehicles ----
    # Paper says unlimited. We set m = sum of ceil(d_i / Q) as upper bound.
    # INFERRED ASSUMPTION: m is set as the sum of ceil(d_i/Q) for all customers,
    # which is an upper bound on the number of vehicles needed.
    m = sum(math.ceil(demand[i] / Q) for i in range(1, n + 1))
    m = max(m, n)  # ensure at least n vehicles available
    K = list(range(m))

    # ---- Big-M ----
    # NOT SPECIFIED IN PAPER: Big-M value.
    # Standard choice: M >= max_{(i,j)} (l_i + s_i + t_{i,j} - e_j)
    big_M = 0
    for i in range(num_vertices):
        if i == n + 1:
            continue
        for j in V_plus[i]:
            val = late[i] + service[i] + t_travel[i, j] - early[j]
            if val > big_M:
                big_M = val
    big_M = big_M + 1  # add small buffer

    # ---- Build Gurobi model ----
    model = gp.Model("SDVRPTWL")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i,j,k] binary: vehicle k traverses edge (i,j)
    x = {}
    # w[i,j,k] >= 0: load weight of vehicle k on edge (i,j)
    w = {}
    # arr[i,k] >= 0: service starting time of vehicle k at vertex i
    arr = {}

    # Only create variables for valid edges (i in V_C union {0}, j in V+(i))
    for k in K:
        for i in range(num_vertices):
            if i == n + 1:
                continue  # no outgoing edges from n+1
            for j in V_plus[i]:
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
                w[i, j, k] = model.addVar(lb=0, ub=Q, name=f"w_{i}_{j}_{k}")

    for k in K:
        for i in range(num_vertices):
            arr[i, k] = model.addVar(lb=early[i], ub=late[i], name=f"a_{i}_{k}")

    model.update()

    # ---- Objective (1) ----
    # min sum_{k} sum_{i in V_C union {0}} sum_{j in V+(i)} c_{i,j} * (a * w_{i,j,k} + b * x_{i,j,k})
    obj = gp.LinExpr()
    for k in K:
        for i in range(num_vertices):
            if i == n + 1:
                continue
            for j in V_plus[i]:
                obj += c[i, j] * (a_cost * w[i, j, k] + b_cost * x[i, j, k])
    model.setObjective(obj, GRB.MINIMIZE)

    # ---- Constraint (2): Demand fulfillment ----
    for i in range(1, n + 1):
        lhs = gp.LinExpr()
        for k in K:
            for j in V_minus[i]:
                if (j, i, k) in w:
                    lhs += w[j, i, k]
            for j in V_plus[i]:
                if (i, j, k) in w:
                    lhs -= w[i, j, k]
        model.addConstr(lhs >= demand[i], name=f"demand_{i}")

    # ---- Constraint (3): Minimum number of vehicles ----
    for i in range(1, n + 1):
        lhs = gp.LinExpr()
        for k in K:
            for j in V_minus[i]:
                if (j, i, k) in x:
                    lhs += x[j, i, k]
        model.addConstr(lhs >= math.ceil(demand[i] / Q), name=f"min_vehicles_{i}")

    # ---- Constraint (4): Each vehicle departs from depot ----
    for k in K:
        lhs = gp.LinExpr()
        for j in V_plus[0]:
            if (0, j, k) in x:
                lhs += x[0, j, k]
        model.addConstr(lhs == 1, name=f"depart_{k}")

    # ---- Constraint (5): Flow conservation ----
    for k in K:
        for i in range(1, n + 1):
            out_flow = gp.LinExpr()
            in_flow = gp.LinExpr()
            for j in V_plus[i]:
                if (i, j, k) in x:
                    out_flow += x[i, j, k]
            for j in V_minus[i]:
                if (j, i, k) in x:
                    in_flow += x[j, i, k]
            model.addConstr(out_flow == in_flow, name=f"flow_eq_{i}_{k}")
            model.addConstr(out_flow <= 1, name=f"flow_le_{i}_{k}")

    # ---- Constraint (6): Each vehicle returns to depot ----
    for k in K:
        lhs = gp.LinExpr()
        for j in V_minus[n + 1]:
            if (j, n + 1, k) in x:
                lhs += x[j, n + 1, k]
        model.addConstr(lhs == 1, name=f"return_{k}")

    # ---- Constraint (7): Capacity on each edge ----
    for k in K:
        for i in range(num_vertices):
            if i == n + 1:
                continue
            for j in V_plus[i]:
                if (i, j, k) in w:
                    model.addConstr(w[i, j, k] <= Q * x[i, j, k],
                                    name=f"cap_{i}_{j}_{k}")

    # ---- Constraint (8): Time window precedence ----
    for k in K:
        for i in range(num_vertices):
            if i == n + 1:
                continue
            for j in V_plus[i]:
                if (i, j, k) in x:
                    model.addConstr(
                        arr[j, k] >= arr[i, k] + service[i] + t_travel[i, j]
                        + big_M * (x[i, j, k] - 1),
                        name=f"tw_{i}_{j}_{k}"
                    )

    # ---- Constraint (9): Time window bounds ----
    # Already enforced by variable bounds (lb=e_i, ub=l_i)

    # ---- Symmetry breaking (optional, helps solver) ----
    # INFERRED ASSUMPTION: We add symmetry-breaking constraints to help
    # the MIP solver since all vehicles are identical. Vehicles are ordered
    # by their departure time from the depot.
    for k in range(len(K) - 1):
        model.addConstr(arr[0, K[k]] <= arr[0, K[k + 1]],
                        name=f"symbreak_{k}")

    # ---- Solve ----
    model.optimize()

    # ---- Extract solution ----
    result = {
        "problem": "SDVRPTWL",
        "solver": "Gurobi",
        "status": model.Status,
        "status_str": "",
    }

    if model.Status == GRB.OPTIMAL:
        result["status_str"] = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT:
        result["status_str"] = "TIME_LIMIT"
    elif model.Status == GRB.INFEASIBLE:
        result["status_str"] = "INFEASIBLE"
        result["objective_value"] = None
        return result
    else:
        result["status_str"] = f"OTHER_{model.Status}"

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["gap"] = model.MIPGap

        # Extract routes
        routes = []
        for k in K:
            route_edges = []
            for i in range(num_vertices):
                if i == n + 1:
                    continue
                for j in V_plus[i]:
                    if (i, j, k) in x and x[i, j, k].X > 0.5:
                        route_edges.append({
                            "from": i,
                            "to": j,
                            "load_weight": round(w[i, j, k].X, 6) if (i, j, k) in w else 0,
                        })

            # Build ordered route from edges
            if route_edges:
                # Check if vehicle actually visits customers
                visits_customer = any(
                    e["from"] in range(1, n + 1) or e["to"] in range(1, n + 1)
                    for e in route_edges
                )
                if visits_customer:
                    # Reconstruct route sequence
                    edge_map = {e["from"]: e for e in route_edges}
                    route_seq = []
                    deliveries = {}
                    current = 0
                    visited = set()
                    while current is not None and current not in visited:
                        visited.add(current)
                        route_seq.append(current)
                        if current in edge_map:
                            nxt = edge_map[current]["to"]
                            current = nxt
                        else:
                            break
                    if current == n + 1:
                        route_seq.append(n + 1)

                    # Compute deliveries: delta_i = sum w_{j,i,k} - sum w_{i,j,k}
                    for i in range(1, n + 1):
                        inflow = sum(
                            w[j, i, k].X for j in V_minus[i]
                            if (j, i, k) in w
                        )
                        outflow = sum(
                            w[i, j, k].X for j in V_plus[i]
                            if (i, j, k) in w
                        )
                        delta = inflow - outflow
                        if delta > 0.001:
                            deliveries[i] = round(delta, 6)

                    if deliveries:
                        routes.append({
                            "vehicle": k,
                            "route": route_seq,
                            "deliveries": deliveries,
                            "edges": route_edges
                        })

        result["routes"] = routes
        result["num_vehicles_used"] = len(routes)
    else:
        result["objective_value"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve SDVRPTWL using Gurobi MIP (arc-flow formulation)"
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

    print(f"Loading instance from: {args.instance_path}")
    data = load_instance(args.instance_path)

    print(f"Problem: {data['problem']}")
    print(f"Customers: {data['num_customers']}")
    print(f"Capacity Q: {data['vehicle_capacity']}")
    print(f"Cost function: f(w) = {data['cost_function']['a']}*w + {data['cost_function']['b']}")
    print(f"Time limit: {args.time_limit} seconds")

    start_time = time.time()
    result = build_and_solve(data, args.time_limit)
    elapsed = time.time() - start_time

    result["computation_time_seconds"] = round(elapsed, 2)

    print(f"\nStatus: {result.get('status_str', 'UNKNOWN')}")
    if result.get("objective_value") is not None:
        print(f"Objective value: {result['objective_value']:.4f}")
        print(f"Vehicles used: {result.get('num_vehicles_used', 'N/A')}")
    print(f"Computation time: {elapsed:.2f}s")

    # Write solution
    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2, default=str)

    print(f"Solution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
