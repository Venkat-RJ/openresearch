"""
Gurobi implementation of the VRPDDP (Vehicle Routing Problem with Divisible
Deliveries and Pickups) mathematical formulation.

Reference: Nagy, Wassan, Speranza, Archetti (2015)
"The Vehicle Routing Problem with Divisible Deliveries and Pickups"
Transportation Science 49(2), pp. 271-294.

The VRPDDP is modelled as a VRPMDP by creating a pair of fictitious co-located
customers for each original customer:
  - Linehaul customer i in L = {1, ..., n} with delivery demand
  - Backhaul customer n+i in B = {n+1, ..., 2n} with pickup demand

Decision variables:
  x_ij in {0,1}: 1 if a vehicle travels from i to j
  R_ij >= 0: delivery goods on arc (i,j)
  P_ij >= 0: pickup goods on arc (i,j)

Objective: Minimise total distance = sum_{i,j} d_ij * x_ij
"""

import json
import argparse
import math
import os
import sys
import re
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
    """Load a VRPDDP instance from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_expanded_distance_matrix(instance):
    """
    Build the expanded distance matrix for the VRPMDP formulation.

    Original nodes: 0 (depot), 1..n (customers)
    Expanded nodes: 0 (depot), 1..n (linehaul), n+1..2n (backhaul)

    Backhaul customer n+i is co-located with linehaul customer i, so:
      d[i][n+j] = d[i][j]   (linehaul to backhaul)
      d[n+i][j] = d[i][j]   (backhaul to linehaul)
      d[n+i][n+j] = d[i][j] (backhaul to backhaul)
      d[i][n+i] = 0          (co-located pair)
      d[n+i][i] = 0          (co-located pair)
    """
    n = instance["num_customers"]
    orig_dist = instance["distance_matrix"]  # (n+1) x (n+1) matrix for nodes 0..n
    size = 2 * n + 1  # nodes 0, 1..n, n+1..2n

    dist = [[0] * size for _ in range(size)]

    for i in range(n + 1):       # i in {0, 1, ..., n}
        for j in range(n + 1):   # j in {0, 1, ..., n}
            d_val = orig_dist[i][j]

            # Original-to-original (depot-depot, depot-linehaul, linehaul-linehaul)
            dist[i][j] = d_val

            # Original-to-backhaul: dist[i][n+j] = dist[i][j] for j >= 1
            if j >= 1:
                dist[i][n + j] = d_val

            # Backhaul-to-original: dist[n+i][j] = dist[i][j] for i >= 1
            if i >= 1:
                dist[n + i][j] = d_val

            # Backhaul-to-backhaul: dist[n+i][n+j] = dist[i][j] for i,j >= 1
            if i >= 1 and j >= 1:
                dist[n + i][n + j] = d_val

    # Co-located pairs: distance between linehaul i and backhaul n+i is 0
    for i in range(1, n + 1):
        dist[i][n + i] = 0
        dist[n + i][i] = 0

    return dist


def build_demands(instance):
    """
    Build the demand array for the expanded node set.

    q[i] for i in L: delivery demand of original customer i
    q[n+i] for n+i in B: pickup demand of original customer i
    q[0] = 0 (depot)
    """
    n = instance["num_customers"]
    size = 2 * n + 1
    q = [0] * size

    for cust in instance["customers"]:
        cid = cust["id"]  # 1-indexed
        q[cid] = cust["delivery_demand"]       # linehaul demand
        q[n + cid] = cust["pickup_demand"]      # backhaul demand

    return q


def solve_vrpddp(instance, time_limit):
    """
    Solve the VRPDDP using the two-index MILP formulation from the paper.
    """
    n = instance["num_customers"]
    C = instance["vehicle_capacity"]

    # Build expanded data
    dist = build_expanded_distance_matrix(instance)
    q = build_demands(instance)

    size = 2 * n + 1  # total number of nodes: 0, 1..n, n+1..2n

    # Define sets
    depot = 0
    L = list(range(1, n + 1))                  # linehaul customers
    B = list(range(n + 1, 2 * n + 1))          # backhaul customers
    LB = L + B                                  # all customers
    V = [depot] + LB                            # all vertices

    print(f"Problem size: n={n}, expanded nodes={size}, capacity={C}")
    print(f"  Linehaul customers (L): {len(L)}")
    print(f"  Backhaul customers (B): {len(B)}")
    print(f"  Total delivery demand: {sum(q[i] for i in L)}")
    print(f"  Total pickup demand: {sum(q[i] for i in B)}")

    # =========================================================================
    # Build Gurobi model
    # =========================================================================
    model = Model("VRPDDP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # ----- Decision variables -----
    # x[i,j] in {0,1}: 1 if vehicle travels from i to j
    x = {}
    for i in V:
        for j in V:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # R[i,j] >= 0: delivery goods on arc (i,j)
    R = {}
    for i in V:
        for j in V:
            R[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"R_{i}_{j}")

    # P[i,j] >= 0: pickup goods on arc (i,j)
    P = {}
    for i in V:
        for j in V:
            P[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"P_{i}_{j}")

    model.update()

    # ----- Objective (1): Minimise total distance -----
    model.setObjective(
        quicksum(dist[i][j] * x[i, j] for i in V for j in V),
        GRB.MINIMIZE
    )

    # ----- Constraints -----

    # (2) In-degree: each customer visited exactly once
    for j in LB:
        model.addConstr(
            quicksum(x[i, j] for i in V) == 1,
            name=f"indegree_{j}"
        )

    # (3) Out-degree: each customer leaves exactly once
    for j in LB:
        model.addConstr(
            quicksum(x[j, i] for i in V) == 1,
            name=f"outdegree_{j}"
        )

    # (4) Delivery flow conservation at linehaul customers:
    #     sum_i R[i,j] - q[j] = sum_i R[j,i],  j in L
    for j in L:
        model.addConstr(
            quicksum(R[i, j] for i in V) - q[j] == quicksum(R[j, i] for i in V),
            name=f"delivery_flow_linehaul_{j}"
        )

    # (5) Delivery flow conservation at backhaul customers (pass-through):
    #     sum_i R[i,j] = sum_i R[j,i],  j in B
    for j in B:
        model.addConstr(
            quicksum(R[i, j] for i in V) == quicksum(R[j, i] for i in V),
            name=f"delivery_flow_backhaul_{j}"
        )

    # (6) Pickup flow conservation at linehaul customers (pass-through):
    #     sum_i P[i,j] = sum_i P[j,i],  j in L
    for j in L:
        model.addConstr(
            quicksum(P[i, j] for i in V) == quicksum(P[j, i] for i in V),
            name=f"pickup_flow_linehaul_{j}"
        )

    # (7) Pickup flow conservation at backhaul customers:
    #     sum_i P[i,j] + q[j] = sum_i P[j,i],  j in B
    for j in B:
        model.addConstr(
            quicksum(P[i, j] for i in V) + q[j] == quicksum(P[j, i] for i in V),
            name=f"pickup_flow_backhaul_{j}"
        )

    # (8) Vehicles start with zero pickup load:
    #     sum_{i in L union B} P[0,i] = 0
    model.addConstr(
        quicksum(P[depot, i] for i in LB) == 0,
        name="zero_initial_pickup"
    )

    # (9) Vehicles finish with zero delivery load:
    #     sum_{i in L union B} R[i,0] = 0
    model.addConstr(
        quicksum(R[i, depot] for i in LB) == 0,
        name="zero_final_delivery"
    )

    # (10) Capacity constraint on each arc:
    #      R[i,j] + P[i,j] <= C * x[i,j]
    for i in V:
        for j in V:
            model.addConstr(
                R[i, j] + P[i, j] <= C * x[i, j],
                name=f"capacity_{i}_{j}"
            )

    # =========================================================================
    # Valid inequalities / cuts
    # =========================================================================

    # (14) No self-loops: x[i,i] = 0
    for i in V:
        model.addConstr(x[i, i] == 0, name=f"no_selfloop_{i}")

    # (15) No arc from backhaul n+i to its corresponding linehaul i:
    #      x[n+i, i] = 0
    for i in L:
        model.addConstr(x[n + i, i] == 0, name=f"no_back_to_line_{i}")

    # (16) Minimum number of vehicles:
    #      sum_{i in L union B} x[0,i] >= ceil( sum_{i in L} q[i] / C )
    total_delivery = sum(q[i] for i in L)
    total_pickup = sum(q[i] for i in B)
    min_vehicles_delivery = math.ceil(total_delivery / C)
    min_vehicles_pickup = math.ceil(total_pickup / C)
    min_vehicles = max(min_vehicles_delivery, min_vehicles_pickup)
    print(f"  Minimum vehicles (delivery): {min_vehicles_delivery}")
    print(f"  Minimum vehicles (pickup):   {min_vehicles_pickup}")
    print(f"  Minimum vehicles (overall):  {min_vehicles}")

    # The paper's constraint (16) uses delivery demand for the lower bound.
    # We add both delivery and pickup lower bounds for tighter relaxation.
    model.addConstr(
        quicksum(x[depot, i] for i in LB) >= min_vehicles_delivery,
        name="min_vehicles_delivery"
    )
    model.addConstr(
        quicksum(x[depot, i] for i in LB) >= min_vehicles_pickup,
        name="min_vehicles_pickup"
    )

    # (17) Subtour elimination for pairs: x[i,j] + x[j,i] <= 1
    for i in LB:
        for j in LB:
            if i < j:
                model.addConstr(
                    x[i, j] + x[j, i] <= 1,
                    name=f"pair_subtour_{i}_{j}"
                )

    # ----- Flow balance at depot: vehicles that leave must return -----
    model.addConstr(
        quicksum(x[depot, j] for j in LB) == quicksum(x[j, depot] for j in LB),
        name="depot_flow_balance"
    )

    # =========================================================================
    # Solve
    # =========================================================================
    print(f"\nSolving with time limit = {time_limit} seconds...")
    model.optimize()

    # =========================================================================
    # Extract solution
    # =========================================================================
    result = {
        "objective_value": None,
        "routes": [],
        "num_routes": 0,
        "status": "",
        "mip_gap": None,
        "solve_time": model.Runtime,
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["mip_gap"] = model.MIPGap

        if model.Status == GRB.OPTIMAL:
            result["status"] = "optimal"
        elif model.Status == GRB.TIME_LIMIT:
            result["status"] = "time_limit"
        else:
            result["status"] = "feasible"

        # Reconstruct routes by following arcs from the depot
        # Build adjacency from selected arcs
        adj = {}
        for i in V:
            for j in V:
                if x[i, j].X > 0.5:
                    adj[i] = j

        # Extract routes starting from depot
        routes = []
        visited_from_depot = set()

        for j in LB:
            if x[depot, j].X > 0.5 and j not in visited_from_depot:
                # Trace route: depot -> j -> ... -> depot
                route = [depot]
                current = j
                route_visited = set()
                while current != depot:
                    if current in route_visited:
                        # Safety: avoid infinite loop from numerical issues
                        print(f"  WARNING: cycle detected in route at node {current}")
                        break
                    route_visited.add(current)
                    visited_from_depot.add(current)
                    route.append(current)
                    current = adj.get(current, depot)
                route.append(depot)
                routes.append(route)

        result["routes"] = routes
        result["num_routes"] = len(routes)

        # Map expanded node IDs back to original customer IDs for readability
        routes_original = []
        for route in routes:
            original_route = []
            for node in route:
                if node == 0:
                    original_route.append({"node": 0, "type": "depot"})
                elif 1 <= node <= n:
                    original_route.append({
                        "node": node,
                        "type": "linehaul",
                        "original_customer": node,
                        "delivery_demand": q[node],
                    })
                else:
                    orig_cust = node - n
                    original_route.append({
                        "node": node,
                        "type": "backhaul",
                        "original_customer": orig_cust,
                        "pickup_demand": q[node],
                    })
            routes_original.append(original_route)

        result["routes_detailed"] = routes_original

        # Compute per-route distances
        route_distances = []
        for route in routes:
            route_dist = 0
            for k in range(len(route) - 1):
                route_dist += dist[route[k]][route[k + 1]]
            route_distances.append(route_dist)
        result["route_distances"] = route_distances

        # Print solution summary
        print(f"\n{'='*60}")
        print(f"Solution found!")
        print(f"  Objective value: {obj_val}")
        print(f"  MIP gap: {model.MIPGap:.4%}")
        print(f"  Solve time: {model.Runtime:.2f} seconds")
        print(f"  Number of routes: {len(routes)}")
        print(f"{'='*60}")
        for idx, route in enumerate(routes):
            desc_parts = []
            for node in route:
                if node == 0:
                    desc_parts.append("depot")
                elif 1 <= node <= n:
                    desc_parts.append(f"L{node}")
                else:
                    desc_parts.append(f"B{node - n}")
            print(f"  Route {idx + 1} (dist={route_distances[idx]}): "
                  f"{' -> '.join(desc_parts)}")
    else:
        if model.Status == GRB.INFEASIBLE:
            result["status"] = "infeasible"
            print("Model is infeasible.")
            # Compute IIS for debugging
            try:
                model.computeIIS()
                print("IIS constraints:")
                for c in model.getConstrs():
                    if c.IISConstr:
                        print(f"  {c.ConstrName}")
            except Exception:
                pass
        elif model.Status == GRB.TIME_LIMIT:
            result["status"] = "time_limit_no_solution"
            print("Time limit reached with no feasible solution.")
        else:
            result["status"] = f"gurobi_status_{model.Status}"
            print(f"Gurobi status: {model.Status}")

    return result


def extract_instance_id(instance_path):
    """Extract the instance number from a path like .../instance_3.json."""
    basename = os.path.basename(instance_path)
    match = re.search(r"(\d+)", basename)
    if match:
        return match.group(1)
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "VRPDDP solver using Gurobi (two-index MILP formulation). "
            "Based on Nagy, Wassan, Speranza, Archetti (2015), "
            "Transportation Science 49(2):271-294."
        )
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
        default=None,
        help=(
            "Path for the output solution JSON file. "
            "Default: gurobi_solution_{i}.json in the instance directory"
        ),
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Maximum solver runtime in seconds (default: 3600)",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Determine solution path
    if args.solution_path is None:
        inst_id = extract_instance_id(args.instance_path)
        inst_dir = os.path.dirname(args.instance_path)
        args.solution_path = os.path.join(inst_dir, f"gurobi_solution_{inst_id}.json")

    print(f"Instance: {args.instance_path}")
    print(f"Solution: {args.solution_path}")
    print(f"Time limit: {args.time_limit} seconds")
    print()

    instance = load_instance(args.instance_path)
    result = solve_vrpddp(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)
    print(f"\nSolution saved to {args.solution_path}")


if __name__ == "__main__":
    main()
