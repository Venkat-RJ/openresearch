#!/usr/bin/env python3
"""
Compact arc-flow formulation for the Inventory-Routing Problem (IRP)
under Maximum-Level (ML) replenishment policy, solved directly by Gurobi.

Based on: Desaulniers, Rakke, and Coelho (2016),
"A Branch-Price-and-Cut Algorithm for the Inventory-Routing Problem",
Transportation Science 50(3), pp. 1060-1076.

NOTE: The paper's own formulation (Eqs. 1-9) is an RDP-based set-partitioning
model designed for column generation. This compact arc-flow formulation is
**NOT EXPLICITLY SPECIFIED IN PAPER** — it is reconstructed from the standard
IRP literature (Archetti et al. 2007, Coelho and Laporte 2014) as referenced
in the paper. The paper uses this type of formulation as a competitor
(branch-and-cut approach).

Assumptions:
- Travel costs c_{ij} are the Euclidean distances (rounded to integer) as per
  the benchmark instances of Archetti et al. (2007). The instance JSON provides
  a precomputed distance matrix which we use directly.
- Holding costs are charged on inventory at the END of each period.
- The fleet is homogeneous (all vehicles have the same capacity Q).
- Each customer can be serviced at most once per period.
- Subtour elimination uses Miller-Tucker-Zemlin (MTZ) formulation.
  (INFERRED ASSUMPTION: The exact subtour elimination used in the competitor
  paper Coelho and Laporte 2014 is not specified; MTZ is a standard choice.)
- Sequence of operations: production occurs before deliveries; customers receive
  deliveries at the beginning of the period and can use them to fulfill demand.
"""

import argparse
import json
import time
import sys
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


try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy is not installed. Please install Gurobi.")
    sys.exit(1)


def load_instance(instance_path):
    """Load an IRP instance from a JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_irp(data, time_limit):
    """
    Solve the IRP using a compact arc-flow formulation with Gurobi.

    Returns a dict with the solution information.
    """
    # ================================================================
    # Extract instance data
    # ================================================================
    n = data["num_customers"]           # number of customers
    T = data["num_periods"]             # number of periods
    K_num = data["num_vehicles"]        # number of vehicles
    Q = data["vehicle_capacity"]        # vehicle capacity

    # Sets
    customers = list(range(1, n + 1))   # 1..n
    depot = 0
    V = list(range(n + 1))              # 0..n (all locations)
    periods = list(range(1, T + 1))     # 1..T
    vehicles = list(range(K_num))       # 0..K-1

    # Distance matrix (precomputed, integer Euclidean distances)
    dist = data["distance_matrix"]

    # Holding costs for each location (index 0 = depot, 1..n = customers)
    h = data["holding_costs"]

    # Customer demands: customer_demands[i-1][p-1] = demand of customer i in period p
    cust_demands = data["customer_demands"]

    # Supplier production: supplier_production[p-1] = production in period p
    supplier_prod = data["supplier_production"]

    # Inventory capacities: index 0 = depot, 1..n = customers
    inv_cap = data["inventory_capacities"]

    # Initial inventories: index 0 = depot, 1..n = customers
    init_inv = data["initial_inventories"]

    # Helper to get demand d_{ip}
    def d(i, p):
        """Demand of customer i (1-indexed) in period p (1-indexed)."""
        if i == 0:
            return supplier_prod[p - 1]
        else:
            return cust_demands[i - 1][p - 1]

    # ================================================================
    # Build Gurobi model
    # ================================================================
    model = gp.Model("IRP_ArcFlow")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)  # Show solver output

    # ----------------------------------------------------------------
    # Decision Variables
    # ----------------------------------------------------------------

    # x[i,j,p,k]: binary, 1 if vehicle k traverses arc (i,j) in period p
    x = {}
    for p in periods:
        for k in vehicles:
            for i in V:
                for j in V:
                    if i != j:
                        x[i, j, p, k] = model.addVar(
                            vtype=GRB.BINARY, name=f"x_{i}_{j}_{p}_{k}")

    # z[i,p,k]: binary, 1 if vehicle k visits customer i in period p
    z = {}
    for i in customers:
        for p in periods:
            for k in vehicles:
                z[i, p, k] = model.addVar(
                    vtype=GRB.BINARY, name=f"z_{i}_{p}_{k}")

    # q[i,p,k]: continuous, quantity delivered to customer i by vehicle k in period p
    q = {}
    for i in customers:
        for p in periods:
            for k in vehicles:
                q[i, p, k] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{p}_{k}")

    # I[i,p]: inventory at location i at end of period p
    I = {}
    for i in V:
        for p in periods:
            I[i, p] = model.addVar(
                lb=0.0, vtype=GRB.CONTINUOUS, name=f"I_{i}_{p}")

    # u[i,p,k]: MTZ auxiliary variable for subtour elimination
    u = {}
    for i in customers:
        for p in periods:
            for k in vehicles:
                u[i, p, k] = model.addVar(
                    lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name=f"u_{i}_{p}_{k}")

    model.update()

    # ----------------------------------------------------------------
    # Objective: minimize total travel + holding costs
    # ----------------------------------------------------------------
    obj = gp.LinExpr()

    # Travel costs
    for p in periods:
        for k in vehicles:
            for i in V:
                for j in V:
                    if i != j:
                        obj += dist[i][j] * x[i, j, p, k]

    # Holding costs (at all locations, end of each period)
    for p in periods:
        for i in V:
            obj += h[i] * I[i, p]

    model.setObjective(obj, GRB.MINIMIZE)

    # ----------------------------------------------------------------
    # Constraints
    # ----------------------------------------------------------------

    # 1. Supplier (depot) inventory balance:
    #    I_{0,p} = I_{0,p-1} + d_{0p} - sum_k sum_i q_{ip}^k
    for p in periods:
        prev_inv = init_inv[0] if p == 1 else I[0, p - 1]
        total_delivered = gp.quicksum(
            q[i, p, k] for i in customers for k in vehicles)
        model.addConstr(
            I[0, p] == prev_inv + d(0, p) - total_delivered,
            name=f"supplier_inv_balance_{p}")

    # 2. Customer inventory balance:
    #    I_{ip} = I_{i,p-1} + sum_k q_{ip}^k - d_{ip}
    for i in customers:
        for p in periods:
            prev_inv = init_inv[i] if p == 1 else I[i, p - 1]
            total_received = gp.quicksum(q[i, p, k] for k in vehicles)
            model.addConstr(
                I[i, p] == prev_inv + total_received - d(i, p),
                name=f"cust_inv_balance_{i}_{p}")

    # 3. Inventory non-negativity is already enforced by lb=0.0 on I variables

    # 4. Customer inventory capacity:
    #    I_{i,p-1} + sum_k q_{ip}^k <= C_i  for i in N, p in P
    for i in customers:
        for p in periods:
            prev_inv = init_inv[i] if p == 1 else I[i, p - 1]
            total_received = gp.quicksum(q[i, p, k] for k in vehicles)
            model.addConstr(
                prev_inv + total_received <= inv_cap[i],
                name=f"cust_inv_cap_{i}_{p}")

    # 5. Supplier inventory capacity:
    #    0 <= I_{0p} <= C_0
    for p in periods:
        model.addConstr(I[0, p] <= inv_cap[0],
                        name=f"supplier_inv_cap_{p}")

    # 6. Delivery only if visited:
    #    q_{ip}^k <= C_i * z_{ip}^k
    for i in customers:
        for p in periods:
            for k in vehicles:
                model.addConstr(
                    q[i, p, k] <= inv_cap[i] * z[i, p, k],
                    name=f"delivery_if_visit_{i}_{p}_{k}")

    # 7. Flow conservation for vehicles:
    #    sum_j x_{ijp}^k = z_{ip}^k  (inflow)
    #    sum_j x_{jip}^k = z_{ip}^k  (outflow)
    for i in customers:
        for p in periods:
            for k in vehicles:
                model.addConstr(
                    gp.quicksum(x[j, i, p, k] for j in V if j != i) == z[i, p, k],
                    name=f"flow_in_{i}_{p}_{k}")
                model.addConstr(
                    gp.quicksum(x[i, j, p, k] for j in V if j != i) == z[i, p, k],
                    name=f"flow_out_{i}_{p}_{k}")

    # 8. Each vehicle leaves depot at most once per period:
    #    sum_j x_{0jp}^k <= 1
    for p in periods:
        for k in vehicles:
            model.addConstr(
                gp.quicksum(x[0, j, p, k] for j in customers) <= 1,
                name=f"depot_leave_{p}_{k}")

    # 9. Each vehicle returns to depot if it leaves:
    #    sum_j x_{0jp}^k = sum_j x_{j0p}^k
    for p in periods:
        for k in vehicles:
            model.addConstr(
                gp.quicksum(x[0, j, p, k] for j in customers) ==
                gp.quicksum(x[j, 0, p, k] for j in customers),
                name=f"depot_return_{p}_{k}")

    # 10. At most one visit per customer per period:
    #     sum_k z_{ip}^k <= 1
    for i in customers:
        for p in periods:
            model.addConstr(
                gp.quicksum(z[i, p, k] for k in vehicles) <= 1,
                name=f"one_visit_{i}_{p}")

    # 11. MTZ subtour elimination:
    #     u_{ip}^k - u_{jp}^k + Q * x_{ijp}^k <= Q - q_{jp}^k
    #     for i,j in N, i != j
    for p in periods:
        for k in vehicles:
            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            u[i, p, k] - u[j, p, k] + Q * x[i, j, p, k]
                            <= Q - q[j, p, k],
                            name=f"mtz_{i}_{j}_{p}_{k}")

    # 12. Vehicle load capacity:
    #     sum_i q_{ip}^k <= Q
    for p in periods:
        for k in vehicles:
            model.addConstr(
                gp.quicksum(q[i, p, k] for i in customers) <= Q,
                name=f"vehicle_cap_{p}_{k}")

    # 13. MTZ variable linking with depot:
    #     q_{ip}^k <= u_{ip}^k for all i in N, k, p
    for i in customers:
        for p in periods:
            for k in vehicles:
                model.addConstr(
                    q[i, p, k] <= u[i, p, k],
                    name=f"mtz_link_{i}_{p}_{k}")

    # ----------------------------------------------------------------
    # Symmetry breaking: order vehicles by first customer visited
    # (INFERRED ASSUMPTION: helps solver performance; not from paper)
    # ----------------------------------------------------------------
    # Vehicles are identical, so we can impose ordering constraints
    for p in periods:
        for k in range(len(vehicles) - 1):
            model.addConstr(
                gp.quicksum(x[0, j, p, k] for j in customers) >=
                gp.quicksum(x[0, j, p, k + 1] for j in customers),
                name=f"sym_break_{p}_{k}")

    # ----------------------------------------------------------------
    # Optimize
    # ----------------------------------------------------------------
    model.optimize()

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    solution = {}

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            solution["objective_value"] = model.ObjVal
            solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "time_limit"
            solution["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
            solution["solve_time"] = model.Runtime

            # Extract routes
            routes = {}
            for p in periods:
                routes[str(p)] = {}
                for k in vehicles:
                    route_arcs = []
                    for i in V:
                        for j in V:
                            if i != j and x[i, j, p, k].X > 0.5:
                                route_arcs.append([i, j])
                    if route_arcs:
                        # Build ordered route
                        route_seq = build_route(route_arcs)
                        deliveries = {}
                        for i in customers:
                            val = q[i, p, k].X
                            if val > 0.001:
                                deliveries[str(i)] = round(val, 2)
                        routes[str(p)][str(k)] = {
                            "route": route_seq,
                            "deliveries": deliveries
                        }

            solution["routes"] = routes

            # Extract inventories
            inventories = {}
            for i in V:
                inventories[str(i)] = {}
                for p in periods:
                    inventories[str(i)][str(p)] = round(I[i, p].X, 2)

            solution["inventories"] = inventories

            # Cost breakdown
            travel_cost = sum(
                dist[i][j] * x[i, j, p, k].X
                for p in periods for k in vehicles
                for i in V for j in V if i != j and x[i, j, p, k].X > 0.5
            )
            holding_cost = sum(
                h[i] * I[i, p].X for p in periods for i in V
            )
            solution["travel_cost"] = round(travel_cost, 2)
            solution["holding_cost"] = round(holding_cost, 2)
        else:
            solution["objective_value"] = None
            solution["status"] = "no_feasible_solution"
            solution["solve_time"] = model.Runtime
    elif model.Status == GRB.INFEASIBLE:
        solution["objective_value"] = None
        solution["status"] = "infeasible"
        solution["solve_time"] = model.Runtime
    else:
        solution["objective_value"] = None
        solution["status"] = f"gurobi_status_{model.Status}"
        solution["solve_time"] = model.Runtime

    return solution


def build_route(arcs):
    """Build an ordered route from a list of arcs starting and ending at depot 0."""
    if not arcs:
        return []

    # Build adjacency from arcs
    adj = {}
    for (i, j) in arcs:
        adj[i] = j

    # Start from depot
    route = [0]
    current = adj.get(0)
    visited = {0}
    while current is not None and current not in visited:
        route.append(current)
        visited.add(current)
        current = adj.get(current)
    route.append(0)  # return to depot
    return route


def main():
    parser = argparse.ArgumentParser(
        description="Solve IRP using compact arc-flow formulation with Gurobi")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)

    print(f"Instance: {data.get('instance_class', 'unknown')}, "
          f"Customers: {data['num_customers']}, "
          f"Periods: {data['num_periods']}, "
          f"Vehicles: {data['num_vehicles']}, "
          f"Capacity: {data['vehicle_capacity']}")

    # Solve
    solution = solve_irp(data, args.time_limit)

    # Write solution
    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"\nSolution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Status: {solution['status']}")
    else:
        print(f"No feasible solution found. Status: {solution['status']}")


if __name__ == "__main__":
    main()
