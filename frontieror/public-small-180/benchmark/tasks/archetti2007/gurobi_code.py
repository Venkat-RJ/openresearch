#!/usr/bin/env python3
"""
Gurobi implementation of the VMIR-OU (Vendor-Managed Inventory Routing
with Order-Up-To level) problem.

Source: Archetti, Bertazzi, Laporte, Speranza (2007)
        "A Branch-and-Cut Algorithm for a Vendor-Managed Inventory-Routing Problem"
        Transportation Science 41(3), pp. 382-391.

This program implements the full MILP model (equations 1-16) with:
  - Valid inequalities (17)-(19), (22)-(24) added statically at root
  - Subtour elimination constraints (12) separated dynamically via lazy callbacks
  - Branching priority: z variables first, then y variables
  - Best-bound-first search strategy
"""

import argparse
import json
import math
import sys
import time
from itertools import combinations

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


def load_instance(path):
    """Load a VMIR-OU instance from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def find_connected_components(n_retailers, z_vals, y_vals, t):
    """
    Given fractional/integer z and y values for time period t,
    find connected components among visited retailers using a
    support graph approach. Used for subtour elimination separation.

    Returns a list of components (sets of retailer indices 1..n)
    that do NOT contain the depot (node 0).
    """
    # Build adjacency from y values
    visited = set()
    for s in range(1, n_retailers + 1):
        if z_vals.get((s, t), 0) > 0.5:
            visited.add(s)

    if len(visited) <= 1:
        return []

    # Build adjacency list among visited retailers
    adj = {s: set() for s in visited}
    for s in visited:
        # Check edge to depot
        # y_{s,0} uses key (s, 0) since s > 0
        pass  # depot connections handled separately

    for i in visited:
        for j in visited:
            if j < i:
                val = y_vals.get((i, j, t), 0)
                if val > 0.5:
                    adj[i].add(j)
                    adj[j].add(i)

    # Check depot connectivity
    depot_connected = set()
    for s in visited:
        # Edge between s and 0: key is (s, 0, t) since s > 0
        val = y_vals.get((s, 0, t), 0)
        if val > 0.5:
            depot_connected.add(s)

    # BFS/DFS to find components
    remaining = set(visited)
    components = []
    while remaining:
        start = next(iter(remaining))
        component = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            remaining.discard(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in component:
                    queue.append(neighbor)
        components.append(component)

    # Return only components NOT connected to depot
    subtour_components = []
    for comp in components:
        if not comp.intersection(depot_connected):
            subtour_components.append(comp)

    return subtour_components


def subtour_callback(model, where):
    """
    Lazy constraint callback for subtour elimination.
    Implements the Padberg-Rinaldi style separation:
    find connected components among visited nodes that don't include depot.

    Uses constraint (12):
      sum_{i in S, j in S, j<i} y^t_{ij} <= sum_{i in S} z_{it} - z_{kt}
      where k = argmax_{j in S} z_{jt}
    """
    if where != GRB.Callback.MIPSOL:
        return

    n = model._n_retailers
    H = model._H
    z = model._z
    y = model._y

    # Get current solution values
    z_vals = {}
    y_vals = {}
    for key in z:
        z_vals[key] = model.cbGetSolution(z[key])
    for key in y:
        y_vals[key] = model.cbGetSolution(y[key])

    for t in range(1, H + 1):
        components = find_connected_components(n, z_vals, y_vals, t)
        for S in components:
            if len(S) < 2:
                continue
            # k = argmax_{j in S} z_{jt}
            k = max(S, key=lambda j: z_vals.get((j, t), 0))

            # LHS: sum_{i in S, j in S, j<i} y^t_{ij}
            lhs = gp.LinExpr()
            for i in S:
                for j in S:
                    if j < i:
                        lhs += y[(i, j, t)]

            # RHS: sum_{i in S} z_{it} - z_{kt}
            rhs = gp.LinExpr()
            for i in S:
                rhs += z[(i, t)]
            rhs -= z[(k, t)]

            model.cbLazy(lhs <= rhs)


def solve_vmir_ou(data, time_limit):
    """Build and solve the VMIR-OU model."""
    n = data['num_retailers']
    H = data['time_horizon']
    supplier = data['supplier']
    retailers = data['retailers']
    C = data['vehicle_capacity']
    dist = data['distance_matrix']

    # Parameters
    B0 = supplier['starting_inventory']
    h0 = supplier['inventory_cost']
    # Production rate: constant each period
    # From instance: production_rate is r_{0t} for each t
    r0 = supplier['production_rate']  # r_{0t} = r0 for all t in T

    # Retailer parameters (indexed 1..n)
    r = {}      # r[s] = demand of retailer s (constant over time)
    U = {}      # U[s] = max inventory
    I0 = {}     # I0[s] = starting inventory
    h = {}      # h[s] = inventory cost
    for ret in retailers:
        s = ret['id']
        r[s] = ret['demand']
        U[s] = ret['max_inventory']
        I0[s] = ret['starting_inventory']
        h[s] = ret['inventory_cost']

    # Transportation cost c[i][j] = dist[i][j], symmetric
    c = {}
    for i in range(n + 1):
        for j in range(n + 1):
            c[i, j] = dist[i][j]

    # Sets
    M = list(range(1, n + 1))         # retailers
    T = list(range(1, H + 1))         # time periods
    T_prime = list(range(1, H + 2))   # extended time horizon {1,...,H+1}
    M_prime = list(range(0, n + 1))   # all vertices including supplier 0

    # Create model
    model = gp.Model("VMIR_OU")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LazyConstraints", 1)
    model.setParam("NodeMethod", -1)

    # =====================
    # DECISION VARIABLES
    # =====================

    # x[s,t]: quantity shipped to retailer s at time t (continuous, >= 0)
    x = {}
    for s in M:
        for t in T:
            x[s, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{s}_{t}")

    # B[t]: supplier inventory at time t in T' (continuous)
    B = {}
    for t in T_prime:
        B[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"B_{t}")

    # I[s,t]: retailer s inventory at time t in T' (continuous)
    I = {}
    for s in M:
        for t in T_prime:
            I[s, t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"I_{s}_{t}")

    # z[i,t]: binary, 1 if vertex i is visited at time t
    z = {}
    for i in M_prime:
        for t in T:
            z[i, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}")

    # y[i,j,t]: for j < i
    #   If i,j both in M (retailers): binary {0,1}
    #   If j=0 (supplier): integer {0,1,2}
    y = {}
    for t in T:
        for i in M_prime:
            for j in M_prime:
                if j < i:
                    if j == 0:
                        # y[i,0,t] in {0,1,2}
                        y[i, j, t] = model.addVar(lb=0, ub=2, vtype=GRB.INTEGER,
                                                   name=f"y_{i}_{j}_{t}")
                    else:
                        # y[i,j,t] in {0,1}
                        y[i, j, t] = model.addVar(vtype=GRB.BINARY,
                                                   name=f"y_{i}_{j}_{t}")

    model.update()

    # =====================
    # BRANCHING PRIORITIES
    # =====================
    # Priority 1: z variables (higher priority = branched first)
    for key in z:
        z[key].BranchPriority = 2
    for key in y:
        y[key].BranchPriority = 1

    # =====================
    # OBJECTIVE FUNCTION (1)
    # =====================
    obj = gp.LinExpr()
    # Supplier inventory cost over T'
    for t in T_prime:
        obj += h0 * B[t]
    # Retailer inventory cost over T'
    for s in M:
        for t in T_prime:
            obj += h[s] * I[s, t]
    # Transportation cost over T
    for t in T:
        for i in M_prime:
            for j in M_prime:
                if j < i:
                    obj += c[i, j] * y[i, j, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # =====================
    # CONSTRAINTS
    # =====================

    # (2) Supplier inventory definition: B_t = B_{t-1} + r_{0,t-1} - sum_s x_{s,t-1}
    # For t=1: B_1 = B_0 + r_{0,0} - sum_s x_{s,0} = B_0 (since r_{00}=0, x_{s0}=0)
    # For t>1: B_t = B_{t-1} + r_0 - sum_s x_{s,t-1}
    for t in T_prime:
        if t == 1:
            model.addConstr(B[t] == B0, name=f"inv_supplier_{t}")
        else:
            model.addConstr(
                B[t] == B[t - 1] + r0 - gp.quicksum(x[s, t - 1] for s in M),
                name=f"inv_supplier_{t}")

    # (3) Stockout at supplier: B_t >= sum_s x_{st}, t in T
    for t in T:
        model.addConstr(
            B[t] >= gp.quicksum(x[s, t] for s in M),
            name=f"stockout_supplier_{t}")

    # (4) Retailer inventory definition: I_{st} = I_{s,t-1} + x_{s,t-1} - r_{s,t-1}
    # For t=1: I_{s,1} = I_{s,0} + x_{s,0} - r_{s,0} = I_{s,0} (x_{s0}=r_{s0}=0)
    for s in M:
        for t in T_prime:
            if t == 1:
                model.addConstr(I[s, t] == I0[s], name=f"inv_retailer_{s}_{t}")
            else:
                model.addConstr(
                    I[s, t] == I[s, t - 1] + x[s, t - 1] - r[s],
                    name=f"inv_retailer_{s}_{t}")

    # (5) Stockout at retailers: I_{st} >= 0, s in M, t in T'
    for s in M:
        for t in T_prime:
            model.addConstr(I[s, t] >= 0, name=f"nonneg_inv_{s}_{t}")

    # (6) Order-up-to: x_{st} >= U_s * z_{st} - I_{st}
    for s in M:
        for t in T:
            model.addConstr(
                x[s, t] >= U[s] * z[s, t] - I[s, t],
                name=f"order_up_to_lb_{s}_{t}")

    # (7) x_{st} <= U_s - I_{st}
    for s in M:
        for t in T:
            model.addConstr(
                x[s, t] <= U[s] - I[s, t],
                name=f"order_up_to_ub_{s}_{t}")

    # (8) x_{st} <= U_s * z_{st}
    for s in M:
        for t in T:
            model.addConstr(
                x[s, t] <= U[s] * z[s, t],
                name=f"order_up_to_link_{s}_{t}")

    # (9) Capacity: sum_s x_{st} <= C, t in T
    for t in T:
        model.addConstr(
            gp.quicksum(x[s, t] for s in M) <= C,
            name=f"capacity_{t}")

    # (10) Supplier inclusion: sum_s x_{st} <= C * z_{0t}
    for t in T:
        model.addConstr(
            gp.quicksum(x[s, t] for s in M) <= C * z[0, t],
            name=f"supplier_link_{t}")

    # (11) Degree constraints:
    # sum_{j in M', j<i} y^t_{ij} + sum_{j in M', j>i} y^t_{ji} = 2*z_{it}
    for i in M_prime:
        for t in T:
            lhs = gp.LinExpr()
            for j in M_prime:
                if j < i:
                    lhs += y[i, j, t]
                elif j > i:
                    lhs += y[j, i, t]
            model.addConstr(lhs == 2 * z[i, t], name=f"degree_{i}_{t}")

    # (13) x_{st} >= 0 -- already enforced by variable bounds

    # =====================
    # VALID INEQUALITIES
    # =====================

    # (17) I_{st} >= (1 - z_{st}) * r_{st}, s in M, t in T
    for s in M:
        for t in T:
            model.addConstr(
                I[s, t] >= (1 - z[s, t]) * r[s],
                name=f"vi17_{s}_{t}")

    # (18) I_{s,t-k} >= (sum_{j=0}^{k} r_{s,t-j}) * (1 - sum_{j=0}^{k} z_{s,t-j})
    # for s in M, t in T, k = 0, 1, ..., t-1
    for s in M:
        for t in T:
            for k in range(0, t):  # k = 0, 1, ..., t-1
                sum_r = sum(r[s] for j in range(k + 1))  # r is constant
                sum_r_val = (k + 1) * r[s]

                sum_z = gp.LinExpr()
                for j in range(k + 1):
                    tj = t - j
                    if tj >= 1:
                        sum_z += z[s, tj]

                t_minus_k = t - k
                if t_minus_k >= 1:
                    inv_var = I[s, t_minus_k]
                elif t_minus_k == 0:
                    # I_{s,0} is a constant = I0[s], skip adding constraint
                    # since I0[s] >= sum_r * (1 - sum_z) is always true when
                    # sum_z could be at most k+1 (all visited)
                    continue
                else:
                    continue

                model.addConstr(
                    inv_var >= sum_r_val * (1 - sum_z),
                    name=f"vi18_{s}_{t}_{k}")

    # (19) I_{st} >= U_s * z_{s,t-k} - sum_{j=t-k}^{t-1} r_{sj}
    # for s in M, t in T, k = 1, 2, ..., t-1
    # Valid only for VMIR-OU
    for s in M:
        for t in T:
            for k in range(1, t):  # k = 1, ..., t-1
                t_minus_k = t - k
                if t_minus_k < 1:
                    continue
                # sum_{j=t-k}^{t-1} r_{sj} = k * r[s] (constant demand)
                sum_r_val = k * r[s]
                model.addConstr(
                    I[s, t] >= U[s] * z[s, t_minus_k] - sum_r_val,
                    name=f"vi19_{s}_{t}_{k}")

    # (22) z_{st} <= z_{0t}, s in M, t in T
    for s in M:
        for t in T:
            model.addConstr(z[s, t] <= z[0, t], name=f"vi22_{s}_{t}")

    # (23) y^t_{i0} <= 2 * z_{it}, i in M, t in T
    for i in M:
        for t in T:
            model.addConstr(y[i, 0, t] <= 2 * z[i, t], name=f"vi23_{i}_{t}")

    # (24) y^t_{ij} <= z_{it}, i in M, j in M, t in T
    # Note: y is defined for j < i, so we add for all i,j in M with j < i
    for i in M:
        for j in M:
            if j < i:
                for t in T:
                    model.addConstr(y[i, j, t] <= z[i, t], name=f"vi24_{i}_{j}_{t}")

    # =====================
    # STORE FOR CALLBACK
    # =====================
    model._z = z
    model._y = y
    model._n_retailers = n
    model._H = H

    # =====================
    # OPTIMIZE
    # =====================
    model.optimize(subtour_callback)

    # =====================
    # EXTRACT SOLUTION
    # =====================
    result = {
        "objective_value": None,
        "status": model.Status,
        "solver": "Gurobi",
        "problem": "VMIR-OU"
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        # Extract delivery schedule
        deliveries = {}
        for t in T:
            deliveries[t] = {}
            for s in M:
                xval = x[s, t].X
                if xval > 1e-6:
                    deliveries[t][s] = round(xval, 4)
        result["deliveries"] = {str(k): {str(sk): sv for sk, sv in v.items()}
                                for k, v in deliveries.items()}

        # Extract routes
        routes = {}
        for t in T:
            route_edges = []
            for i in M_prime:
                for j in M_prime:
                    if j < i:
                        yval = y[i, j, t].X
                        if yval > 0.5:
                            route_edges.append((j, i, round(yval)))
            if route_edges:
                routes[t] = route_edges
        result["routes"] = {str(k): v for k, v in routes.items()}

        # Extract inventory levels
        supplier_inv = {}
        for t in T_prime:
            supplier_inv[t] = round(B[t].X, 4)
        result["supplier_inventory"] = {str(k): v for k, v in supplier_inv.items()}

        retailer_inv = {}
        for s in M:
            retailer_inv[s] = {}
            for t in T_prime:
                retailer_inv[s][t] = round(I[s, t].X, 4)
        result["retailer_inventory"] = {
            str(s): {str(t): v for t, v in inv.items()}
            for s, inv in retailer_inv.items()
        }

        if model.Status == GRB.OPTIMAL:
            result["status_str"] = "OPTIMAL"
        elif model.Status == GRB.TIME_LIMIT:
            result["status_str"] = "TIME_LIMIT"
            if hasattr(model, 'MIPGap'):
                result["mip_gap"] = model.MIPGap
        else:
            result["status_str"] = f"STATUS_{model.Status}"
    else:
        result["status_str"] = "NO_SOLUTION_FOUND"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve VMIR-OU problem using Gurobi (Archetti et al. 2007)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = solve_vmir_ou(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2, default=str)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status_str']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
