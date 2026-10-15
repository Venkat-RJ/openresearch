#!/usr/bin/env python3
"""
Gurobi implementation of the Two-Echelon Capacitated Vehicle Routing Problem
with Grouping Constraints (2E-CVRPGC).

Based on: Liu, Luo, Qin, Lim (2018). "A branch-and-cut algorithm for the
two-echelon capacitated vehicle routing problem with grouping constraints."
European Journal of Operational Research, 266(2), 487-497.

This implements the arc-flow MILP formulation (constraints 1-23) from the paper.
"""

import argparse
import json
import math
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


def load_instance(instance_path):
    """Load a 2E-CVRPGC instance from a JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def euclidean_distance(n1, n2):
    """Compute Euclidean distance between two nodes."""
    return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)


def solve_2e_cvrpgc(instance_path, solution_path, time_limit):
    """Build and solve the 2E-CVRPGC MILP model using Gurobi."""
    data = load_instance(instance_path)

    # -------------------------------------------------------------------------
    # Extract instance data
    # -------------------------------------------------------------------------
    depot = data['depot']
    satellites = data['satellites']
    customers = data['customers']
    groups = data['groups']

    Q1 = data['Q1']  # First echelon vehicle capacity
    Q2 = data['Q2']  # Second echelon vehicle capacity
    T1 = data['T1']  # Max first echelon vehicles
    T2 = data['T2']  # Max second echelon vehicles

    num_satellites = data['num_satellites']
    num_customers = data['num_customers']
    num_groups = data['num_groups']

    # Build node dictionaries indexed by id
    all_nodes = {}
    all_nodes[depot['id']] = depot

    for s in satellites:
        all_nodes[s['id']] = s

    for c in customers:
        all_nodes[c['id']] = c

    depot_id = depot['id']
    sat_ids = [s['id'] for s in satellites]
    cust_ids = [c['id'] for c in customers]

    # Satellite parameters
    sat_capacity = {s['id']: s['capacity'] for s in satellites}       # Q_s
    sat_max_vehicles = {s['id']: s['max_vehicles'] for s in satellites}  # T_s
    sat_handling_cost = {s['id']: s['handling_cost'] for s in satellites}  # h_s

    # Customer parameters
    cust_demand = {c['id']: c['demand'] for c in customers}  # d_i

    # Group membership: group_id -> list of customer ids
    group_members = {}
    for g in groups:
        group_members[g['group_id']] = g['customer_ids']

    # Customer to group mapping
    cust_to_group = {}
    for c in customers:
        cust_to_group[c['id']] = c['group']

    # -------------------------------------------------------------------------
    # Compute travel costs (Euclidean distances)
    # -------------------------------------------------------------------------
    # NOTE: The paper does not specify whether distances are rounded.
    # **INFERRED ASSUMPTION**: We use continuous floating-point Euclidean distances.
    cost = {}
    for i_id in all_nodes:
        for j_id in all_nodes:
            if i_id != j_id:
                cost[i_id, j_id] = euclidean_distance(all_nodes[i_id], all_nodes[j_id])

    # -------------------------------------------------------------------------
    # Define arc sets
    # -------------------------------------------------------------------------
    # A_1: arcs connecting depot and satellites (both directions)
    # A_1 = {(i,j) | i, j in V_0 ∪ V_S}
    V0_VS = [depot_id] + sat_ids
    A1 = [(i, j) for i in V0_VS for j in V0_VS if i != j]

    # A_2: arcs connecting satellites to customers and between customers
    # A_2 = {(i,j) | i, j in V_S ∪ V_C, i ≠ j} \ {(i,j) | i, j in V_S, i ≠ j}
    VS_VC = sat_ids + cust_ids
    A2 = [(i, j) for i in VS_VC for j in VS_VC
           if i != j and not (i in sat_ids and j in sat_ids)]

    # Vehicle set for first echelon
    K1 = list(range(T1))

    # |V_S| for MTZ big-M
    n_sat = len(sat_ids)

    # -------------------------------------------------------------------------
    # Create Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("2E-CVRPGC")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # -------------------------------------------------------------------------
    # Decision variables
    # -------------------------------------------------------------------------
    # x[i,j,k] ∈ {0,1}: arc (i,j) traveled by first-echelon vehicle k
    x = {}
    for (i, j) in A1:
        for k in K1:
            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    # w[s,k] >= 0: quantity delivered to satellite s by vehicle k
    w = {}
    for s in sat_ids:
        for k in K1:
            w[s, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{s}_{k}")

    # u[s,k] >= 0: position of satellite s in route of vehicle k (MTZ)
    u = {}
    for s in sat_ids:
        for k in K1:
            u[s, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_{s}_{k}")

    # f[i,j] >= 0: load on arc (i,j) in second echelon
    f = {}
    for (i, j) in A2:
        f[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_{i}_{j}")

    # y[i,j] ∈ {0,1}: second echelon vehicle travels arc (i,j)
    y = {}
    for (i, j) in A2:
        y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    # z[i,s] ∈ {0,1}: customer i served by vehicle from satellite s
    z = {}
    for i in cust_ids:
        for s in sat_ids:
            z[i, s] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{s}")

    # v[h,s] ∈ {0,1}: group h assigned to satellite s
    v = {}
    for g in groups:
        h = g['group_id']
        for s in sat_ids:
            v[h, s] = model.addVar(vtype=GRB.BINARY, name=f"v_{h}_{s}")

    model.update()

    # -------------------------------------------------------------------------
    # Objective function (1): minimize routing + handling + routing costs
    # -------------------------------------------------------------------------
    obj = gp.LinExpr()

    # First echelon routing cost
    for (i, j) in A1:
        for k in K1:
            obj += cost[i, j] * x[i, j, k]

    # Handling cost at satellites
    for s in sat_ids:
        for k in K1:
            obj += sat_handling_cost[s] * w[s, k]

    # Second echelon routing cost
    for (i, j) in A2:
        obj += cost[i, j] * y[i, j]

    model.setObjective(obj, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------

    # (2) Flow conservation for first echelon
    # For each node i in V_0 ∪ V_S, for each vehicle k:
    # sum_{(i,j) in A1} x[i,j,k] = sum_{(j,i) in A1} x[j,i,k]
    for i in V0_VS:
        for k in K1:
            out_arcs = [(i, j) for j in V0_VS if j != i]
            in_arcs = [(j, i) for j in V0_VS if j != i]
            model.addConstr(
                gp.quicksum(x[a[0], a[1], k] for a in out_arcs) ==
                gp.quicksum(x[a[0], a[1], k] for a in in_arcs),
                name=f"flow_cons1_{i}_{k}"
            )

    # (3) Each satellite visited at most once per vehicle
    # For each satellite i, vehicle k: sum_{(i,j) in A1} x[i,j,k] <= 1
    for i in sat_ids:
        for k in K1:
            out_arcs = [(i, j) for j in V0_VS if j != i]
            model.addConstr(
                gp.quicksum(x[a[0], a[1], k] for a in out_arcs) <= 1,
                name=f"visit_once_{i}_{k}"
            )

    # (4) Subtour elimination (MTZ) for first echelon
    # u[i,k] + 1 <= u[j,k] + |V_S| * (1 - x[i,j,k])
    # for all (i,j) in A1, k in K1
    # Note: We apply MTZ only for arcs between satellites (and from depot to satellites)
    # The position variables u are only defined for satellites.
    # For arcs (i,j) where i is the depot, we need to handle specially.
    for (i, j) in A1:
        for k in K1:
            if i == depot_id and j in sat_ids:
                # u[j,k] >= 1 - n_sat*(1 - x[i,j,k])  =>  0 + 1 <= u[j,k] + n_sat*(1-x)
                # We treat depot position as 0
                model.addConstr(
                    1 <= u[j, k] + n_sat * (1 - x[i, j, k]),
                    name=f"mtz_{i}_{j}_{k}"
                )
            elif i in sat_ids and j in sat_ids:
                model.addConstr(
                    u[i, k] + 1 <= u[j, k] + n_sat * (1 - x[i, j, k]),
                    name=f"mtz_{i}_{j}_{k}"
                )
            # Arcs (sat, depot) and (depot, depot) don't need MTZ
            # as depot is the start/end point

    # Upper bound on u variables
    for s in sat_ids:
        for k in K1:
            model.addConstr(u[s, k] <= n_sat, name=f"u_ub_{s}_{k}")

    # (5) Delivery only if satellite is visited
    # w[s,k] <= Q_s * sum_{(s,i) in A1} x[s,i,k]
    for s in sat_ids:
        for k in K1:
            out_arcs = [(s, j) for j in V0_VS if j != s]
            model.addConstr(
                w[s, k] <= sat_capacity[s] * gp.quicksum(x[a[0], a[1], k] for a in out_arcs),
                name=f"delivery_visit_{s}_{k}"
            )

    # (6) Vehicle capacity constraint (first echelon)
    # sum_{s in V_S} w[s,k] <= Q1
    for k in K1:
        model.addConstr(
            gp.quicksum(w[s, k] for s in sat_ids) <= Q1,
            name=f"veh_cap1_{k}"
        )

    # (7) Satellite capacity constraint
    # sum_{k in K1} w[s,k] <= Q_s
    for s in sat_ids:
        model.addConstr(
            gp.quicksum(w[s, k] for k in K1) <= sat_capacity[s],
            name=f"sat_cap_{s}"
        )

    # (8) Flow conservation for second echelon: each customer visited exactly once
    # sum_{(i,j) in A2} y[i,j] = 1 for all i in V_C (out-degree)
    # sum_{(j,i) in A2} y[j,i] = 1 for all i in V_C (in-degree)
    for i in cust_ids:
        out_arcs = [(i, j) for j in VS_VC if (i, j) in y]
        in_arcs = [(j, i) for j in VS_VC if (j, i) in y]
        model.addConstr(
            gp.quicksum(y[a] for a in out_arcs) == 1,
            name=f"cust_out_{i}"
        )
        model.addConstr(
            gp.quicksum(y[a] for a in in_arcs) == 1,
            name=f"cust_in_{i}"
        )

    # (9) Vehicle limit per satellite (second echelon)
    # sum_{(s,i) in A2} y[s,i] <= T_s
    for s in sat_ids:
        out_arcs = [(s, j) for j in cust_ids if (s, j) in y]
        model.addConstr(
            gp.quicksum(y[a] for a in out_arcs) <= sat_max_vehicles[s],
            name=f"veh_limit_sat_{s}"
        )

    # (10) Total vehicle limit (second echelon)
    # sum_{s in V_S} sum_{(s,i) in A2} y[s,i] <= T2
    model.addConstr(
        gp.quicksum(
            y[s, j]
            for s in sat_ids
            for j in cust_ids
            if (s, j) in y
        ) <= T2,
        name="total_veh_limit2"
    )

    # (11) Flow conservation for demand satisfaction
    # sum_{(j,i) in A2} f[j,i] = sum_{(i,j) in A2} f[i,j] + d_i, for i in V_C
    for i in cust_ids:
        in_arcs = [(j, i) for j in VS_VC if (j, i) in f]
        out_arcs = [(i, j) for j in VS_VC if (i, j) in f]
        model.addConstr(
            gp.quicksum(f[a] for a in in_arcs) ==
            gp.quicksum(f[a] for a in out_arcs) + cust_demand[i],
            name=f"flow_demand_{i}"
        )

    # (12) Flow bounds (Gavish) for second echelon
    # d_j * y[i,j] <= f[i,j] <= (Q2 - d_i) * y[i,j]
    for (i, j) in A2:
        d_i = cust_demand.get(i, 0)  # satellites have 0 demand
        d_j = cust_demand.get(j, 0)
        model.addConstr(
            f[i, j] >= d_j * y[i, j],
            name=f"flow_lb_{i}_{j}"
        )
        model.addConstr(
            f[i, j] <= (Q2 - d_i) * y[i, j],
            name=f"flow_ub_{i}_{j}"
        )

    # (13) If two customers are successively visited, same satellite assignment
    # z[j,s] >= z[i,s] + y[i,j] - 1, for (i,j) in A2
    # This only applies when both i and j are in V_C (customers)
    # and also for arcs from satellite to customer
    for (i, j) in A2:
        for s in sat_ids:
            if i in cust_ids and j in cust_ids:
                # Both customers
                model.addConstr(
                    z[j, s] >= z[i, s] + y[i, j] - 1,
                    name=f"same_sat_{i}_{j}_{s}"
                )
            elif i in sat_ids and j in cust_ids:
                # Arc from satellite i to customer j
                # z[j,s] >= z[i,s] + y[i,j] - 1
                # z[i,s] for satellite: use constraint (14) z[s,s] = 1
                # So if i == s: z[j,s] >= 1 + y[i,j] - 1 = y[i,j]
                # If i != s: z[j,s] >= 0 + y[i,j] - 1 (trivially satisfied when y=0 or y=1)
                if i == s:
                    model.addConstr(
                        z[j, s] >= y[i, j],
                        name=f"same_sat_{i}_{j}_{s}"
                    )
                # If i != s, the constraint is z[j,s] >= 0 + y[i,j] - 1
                # which is trivially satisfied for binary z, so we can skip it.
                # But let's add it for completeness.
                else:
                    model.addConstr(
                        z[j, s] >= y[i, j] - 1,
                        name=f"same_sat_{i}_{j}_{s}"
                    )
            elif i in cust_ids and j in sat_ids:
                # Arc from customer i back to satellite j: z[j,s] >= z[i,s] + y[i,j] - 1.
                # s == j case: z[j,j] = 1 by (14), trivially satisfied.
                # s != j case: z[j,s] = 0 (a satellite is not assigned to another
                # satellite), so the constraint reduces to z[i,s] + y[i,j] <= 1,
                # which forces the route to terminate at the satellite where
                # customer i is assigned (enforces 13-14 on returns).
                if s != j:
                    model.addConstr(
                        z[i, s] + y[i, j] <= 1,
                        name=f"same_sat_{i}_{j}_{s}"
                    )

    # (14) Each satellite is assigned to itself: z[s,s] = 1
    # Note: z is defined for customers only in the model, but the paper says
    # z_{s,s} = 1 for s in V_S. We handle this via constraint (13) above
    # by checking when arc source is a satellite.
    # The paper defines z_{i,s} for i in V_C, s in V_S.
    # Constraint (14) z_{s,s} = 1 means the satellite node is assigned to itself.
    # This is implicitly handled above in constraint (13) for satellite->customer arcs.

    # (15) Grouping constraint: customers in same group -> same satellite
    # z[i,s] = v[h,s] for all i in C_h, C_h in C, s in V_S
    for g in groups:
        h = g['group_id']
        for cid in g['customer_ids']:
            for s in sat_ids:
                model.addConstr(
                    z[cid, s] == v[h, s],
                    name=f"group_{h}_{cid}_{s}"
                )

    # Each customer is assigned to exactly one satellite.
    # This is implied by the paper's formulation but must be stated explicitly
    # to prevent z (and v) variables from being trivially over-assigned (all 1s),
    # which would decouple the grouping constraint (15) from the actual routing.
    for i in cust_ids:
        model.addConstr(
            gp.quicksum(z[i, s] for s in sat_ids) == 1,
            name=f"assign_one_{i}"
        )

    # (16) Linking constraint: quantity delivered = flow out to customers
    # sum_{k in K1} w[s,k] = sum_{(s,i) in A2} f[s,i]
    for s in sat_ids:
        out_arcs = [(s, j) for j in cust_ids if (s, j) in f]
        model.addConstr(
            gp.quicksum(w[s, k] for k in K1) ==
            gp.quicksum(f[a] for a in out_arcs),
            name=f"link_{s}"
        )

    # -------------------------------------------------------------------------
    # Symmetry breaking for first echelon vehicles (optional, helps performance)
    # **INFERRED ASSUMPTION**: We add a simple symmetry-breaking constraint
    # ordering the usage of vehicles. This is not specified in the paper but
    # is a standard technique to help Gurobi.
    # -------------------------------------------------------------------------
    for k in range(len(K1) - 1):
        out_arcs_k = [(depot_id, j) for j in sat_ids]
        out_arcs_k1 = [(depot_id, j) for j in sat_ids]
        model.addConstr(
            gp.quicksum(x[depot_id, j, K1[k]] for j in sat_ids) >=
            gp.quicksum(x[depot_id, j, K1[k + 1]] for j in sat_ids),
            name=f"sym_break_{k}"
        )

    # -------------------------------------------------------------------------
    # Optimize
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    solution = {}

    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        solution['status'] = 'optimal' if model.Status == GRB.OPTIMAL else 'feasible'
        solution['mip_gap'] = model.MIPGap if hasattr(model, 'MIPGap') else None
        solution['runtime'] = model.Runtime

        # Extract first echelon routes
        first_echelon_routes = []
        for k in K1:
            route_arcs = []
            for (i, j) in A1:
                if x[i, j, k].X > 0.5:
                    route_arcs.append((i, j))
            if route_arcs:
                # Reconstruct route from arcs
                route = reconstruct_route(route_arcs, depot_id)
                deliveries = {}
                for s in sat_ids:
                    if w[s, k].X > 1e-6:
                        deliveries[str(s)] = round(w[s, k].X, 4)
                first_echelon_routes.append({
                    'vehicle': k,
                    'route': route,
                    'deliveries': deliveries
                })
        solution['first_echelon_routes'] = first_echelon_routes

        # Extract second echelon routes
        second_echelon_routes = []
        # Find routes starting from each satellite
        for s in sat_ids:
            # Find arcs leaving this satellite
            start_arcs = [(s, j) for j in cust_ids if (s, j) in y and y[s, j].X > 0.5]
            for (_, first_cust) in start_arcs:
                route = [s, first_cust]
                current = first_cust
                while current not in sat_ids:
                    for j in VS_VC:
                        if (current, j) in y and y[current, j].X > 0.5:
                            route.append(j)
                            current = j
                            break
                    else:
                        break
                second_echelon_routes.append({
                    'satellite': s,
                    'route': route
                })
        solution['second_echelon_routes'] = second_echelon_routes

        # Extract group-satellite assignments from actual routes.
        # NOTE: The v and z variables may be over-assigned (all 1s) because
        # the formulation lacks a constraint sum_s z[i,s] = 1, and z/v do not
        # appear in the objective.  Instead, derive assignments from the
        # second-echelon routes: each customer is assigned to the satellite
        # that starts its route (constraint 13 + 14 ensure this).
        cust_to_satellite = {}
        for route_info in second_echelon_routes:
            sat = route_info['satellite']
            for node in route_info['route']:
                if node in cust_ids:
                    cust_to_satellite[node] = sat
        group_assignments = {}
        for g in groups:
            h = g['group_id']
            for cid in g['customer_ids']:
                if cid in cust_to_satellite:
                    group_assignments[str(h)] = cust_to_satellite[cid]
                    break
        solution['group_assignments'] = group_assignments

    else:
        solution['objective_value'] = None
        solution['status'] = 'infeasible'
        solution['runtime'] = model.Runtime

    # Write solution to file
    with open(solution_path, 'w') as f_out:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f_out, indent=2)

    print(f"Solution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Status: {solution['status']}")
    else:
        print("No feasible solution found.")

    return solution


def reconstruct_route(arcs, depot_id):
    """Reconstruct a route from a list of arcs starting and ending at depot."""
    if not arcs:
        return []

    adj = {}
    for (i, j) in arcs:
        adj[i] = j

    route = [depot_id]
    current = depot_id
    visited = set()
    while current in adj and adj[current] not in visited:
        nxt = adj[current]
        route.append(nxt)
        visited.add(nxt)
        current = nxt
        if current == depot_id:
            break

    return route


def main():
    parser = argparse.ArgumentParser(
        description="Solve the 2E-CVRPGC using Gurobi (MILP formulation)"
    )
    parser.add_argument(
        '--instance_path', type=str, required=True,
        help='Path to the JSON instance file'
    )
    parser.add_argument(
        '--solution_path', type=str, required=True,
        help='Path to write the solution JSON file'
    )
    parser.add_argument(
        '--time_limit', type=int, required=True,
        help='Maximum solver runtime in seconds'
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_2e_cvrpgc(args.instance_path, args.solution_path, args.time_limit)


if __name__ == '__main__':
    main()
