#!/usr/bin/env python3
"""
Gurobi implementation of the VRDAP (Vehicle Routing with Demand Allocation Problem).
MIP formulation from Reihaneh & Ghoniem (2019), Appendix A, Equations (4a)-(4l).

This implements the compact (base) MIP formulation directly solvable by Gurobi.
"""

import json
import argparse
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
    """Load the VRDAP instance from a JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_vrdap(data, time_limit):
    """
    Build and solve the VRDAP MIP formulation (4a)-(4l).

    Sets:
      S: delivery sites (node IDs 1..|S|)
      K: customers (IDs 1..|K|)
      V: vehicle tours (IDs 1..|V|)
      N = S ∪ {0} (depot is node 0)
      E = {(i,j) : i,j ∈ N, i ≠ j}

    Parameters read from instance:
      c[i][j]:  routing cost matrix (|N| x |N|), indexed by node IDs
      f[k][s]:  assignment cost matrix (|K| x |S|), 0-indexed
      d[k]:     demand of customer k
      Q:        vehicle capacity
    """
    num_sites = data['num_delivery_sites']
    num_customers = data['num_customers']
    Q = data['vehicle_capacity']
    num_vehicles = data['num_vehicles']

    customers = data['customers']
    c = data['routing_cost_matrix']       # c[i][j], i,j in {0,...,|S|}
    f_cost = data['assignment_cost_matrix']  # f[k_idx][s_idx], 0-indexed

    # --- Index sets ---
    S = list(range(1, num_sites + 1))           # delivery site node IDs
    K = list(range(1, num_customers + 1))       # customer IDs
    V = list(range(1, num_vehicles + 1))        # vehicle (tour) IDs
    N = [0] + S                                 # all nodes

    # Customer demands indexed by customer ID
    d = {}
    for cust in customers:
        d[cust['id']] = cust['demand']

    # --- Build Gurobi model ---
    model = gp.Model("VRDAP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # ----- Decision Variables -----

    # s[i,k] ∈ {0,1}: customer k assigned to delivery site i
    s = {}
    for i in S:
        for k in K:
            s[i, k] = model.addVar(vtype=GRB.BINARY, name=f"s_{i}_{k}")

    # e[v,i,j] ∈ {0,1}: arc (i,j) included in vehicle tour v
    e = {}
    for v_idx in V:
        for i in N:
            for j in N:
                if i != j:
                    e[v_idx, i, j] = model.addVar(vtype=GRB.BINARY,
                                                   name=f"e_{v_idx}_{i}_{j}")

    # theta[v,i,k] ≥ 0: site i visited by tour v AND customer k assigned to site i
    # (Declared continuous; automatically binary due to constraints 4b, 4k and
    #  binariness of s, e — see paper's Reproduction-Critical Note 1.)
    theta = {}
    for v_idx in V:
        for i in S:
            for k in K:
                theta[v_idx, i, k] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0,
                    name=f"theta_{v_idx}_{i}_{k}")

    # q[v,i] ≥ 0: cumulative deliveries upon serving site i in tour v
    q = {}
    for v_idx in V:
        for i in S:
            q[v_idx, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                        name=f"q_{v_idx}_{i}")

    model.update()

    # ----- Objective (4a) -----
    # min Σ_v Σ_(i,j)∈E c_ij e^v_ij  +  Σ_i∈S Σ_k∈K f_ik s_ik
    obj = gp.LinExpr()
    for v_idx in V:
        for i in N:
            for j in N:
                if i != j:
                    obj += c[i][j] * e[v_idx, i, j]
    for i in S:
        for k in K:
            # f_cost is 0-indexed: row = customer_id - 1, col = site_id - 1
            obj += f_cost[k - 1][i - 1] * s[i, k]
    model.setObjective(obj, GRB.MINIMIZE)

    # ----- Constraints -----

    # (4b) Each delivery site visited by at most one vehicle tour
    # Σ_v Σ_{i∈N\{j}} e^v_ij ≤ 1,  ∀ j ∈ S
    for j in S:
        model.addConstr(
            gp.quicksum(e[v_idx, i, j]
                        for v_idx in V for i in N if i != j) <= 1,
            name=f"c4b_{j}")

    # (4c) Each vehicle tour has at most one arc leaving the central depot
    # Σ_{j∈S} e^v_{0j} ≤ 1,  ∀ v ∈ V
    for v_idx in V:
        model.addConstr(
            gp.quicksum(e[v_idx, 0, j] for j in S) <= 1,
            name=f"c4c_{v_idx}")

    # (4d) Flow balance for depot and every delivery site
    # Σ_{i∈N\{j}} e^v_ij − Σ_{i∈N\{j}} e^v_ji = 0,  ∀ v ∈ V, j ∈ N
    for v_idx in V:
        for j in N:
            model.addConstr(
                gp.quicksum(e[v_idx, i, j] for i in N if i != j)
                - gp.quicksum(e[v_idx, j, i] for i in N if i != j) == 0,
                name=f"c4d_{v_idx}_{j}")

    # (4e) Subtour elimination (lifted Miller-Tucker-Zemlin) + cumulative delivery
    # q^v_j ≥ q^v_i + Σ_k d_k s_{jk} − 2Q(1−e^v_ij) + Q e^v_ji,
    #   ∀ v ∈ V, i,j ∈ S, i ≠ j
    for v_idx in V:
        for i in S:
            for j in S:
                if i != j:
                    model.addConstr(
                        q[v_idx, j] >= q[v_idx, i]
                        + gp.quicksum(d[k] * s[j, k] for k in K)
                        - 2 * Q * (1 - e[v_idx, i, j])
                        + Q * e[v_idx, j, i],
                        name=f"c4e_{v_idx}_{i}_{j}")

    # (4f) Linking θ and (s, e) — lower bound
    # θ^v_ik ≥ s_ik + Σ_{j∈N\{i}} e^v_ij − 1
    for v_idx in V:
        for i in S:
            for k in K:
                model.addConstr(
                    theta[v_idx, i, k] >= s[i, k]
                    + gp.quicksum(e[v_idx, i, j] for j in N if j != i) - 1,
                    name=f"c4f_{v_idx}_{i}_{k}")

    # (4g) Linking θ and e — upper bound
    # θ^v_ik ≤ Σ_{j∈N\{i}} e^v_ij
    for v_idx in V:
        for i in S:
            for k in K:
                model.addConstr(
                    theta[v_idx, i, k]
                    <= gp.quicksum(e[v_idx, i, j] for j in N if j != i),
                    name=f"c4g_{v_idx}_{i}_{k}")

    # (4h) Linking θ and s — upper bound
    # θ^v_ik ≤ s_ik
    for v_idx in V:
        for i in S:
            for k in K:
                model.addConstr(
                    theta[v_idx, i, k] <= s[i, k],
                    name=f"c4h_{v_idx}_{i}_{k}")

    # (4i) Upper bound on cumulative delivery at visited site
    # q^v_i ≤ Q Σ_{j∈N\{i}} e^v_ij
    for v_idx in V:
        for i in S:
            model.addConstr(
                q[v_idx, i]
                <= Q * gp.quicksum(e[v_idx, i, j] for j in N if j != i),
                name=f"c4i_{v_idx}_{i}")

    # (4j) Lower bound on cumulative delivery (demand assigned via θ)
    # Σ_k d_k θ^v_ik ≤ q^v_i
    for v_idx in V:
        for i in S:
            model.addConstr(
                gp.quicksum(d[k] * theta[v_idx, i, k] for k in K)
                <= q[v_idx, i],
                name=f"c4j_{v_idx}_{i}")

    # (4k) Every customer assigned to exactly one delivery site and one tour
    # Σ_v Σ_i θ^v_ik = 1,  ∀ k ∈ K
    for k in K:
        model.addConstr(
            gp.quicksum(theta[v_idx, i, k]
                        for v_idx in V for i in S) == 1,
            name=f"c4k_{k}")

    # ----- Solve -----
    model.optimize()

    # ----- Extract solution -----
    solution = {}
    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        solution['status'] = model.Status
        solution['mip_gap'] = model.MIPGap if hasattr(model, 'MIPGap') else None

        # Extract routes per vehicle
        routes = {}
        for v_idx in V:
            arcs = []
            for i in N:
                for j in N:
                    if i != j and e[v_idx, i, j].X > 0.5:
                        arcs.append([i, j])
            if arcs:
                # Reconstruct ordered route from arcs
                route_seq = _reconstruct_route(arcs)
                routes[str(v_idx)] = {"arcs": arcs, "sequence": route_seq}
        solution['routes'] = routes

        # Extract customer-to-site assignments
        assignments = {}
        for k in K:
            for i in S:
                if s[i, k].X > 0.5:
                    assignments[str(k)] = i
                    break
        solution['assignments'] = assignments

        # Extract customer-to-tour assignments via theta
        customer_tours = {}
        for k in K:
            for v_idx in V:
                for i in S:
                    if theta[v_idx, i, k].X > 0.5:
                        customer_tours[str(k)] = {
                            "vehicle": v_idx,
                            "site": i
                        }
                        break
        solution['customer_tours'] = customer_tours
    else:
        solution['objective_value'] = None
        solution['status'] = model.Status

    return solution


def _reconstruct_route(arcs):
    """Reconstruct an ordered node sequence from a list of arcs."""
    if not arcs:
        return []
    adj = {}
    for (i, j) in arcs:
        adj[i] = j
    # Start from depot (node 0)
    route = [0]
    current = 0
    visited = {0}
    while current in adj:
        nxt = adj[current]
        route.append(nxt)
        if nxt in visited:
            break
        visited.add(nxt)
        current = nxt
    return route


def main():
    parser = argparse.ArgumentParser(
        description="VRDAP solver using Gurobi (compact MIP formulation 4a-4l)")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    solution = solve_vrdap(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution['objective_value'] is not None:
        print(f"Objective value: {solution['objective_value']}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == '__main__':
    main()
