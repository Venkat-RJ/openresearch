"""
Gurobi implementation of the Dial-a-Ride Problem (DARP)
Based on: Cordeau (2006), "A Branch-and-Cut Algorithm for the Dial-a-Ride Problem",
Operations Research 54(3), pp. 573-586.

Uses the aggregate formulation (constraints 17-24) as described in the paper.
"""

import argparse
import json
import math
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


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_instance(instance_path):
    """Load a DARP instance from JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(instance_path, solution_path, time_limit):
    """Build and solve the DARP model using Gurobi."""
    data = load_instance(instance_path)

    n = data["num_users"]  # number of users
    K_size = data["num_vehicles"]
    Q = data["vehicle_capacity"]
    L = data["maximum_ride_time"]
    T_max = data["maximum_route_duration"]

    nodes = data["nodes"]
    num_nodes = data["num_nodes"]  # = 2n + 2

    # Node indices:
    # 0: origin depot
    # 1..n: pickup nodes (P)
    # n+1..2n: dropoff nodes (D)
    # 2n+1: destination depot
    origin_depot = 0
    dest_depot = 2 * n + 1

    P = list(range(1, n + 1))
    D = list(range(n + 1, 2 * n + 1))
    N = list(range(num_nodes))  # all nodes including depots
    K = list(range(K_size))

    # Extract node data
    x_coord = {}
    y_coord = {}
    e = {}  # earliest time
    l = {}  # latest time
    d = {}  # service duration
    q = {}  # load

    for node in nodes:
        nid = node["node_id"]
        x_coord[nid] = node["x"]
        y_coord[nid] = node["y"]
        e[nid] = node["earliest_time"]
        l[nid] = node["latest_time"]
        d[nid] = node["service_duration"]
        q[nid] = node["load"]

    # Compute travel times/costs (Euclidean distance)
    t = {}
    c = {}
    for i in N:
        for j in N:
            if i != j:
                dist = euclidean_distance(x_coord[i], y_coord[i], x_coord[j], y_coord[j])
                t[i, j] = dist
                c[i, j] = dist

    # =========================================================================
    # ARC ELIMINATION (Section 5.1.2)
    # =========================================================================
    # Build the set of feasible arcs A
    A = []
    for i in N:
        for j in N:
            if i == j:
                continue
            # Remove arcs (0, n+i) for i in P: depot directly to dropoff
            if i == origin_depot and j in D:
                continue
            # Remove arcs (i, 2n+1) for i in P: pickup directly to dest depot
            if i in P and j == dest_depot:
                continue
            # Remove arcs (n+i, i) for i in P: dropoff back to own pickup
            if i in D and j in P and i == j + n:
                continue
            # Remove arc (i,j) if e_i + d_i + t_ij > l_j
            if e[i] + d[i] + t[i, j] > l[j] + 1e-9:
                continue
            # Remove arcs between depot nodes that don't make sense
            if i == dest_depot:
                continue  # no arcs out of destination depot
            if j == origin_depot:
                continue  # no arcs into origin depot
            A.append((i, j))

    A_set = set(A)

    # =========================================================================
    # BUILD GUROBI MODEL
    # =========================================================================
    model = gp.Model("DARP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision Variables ---
    # x[k,i,j]: binary, vehicle k traverses arc (i,j)
    x = {}
    for k in K:
        for (i, j) in A:
            x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # B_i: aggregate time variable for non-depot nodes
    B = {}
    for i in P + D:
        B[i] = model.addVar(lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS, name=f"B_{i}")

    # B_k_0, B_k_{2n+1}: per-vehicle time at depots
    B_depot_start = {}
    B_depot_end = {}
    for k in K:
        B_depot_start[k] = model.addVar(lb=e[origin_depot], ub=l[origin_depot],
                                         vtype=GRB.CONTINUOUS, name=f"B_{k}_0")
        B_depot_end[k] = model.addVar(lb=e[dest_depot], ub=l[dest_depot],
                                       vtype=GRB.CONTINUOUS, name=f"B_{k}_{dest_depot}")

    # Q_i: aggregate load for non-depot nodes (homogeneous fleet)
    Qvar = {}
    for i in P + D:
        lb_q = max(0, q[i])
        ub_q = min(Q, Q + q[i])
        Qvar[i] = model.addVar(lb=lb_q, ub=ub_q, vtype=GRB.CONTINUOUS, name=f"Q_{i}")

    # Q_k_0, Q_k_{2n+1}: per-vehicle load at depots
    Q_depot_start = {}
    Q_depot_end = {}
    for k in K:
        Q_depot_start[k] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS,
                                         name=f"Q_{k}_0")
        Q_depot_end[k] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS,
                                       name=f"Q_{k}_{dest_depot}")

    # L_i: aggregate ride time for users
    L_var = {}
    for i in P:
        t_direct = t[i, n + i]
        L_var[i] = model.addVar(lb=t_direct, ub=L, vtype=GRB.CONTINUOUS,
                                name=f"L_{i}")

    model.update()

    # --- Objective (1): minimize total routing cost ---
    model.setObjective(
        gp.quicksum(c[i, j] * x[k, i, j]
                    for k in K for (i, j) in A if (k, i, j) in x),
        GRB.MINIMIZE
    )

    # --- Constraint (2): each request served exactly once ---
    for i in P:
        model.addConstr(
            gp.quicksum(x[k, i, j] for k in K for j in N
                        if (i, j) in A_set and (k, i, j) in x) == 1,
            name=f"serve_{i}"
        )

    # --- Constraint (3): same vehicle for pickup and dropoff ---
    for i in P:
        for k in K:
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in N if (i, j) in A_set and (k, i, j) in x) -
                gp.quicksum(x[k, n + i, j] for j in N if (n + i, j) in A_set and (k, n + i, j) in x) == 0,
                name=f"pair_{i}_{k}"
            )

    # --- Constraint (4): each vehicle starts at origin depot ---
    for k in K:
        model.addConstr(
            gp.quicksum(x[k, origin_depot, j] for j in N
                        if (origin_depot, j) in A_set and (k, origin_depot, j) in x) == 1,
            name=f"start_{k}"
        )

    # --- Constraint (5): flow conservation at pickup and dropoff nodes ---
    for i in P + D:
        for k in K:
            model.addConstr(
                gp.quicksum(x[k, j, i] for j in N if (j, i) in A_set and (k, j, i) in x) -
                gp.quicksum(x[k, i, j] for j in N if (i, j) in A_set and (k, i, j) in x) == 0,
                name=f"flow_{i}_{k}"
            )

    # --- Constraint (6): each vehicle ends at destination depot ---
    for k in K:
        model.addConstr(
            gp.quicksum(x[k, i, dest_depot] for i in N
                        if (i, dest_depot) in A_set and (k, i, dest_depot) in x) == 1,
            name=f"end_{k}"
        )

    # --- Constraint (17): time from depot to non-depot node ---
    # B_j >= (B_k_0 + d_0 + t_{0,j}) * x^k_{0,j}
    # Linearized: B_j >= B_k_0 + d[0] + t[0,j] - M*(1 - x^k_{0,j})
    for j in P + D:
        for k in K:
            if (origin_depot, j) in A_set and (k, origin_depot, j) in x:
                M_val = max(0, l[origin_depot] + d[origin_depot] + t[origin_depot, j] - e[j])
                model.addConstr(
                    B[j] >= B_depot_start[k] + d[origin_depot] + t[origin_depot, j]
                    - M_val * (1 - x[k, origin_depot, j]),
                    name=f"time_depot_to_{j}_{k}"
                )

    # --- Constraint (18): time consistency between non-depot nodes ---
    # B_j >= (B_i + d_i + t_{ij}) * sum_k x^k_{ij}
    # Linearized: B_j >= B_i + d_i + t_{ij} - M_{ij}*(1 - sum_k x^k_{ij})
    for i in P + D:
        for j in P + D:
            if i == j:
                continue
            if (i, j) not in A_set:
                continue
            sum_x = gp.quicksum(x[k, i, j] for k in K if (k, i, j) in x)
            M_val = max(0, l[i] + d[i] + t[i, j] - e[j])
            if M_val > 0:
                model.addConstr(
                    B[j] >= B[i] + d[i] + t[i, j] - M_val * (1 - sum_x),
                    name=f"time_{i}_{j}"
                )

    # --- Constraint (19): time from non-depot node to dest depot ---
    # B^k_{2n+1} >= (B_i + d_i + t_{i,2n+1}) * x^k_{i,2n+1}
    # Linearized: B^k_{2n+1} >= B_i + d_i + t_{i,2n+1} - M*(1 - x^k_{i,2n+1})
    for i in P + D:
        for k in K:
            if (i, dest_depot) in A_set and (k, i, dest_depot) in x:
                M_val = max(0, l[i] + d[i] + t[i, dest_depot] - e[dest_depot])
                model.addConstr(
                    B_depot_end[k] >= B[i] + d[i] + t[i, dest_depot]
                    - M_val * (1 - x[k, i, dest_depot]),
                    name=f"time_to_depot_{i}_{k}"
                )

    # --- Constraint (20): ride time definition ---
    # L_i = B_{n+i} - (B_i + d_i) for i in P
    for i in P:
        model.addConstr(L_var[i] == B[n + i] - B[i] - d[i],
                        name=f"ridetime_{i}")

    # --- Constraint (10): maximum route duration ---
    # B^k_{2n+1} - B^k_0 <= T_k
    for k in K:
        model.addConstr(B_depot_end[k] - B_depot_start[k] <= T_max,
                        name=f"duration_{k}")

    # --- Constraint (21): load from depot to non-depot node ---
    # Q_j >= (Q^k_0 + q_j) * x^k_{0,j}
    # Since Q^k_0 = 0: Q_j >= q_j * x^k_{0,j}
    # Linearized: Q_j >= q_j - W*(1 - x^k_{0,j})
    for j in P + D:
        for k in K:
            if (origin_depot, j) in A_set and (k, origin_depot, j) in x:
                W_val = min(Q, Q + q[origin_depot])  # = Q since q_0 = 0
                model.addConstr(
                    Qvar[j] >= q[j] - W_val * (1 - x[k, origin_depot, j]),
                    name=f"load_depot_to_{j}_{k}"
                )

    # --- Constraint (24): lifted load constraints between non-depot nodes ---
    # Q_j >= Q_i + q_j - W_{ij}*(1 - sum_k x^k_{ij}) + (W_{ij} - q_i - q_j)*sum_k x^k_{ji}
    for i in P + D:
        for j in P + D:
            if i == j:
                continue
            if (i, j) not in A_set:
                continue
            W_ij = min(Q, Q + q[i])
            sum_x_ij = gp.quicksum(x[k, i, j] for k in K if (k, i, j) in x)
            sum_x_ji = gp.LinExpr(0)
            if (j, i) in A_set:
                sum_x_ji = gp.quicksum(x[k, j, i] for k in K if (k, j, i) in x)
            model.addConstr(
                Qvar[j] >= Qvar[i] + q[j] - W_ij * (1 - sum_x_ij)
                + (W_ij - q[i] - q[j]) * sum_x_ji,
                name=f"load_{i}_{j}"
            )

    # --- Constraint (23): load from non-depot node to dest depot ---
    # Q^k_{2n+1} >= (Q_i + q_{2n+1}) * x^k_{i,2n+1}
    # Since q_{2n+1}=0 and Q^k_{2n+1}=0, this is always satisfied when load
    # must be 0 at dest depot. We add it for correctness.
    # Actually Q_depot_end is fixed at 0, so this constrains Q_i to be <= 0 when
    # x^k_{i,2n+1} = 1, but Q_i >= 0 so Q_i = 0 at the last stop before depot.
    # This is automatically handled by the load bounds and flow.

    # =========================================================================
    # SOLVE
    # =========================================================================
    model.optimize()

    # =========================================================================
    # EXTRACT SOLUTION
    # =========================================================================
    result = {}
    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["status"] = model.Status
        result["status_description"] = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(model.Status, f"STATUS_{model.Status}")
        result["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None

        # Extract routes
        routes = {}
        for k in K:
            route = []
            current = origin_depot
            visited = set()
            while current != dest_depot and current not in visited:
                visited.add(current)
                route.append(current)
                found_next = False
                for j in N:
                    if (current, j) in A_set and (k, current, j) in x:
                        if x[k, current, j].X > 0.5:
                            current = j
                            found_next = True
                            break
                if not found_next:
                    break
            route.append(dest_depot)
            routes[k] = route

        result["routes"] = {str(k): routes[k] for k in K}

        # Extract service times
        service_times = {}
        for i in P + D:
            service_times[str(i)] = B[i].X
        for k in K:
            service_times[f"depot_start_{k}"] = B_depot_start[k].X
            service_times[f"depot_end_{k}"] = B_depot_end[k].X
        result["service_times"] = service_times

        # Extract ride times
        ride_times = {}
        for i in P:
            ride_times[str(i)] = L_var[i].X
        result["ride_times"] = ride_times

    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["status_description"] = "NO_SOLUTION_FOUND"

    result["solve_time_seconds"] = model.Runtime
    result["num_variables"] = model.NumVars
    result["num_constraints"] = model.NumConstrs

    # Write solution
    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']:.2f}")
    print(f"Solver status: {result['status_description']}")
    print(f"Solve time: {result['solve_time_seconds']:.2f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for the Dial-a-Ride Problem (DARP) - Cordeau (2006)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the solution JSON file will be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    build_and_solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
