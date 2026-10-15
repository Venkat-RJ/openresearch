"""
Gurobi implementation of the PDPTWL (Pickup and Delivery Problem with Time Windows
and LIFO Loading) compact three-index formulation.

Based on: Cherkesly, Desaulniers, and Laporte (2015), Transportation Science.
Formulation: Constraints (1)-(13) from the paper.

Assumptions (inferred):
  - Travel costs c_{ij} = Euclidean distance (NOT SPECIFIED IN PAPER whether rounded).
    We use the costs provided in the instance JSON directly.
  - Travel times t_{ij} provided in the instance JSON directly.
  - Vehicle fixed cost is included in c_{0,j} arcs (already in instance data).
  - Big-M linearization used for constraints (8) and (10).

Full LIFO is enforced by exact lazy separation of every incumbent route that
contains a pickup/pickup/delivery pattern inconsistent with stack order.
"""

import argparse
import json
import math
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
    """Load problem instance from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def lifo_lazy_callback(model, where):
    """Separate a minimal infeasible path for every observed LIFO violation."""
    if where != GRB.Callback.MIPSOL:
        return
    values = model.cbGetSolution(model._lifo_x)
    for vehicle in model._lifo_vehicles:
        successor = {
            i: j
            for (i, j) in model._lifo_arcs
            if values[vehicle, i, j] > 0.5
        }
        sequence = [model._lifo_origin]
        while sequence[-1] in successor and len(sequence) <= model._lifo_node_count:
            sequence.append(successor[sequence[-1]])
            if sequence[-1] == model._lifo_destination:
                break
        stack = []
        pickup_position = {}
        for position, node in enumerate(sequence):
            if 1 <= node <= model._lifo_request_count:
                stack.append(node)
                pickup_position[node] = position
                continue
            if not (model._lifo_request_count < node <= 2 * model._lifo_request_count):
                continue
            request = node - model._lifo_request_count
            if request not in stack:
                continue
            if stack[-1] != request:
                first = pickup_position[request]
                path = list(zip(sequence[first:position], sequence[first + 1:position + 1]))
                model.cbLazy(
                    gp.quicksum(model._lifo_x[vehicle, i, j] for i, j in path)
                    <= len(path) - 1
                )
            stack.remove(request)


def build_model(data, time_limit):
    """Build the PDPTWL compact formulation Gurobi model."""

    n = data["n_requests"]
    n_nodes = data["n_nodes"]  # 2n + 2
    Q = data["vehicle_capacity"]
    planning_horizon = data["planning_horizon"]

    # Build node data
    nodes = {}
    for node_data in data["nodes"]:
        nid = node_data["node_id"]
        nodes[nid] = node_data

    P = list(range(1, n + 1))           # Pickup nodes
    D = list(range(n + 1, 2 * n + 1))   # Delivery nodes
    depot_o = 0
    depot_d = 2 * n + 1
    N = list(range(n_nodes))             # All nodes

    # Build arc set from instance data
    arc_set = set()
    arc_cost = {}
    arc_time = {}
    for arc in data["arcs"]:
        i, j = arc["from"], arc["to"]
        arc_set.add((i, j))
        arc_cost[(i, j)] = arc["cost"]
        arc_time[(i, j)] = arc["travel_time"]

    A = list(arc_set)

    # Determine number of vehicles K (upper bound: n vehicles)
    # The paper says K is unrestricted; we use n as an upper bound
    K_size = n
    K = list(range(K_size))

    # Big-M values for linearization
    M_time = planning_horizon[1] + max(
        node["service_time"] for node in data["nodes"]
    ) + max(arc["travel_time"] for arc in data["arcs"])
    M_load = Q

    # --- Create model ---
    model = gp.Model("PDPTWL")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    model.setParam("LazyConstraints", 1)

    # --- Decision variables ---
    # x[k,i,j] binary: vehicle k uses arc (i,j)
    x = {}
    for k in K:
        for (i, j) in A:
            x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # T[k,i] continuous: service start time at node i for vehicle k
    T = {}
    for k in K:
        for i in N:
            lb = nodes[i]["tw_early"]
            ub = nodes[i]["tw_late"]
            T[k, i] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS,
                                   name=f"T_{k}_{i}")

    # Q_var[k,i] continuous: load of vehicle k upon leaving node i
    Q_var = {}
    for k in K:
        for i in N:
            q_i = nodes[i]["load"]
            lb = max(0, q_i)
            ub = min(Q, Q + q_i)
            Q_var[k, i] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS,
                                       name=f"Q_{k}_{i}")

    model.update()

    # --- Objective (1): minimize total cost ---
    model.setObjective(
        gp.quicksum(arc_cost[(i, j)] * x[k, i, j]
                    for k in K for (i, j) in A),
        GRB.MINIMIZE
    )

    # --- Constraints ---

    # (2) Each pickup visited exactly once
    for i in P:
        outgoing = [(i, j) for (ii, j) in A if ii == i]
        model.addConstr(
            gp.quicksum(x[k, i, j] for k in K for (_, j) in outgoing) == 1,
            name=f"visit_{i}"
        )

    # (3) Pairing: if vehicle k visits pickup i, it must visit delivery n+i
    for i in P:
        ni = n + i  # delivery node for request i
        for k in K:
            out_pickup = [(i, j) for (ii, j) in A if ii == i]
            out_delivery = [(ni, j) for (ii, j) in A if ii == ni]
            model.addConstr(
                gp.quicksum(x[k, i, j] for (_, j) in out_pickup)
                - gp.quicksum(x[k, ni, j] for (_, j) in out_delivery) == 0,
                name=f"pair_{i}_{k}"
            )

    # (4) Each vehicle leaves origin depot exactly once
    out_depot = [(depot_o, j) for (ii, j) in A if ii == depot_o]
    for k in K:
        model.addConstr(
            gp.quicksum(x[k, depot_o, j] for (_, j) in out_depot) == 1,
            name=f"depot_out_{k}"
        )

    # (5) Flow conservation at pickup and delivery nodes
    for i in P + D:
        in_arcs = [(ii, i) for (ii, jj) in A if jj == i]
        out_arcs = [(i, j) for (ii, j) in A if ii == i]
        for k in K:
            model.addConstr(
                gp.quicksum(x[k, ii, i] for (ii, _) in in_arcs)
                - gp.quicksum(x[k, i, j] for (_, j) in out_arcs) == 0,
                name=f"flow_{i}_{k}"
            )

    # (6) Each vehicle arrives at destination depot exactly once
    in_depot = [(i, depot_d) for (i, jj) in A if jj == depot_d]
    for k in K:
        model.addConstr(
            gp.quicksum(x[k, i, depot_d] for (i, _) in in_depot) == 1,
            name=f"depot_in_{k}"
        )

    # (8) Load constraints (linearized with big-M)
    # Q_var[k,j] >= Q_var[k,i] + q_j - M*(1 - x[k,i,j])
    for k in K:
        for (i, j) in A:
            q_j = nodes[j]["load"]
            model.addConstr(
                Q_var[k, j] >= Q_var[k, i] + q_j - M_load * (1 - x[k, i, j]),
                name=f"load_{k}_{i}_{j}"
            )

    # (9) Load bounds are set via variable bounds already

    # (10) Time constraints (linearized with big-M)
    # T[k,j] >= T[k,i] + s_i + t_{ij} - M*(1 - x[k,i,j])
    for k in K:
        for (i, j) in A:
            s_i = nodes[i]["service_time"]
            t_ij = arc_time[(i, j)]
            model.addConstr(
                T[k, j] >= T[k, i] + s_i + t_ij - M_time * (1 - x[k, i, j]),
                name=f"time_{k}_{i}_{j}"
            )

    # (11) Time window bounds are set via variable bounds already

    # (12) Precedence: T[k,i] + s_i + t_{i,n+i} <= T[k,n+i]
    for i in P:
        ni = n + i
        s_i = nodes[i]["service_time"]
        if (i, ni) in arc_time:
            t_i_ni = arc_time[(i, ni)]
        else:
            # Compute Euclidean distance if arc not in set
            # INFERRED ASSUMPTION: direct travel time between pickup and delivery
            xi, yi = nodes[i]["x"], nodes[i]["y"]
            xni, yni = nodes[ni]["x"], nodes[ni]["y"]
            t_i_ni = math.sqrt((xi - xni) ** 2 + (yi - yni) ** 2)
        for k in K:
            model.addConstr(
                T[k, i] + s_i + t_i_ni <= T[k, ni],
                name=f"prec_{i}_{k}"
            )

    # --- LIFO constraints (7) ---
    # We add LIFO constraints via infeasible path inequalities.
    # For a compact formulation, we enumerate short LIFO-violating patterns.
    # A LIFO violation occurs when pickup i is visited before pickup j,
    # but delivery n+j is visited before delivery n+i (i.e., j is delivered
    # before i, violating the "last picked up, first delivered" rule).
    #
    # For each pair (i, j) in P x P with i != j, the LIFO constraint says:
    # If vehicle k picks up i then j (in that order), it must deliver j before i.
    #
    # We enforce: for each vehicle k, for each pair i,j in P:
    #   x[k,i,j] = 1 implies n+j must be delivered before n+i
    #   This is implicitly handled by the arc structure (no arcs from pickup i
    #   to delivery n+j for j != i), but we add explicit LIFO ordering constraints.
    #
    # The arc set already excludes (i, n+j) for i in P, j in P, j != i,
    # which handles direct pickup-to-wrong-delivery violations.
    # For indirect violations, we add constraints for delivery ordering:
    # If both i and j are onboard (picked up), then n+j must come before n+i
    # if j was picked up after i. We model this with time-based constraints:
    # T[k,n+j] <= T[k,n+i] + M*(2 - x[k,i,j] - (sum of x indicating j picked after i))
    #
    # INFERRED ASSUMPTION: The arc structure already restricts most LIFO violations.
    # For the compact formulation, the paper notes that the explicit LIFO
    # constraints (7) involve exponentially many subsets. We add pairwise
    # LIFO constraints as a practical compromise for the compact model.
    for i in P:
        for j in P:
            if i == j:
                continue
            ni = n + i
            nj = n + j
            # If arc (i,j) exists (pickup i then pickup j), then delivery j
            # must happen before delivery i: T[k,nj] <= T[k,ni]
            if (i, j) in arc_set:
                for k in K:
                    model.addConstr(
                        T[k, nj] <= T[k, ni] + M_time * (1 - x[k, i, j]),
                        name=f"lifo_{i}_{j}_{k}"
                    )

    # --- Symmetry breaking ---
    # Break symmetry between identical vehicles by ordering their first pickup
    # INFERRED ASSUMPTION: Standard symmetry-breaking technique
    for k in range(len(K) - 1):
        # Vehicle k's departure time from depot <= vehicle k+1's departure time
        model.addConstr(
            T[k, depot_o] <= T[k + 1, depot_o],
            name=f"sym_{k}"
        )

    model.update()
    model._lifo_x = gp.tupledict(x)
    model._lifo_arcs = A
    model._lifo_vehicles = K
    model._lifo_origin = depot_o
    model._lifo_destination = depot_d
    model._lifo_request_count = n
    model._lifo_node_count = len(N)
    return model, x, T, Q_var, A, K, N, P, D, n, depot_o, depot_d, nodes


def extract_solution(model, x, T, Q_var, A, K, N, P, D, n, depot_o, depot_d, nodes):
    """Extract solution from solved model."""
    if model.SolCount == 0:
        return None

    obj_val = model.ObjVal

    routes = []
    for k in K:
        route_arcs = []
        for (i, j) in A:
            if x[k, i, j].X > 0.5:
                route_arcs.append((i, j))

        if not route_arcs:
            continue

        # Build route sequence from arcs
        # Check if this is a non-empty route (not just depot->depot path)
        visits_customer = any(i in P or i in D or j in P or j in D
                              for (i, j) in route_arcs)
        if not visits_customer:
            continue

        # Build adjacency
        adj = {}
        for (i, j) in route_arcs:
            adj[i] = j

        # Trace route from depot_o
        route = [depot_o]
        current = depot_o
        visited_count = 0
        while current in adj and visited_count < len(N):
            nxt = adj[current]
            route.append(nxt)
            current = nxt
            visited_count += 1
            if current == depot_d:
                break

        # Get timing and load info
        route_detail = []
        for node_id in route:
            detail = {
                "node_id": node_id,
                "arrival_time": round(T[k, node_id].X, 2),
                "load_after": round(Q_var[k, node_id].X, 2)
            }
            route_detail.append(detail)

        route_cost = sum(
            model.getAttr("Obj", [x[k, i, j]])[0] * x[k, i, j].X
            for (i, j) in route_arcs
        )

        routes.append({
            "vehicle_id": k,
            "route_sequence": route,
            "route_details": route_detail,
        })

    solution = {
        "objective_value": round(obj_val, 2),
        "num_vehicles": len(routes),
        "status": model.Status,
        "status_description": {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "time_limit",
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        }.get(model.Status, f"other_{model.Status}"),
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "best_bound": model.ObjBound if model.SolCount > 0 else None,
        "lifo_enforcement": "exact_lazy_infeasible_path_separation",
        "solve_time": model.Runtime,
        "routes": routes,
    }
    return solution


def main():
    parser = argparse.ArgumentParser(
        description="PDPTWL Gurobi compact formulation solver"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from: {args.instance_path}")
    data = load_instance(args.instance_path)

    print(f"Building model for {data['n_requests']} requests...")
    model, x, T, Q_var, A, K, N, P, D, n, depot_o, depot_d, nodes = \
        build_model(data, args.time_limit)

    print(f"Solving with time limit = {args.time_limit}s...")
    model.optimize(lifo_lazy_callback)

    print(f"Status: {model.Status}, Solutions found: {model.SolCount}")

    if model.SolCount > 0:
        solution = extract_solution(
            model, x, T, Q_var, A, K, N, P, D, n, depot_o, depot_d, nodes
        )
        print(f"Objective value: {solution['objective_value']}")
        print(f"Number of vehicles: {solution['num_vehicles']}")
    else:
        solution = {
            "objective_value": None,
            "num_vehicles": None,
            "status": model.Status,
            "status_description": "no_solution_found",
            "mip_gap": None,
            "best_bound": None,
            "lifo_enforcement": "exact_lazy_infeasible_path_separation",
            "solve_time": model.Runtime,
            "routes": [],
        }
        print("No feasible solution found.")

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)
    print(f"Solution saved to: {args.solution_path}")


if __name__ == "__main__":
    main()
