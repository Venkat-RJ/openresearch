"""
VRPTW-CNC (Vehicle Routing Problem with Time Windows and Convex Node Costs)
solved using Gurobi as a direct MIP solver.

The paper (Spliet et al., 2020) uses set partitioning formulation (3a-3d) solved
via Branch-Cut-and-Price. For direct Gurobi solving, we use an equivalent compact
arc-based formulation with quadratic inconvenience costs in the objective.

The inconvenience cost for customer i is:
    f_i(t_i) = rho * (t_i - mid_i)^2
where mid_i = (e_i + l_i) / 2 is the midpoint of the time window [e_i, l_i].
"""

import argparse
import json
import sys

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


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def solve_vrptw_cnc(instance, time_limit=3600):
    # ---------------------------------------------------------------
    # Extract instance data
    # ---------------------------------------------------------------
    n = instance["num_customers"]
    K = instance["num_vehicles"]
    Q = instance["vehicle_capacity"]
    rho = instance["inconvenience_cost_function"]["rho"]
    dist = instance["distance_matrix"]  # (n+2) x (n+2), indices 0..n+1

    depot = instance["depot"]
    depot_end = instance["depot_end"]
    customers = instance["customers"]

    depot_id = depot["id"]          # 0
    depot_end_id = depot_end["id"]  # n+1

    # Number of vertices: 0, 1, ..., n, n+1
    num_vertices = n + 2
    assert depot_end_id == n + 1, f"Expected depot_end id = {n+1}, got {depot_end_id}"
    assert len(dist) == num_vertices
    assert len(dist[0]) == num_vertices

    # Customer set N = {1, ..., n} (by their matrix indices, same as id)
    N = [c["id"] for c in customers]
    V = list(range(num_vertices))  # {0, 1, ..., n+1}

    # Time windows: indexed by vertex id
    e = {}
    l = {}
    demand = {}
    service_time = {}

    e[0] = depot["time_window"][0]
    l[0] = depot["time_window"][1]
    demand[0] = 0
    service_time[0] = 0

    e[depot_end_id] = depot_end["time_window"][0]
    l[depot_end_id] = depot_end["time_window"][1]
    demand[depot_end_id] = 0
    service_time[depot_end_id] = 0

    for c in customers:
        cid = c["id"]
        e[cid] = c["time_window"][0]
        l[cid] = c["time_window"][1]
        demand[cid] = c["demand"]
        service_time[cid] = c["service_time"]

    mid = {i: (e[i] + l[i]) / 2.0 for i in N}

    # ---------------------------------------------------------------
    # Build arc set A
    # Arcs from depot 0 to customers and depot_end,
    # arcs between customers, arcs from customers to depot_end.
    # No arcs into depot 0 or out of depot_end.
    # No self-loops.
    # ---------------------------------------------------------------
    vehicles = list(range(K))

    # Big-M for time precedence constraints
    max_l = max(l[i] for i in V)
    max_service = max(service_time[i] for i in V)
    max_dist = max(dist[i][j] for i in V for j in V)
    M = max_l + max_service + max_dist

    # ---------------------------------------------------------------
    # Create Gurobi model
    # ---------------------------------------------------------------
    model = gp.Model("VRPTW_CNC")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # ---------------------------------------------------------------
    # Decision variables
    # ---------------------------------------------------------------
    # x[i,j,k]: binary, vehicle k traverses arc (i,j)
    x = {}
    for k in vehicles:
        for i in V:
            for j in V:
                if i == j:
                    continue
                if j == 0:
                    continue  # no arcs into depot start
                if i == depot_end_id:
                    continue  # no arcs out of depot end
                x[i, j, k] = model.addVar(
                    vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}"
                )

    # t[i]: continuous, service start time at vertex i
    t = {}
    for i in V:
        t[i] = model.addVar(
            lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS, name=f"t_{i}"
        )

    model.update()

    # ---------------------------------------------------------------
    # Objective: minimize routing cost + inconvenience cost
    #
    # routing_cost = sum_{(i,j,k)} c_{ij} * x[i,j,k]
    # inconvenience_cost = sum_{i in N} rho * (t[i] - mid_i)^2
    # ---------------------------------------------------------------
    obj = gp.QuadExpr()

    # Routing cost (linear)
    for (i, j, k), var in x.items():
        obj += dist[i][j] * var

    # Inconvenience cost (quadratic)
    for i in N:
        # rho * (t[i] - mid_i)^2 = rho * (t[i]^2 - 2*mid_i*t[i] + mid_i^2)
        obj += rho * (t[i] * t[i] - 2.0 * mid[i] * t[i] + mid[i] * mid[i])

    model.setObjective(obj, GRB.MINIMIZE)

    # ---------------------------------------------------------------
    # Constraints
    # ---------------------------------------------------------------

    # (1) Each customer visited exactly once
    for i in N:
        model.addConstr(
            gp.quicksum(
                x[i, j, k]
                for k in vehicles
                for j in V
                if (i, j, k) in x
            ) == 1,
            name=f"visit_{i}",
        )

    # (2) Flow conservation at each customer for each vehicle
    for j in N:
        for k in vehicles:
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in V if (i, j, k) in x)
                == gp.quicksum(x[j, i, k] for i in V if (j, i, k) in x),
                name=f"flow_{j}_{k}",
            )

    # (3) Each vehicle leaves depot at most once
    for k in vehicles:
        model.addConstr(
            gp.quicksum(x[0, j, k] for j in V if (0, j, k) in x) <= 1,
            name=f"depot_leave_{k}",
        )

    # (4) Vehicle that leaves depot must return to depot_end
    for k in vehicles:
        model.addConstr(
            gp.quicksum(x[0, j, k] for j in V if (0, j, k) in x)
            == gp.quicksum(
                x[i, depot_end_id, k] for i in V if (i, depot_end_id, k) in x
            ),
            name=f"depot_return_{k}",
        )

    # (5) Time windows are enforced via variable bounds (lb=e[i], ub=l[i])

    # (6) Time precedence (subtour elimination with big-M):
    #     t[i] + service_time[i] + dist[i][j] <= t[j] + M*(1 - x[i,j,k])
    for (i, j, k), var in x.items():
        if j == depot_end_id and i == 0:
            # Depot-to-depot_end: no meaningful precedence needed
            continue
        model.addConstr(
            t[i] + service_time[i] + dist[i][j] - M * (1 - var) <= t[j],
            name=f"time_prec_{i}_{j}_{k}",
        )

    # (7) Vehicle capacity
    for k in vehicles:
        model.addConstr(
            gp.quicksum(
                demand[i] * gp.quicksum(
                    x[i, j, k] for j in V if (i, j, k) in x
                )
                for i in N
            ) <= Q,
            name=f"capacity_{k}",
        )

    # ---------------------------------------------------------------
    # Symmetry breaking: order vehicles by first customer visited
    # Vehicle k is "used" only if vehicle k-1 is used, and among
    # used vehicles, the smallest-index customer on vehicle k-1
    # should be less than on vehicle k. We approximate this by simply
    # requiring: if vehicle k leaves the depot, then vehicle k-1 also
    # leaves the depot.
    # ---------------------------------------------------------------
    for k in vehicles:
        if k == 0:
            continue
        model.addConstr(
            gp.quicksum(x[0, j, k] for j in V if (0, j, k) in x)
            <= gp.quicksum(x[0, j, k - 1] for j in V if (0, j, k - 1) in x),
            name=f"sym_break_{k}",
        )

    # ---------------------------------------------------------------
    # Optimize
    # ---------------------------------------------------------------
    model.optimize()

    if model.SolCount == 0:
        print("No feasible solution found.")
        return {
            "objective_value": None,
            "routes": [],
            "schedule": {},
            "routing_cost": None,
            "inconvenience_cost": None,
            "mip_gap": None,
            "solve_time": model.Runtime,
            "status": model.Status,
        }

    # ---------------------------------------------------------------
    # Extract solution
    # ---------------------------------------------------------------
    sol_x = {}
    for (i, j, k), var in x.items():
        val = var.X
        if val > 0.5:
            sol_x[i, j, k] = 1

    sol_t = {i: t[i].X for i in V}

    # Build routes per vehicle
    routes = []
    for k in vehicles:
        # Check if vehicle k is used
        arcs_k = [(i, j) for (i, j, kk) in sol_x if kk == k]
        if not arcs_k:
            continue

        # Build successor map
        succ = {i: j for (i, j) in arcs_k}
        if 0 not in succ:
            continue

        route = [0]
        current = 0
        visited = set()
        while current in succ and current != depot_end_id:
            nxt = succ[current]
            route.append(nxt)
            if nxt in visited:
                break  # safety: avoid infinite loop
            visited.add(nxt)
            current = nxt

        routes.append(route)

    # Compute cost breakdown
    routing_cost = 0.0
    for (i, j, k) in sol_x:
        routing_cost += dist[i][j]

    inconvenience_cost = 0.0
    for i in N:
        inconvenience_cost += rho * (sol_t[i] - mid[i]) ** 2

    schedule = {i: sol_t[i] for i in N}

    solution = {
        "objective_value": model.ObjVal,
        "routes": routes,
        "schedule": schedule,
        "routing_cost": routing_cost,
        "inconvenience_cost": inconvenience_cost,
        "mip_gap": model.MIPGap if hasattr(model, "MIPGap") else None,
        "solve_time": model.Runtime,
        "status": model.Status,
    }

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve VRPTW-CNC using Gurobi (compact arc-based formulation)"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the instance JSON file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path to write the solution JSON file",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Solver time limit in seconds (default: 3600)",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    print(f"Loaded instance: {instance.get('description', 'unknown')}")
    print(f"  Customers: {instance['num_customers']}")
    print(f"  Vehicles:  {instance['num_vehicles']}")
    print(f"  Capacity:  {instance['vehicle_capacity']}")
    print(f"  Rho:       {instance['inconvenience_cost_function']['rho']}")

    solution = solve_vrptw_cnc(instance, time_limit=args.time_limit)

    has_incumbent = solution.get("objective_value") is not None
    if has_incumbent:
        print(f"\nSolution found:")
        print(f"  Objective value:     {solution['objective_value']:.4f}")
        print(f"  Routing cost:        {solution['routing_cost']:.4f}")
        print(f"  Inconvenience cost:  {solution['inconvenience_cost']:.4f}")
        print(f"  Number of routes:    {len(solution['routes'])}")
        print(f"  MIP gap:             {solution['mip_gap']}")
        print(f"  Solve time:          {solution['solve_time']:.2f}s")
        for idx, route in enumerate(solution["routes"]):
            print(f"  Route {idx + 1}: {route}")
    else:
        print(f"\nNo incumbent — Gurobi status={solution['status']}, "
              f"runtime={solution['solve_time']:.2f}s. Writing empty solution.")

    serializable_solution = {
        "objective_value": solution["objective_value"],
        "routes": solution["routes"],
        "schedule": {str(k): v for k, v in solution["schedule"].items()},
        "routing_cost": solution["routing_cost"],
        "inconvenience_cost": solution["inconvenience_cost"],
        "mip_gap": solution["mip_gap"],
        "solve_time": solution["solve_time"],
        "status": solution["status"],
    }

    with open(args.solution_path, "w") as f:
        serializable_solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(serializable_solution, f, indent=2)

    print(f"\nSolution written to {args.solution_path}")


if __name__ == "__main__":
    main()
