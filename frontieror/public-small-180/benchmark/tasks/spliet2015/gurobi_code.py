"""
Gurobi-based solver for the Time Window Assignment Vehicle Routing Problem (TWAVRP).

Reference: Spliet & Gabor (2014), "The Time Window Assignment Vehicle Routing Problem",
           Transportation Science.

This implements a compact arc-flow MILP formulation equivalent to the route-based
formulation (1)-(6) in the paper. The route-based formulation has an exponential
(infinite for continuous time) number of variables and requires column generation.
Instead, we use arc-flow variables z[i,j,omega] and arrival time variables t[v,omega]
to model the same problem compactly.

ASSUMPTION (INFERRED): The paper's formulation is route-based with an infinite number
of route variables. We use a standard arc-flow VRP formulation that is equivalent for
integer solutions. This is a standard transformation; see e.g., Desrochers et al. (1992).

ASSUMPTION (INFERRED): Service time at each customer is 0. The paper does not specify
a positive service duration, and the test instances use travel time = Euclidean distance
with wide time windows, suggesting zero service time. **NOT SPECIFIED IN PAPER**.

ASSUMPTION (INFERRED): The ending depot n+1 has the same exogenous time window as
the starting depot 0, i.e., [6, 22]. **NOT SPECIFIED IN PAPER** but standard.
"""

import argparse
import json
import math
import sys
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


try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Error: gurobipy is required. Install Gurobi and its Python interface.")
    sys.exit(1)


def load_instance(instance_path):
    """Load instance from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_twavrp(data, time_limit):
    """
    Solve the TWAVRP using a compact arc-flow MILP formulation.

    Sets:
      V = {0, 1, ..., n, n+1} where 0 = start depot, n+1 = end depot, 1..n = customers
      V' = {1, ..., n} = customers
      Omega = set of scenarios

    Decision variables:
      z[i,j,omega] in {0,1}: 1 if arc (i,j) is used in scenario omega
      t[v,omega] >= 0: arrival/service time at node v in scenario omega
      y[v]: start of endogenous time window for customer v in V'

    Objective: min sum_{omega} p_omega * sum_{(i,j)} c_ij * z[i,j,omega]
    """
    n = data["num_customers"]
    customers_data = data["customers"]
    depot = data["depot"]
    scenarios = data["scenarios"]
    Q = data["vehicle_capacity"]
    dist = data["distance_matrix"]

    num_scenarios = len(scenarios)

    # Node indices: 0 = start depot, 1..n = customers, n+1 = end depot
    V = list(range(n + 2))
    V_prime = list(range(1, n + 1))
    depot_start = 0
    depot_end = n + 1

    # Exogenous time windows
    tw = {}
    tw[depot_start] = (depot["exogenous_time_window"][0], depot["exogenous_time_window"][1])
    for c in customers_data:
        tw[c["id"]] = (c["exogenous_time_window"][0], c["exogenous_time_window"][1])
    tw[depot_end] = tw[depot_start]

    # Endogenous time window widths
    w = {}
    for c in customers_data:
        w[c["id"]] = c["endogenous_time_window_width"]

    # Travel costs = travel times = Euclidean distance (per paper Section 4.1)
    c_travel = {}
    t_travel = {}
    for i in V:
        for j in V:
            c_travel[i, j] = dist[i][j]
            t_travel[i, j] = dist[i][j]

    # Demands per scenario
    demands = {}
    for s_idx, sc in enumerate(scenarios):
        for c_idx, cust in enumerate(customers_data):
            demands[cust["id"], s_idx] = sc["demands"][c_idx]
        demands[depot_start, s_idx] = 0
        demands[depot_end, s_idx] = 0

    probs = [sc["probability"] for sc in scenarios]

    # Create arcs: no self-loops, no arcs INTO start depot, no arcs OUT OF end depot,
    # no arc from start depot directly to end depot (would be empty route - not needed
    # since all customers must be visited)
    arcs = []
    for i in V:
        for j in V:
            if i == j:
                continue
            if j == depot_start:
                continue
            if i == depot_end:
                continue
            # Allow depot_start -> depot_end only if there could be empty routes,
            # but since each customer must be visited exactly once, skip it
            if i == depot_start and j == depot_end:
                continue
            arcs.append((i, j))

    # ---- Build Gurobi Model ----
    model = gp.Model("TWAVRP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision Variables

    # z[i,j,omega]: binary, 1 if arc (i,j) used in scenario omega
    z = {}
    for omega in range(num_scenarios):
        for (i, j) in arcs:
            z[i, j, omega] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}_{omega}")

    # t[v, omega]: continuous service/arrival time at node v in scenario omega
    t_var = {}
    for omega in range(num_scenarios):
        for v in V:
            t_var[v, omega] = model.addVar(
                lb=tw[v][0], ub=tw[v][1],
                vtype=GRB.CONTINUOUS, name=f"t_{v}_{omega}")

    # y[v]: start of endogenous time window (first-stage decision)
    y = {}
    for v in V_prime:
        y[v] = model.addVar(
            lb=tw[v][0], ub=tw[v][1] - w[v],
            vtype=GRB.CONTINUOUS, name=f"y_{v}")

    model.update()

    # ---- Objective ----
    model.setObjective(
        gp.quicksum(
            probs[omega] * c_travel[i, j] * z[i, j, omega]
            for omega in range(num_scenarios)
            for (i, j) in arcs
        ),
        GRB.MINIMIZE
    )

    # ---- Constraints per scenario ----
    for omega in range(num_scenarios):

        # (2) Each customer visited exactly once: in-degree = 1
        for v in V_prime:
            model.addConstr(
                gp.quicksum(z[i, v, omega] for i in V if (i, v, omega) in z) == 1,
                name=f"in_{v}_{omega}")

        # Flow conservation: out-degree = in-degree for customers
        for v in V_prime:
            model.addConstr(
                gp.quicksum(z[v, j, omega] for j in V if (v, j, omega) in z) == 1,
                name=f"out_{v}_{omega}")

        # Depot: vehicles leaving start = vehicles arriving at end
        model.addConstr(
            gp.quicksum(z[depot_start, j, omega] for j in V if (depot_start, j, omega) in z) ==
            gp.quicksum(z[i, depot_end, omega] for i in V if (i, depot_end, omega) in z),
            name=f"depot_balance_{omega}")

        # Time precedence (MTZ-like): t[j] >= t[i] + t_ij - M*(1-z[i,j])
        # Big-M: M_ij = max(0, tw[i][1] + t_travel[i,j] - tw[j][0])
        # This is valid because: t[i] <= tw[i][1], t[j] >= tw[j][0]
        # When z=0: t[j] >= t[i] + t_ij - M is trivially satisfied
        for (i, j) in arcs:
            if i == depot_end or j == depot_start:
                continue
            M_ij = max(0, tw[i][1] + t_travel[i, j] - tw[j][0])
            model.addConstr(
                t_var[j, omega] >= t_var[i, omega] + t_travel[i, j] - M_ij * (1 - z[i, j, omega]),
                name=f"time_{i}_{j}_{omega}")

        # Subtour elimination via single-commodity flow
        # u[v, omega] = cumulative load when arriving at customer v
        # u[v] >= d_v, u[v] <= Q
        # u[j] >= u[i] + d[j] - Q*(1 - z[i,j])  for customer-to-customer arcs

    # MTZ-like capacity constraints
    u = {}
    for omega in range(num_scenarios):
        for v in V_prime:
            u[v, omega] = model.addVar(
                lb=demands[v, omega], ub=Q,
                vtype=GRB.CONTINUOUS, name=f"u_{v}_{omega}")
    model.update()

    for omega in range(num_scenarios):
        for (i, j) in arcs:
            if i in V_prime and j in V_prime:
                # u[j] >= u[i] + d[j] - Q*(1 - z[i,j])
                model.addConstr(
                    u[j, omega] >= u[i, omega] + demands[j, omega] - Q * (1 - z[i, j, omega]),
                    name=f"cap_{i}_{j}_{omega}")
            elif i == depot_start and j in V_prime:
                # u[j] >= d[j] (already by lb, but also from depot route start)
                # u[j] <= Q * z[0,j] + Q * (1 - z[0,j]) is trivially Q
                # We only need: when z[0,j]=1, u[j] = d[j] (start of route)
                # Actually u[j] >= d[j] always, and the chain propagation handles the rest
                pass

    # Endogenous time window constraints: linking y and t
    for omega in range(num_scenarios):
        for v in V_prime:
            # (3) t[v,omega] >= y[v]
            model.addConstr(t_var[v, omega] >= y[v], name=f"tw_lb_{v}_{omega}")
            # (4) t[v,omega] <= y[v] + w[v]
            model.addConstr(t_var[v, omega] <= y[v] + w[v], name=f"tw_ub_{v}_{omega}")

    # ---- Solve ----
    model.optimize()

    # ---- Extract Solution ----
    result = {
        "objective_value": None,
        "status": None,
        "time_windows": {},
        "routes": {}
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract time window assignments
        for v in V_prime:
            yv = y[v].X
            result["time_windows"][str(v)] = [round(yv, 4), round(yv + w[v], 4)]

        # Extract routes per scenario
        for omega in range(num_scenarios):
            scenario_routes = []
            # Build adjacency from solution
            next_node = {}
            for (i, j) in arcs:
                if z[i, j, omega].X > 0.5:
                    if i not in next_node:
                        next_node[i] = []
                    next_node[i].append(j)

            # Trace routes from depot_start
            if depot_start in next_node:
                for first_cust in next_node[depot_start]:
                    route = [depot_start]
                    current = first_cust
                    visited_in_route = set()
                    while current != depot_end:
                        if current in visited_in_route:
                            break  # safety
                        route.append(current)
                        visited_in_route.add(current)
                        nxt = next_node.get(current, [depot_end])
                        current = nxt[0] if nxt else depot_end
                    route.append(depot_end)

                    # Get arrival times
                    times = {}
                    for node in route:
                        times[str(node)] = round(t_var[node, omega].X, 4)

                    route_cost = sum(c_travel[route[k], route[k+1]]
                                     for k in range(len(route) - 1))

                    scenario_routes.append({
                        "route": route,
                        "arrival_times": times,
                        "cost": round(route_cost, 4)
                    })

            result["routes"][str(omega + 1)] = scenario_routes

        if model.Status == GRB.OPTIMAL:
            result["optimality_gap"] = 0.0
        else:
            result["optimality_gap"] = model.MIPGap

    else:
        result["objective_value"] = None
        result["status"] = "infeasible_or_no_solution"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve the TWAVRP using Gurobi (compact arc-flow MILP formulation)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    print(f"Loaded instance with {data['num_customers']} customers and "
          f"{len(data['scenarios'])} scenarios.")

    result = solve_twavrp(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']:.4f}")
        print(f"Status: {result['status']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
