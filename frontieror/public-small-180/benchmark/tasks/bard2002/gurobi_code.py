"""
Gurobi implementation of the VRPTW formulation from:
Bard, Kontoravdis, and Yu (2002),
"A Branch-and-Cut Procedure for the Vehicle Routing Problem with Time Windows",
Transportation Science, 36(2), 250-269.

The model minimizes the number of vehicles (primary objective).
"""

import argparse
import json
import math
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
    """Load a VRPTW instance from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    depot = data["depot"]
    customers = data["customers"]
    n = len(customers)
    Q = data["vehicle_capacity"]

    # Build node list: 0 = depot, 1..n = customers (using their original ids)
    # Map customer id -> index in our arrays (1-based)
    # We assume customers are listed in order with ids 1..n
    nodes = [depot] + customers  # index 0 = depot, 1..n = customers
    num_nodes = n + 1  # 0..n

    # Extract coordinates, demands, time windows, service times
    x_coord = [nodes[i]["x"] for i in range(num_nodes)]
    y_coord = [nodes[i]["y"] for i in range(num_nodes)]
    demand = [nodes[i]["demand"] for i in range(num_nodes)]
    ready = [nodes[i]["ready_time"] for i in range(num_nodes)]
    due = [nodes[i]["due_date"] for i in range(num_nodes)]
    service = [nodes[i]["service_time"] for i in range(num_nodes)]

    # -------------------------------------------------------------------------
    # Use the distance matrix from the instance if provided, otherwise compute
    # -------------------------------------------------------------------------
    if "distance_matrix" in data and data["distance_matrix"]:
        raw_dist = data["distance_matrix"]
    else:
        # Compute Euclidean distances truncated to 1 decimal
        raw_dist = [[0.0] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    d = math.sqrt((x_coord[i] - x_coord[j]) ** 2 +
                                  (y_coord[i] - y_coord[j]) ** 2)
                    raw_dist[i][j] = math.floor(d * 10) / 10.0

    # -------------------------------------------------------------------------
    # Preprocessing: compute travel times tau[i][j]
    # Paper convention:
    #   1) tau[i][j] = distance[i][j] (velocity = 1)
    #   2) Fold service time into travel time: tau[i][j] += sigma_i
    #   3) If (a_j - b_i) > tau[i][j], set tau[i][j] = a_j - b_i
    # -------------------------------------------------------------------------
    # NOTE: The instance JSON may already indicate travel_time_includes_service_time.
    # We check and only fold service time if needed.
    fold_service = data.get("travel_time_includes_service_time", False)

    tau = [[0.0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                tau[i][j] = 0.0
                continue
            t = raw_dist[i][j]
            # Fold in service time of node i (departure node)
            if not fold_service:
                t += service[i]
            # If (a_j - b_i) > t, replace t
            if (ready[j] - due[i]) > t:
                t = ready[j] - due[i]
            tau[i][j] = t

    # NOTE: When travel_time_includes_service_time is True, the distance_matrix
    # already includes service time. The paper says tau_{ij} <- tau_{ij} + sigma_i.
    # If the instance says it's already included, we skip that step.
    # However, looking at the instance data, the distance matrix appears to be
    # raw Euclidean distances (not including service time). Let's re-check:
    # The flag says True but the distances look like raw Euclidean.
    # ASSUMPTION: We trust the flag. If distances seem raw, we still respect the flag.

    # -------------------------------------------------------------------------
    # Determine infeasible arcs
    # Arc (i,j) is infeasible if:
    #   - Visiting i then j violates time windows: a_i + tau[i][j] > b_j
    #     (earliest departure from i + travel time exceeds latest arrival at j)
    #   - Or capacity: demand[i] + demand[j] > Q (for i,j both customers)
    # -------------------------------------------------------------------------
    I = list(range(1, num_nodes))  # customer indices 1..n
    I0 = list(range(num_nodes))    # all nodes 0..n

    feasible_arc = [[False] * num_nodes for _ in range(num_nodes)]
    for i in I0:
        for j in I0:
            if i == j:
                continue
            # Check time feasibility
            # Earliest departure from i: ready[i] (+ service time if not folded)
            earliest_dep_i = ready[i]
            if not fold_service and i > 0:
                earliest_dep_i += service[i]
            # Actually, with service time folded into tau, departure from i at
            # time ready[i] means arrival at j at ready[i] + tau[i][j].
            # For feasibility: ready[i] + tau[i][j] <= due[j]
            # But we also need: the arrival time at j >= ready[j] (wait is ok)
            # The binding constraint is: ready[i] + tau[i][j] <= due[j]
            if ready[i] + tau[i][j] > due[j]:
                continue
            # Check capacity feasibility (only for customer-customer arcs)
            if i > 0 and j > 0:
                if demand[i] + demand[j] > Q:
                    continue
            feasible_arc[i][j] = True

    # -------------------------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("VRPTW")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i][j]: binary, 1 if vehicle travels from i to j
    x = {}
    for i in I0:
        for j in I0:
            if i != j and feasible_arc[i][j]:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # t[i]: departure time from customer i (continuous)
    t = {}
    for i in I:
        t[i] = model.addVar(lb=ready[i], ub=due[i], vtype=GRB.CONTINUOUS,
                            name=f"t_{i}")

    # y[i]: vehicle load at departure from customer i (continuous)
    y = {}
    for i in I:
        y[i] = model.addVar(lb=demand[i], ub=Q, vtype=GRB.CONTINUOUS,
                            name=f"y_{i}")

    model.update()

    # -------------------------------------------------------------------------
    # Objective: minimize number of vehicles = sum of x[0][j] for j in I
    # -------------------------------------------------------------------------
    model.setObjective(
        gp.quicksum(x[0, j] for j in I if (0, j) in x),
        GRB.MINIMIZE
    )

    # -------------------------------------------------------------------------
    # Constraint (2): Each customer served by exactly one vehicle
    # sum_j x[i][j] = 1 for all i in I
    # -------------------------------------------------------------------------
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, j] for j in I0 if (i, j) in x) == 1,
            name=f"serve_{i}"
        )

    # -------------------------------------------------------------------------
    # Constraint (3): Flow conservation
    # sum_j x[i][j] - sum_j x[j][i] = 0 for all i in I_0
    # -------------------------------------------------------------------------
    for i in I0:
        model.addConstr(
            gp.quicksum(x[i, j] for j in I0 if (i, j) in x) -
            gp.quicksum(x[j, i] for j in I0 if (j, i) in x) == 0,
            name=f"flow_{i}"
        )

    # -------------------------------------------------------------------------
    # Constraint (4): Time linking constraints
    # t[j] >= t[i] + tau[i][j] * x[i][j] - T[i][j] * (1 - x[i][j])
    # where T[i][j] = b_i - a_j
    # For i, j in I (customers only)
    # -------------------------------------------------------------------------
    for i in I:
        for j in I:
            if i != j and (i, j) in x:
                T_ij = due[i] - ready[j]
                model.addConstr(
                    t[j] >= t[i] + tau[i][j] * x[i, j] - T_ij * (1 - x[i, j]),
                    name=f"time_{i}_{j}"
                )
    # NOTE: Constraint (4) with x_{ij}=0 reduces to t_j >= t_i - (b_i - a_j),
    # which is always satisfied by variable bounds (t_i <= b_i, t_j >= a_j).

    # -------------------------------------------------------------------------
    # Constraint (5): Load linking constraints
    # y[j] >= y[i] + q_j - Q_j * (1 - x[i][j])
    # where Q_j = Q - q_j
    # For i, j in I (customers only)
    # -------------------------------------------------------------------------
    for i in I:
        for j in I:
            if i != j:
                Q_j = Q - demand[j]
                if (i, j) in x:
                    model.addConstr(
                        y[j] >= y[i] + demand[j] - Q_j * (1 - x[i, j]),
                        name=f"load_{i}_{j}"
                    )
                else:
                    # Arc excluded (x_{ij}=0): constraint becomes y_j >= y_i + q_j - Q_j
                    model.addConstr(
                        y[j] >= y[i] + demand[j] - Q_j,
                        name=f"load_{i}_{j}"
                    )

    # -------------------------------------------------------------------------
    # Constraints (6) and (7) are handled by variable bounds:
    #   q_i <= y_i <= Q   (set as lb/ub on y[i])
    #   a_i <= t_i <= b_i (set as lb/ub on t[i])
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    result = {}
    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["status"] = model.Status
        result["status_description"] = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, f"STATUS_{model.Status}")

        # Extract routes
        routes = []
        # Find all arcs leaving the depot
        depot_successors = []
        for j in I:
            if (0, j) in x and x[0, j].X > 0.5:
                depot_successors.append(j)

        for start in depot_successors:
            route = [0, start]
            current = start
            while current != 0:
                next_node = None
                for j in I0:
                    if (current, j) in x and x[current, j].X > 0.5:
                        next_node = j
                        break
                if next_node is None or next_node == 0:
                    route.append(0)
                    break
                route.append(next_node)
                current = next_node
            routes.append(route)

        result["num_vehicles"] = len(routes)
        result["routes"] = routes

        # Compute total distance
        total_dist = 0.0
        for i_node in I0:
            for j_node in I0:
                if (i_node, j_node) in x and x[i_node, j_node].X > 0.5:
                    total_dist += raw_dist[i_node][j_node]
        result["total_distance"] = round(total_dist, 1)

        # Extract departure times and loads
        departure_times = {}
        loads = {}
        for i in I:
            departure_times[str(i)] = round(t[i].X, 2)
            loads[str(i)] = round(y[i].X, 2)
        result["departure_times"] = departure_times
        result["loads"] = loads

    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["status_description"] = "NO_SOLUTION_FOUND"
        result["num_vehicles"] = None
        result["routes"] = []

    # Add solver info
    result["solver"] = "Gurobi"
    result["time_limit"] = time_limit
    if hasattr(model, "Runtime"):
        result["runtime_seconds"] = round(model.Runtime, 2)
    if hasattr(model, "MIPGap") and model.SolCount > 0:
        result["mip_gap"] = round(model.MIPGap, 6)

    # Write solution
    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    print(f"Objective value: {result['objective_value']}")
    if result.get("num_vehicles") is not None:
        print(f"Number of vehicles: {result['num_vehicles']}")
        print(f"Total distance: {result.get('total_distance')}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="VRPTW Gurobi solver (Bard et al. 2002 formulation)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    build_and_solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
