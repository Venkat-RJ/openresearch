"""
Feasibility checker for the Dial-a-Ride Problem (DARP).
Based on: Cordeau (2006), "A Branch-and-Cut Algorithm for the Dial-a-Ride Problem",
Operations Research 54(3), pp. 573-586.

Checks constraints (2)-(14) from the mathematical formulation, plus
constraint (15) -- objective consistency: the reported objective_value
must match the recomputed routing cost (sum of Euclidean distances along
all arcs in the routes), which is the deterministic objective of (1).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'ride_times', 'routes', 'service_times')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('ride_times', 'service_times')
_FRONTIEROR_DERIVED_FIELD_RULES = ()
_FRONTIEROR_CONDITIONAL_FIELD_RULES = ()


def _frontieror_validate_solution_contract(solution):
    """Return a structured rejection when the public solution contract is absent."""
    violations = []
    if not isinstance(solution, dict):
        violations.append("solution must be a JSON object")
    else:
        missing = [
            field
            for field in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS
            if field not in solution or solution[field] is None
        ]
        if missing:
            violations.append(
                "missing required solution field(s): " + ", ".join(missing)
            )

        for discriminator, expected_value, fields in _FRONTIEROR_CONDITIONAL_FIELD_RULES:
            if solution.get(discriminator) != expected_value:
                continue
            conditional_missing = [
                field
                for field in fields
                if field not in solution or solution[field] is None
            ]
            if conditional_missing:
                violations.append(
                    "missing conditionally required solution field(s): "
                    + ", ".join(conditional_missing)
                )

        empty = [
            field
            for field in _FRONTIEROR_NONEMPTY_SOLUTION_FIELDS
            if field in solution and solution[field] in (None, [], {})
        ]
        if empty:
            violations.append(
                "empty required decision witness field(s): " + ", ".join(empty)
            )

        for target, operation, source, source_key in _FRONTIEROR_DERIVED_FIELD_RULES:
            try:
                if operation == "assignment_key_set":
                    reported_set = set(solution[target])
                    expected_set = {entry[source_key] for entry in solution[source]}
                    if reported_set != expected_set:
                        violations.append(
                            f"{target}={sorted(reported_set)} does not match the "
                            f"decision-derived value {sorted(expected_set)}"
                        )
                    continue

                reported = float(solution[target])
                if operation == "length":
                    expected = float(len(solution[source]))
                elif operation == "mapping_value":
                    expected = float(solution[source][source_key])
                else:
                    raise ValueError("unknown derived-field operation")
            except (KeyError, TypeError, ValueError, OverflowError):
                violations.append(
                    f"cannot derive {target} from decision witness field {source}"
                )
                continue
            if (
                reported != reported
                or expected != expected
                or reported in (float("inf"), float("-inf"))
                or expected in (float("inf"), float("-inf"))
                or abs(reported - expected) > 1e-6
            ):
                violations.append(
                    f"{target}={reported} does not match the decision-derived "
                    f"value {expected}"
                )

        if "objective_value" in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS:
            objective = solution.get("objective_value")
            if (
                isinstance(objective, bool)
                or not isinstance(objective, (int, float))
                or objective != objective
                or objective in (float("inf"), float("-inf"))
            ):
                violations.append("objective_value must be a finite JSON number")

    if not violations:
        return None
    return {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": violations,
        "violation_magnitudes": [],
    }


def _frontieror_write_unexpected_rejection(error):
    """Turn malformed candidate/checker exceptions into a normal rejection."""
    import json as _frontieror_json
    import sys as _frontieror_sys

    result_path = None
    for index, argument in enumerate(_frontieror_sys.argv):
        if argument == "--result_path" and index + 1 < len(_frontieror_sys.argv):
            result_path = _frontieror_sys.argv[index + 1]
            break
        if argument.startswith("--result_path="):
            result_path = argument.split("=", 1)[1]
            break
    if not result_path:
        return False
    try:
        with open(result_path, "w") as result_handle:
            _frontieror_json.dump(
                {
                    "feasible": False,
                    "violated_constraints": ["malformed_solution"],
                    "violations": [
                        "checker rejected malformed candidate: "
                        + type(error).__name__
                        + ": "
                        + str(error)[:500]
                    ],
                    "violation_magnitudes": [],
                },
                result_handle,
                indent=2,
            )
    except Exception:
        return False
    return True



def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance_path, solution_path, result_path):
    tol = 1e-5
    eps = 1e-5

    data = load_json(instance_path)
    sol = load_json(solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
    if _frontieror_contract_result is not None:
        with open(result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return _frontieror_contract_result

    n = data["num_users"]
    K_size = data["num_vehicles"]
    Q = data["vehicle_capacity"]
    L_max = data["maximum_ride_time"]
    T_max = data["maximum_route_duration"]
    num_nodes = data["num_nodes"]

    origin_depot = 0
    dest_depot = 2 * n + 1

    P = list(range(1, n + 1))
    D = list(range(n + 1, 2 * n + 1))
    N_all = list(range(num_nodes))
    K = list(range(K_size))

    # Extract node data
    nodes_by_id = {}
    for node in data["nodes"]:
        nodes_by_id[node["node_id"]] = node

    x_coord = {}
    y_coord = {}
    e = {}
    l_tw = {}
    d_serv = {}
    q = {}
    for nid, node in nodes_by_id.items():
        x_coord[nid] = node["x"]
        y_coord[nid] = node["y"]
        e[nid] = node["earliest_time"]
        l_tw[nid] = node["latest_time"]
        d_serv[nid] = node["service_duration"]
        q[nid] = node["load"]

    # Travel times / costs (Euclidean distance)
    t_matrix = {}
    for i in N_all:
        for j in N_all:
            if i != j:
                t_matrix[i, j] = euclidean_distance(
                    x_coord[i], y_coord[i], x_coord[j], y_coord[j]
                )

    # =========================================================================
    # Parse solution
    # =========================================================================
    if "routes" not in sol or sol["routes"] is None:
        result = {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No solution data in solution file"],
            "violation_magnitudes": [],
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("No solution data in solution file")
        print(f"Result written to {result_path}")
        return result

    routes = {}
    for k_str, route in sol["routes"].items():
        routes[int(k_str)] = route

    service_times = {}
    for key, val in sol["service_times"].items():
        service_times[key] = val

    ride_times = {}
    if "ride_times" in sol:
        for key, val in sol["ride_times"].items():
            ride_times[key] = val

    assignments = {}
    if "assignments" in sol:
        for key, val in sol["assignments"].items():
            assignments[int(key)] = val

    # Build B (service begin times) for each node
    B = {}
    for i in P + D:
        key = str(i)
        if key in service_times:
            B[i] = service_times[key]

    B_depot_start = {}
    B_depot_end = {}
    for k in K:
        ds_key = f"depot_start_{k}"
        de_key = f"depot_end_{k}"
        if ds_key in service_times:
            B_depot_start[k] = service_times[ds_key]
        if de_key in service_times:
            B_depot_end[k] = service_times[de_key]

    # Build x (binary routing variables) from routes
    x_var = {}
    for k in K:
        route = routes.get(k, [])
        for idx in range(len(route) - 1):
            i_node = route[idx]
            j_node = route[idx + 1]
            x_var[k, i_node, j_node] = 1

    # Build load at each node by traversing routes
    Q_var = {}
    for k in K:
        route = routes.get(k, [])
        load = 0
        for node in route:
            load += q.get(node, 0)
            Q_var[k, node] = load

    # Build ride times from solution or compute from B
    L_var = {}
    for i in P:
        key = str(i)
        if key in ride_times:
            L_var[i] = ride_times[key]
        elif i in B and (n + i) in B:
            L_var[i] = B[n + i] - (B[i] + d_serv[i])

    violations = []
    violated_constraints = set()
    violation_magnitudes = []

    def add_violation(constraint_idx, message, lhs, rhs, violation_amount):
        violated_constraints.add(constraint_idx)
        violations.append(message)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio,
        })

    # A partial timing map is not a valid witness: all timing checks below use
    # dictionary lookups, so missing keys must be rejected before those checks
    # can legitimately use their defensive `continue` branches.
    expected_route_keys = set(K)
    if set(routes) != expected_route_keys:
        add_violation(
            1,
            "Route map must contain exactly one route for every vehicle: "
            f"expected={sorted(expected_route_keys)}, got={sorted(routes)}",
            float(len(routes)), float(len(expected_route_keys)),
            float(len(set(routes) ^ expected_route_keys)),
        )

    expected_service_keys = {str(i) for i in P + D}
    expected_service_keys.update(f"depot_start_{k}" for k in K)
    expected_service_keys.update(f"depot_end_{k}" for k in K)
    actual_service_keys = set(service_times)
    if actual_service_keys != expected_service_keys:
        missing = sorted(expected_service_keys - actual_service_keys)
        extra = sorted(actual_service_keys - expected_service_keys)
        add_violation(
            7,
            "service_times must exactly cover every request node and both "
            f"depot endpoints per vehicle; missing={missing[:10]}, extra={extra[:10]}",
            float(len(actual_service_keys)), float(len(expected_service_keys)),
            float(len(actual_service_keys ^ expected_service_keys)),
        )

    expected_ride_keys = {str(i) for i in P}
    actual_ride_keys = set(ride_times)
    if actual_ride_keys != expected_ride_keys:
        missing = sorted(expected_ride_keys - actual_ride_keys)
        extra = sorted(actual_ride_keys - expected_ride_keys)
        add_violation(
            9,
            "ride_times must exactly cover every pickup request; "
            f"missing={missing[:10]}, extra={extra[:10]}",
            float(len(actual_ride_keys)), float(len(expected_ride_keys)),
            float(len(actual_ride_keys ^ expected_ride_keys)),
        )

    # =========================================================================
    # Constraint (2): Each request served exactly once
    # sum_{k in K} sum_{j in N} x^k_{ij} = 1  for all i in P
    # =========================================================================
    for i in P:
        total = sum(
            x_var.get((k, i, j), 0) for k in K for j in N_all if j != i
        )
        violation_amount = abs(total - 1)
        if violation_amount > tol:
            add_violation(
                2,
                f"Constraint (2): Pickup node {i} (user {i}) served {total} times instead of exactly 1",
                total, 1.0, violation_amount,
            )

    # =========================================================================
    # Constraint (3): Same vehicle serves pickup and dropoff
    # sum_{j in N} x^k_{ij} - sum_{j in N} x^k_{n+i,j} = 0
    #   for all i in P, k in K
    # =========================================================================
    for i in P:
        for k in K:
            out_pickup = sum(
                x_var.get((k, i, j), 0) for j in N_all if j != i
            )
            out_dropoff = sum(
                x_var.get((k, n + i, j), 0) for j in N_all if j != n + i
            )
            diff = out_pickup - out_dropoff
            violation_amount = abs(diff)
            if violation_amount > tol:
                add_violation(
                    3,
                    f"Constraint (3): User {i}, vehicle {k}: pickup outflow={out_pickup}, dropoff outflow={out_dropoff}, diff={diff}",
                    diff, 0.0, violation_amount,
                )

    # =========================================================================
    # Constraint (4): Each vehicle route starts at origin depot
    # sum_{j in N} x^k_{0,j} = 1  for all k in K
    # =========================================================================
    for k in K:
        total = sum(
            x_var.get((k, origin_depot, j), 0) for j in N_all if j != origin_depot
        )
        violation_amount = abs(total - 1)
        if violation_amount > tol:
            add_violation(
                4,
                f"Constraint (4): Vehicle {k} has {total} arcs leaving origin depot instead of 1",
                total, 1.0, violation_amount,
            )

    # =========================================================================
    # Constraint (5): Flow conservation at pickup and dropoff nodes
    # sum_{j in N} x^k_{j,i} - sum_{j in N} x^k_{i,j} = 0
    #   for all i in P union D, k in K
    # =========================================================================
    for i in P + D:
        for k in K:
            inflow = sum(
                x_var.get((k, j, i), 0) for j in N_all if j != i
            )
            outflow = sum(
                x_var.get((k, i, j), 0) for j in N_all if j != i
            )
            diff = inflow - outflow
            violation_amount = abs(diff)
            if violation_amount > tol:
                add_violation(
                    5,
                    f"Constraint (5): Node {i}, vehicle {k}: inflow={inflow}, outflow={outflow}, diff={diff}",
                    diff, 0.0, violation_amount,
                )

    # =========================================================================
    # Constraint (6): Each vehicle route ends at destination depot
    # sum_{i in N} x^k_{i,2n+1} = 1  for all k in K
    # =========================================================================
    for k in K:
        total = sum(
            x_var.get((k, i, dest_depot), 0) for i in N_all if i != dest_depot
        )
        violation_amount = abs(total - 1)
        if violation_amount > tol:
            add_violation(
                6,
                f"Constraint (6): Vehicle {k} has {total} arcs entering destination depot instead of 1",
                total, 1.0, violation_amount,
            )

    # =========================================================================
    # Constraint (7): Time consistency
    # B^k_j >= (B^k_i + d_i + t_{ij}) * x^k_{ij}
    #   for all i in N, j in N, k in K
    # Only active when x^k_{ij} = 1.
    # =========================================================================
    for k in K:
        route = routes.get(k, [])
        for idx in range(len(route) - 1):
            i_node = route[idx]
            j_node = route[idx + 1]
            # Get B values
            if i_node == origin_depot:
                B_i = B_depot_start.get(k, 0)
            elif i_node == dest_depot:
                B_i = B_depot_end.get(k, 0)
            else:
                B_i = B.get(i_node)
            if j_node == origin_depot:
                B_j = B_depot_start.get(k, 0)
            elif j_node == dest_depot:
                B_j = B_depot_end.get(k, 0)
            else:
                B_j = B.get(j_node)

            if B_i is None or B_j is None:
                continue
            if (i_node, j_node) not in t_matrix:
                continue

            required = B_i + d_serv[i_node] + t_matrix[i_node, j_node]
            violation_amount = required - B_j
            if violation_amount > tol:
                add_violation(
                    7,
                    f"Constraint (7): Vehicle {k}, arc ({i_node}->{j_node}): B[{j_node}]={B_j:.6f} < {required:.6f} (B[{i_node}]+d+t)",
                    B_j, required, violation_amount,
                )

    # =========================================================================
    # Constraint (8): Load consistency
    # Q^k_j >= (Q^k_i + q_j) * x^k_{ij}
    #   for all i in N, j in N, k in K
    # Only active when x^k_{ij} = 1.
    # =========================================================================
    for k in K:
        route = routes.get(k, [])
        for idx in range(len(route) - 1):
            i_node = route[idx]
            j_node = route[idx + 1]
            Q_i = Q_var.get((k, i_node))
            Q_j = Q_var.get((k, j_node))
            if Q_i is None or Q_j is None:
                continue
            required = Q_i + q.get(j_node, 0)
            violation_amount = required - Q_j
            if violation_amount > tol:
                add_violation(
                    8,
                    f"Constraint (8): Vehicle {k}, arc ({i_node}->{j_node}): Q[{j_node}]={Q_j} < {required} (Q[{i_node}]+q[{j_node}])",
                    Q_j, required, violation_amount,
                )

    # =========================================================================
    # Constraint (9): Ride time definition
    # L^k_i = B^k_{n+i} - (B^k_i + d_i)  for all i in P, k in K
    # =========================================================================
    for i in P:
        if i not in B or (n + i) not in B:
            continue
        expected = B[n + i] - (B[i] + d_serv[i])
        actual = L_var.get(i)
        if actual is not None:
            violation_amount = abs(actual - expected)
            if violation_amount > tol:
                add_violation(
                    9,
                    f"Constraint (9): User {i}: ride time L={actual:.6f} != B[{n+i}]-B[{i}]-d={expected:.6f}",
                    actual, expected, violation_amount,
                )

    # =========================================================================
    # Constraint (10): Maximum route duration
    # B^k_{2n+1} - B^k_0 <= T_k  for all k in K
    # =========================================================================
    for k in K:
        if k not in B_depot_start or k not in B_depot_end:
            continue
        duration = B_depot_end[k] - B_depot_start[k]
        rhs = T_max
        violation_amount = duration - rhs
        if violation_amount > tol:
            add_violation(
                10,
                f"Constraint (10): Vehicle {k}: route duration={duration:.6f} > T_max={rhs}",
                duration, rhs, violation_amount,
            )

    # =========================================================================
    # Constraint (11): Time windows
    # e_i <= B^k_i <= l_i  for all i in N, k in K
    # =========================================================================
    # Check non-depot nodes
    for i in P + D:
        if i not in B:
            continue
        # Lower bound: e_i <= B_i
        lb_violation = e[i] - B[i]
        if lb_violation > tol:
            add_violation(
                11,
                f"Constraint (11): Node {i}: B={B[i]:.6f} < earliest={e[i]:.6f}",
                B[i], e[i], lb_violation,
            )
        # Upper bound: B_i <= l_i
        ub_violation = B[i] - l_tw[i]
        if ub_violation > tol:
            add_violation(
                11,
                f"Constraint (11): Node {i}: B={B[i]:.6f} > latest={l_tw[i]:.6f}",
                B[i], l_tw[i], ub_violation,
            )
    # Check depot nodes
    for k in K:
        if k in B_depot_start:
            lb_viol = e[origin_depot] - B_depot_start[k]
            if lb_viol > tol:
                add_violation(
                    11,
                    f"Constraint (11): Vehicle {k} origin depot: B={B_depot_start[k]:.6f} < earliest={e[origin_depot]:.6f}",
                    B_depot_start[k], e[origin_depot], lb_viol,
                )
            ub_viol = B_depot_start[k] - l_tw[origin_depot]
            if ub_viol > tol:
                add_violation(
                    11,
                    f"Constraint (11): Vehicle {k} origin depot: B={B_depot_start[k]:.6f} > latest={l_tw[origin_depot]:.6f}",
                    B_depot_start[k], l_tw[origin_depot], ub_viol,
                )
        if k in B_depot_end:
            lb_viol = e[dest_depot] - B_depot_end[k]
            if lb_viol > tol:
                add_violation(
                    11,
                    f"Constraint (11): Vehicle {k} dest depot: B={B_depot_end[k]:.6f} < earliest={e[dest_depot]:.6f}",
                    B_depot_end[k], e[dest_depot], lb_viol,
                )
            ub_viol = B_depot_end[k] - l_tw[dest_depot]
            if ub_viol > tol:
                add_violation(
                    11,
                    f"Constraint (11): Vehicle {k} dest depot: B={B_depot_end[k]:.6f} > latest={l_tw[dest_depot]:.6f}",
                    B_depot_end[k], l_tw[dest_depot], ub_viol,
                )

    # =========================================================================
    # Constraint (12): Ride time bounds
    # t_{i,n+i} <= L^k_i <= L  for all i in P, k in K
    # =========================================================================
    for i in P:
        if i not in L_var:
            continue
        ride = L_var[i]
        t_direct = t_matrix[i, n + i]
        # Lower bound: t_{i,n+i} <= L_i
        lb_violation = t_direct - ride
        if lb_violation > tol:
            add_violation(
                12,
                f"Constraint (12): User {i}: ride time={ride:.6f} < direct travel time={t_direct:.6f}",
                ride, t_direct, lb_violation,
            )
        # Upper bound: L_i <= L
        ub_violation = ride - L_max
        if ub_violation > tol:
            add_violation(
                12,
                f"Constraint (12): User {i}: ride time={ride:.6f} > max ride time={L_max}",
                ride, float(L_max), ub_violation,
            )

    # =========================================================================
    # Constraint (13): Capacity bounds
    # max{0, q_i} <= Q^k_i <= min{Q_k, Q_k + q_i}
    #   for all i in N, k in K
    # =========================================================================
    for k in K:
        route = routes.get(k, [])
        for node in route:
            Q_node = Q_var.get((k, node))
            if Q_node is None:
                continue
            lb = max(0, q.get(node, 0))
            ub = min(Q, Q + q.get(node, 0))
            lb_violation = lb - Q_node
            if lb_violation > tol:
                add_violation(
                    13,
                    f"Constraint (13): Vehicle {k}, node {node}: load={Q_node} < lower bound={lb}",
                    float(Q_node), float(lb), lb_violation,
                )
            ub_violation = Q_node - ub
            if ub_violation > tol:
                add_violation(
                    13,
                    f"Constraint (13): Vehicle {k}, node {node}: load={Q_node} > upper bound={ub} (capacity={Q})",
                    float(Q_node), float(ub), ub_violation,
                )

    # =========================================================================
    # Constraint (14): Binary variables
    # x^k_{ij} in {0, 1}  for all i in N, j in N, k in K
    # Since we reconstruct x from routes as 0/1, check that each arc is used
    # at most once and values are binary.
    # =========================================================================
    for key, val in x_var.items():
        violation_amount = abs(val - round(val))
        if violation_amount > tol:
            k, i_node, j_node = key
            add_violation(
                14,
                f"Constraint (14): x[{k},{i_node},{j_node}]={val} is not binary",
                val, round(val), violation_amount,
            )

    # =========================================================================
    # Constraint (15): Objective consistency
    # The reported objective_value must equal the recomputed routing cost,
    # which is sum_{k in K} sum_{(i,j) in route_k} c_{ij} where
    # c_{ij} = t_{ij} = Euclidean distance between nodes i and j
    # (per math_model.txt reproduction note 1; cost = travel time for all
    # computational experiments). Full recompute applies because the route
    # variables fully determine which arcs are traversed.
    # =========================================================================
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for k in K:
                route = routes.get(k, [])
                for idx in range(len(route) - 1):
                    i_node = route[idx]
                    j_node = route[idx + 1]
                    if (i_node, j_node) in t_matrix:
                        true_obj += t_matrix[i_node, j_node]
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor.
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                add_violation(
                    15,
                    f"Constraint (15): Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum of arc Euclidean distances="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol_obj:.6g})",
                    reported, true_obj, abs_diff,
                )

    # =========================================================================
    # Compile results
    # =========================================================================
    unique_violated = sorted(violated_constraints)
    feasible = len(unique_violated) == 0

    # Deduplicate violation messages per constraint
    seen_messages = set()
    unique_violations = []
    for msg in violations:
        if msg not in seen_messages:
            seen_messages.add(msg)
            unique_violations.append(msg)

    result = {
        "feasible": feasible,
        "violated_constraints": unique_violated,
        "violations": unique_violations,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    if not feasible:
        print(f"Violated constraints: {unique_violated}")
        for msg in unique_violations:
            print(f"  - {msg}")
    print(f"Result written to {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for DARP (Cordeau 2006)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
    args = parser.parse_args()
    check_feasibility(args.instance_path, args.solution_path, args.result_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
