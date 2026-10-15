"""
Feasibility checker for the VRPTW formulation from:
Bard, Kontoravdis, and Yu (2002),
"A Branch-and-Cut Procedure for the Vehicle Routing Problem with Time Windows",
Transportation Science, 36(2), 250-269.

Checks all hard constraints (2)-(8) from the mathematical formulation,
plus constraint (9): objective consistency (Tier C anti-gaming defense).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('departure_times', 'loads', 'num_vehicles', 'objective_value', 'routes', 'total_distance')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('departure_times', 'loads', 'num_vehicles', 'total_distance')
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



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_distance_matrix(data):
    """Build or load distance matrix from instance data."""
    depot = data["depot"]
    customers = data["customers"]
    n = len(customers)
    num = n + 1
    nodes = [depot] + customers

    if "distance_matrix" in data and data["distance_matrix"]:
        return data["distance_matrix"]

    x_c = [nodes[i]["x"] for i in range(num)]
    y_c = [nodes[i]["y"] for i in range(num)]
    raw_dist = [[0.0] * num for _ in range(num)]
    for i in range(num):
        for j in range(num):
            if i != j:
                d = math.sqrt((x_c[i] - x_c[j]) ** 2 + (y_c[i] - y_c[j]) ** 2)
                raw_dist[i][j] = math.floor(d * 10) / 10.0
    return raw_dist


def build_tau(data, raw_dist):
    """Build travel time matrix with service time folded in, per paper preprocessing."""
    depot = data["depot"]
    customers = data["customers"]
    n = len(customers)
    num = n + 1
    nodes = [depot] + customers

    ready = [nodes[i]["ready_time"] for i in range(num)]
    due = [nodes[i]["due_date"] for i in range(num)]
    service = [nodes[i]["service_time"] for i in range(num)]

    fold = data.get("travel_time_includes_service_time", False)

    tau = [[0.0] * num for _ in range(num)]
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            t = raw_dist[i][j]
            if not fold:
                t += service[i]
            if (ready[j] - due[i]) > t:
                t = ready[j] - due[i]
            tau[i][j] = t
    return tau


def extract_x_from_routes(routes, num):
    """Extract binary x_{ij} variables from routes."""
    x = {}
    for route in routes:
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            x[i, j] = 1
    return x


def compute_departure_times_and_loads(routes, tau, ready, demand):
    """
    Compute departure times (t_i) and loads (y_i) from routes.
    t_i = departure time from customer i (max of arrival, ready_time).
    y_i = accumulated load at departure from customer i.
    """
    t = {}
    y = {}
    for route in routes:
        current_time = 0.0
        current_load = 0
        for k in range(len(route) - 1):
            i = route[k]
            j = route[k + 1]
            arr = current_time + tau[i][j]
            if j != 0:
                dep = max(arr, ready[j])
                current_time = dep
                current_load += demand[j]
                t[j] = dep
                y[j] = current_load
    return t, y


def check_feasibility(data, solution):
    """
    Check all hard constraints (2)-(8) from the VRPTW formulation,
    plus constraint (9) objective-consistency check.
    Returns feasibility result dictionary.
    """
    tol = 1e-5
    eps = 1e-5

    depot = data["depot"]
    customers = data["customers"]
    n = len(customers)
    num = n + 1  # depot + customers
    nodes = [depot] + customers

    Q = data["vehicle_capacity"]
    demand = [nodes[i]["demand"] for i in range(num)]
    ready = [nodes[i]["ready_time"] for i in range(num)]
    due = [nodes[i]["due_date"] for i in range(num)]

    raw_dist = build_distance_matrix(data)
    tau = build_tau(data, raw_dist)

    routes = solution.get("routes", [])
    x = extract_x_from_routes(routes, num)

    # Use provided departure_times and loads if available, otherwise compute
    if "departure_times" in solution and solution["departure_times"]:
        t = {int(k): v for k, v in solution["departure_times"].items()}
    else:
        t, _ = compute_departure_times_and_loads(routes, tau, ready, demand)

    if "loads" in solution and solution["loads"]:
        y = {int(k): v for k, v in solution["loads"].items()}
    else:
        _, y = compute_departure_times_and_loads(routes, tau, ready, demand)

    I = list(range(1, num))     # customer indices
    I0 = list(range(num))       # depot + customer indices

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint (2): Each customer served exactly once
    #   sum_{j in I_0} x_{ij} = 1, for all i in I
    # =========================================================================
    for i in I:
        lhs_val = sum(x.get((i, j), 0) for j in I0 if j != i)
        rhs_val = 1.0
        violation_amount = abs(lhs_val - rhs_val)
        if violation_amount > tol:
            violated_constraints.add(2)
            violations.append(
                f"Constraint (2): Customer {i} is visited {int(lhs_val)} time(s) "
                f"instead of exactly 1"
            )
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (3): Flow conservation
    #   sum_{j in I_0} x_{ij} - sum_{j in I_0} x_{ji} = 0, for all i in I_0
    # =========================================================================
    for i in I0:
        out_flow = sum(x.get((i, j), 0) for j in I0 if j != i)
        in_flow = sum(x.get((j, i), 0) for j in I0 if j != i)
        lhs_val = out_flow - in_flow
        rhs_val = 0.0
        violation_amount = abs(lhs_val - rhs_val)
        if violation_amount > tol:
            violated_constraints.add(3)
            node_label = f"depot (node 0)" if i == 0 else f"customer {i}"
            violations.append(
                f"Constraint (3): Flow conservation violated at {node_label}: "
                f"out_flow={out_flow}, in_flow={in_flow}, difference={lhs_val}"
            )
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (4): Time linking
    #   t_j >= t_i + tau_{ij} * x_{ij} - T_{ij} * (1 - x_{ij}), for i,j in I
    #   where T_{ij} = b_i - a_j
    #   Only active (binding) when x_{ij} = 1, so we check: t_j >= t_i + tau_{ij}
    # =========================================================================
    for i in I:
        for j in I:
            if i == j:
                continue
            x_ij = x.get((i, j), 0)
            T_ij = due[i] - ready[j]
            rhs_val = t.get(i, 0.0) + tau[i][j] * x_ij - T_ij * (1 - x_ij)
            lhs_val = t.get(j, 0.0)
            # This is a >= constraint: lhs >= rhs, so violation = rhs - lhs
            violation_amount = rhs_val - lhs_val
            if violation_amount > tol:
                violated_constraints.add(4)
                violations.append(
                    f"Constraint (4): Time linking violated for arc ({i},{j}): "
                    f"t_{j}={lhs_val:.2f} < t_{i} + tau*x - T*(1-x) = {rhs_val:.2f}"
                )
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # =========================================================================
    # Constraint (5): Load linking
    #   y_j >= y_i + q_j - Q_j * (1 - x_{ij}), for i,j in I
    #   where Q_j = Q - q_j
    # =========================================================================
    for i in I:
        for j in I:
            if i == j:
                continue
            x_ij = x.get((i, j), 0)
            Q_j = Q - demand[j]
            rhs_val = y.get(i, 0.0) + demand[j] - Q_j * (1 - x_ij)
            lhs_val = y.get(j, 0.0)
            # >= constraint: violation = rhs - lhs
            violation_amount = rhs_val - lhs_val
            if violation_amount > tol:
                violated_constraints.add(5)
                violations.append(
                    f"Constraint (5): Load linking violated for arc ({i},{j}): "
                    f"y_{j}={lhs_val:.2f} < y_{i} + q_{j} - Q_{j}*(1-x) = {rhs_val:.2f}"
                )
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # =========================================================================
    # Constraint (6): Vehicle capacity
    #   q_i <= y_i <= Q, for all i in I
    #   Two parts: (a) y_i >= q_i and (b) y_i <= Q
    # =========================================================================
    for i in I:
        y_i = y.get(i, 0.0)
        # (a) y_i >= q_i
        lhs_val_a = y_i
        rhs_val_a = float(demand[i])
        violation_amount_a = rhs_val_a - lhs_val_a  # >= constraint
        if violation_amount_a > tol:
            violated_constraints.add(6)
            violations.append(
                f"Constraint (6): Load at customer {i} too low: "
                f"y_{i}={lhs_val_a:.2f} < q_{i}={rhs_val_a:.2f}"
            )
            normalizer = max(abs(rhs_val_a), eps)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs_val_a,
                "rhs": rhs_val_a,
                "raw_excess": violation_amount_a,
                "normalizer": normalizer,
                "ratio": violation_amount_a / normalizer,
            })

        # (b) y_i <= Q
        lhs_val_b = y_i
        rhs_val_b = float(Q)
        violation_amount_b = lhs_val_b - rhs_val_b  # <= constraint
        if violation_amount_b > tol:
            violated_constraints.add(6)
            violations.append(
                f"Constraint (6): Load at customer {i} exceeds capacity: "
                f"y_{i}={lhs_val_b:.2f} > Q={rhs_val_b:.2f}"
            )
            normalizer = max(abs(rhs_val_b), eps)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs_val_b,
                "rhs": rhs_val_b,
                "raw_excess": violation_amount_b,
                "normalizer": normalizer,
                "ratio": violation_amount_b / normalizer,
            })

    # =========================================================================
    # Constraint (7): Time window
    #   a_i <= t_i <= b_i, for all i in I
    #   Two parts: (a) t_i >= a_i and (b) t_i <= b_i
    # =========================================================================
    for i in I:
        t_i = t.get(i, 0.0)
        # (a) t_i >= a_i
        lhs_val_a = t_i
        rhs_val_a = float(ready[i])
        violation_amount_a = rhs_val_a - lhs_val_a  # >= constraint
        if violation_amount_a > tol:
            violated_constraints.add(7)
            violations.append(
                f"Constraint (7): Departure time at customer {i} too early: "
                f"t_{i}={lhs_val_a:.2f} < a_{i}={rhs_val_a:.2f}"
            )
            normalizer = max(abs(rhs_val_a), eps)
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": lhs_val_a,
                "rhs": rhs_val_a,
                "raw_excess": violation_amount_a,
                "normalizer": normalizer,
                "ratio": violation_amount_a / normalizer,
            })

        # (b) t_i <= b_i
        lhs_val_b = t_i
        rhs_val_b = float(due[i])
        violation_amount_b = lhs_val_b - rhs_val_b  # <= constraint
        if violation_amount_b > tol:
            violated_constraints.add(7)
            violations.append(
                f"Constraint (7): Departure time at customer {i} too late: "
                f"t_{i}={lhs_val_b:.2f} > b_{i}={rhs_val_b:.2f}"
            )
            normalizer = max(abs(rhs_val_b), eps)
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": lhs_val_b,
                "rhs": rhs_val_b,
                "raw_excess": violation_amount_b,
                "normalizer": normalizer,
                "ratio": violation_amount_b / normalizer,
            })

    # =========================================================================
    # Constraint (8): Binary constraint
    #   x_{ij} in {0, 1}, for all i,j in I_0
    # =========================================================================
    for i in I0:
        for j in I0:
            if i == j:
                continue
            x_ij = x.get((i, j), 0)
            if x_ij not in (0, 1):
                violation_amount = min(abs(x_ij - 0), abs(x_ij - 1))
                if violation_amount > tol:
                    violated_constraints.add(8)
                    violations.append(
                        f"Constraint (8): x_{{{i},{j}}} = {x_ij} is not binary"
                    )
                    normalizer = max(1.0, eps)
                    violation_magnitudes.append({
                        "constraint": 8,
                        "lhs": float(x_ij),
                        "rhs": round(x_ij),
                        "raw_excess": violation_amount,
                        "normalizer": normalizer,
                        "ratio": violation_amount / normalizer,
                    })

    # =========================================================================
    # Constraint (9): Objective-consistency (Tier C anti-gaming)
    #   Recompute the VRPTW objective from the solution routes and compare to
    #   the reported objective_value. The math model defines the objective as
    #     min sum_{i=1..n} x_{0i}  =  number of vehicles dispatched from depot
    #   which equals the number of (0, j) arcs with j in I, i.e., the count
    #   of routes that actually leave the depot to visit a customer.
    #   Since the objective is an integer count, use tol=0.5 (any integer
    #   mismatch by >= 1 fires).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(x.get((0, j), 0) for j in I))
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5
            if abs_diff > obj_tol:
                violated_constraints.add(9)
                violations.append(
                    f"Constraint (9): Objective consistency violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed sum_{{j in I}} x_{{0,j}}={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 9,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # Bind serialized summary fields to the route witness.  They are useful to
    # consumers, but are not independent decisions and must not be accepted
    # after being corrupted while the routes remain unchanged.
    derived_num_vehicles = len(routes)
    reported_num_vehicles = solution.get("num_vehicles")
    if (isinstance(reported_num_vehicles, bool)
            or not isinstance(reported_num_vehicles, (int, float))
            or abs(float(reported_num_vehicles) - derived_num_vehicles) > tol):
        violated_constraints.add(9)
        violations.append(
            f"Constraint (9): num_vehicles={reported_num_vehicles!r} does not "
            f"equal len(routes)={derived_num_vehicles}"
        )
        violation_magnitudes.append({
            "constraint": 9,
            "lhs": float(reported_num_vehicles)
            if isinstance(reported_num_vehicles, (int, float))
            and not isinstance(reported_num_vehicles, bool) else 0.0,
            "rhs": float(derived_num_vehicles),
            "raw_excess": 1.0,
            "normalizer": max(float(derived_num_vehicles), eps),
            "ratio": 1.0 / max(float(derived_num_vehicles), eps),
        })

    derived_total_distance = round(sum(
        float(raw_dist[route[pos]][route[pos + 1]])
        for route in routes
        for pos in range(len(route) - 1)
    ), 1)
    reported_total_distance = solution.get("total_distance")
    distance_ok = (
        not isinstance(reported_total_distance, bool)
        and isinstance(reported_total_distance, (int, float))
        and abs(float(reported_total_distance) - derived_total_distance) <= 0.05
    )
    if not distance_ok:
        violated_constraints.add(9)
        violations.append(
            f"Constraint (9): total_distance={reported_total_distance!r} does "
            f"not equal the route-derived distance={derived_total_distance}"
        )
        violation_magnitudes.append({
            "constraint": 9,
            "lhs": float(reported_total_distance)
            if isinstance(reported_total_distance, (int, float))
            and not isinstance(reported_total_distance, bool) else 0.0,
            "rhs": float(derived_total_distance),
            "raw_excess": abs(float(reported_total_distance)
                              - derived_total_distance)
            if isinstance(reported_total_distance, (int, float))
            and not isinstance(reported_total_distance, bool) else 1.0,
            "normalizer": max(abs(float(derived_total_distance)), eps),
            "ratio": (abs(float(reported_total_distance)
                          - derived_total_distance)
                      / max(abs(float(derived_total_distance)), eps))
            if isinstance(reported_total_distance, (int, float))
            and not isinstance(reported_total_distance, bool) else 1.0,
        })

    # Build result
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for VRPTW (Bard et al. 2002)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    data = load_json(args.instance_path)
    solution = load_json(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return
    result = check_feasibility(data, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE - no constraints violated")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
