"""
Feasibility checker for TSP-D (Roberti & Ruthmair, 2021).

Checks the hard constraints (1b)-(1o) from the compact MILP formulation,
plus an objective-consistency check (constraint 15) that recomputes the
true tour-completion time a_{0'} from the truck/drone routes and the
instance travel times, and rejects the solution if the reported
objective_value deviates from the recomputed value.

Supports both efficient_algorithm and Gurobi solution formats.

Usage:
    python feasibility_check_new.py \
        --instance_path instance_1.json \
        --solution_path efficient_solution_1.json \
        --result_path efficient_feasi_result_1.json
"""

import argparse
import json
import sys
from itertools import permutations

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('arrival_times', 'customer_types', 'drone_route', 'objective_value', 'truck_route')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ()
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


TOL = 1e-5
EPS = 1e-5


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def load_solution(path):
    with open(path, "r") as f:
        return json.load(f)


def _is_binary(val):
    return abs(val - 0.0) <= TOL or abs(val - 1.0) <= TOL


# ------------------------------------------------------------------ #
#  Reconstruct MILP decision variables from compact solutions
# ------------------------------------------------------------------ #

def reconstruct_variables_efficient(instance, solution):
    """
    Directly read MILP variables from the ORIGINAL TSP-D solution
    structure (truck_route, drone_route, customer_types, arrival_times).

    The earlier DP-based reconstruction searched for a minimum-cost
    feasible assignment of drone customers to truck segments instead
    of verifying the solution as given; that could accept an
    algorithm's infeasible assignment whenever some other assignment
    of the same drone_customers happened to be feasible.
    """
    return reconstruct_variables_gurobi(instance, solution)


def reconstruct_variables_gurobi(instance, solution):
    """
    Reconstruct x^T, x^D, y^T, y^D, y^C, a from Gurobi solution format.
    Gurobi provides: truck_route, drone_route, customer_types, arrival_times.
    """
    n = instance["num_customers"]

    truck_route = solution["truck_route"]
    drone_route = solution["drone_route"]
    customer_types = solution.get("customer_types", {})
    arrival_times_raw = solution.get("arrival_times", {})

    y_T = {}
    y_D = {}
    y_C = {}
    for i in range(1, n + 1):
        ct = customer_types.get(str(i), "")
        y_T[i] = 1 if ct == "truck" else 0
        y_D[i] = 1 if ct == "drone" else 0
        y_C[i] = 1 if ct == "combined" else 0

    # Build truck arcs
    x_T = {}
    for idx in range(len(truck_route) - 1):
        u, v = truck_route[idx], truck_route[idx + 1]
        x_T[(u, v)] = 1

    # Build drone arcs
    x_D = {}
    for idx in range(len(drone_route) - 1):
        u, v = drone_route[idx], drone_route[idx + 1]
        x_D[(u, v)] = 1

    # Arrival times: Gurobi uses node n+1 for depot copy 0'
    a = {}
    for key, val in arrival_times_raw.items():
        node = int(key)
        if node <= n:
            a[node] = val
        else:
            a["0prime"] = val

    return x_T, x_D, y_T, y_D, y_C, a


def detect_solution_type(solution):
    """Detect whether this is an efficient_algorithm or Gurobi solution."""
    if "solver" in solution and solution["solver"] == "Gurobi":
        return "gurobi"
    if "drone_customers" in solution:
        return "efficient"
    return "efficient"


# ------------------------------------------------------------------ #
#  Objective recompute (Tier C anti-exploit)
# ------------------------------------------------------------------ #

def _recompute_a_0prime(instance, solution):
    """
    Recompute the true tour-completion time a_{0'} from the routes and
    travel times.  Truck and drone are synchronized at the depot and at
    every combined customer; between consecutive sync points each
    vehicle travels its sub-route independently, and the arrival time
    at the next sync point is the max of the two cumulative travel
    times.  Returns the recomputed a_{0'}, or None if the routes are
    malformed or their sync-point sequences do not agree (in which
    case other constraints will already catch the infeasibility).
    """
    truck_times = instance.get("truck_travel_times")
    drone_times = instance.get("drone_travel_times")
    truck_route = solution.get("truck_route") or []
    drone_route = solution.get("drone_route") or []
    customer_types = solution.get("customer_types") or {}

    if truck_times is None or drone_times is None:
        return None
    if len(truck_route) < 2 or len(drone_route) < 2:
        return None

    combined = set()
    for k, v in customer_types.items():
        if v == "combined":
            try:
                combined.add(int(k))
            except (TypeError, ValueError):
                pass

    def sync_idxs(route):
        idxs = [0]
        for i in range(1, len(route) - 1):
            if route[i] in combined:
                idxs.append(i)
        idxs.append(len(route) - 1)
        return idxs

    t_idxs = sync_idxs(truck_route)
    d_idxs = sync_idxs(drone_route)
    if len(t_idxs) != len(d_idxs):
        return None
    if [truck_route[i] for i in t_idxs] != [drone_route[i] for i in d_idxs]:
        return None

    total = 0.0
    for k in range(len(t_idxs) - 1):
        try:
            t_seg = sum(
                float(truck_times[truck_route[i]][truck_route[i + 1]])
                for i in range(t_idxs[k], t_idxs[k + 1])
            )
            d_seg = sum(
                float(drone_times[drone_route[i]][drone_route[i + 1]])
                for i in range(d_idxs[k], d_idxs[k + 1])
            )
        except (IndexError, TypeError, ValueError):
            return None
        total += max(t_seg, d_seg)
    return total


# ------------------------------------------------------------------ #
#  Constraint checking
# ------------------------------------------------------------------ #

def add_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs, raw_excess):
    """Record a constraint violation."""
    violations_list.append(message)
    normalizer = max(abs(rhs), EPS)
    ratio = raw_excess / normalizer
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(raw_excess),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def check_feasibility(instance, solution):
    """Check all hard constraints of the TSP-D MILP formulation."""
    n = instance["num_customers"]
    truck_times = instance["truck_travel_times"]
    drone_times = instance["drone_travel_times"]

    sol_type = detect_solution_type(solution)

    # Check for empty/null solutions (e.g., Gurobi timeout with no solution)
    if solution.get("objective_value") is None:
        return {
            "feasible": False,
            "violated_constraints": [7],
            "violations": [
                "No solution found (objective_value is null); "
                "all customers unserved."
            ],
            "violation_magnitudes": [{
                "constraint": 7,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            }],
        }

    truck_route = solution.get("truck_route", [])
    if not truck_route or len(truck_route) < 2:
        return {
            "feasible": False,
            "violated_constraints": [3],
            "violations": ["Truck route is empty or trivial."],
            "violation_magnitudes": [{
                "constraint": 3,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            }],
        }

    # Domain check: nodes are 0 (depot origin), 1..n (customers), n+1 (depot
    # destination, encoded as "0prime" internally). Truck and drone route
    # entries, customer_types keys, and arrival_times keys must all lie in
    # this set.
    valid_nodes = set(range(0, n + 2))
    customers_set = set(range(1, n + 1))
    drone_route = solution.get("drone_route", [])
    customer_types = solution.get("customer_types", {})
    arrival_times_raw = solution.get("arrival_times", {})

    domain_violations = []
    for node in truck_route:
        if node not in valid_nodes:
            domain_violations.append(
                f"truck_route node {node!r} not in valid set [0, {n + 1}]"
            )
    for node in drone_route:
        if node not in valid_nodes:
            domain_violations.append(
                f"drone_route node {node!r} not in valid set [0, {n + 1}]"
            )
    for k_str in customer_types:
        try:
            k_int = int(k_str)
        except (TypeError, ValueError):
            k_int = None
        if k_int not in customers_set:
            domain_violations.append(
                f"customer_types key {k_str!r} not a valid customer id"
            )
    valid_type_set = {"truck", "drone", "combined"}
    for v in customer_types.values():
        if v not in valid_type_set:
            domain_violations.append(
                f"customer_types value {v!r} not in {valid_type_set}"
            )
    for k_str in arrival_times_raw:
        try:
            k_int = int(k_str)
        except (TypeError, ValueError):
            k_int = None
        if k_int not in valid_nodes:
            domain_violations.append(
                f"arrival_times key {k_str!r} not in valid set [0, {n + 1}]"
            )

    if domain_violations:
        return {
            "feasible": False,
            "violated_constraints": [0],
            "violations": [f"Domain: {m}" for m in domain_violations],
            "violation_magnitudes": [{
                "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            } for _ in domain_violations],
        }

    if sol_type == "gurobi":
        x_T, x_D, y_T, y_D, y_C, a = reconstruct_variables_gurobi(instance, solution)
    else:
        x_T, x_D, y_T, y_D, y_C, a = reconstruct_variables_efficient(instance, solution)

    customers = list(range(1, n + 1))

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def x_T_val(i, j):
        return x_T.get((i, j), 0)

    def x_D_val(i, j):
        return x_D.get((i, j), 0)

    def a_val(node):
        if node == "0prime":
            return a.get("0prime", 0)
        return a.get(node, 0)

    # Big-M
    total_truck = sum(sum(row) for row in truck_times)
    M = total_truck

    # ------------------------------------------------------------------
    # Constraint 1 (1b): Truck flow conservation
    # sum_{(i,j) in A} x^T_{ij} = sum_{(j,i) in A} x^T_{ji}  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        out_flow = sum(x_T_val(i, j) for j in customers if j != i) + x_T_val(i, 0)
        in_flow = x_T_val(0, i) + sum(x_T_val(j, i) for j in customers if j != i)
        diff = abs(out_flow - in_flow)
        if diff > TOL:
            violated_constraints.add(1)
            add_violation(violations, violation_magnitudes, 1,
                          f"Truck flow conservation violated at customer {i}: "
                          f"out_flow={out_flow}, in_flow={in_flow}",
                          out_flow, in_flow, diff)

    # ------------------------------------------------------------------
    # Constraint 2 (1c): Truck arc-customer linking
    # sum_{(i,j) in A} x^T_{ij} = y^T_i + y^C_i  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        out_flow = sum(x_T_val(i, j) for j in customers if j != i) + x_T_val(i, 0)
        rhs_val = y_T.get(i, 0) + y_C.get(i, 0)
        diff = abs(out_flow - rhs_val)
        if diff > TOL:
            violated_constraints.add(2)
            add_violation(violations, violation_magnitudes, 2,
                          f"Truck arc-customer linking violated at customer {i}: "
                          f"out_flow={out_flow}, y^T+y^C={rhs_val}",
                          out_flow, rhs_val, diff)

    # ------------------------------------------------------------------
    # Constraint 3 (1d): Truck leaves and returns to depot exactly once
    # ------------------------------------------------------------------
    truck_out_depot = sum(x_T_val(0, j) for j in customers)
    if abs(truck_out_depot - 1) > TOL:
        violated_constraints.add(3)
        add_violation(violations, violation_magnitudes, 3,
                      f"Truck does not leave depot exactly once: "
                      f"sum x^T_{{0j}} = {truck_out_depot}",
                      truck_out_depot, 1.0, abs(truck_out_depot - 1))

    truck_in_depot = sum(x_T_val(i, 0) for i in customers)
    if abs(truck_in_depot - 1) > TOL:
        violated_constraints.add(3)
        add_violation(violations, violation_magnitudes, 3,
                      f"Truck does not return to depot exactly once: "
                      f"sum x^T_{{i0'}} = {truck_in_depot}",
                      truck_in_depot, 1.0, abs(truck_in_depot - 1))

    # ------------------------------------------------------------------
    # Constraint 4 (1e): Drone flow conservation
    # ------------------------------------------------------------------
    for i in customers:
        out_flow = sum(x_D_val(i, j) for j in customers if j != i) + x_D_val(i, 0)
        in_flow = x_D_val(0, i) + sum(x_D_val(j, i) for j in customers if j != i)
        diff = abs(out_flow - in_flow)
        if diff > TOL:
            violated_constraints.add(4)
            add_violation(violations, violation_magnitudes, 4,
                          f"Drone flow conservation violated at customer {i}: "
                          f"out_flow={out_flow}, in_flow={in_flow}",
                          out_flow, in_flow, diff)

    # ------------------------------------------------------------------
    # Constraint 5 (1f): Drone arc-customer linking
    # ------------------------------------------------------------------
    for i in customers:
        out_flow = sum(x_D_val(i, j) for j in customers if j != i) + x_D_val(i, 0)
        rhs_val = y_D.get(i, 0) + y_C.get(i, 0)
        diff = abs(out_flow - rhs_val)
        if diff > TOL:
            violated_constraints.add(5)
            add_violation(violations, violation_magnitudes, 5,
                          f"Drone arc-customer linking violated at customer {i}: "
                          f"out_flow={out_flow}, y^D+y^C={rhs_val}",
                          out_flow, rhs_val, diff)

    # ------------------------------------------------------------------
    # Constraint 6 (1g): Drone leaves and returns to depot exactly once
    # ------------------------------------------------------------------
    drone_out_depot = sum(x_D_val(0, j) for j in customers)
    if abs(drone_out_depot - 1) > TOL:
        violated_constraints.add(6)
        add_violation(violations, violation_magnitudes, 6,
                      f"Drone does not leave depot exactly once: "
                      f"sum x^D_{{0j}} = {drone_out_depot}",
                      drone_out_depot, 1.0, abs(drone_out_depot - 1))

    drone_in_depot = sum(x_D_val(i, 0) for i in customers)
    if abs(drone_in_depot - 1) > TOL:
        violated_constraints.add(6)
        add_violation(violations, violation_magnitudes, 6,
                      f"Drone does not return to depot exactly once: "
                      f"sum x^D_{{i0'}} = {drone_in_depot}",
                      drone_in_depot, 1.0, abs(drone_in_depot - 1))

    # ------------------------------------------------------------------
    # Constraint 7 (1h): Each customer visited exactly once
    # ------------------------------------------------------------------
    for i in customers:
        total = y_T.get(i, 0) + y_D.get(i, 0) + y_C.get(i, 0)
        diff = abs(total - 1)
        if diff > TOL:
            violated_constraints.add(7)
            add_violation(violations, violation_magnitudes, 7,
                          f"Customer {i} not visited exactly once: "
                          f"y^T+y^D+y^C={total}",
                          total, 1.0, diff)

    # ------------------------------------------------------------------
    # Constraint 8 (1i): Truck arrival time / subtour elimination
    # a_i + t^T_{ij} <= a_j + M(1 - x^T_{ij})  for all (i,j) in A
    # ------------------------------------------------------------------
    for (i, j), val in x_T.items():
        if val < 0.5:
            continue
        ai = a_val(i)
        aj = a_val("0prime") if j == 0 else a_val(j)
        t_ij = truck_times[i][j]
        lhs = ai + t_ij
        rhs_val = aj + M * (1 - val)
        excess = lhs - rhs_val
        if excess > TOL:
            violated_constraints.add(8)
            j_label = "0'" if j == 0 else str(j)
            add_violation(violations, violation_magnitudes, 8,
                          f"Truck subtour/timing on arc ({i},{j_label}): "
                          f"a[{i}]={ai} + t^T={t_ij} = {lhs} > a[{j_label}]={aj}",
                          lhs, rhs_val, excess)

    # ------------------------------------------------------------------
    # Constraint 9 (1j): Drone arrival time / subtour elimination
    # a_i + t^D_{ij} <= a_j + M(1 - x^D_{ij})  for all (i,j) in A
    # ------------------------------------------------------------------
    for (i, j), val in x_D.items():
        if val < 0.5:
            continue
        ai = a_val(i)
        aj = a_val("0prime") if j == 0 else a_val(j)
        t_ij = drone_times[i][j]
        lhs = ai + t_ij
        rhs_val = aj + M * (1 - val)
        excess = lhs - rhs_val
        if excess > TOL:
            violated_constraints.add(9)
            j_label = "0'" if j == 0 else str(j)
            add_violation(violations, violation_magnitudes, 9,
                          f"Drone subtour/timing on arc ({i},{j_label}): "
                          f"a[{i}]={ai} + t^D={t_ij} = {lhs} > a[{j_label}]={aj}",
                          lhs, rhs_val, excess)

    # ------------------------------------------------------------------
    # Constraint 10 (1k): Drone leg single drone customer
    # x^D_{ij} + x^D_{ji} <= y^C_i + y^C_j  for all i,j in N, i < j
    # ------------------------------------------------------------------
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            lhs = x_D_val(i, j) + x_D_val(j, i)
            rhs_val = y_C.get(i, 0) + y_C.get(j, 0)
            excess = lhs - rhs_val
            if excess > TOL:
                violated_constraints.add(10)
                add_violation(violations, violation_magnitudes, 10,
                              f"Drone leg constraint violated for ({i},{j}): "
                              f"x^D_ij+x^D_ji={lhs} > y^C_i+y^C_j={rhs_val}",
                              lhs, rhs_val, excess)

    # ------------------------------------------------------------------
    # Constraint 11 (1l): No loops rooted at depot
    # x^D_{0i} + x^D_{i0'} <= 1  for all i in N
    # ------------------------------------------------------------------
    for i in customers:
        lhs = x_D_val(0, i) + x_D_val(i, 0)
        if lhs - 1 > TOL:
            violated_constraints.add(11)
            add_violation(violations, violation_magnitudes, 11,
                          f"Drone depot loop for customer {i}: "
                          f"x^D_{{0,{i}}} + x^D_{{{i},0'}} = {lhs} > 1",
                          lhs, 1.0, lhs - 1)

    # ------------------------------------------------------------------
    # Constraint 12 (1m): Binary domain x^T, x^D
    # ------------------------------------------------------------------
    for (i, j), val in x_T.items():
        if not _is_binary(val):
            violated_constraints.add(12)
            add_violation(violations, violation_magnitudes, 12,
                          f"x^T_{{{i},{j}}} = {val} not binary",
                          val, round(val), abs(val - round(val)))
    for (i, j), val in x_D.items():
        if not _is_binary(val):
            violated_constraints.add(12)
            add_violation(violations, violation_magnitudes, 12,
                          f"x^D_{{{i},{j}}} = {val} not binary",
                          val, round(val), abs(val - round(val)))

    # ------------------------------------------------------------------
    # Constraint 13 (1n): Binary domain y^T, y^D, y^C
    # ------------------------------------------------------------------
    for i in customers:
        for label, ydict in [("y^T", y_T), ("y^D", y_D), ("y^C", y_C)]:
            val = ydict.get(i, 0)
            if not _is_binary(val):
                violated_constraints.add(13)
                add_violation(violations, violation_magnitudes, 13,
                              f"{label}_{i} = {val} not binary",
                              val, round(val), abs(val - round(val)))

    # ------------------------------------------------------------------
    # Constraint 14 (1o): Non-negative arrival times
    # a_i >= 0 for all i in V
    # ------------------------------------------------------------------
    for node, val in a.items():
        if isinstance(val, (int, float)) and val < -TOL:
            violated_constraints.add(14)
            node_label = "0'" if node == "0prime" else str(node)
            add_violation(violations, violation_magnitudes, 14,
                          f"Arrival time a[{node_label}] = {val} < 0",
                          val, 0.0, abs(val))

    # ------------------------------------------------------------------
    # Constraint 15: Objective consistency (Tier C anti-exploit)
    # Reported objective_value must equal the tour-completion time
    # a_{0'} recomputed from the truck/drone routes and the instance
    # travel times, with truck and drone synchronised at the depot and
    # at every combined customer.  The recompute is exact when the two
    # routes share a common sync-point sequence; if not, the routes are
    # already infeasible under the other constraints and we skip this
    # check (no false positives).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    true_obj = _recompute_a_0prime(instance, solution)
    if true_obj is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor; travel
            # times are integers so an honest solution matches exactly
            # modulo Gurobi's float-arithmetic noise.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(15)
                add_violation(
                    violations, violation_magnitudes, 15,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed a_{{0'}}={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})",
                    reported, true_obj, abs_diff,
                )

    violated_list = sorted(violated_constraints)
    feasible = len(violated_list) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_list,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for TSP-D (Roberti & Ruthmair, 2021)"
    )
    parser.add_argument(
        "--instance_path", required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
