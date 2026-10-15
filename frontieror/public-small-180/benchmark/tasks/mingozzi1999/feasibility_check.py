"""
Feasibility checker for the Vehicle Routing Problem with Backhauls (VRPB).

Reference: Mingozzi, Giorgi, Baldacci (1999),
"An Exact Method for the Vehicle Routing Problem with Backhauls",
Transportation Science 33(3):315-329.

Constraints numbered 1-6 match the paper's formulation order (Eqs. 4-9):
  Constraint 1 (Eq. 4): Each linehaul customer visited exactly once.
  Constraint 2 (Eq. 5): Each backhaul customer visited exactly once.
  Constraint 3 (Eq. 6): Linehaul-path-end / arc linkage for each i in L.
  Constraint 4 (Eq. 7): Backhaul-path-start / arc linkage for each j in B.
  Constraint 5 (Eq. 8): Exactly M connecting arcs (i.e., exactly M routes).
  Constraint 6 (Eq. 9): Integrality / binary (routes are well-formed).
  Constraint 7 (Eq. 3): Objective consistency — reported objective_value
                        must equal the sum of distance_matrix[i][j] over
                        consecutive arcs of all routes (Tier C anti-gaming).

Additionally, from the feasibility conditions and reproduction-critical comments:
  - Each route must start and end at the depot.
  - All linehaul customers in a route must precede all backhaul customers.
  - Linehaul load on each route must not exceed Q.
  - Backhaul load on each route must not exceed Q.
  - Routes containing only backhaul customers are NOT allowed.
These structural requirements are embedded in the definition of feasible
paths in L and B and in the arc set A_0, so violations of these are
reported under the corresponding constraint that they make infeasible.
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'routes')
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def check_feasibility(instance, solution):
    violations = []
    violation_magnitudes = []

    # ---- Extract instance data ----
    depot = instance["depot"]["index"]
    Q = instance["vehicle_capacity"]
    M = instance["num_vehicles"]
    n = instance["num_linehaul_customers"]
    m = instance["num_backhaul_customers"]

    linehaul_set = set()
    backhaul_set = set()
    demands = {}
    demands[depot] = 0

    for c in instance["linehaul_customers"]:
        idx = c["index"]
        linehaul_set.add(idx)
        demands[idx] = c["demand"]

    for c in instance["backhaul_customers"]:
        idx = c["index"]
        backhaul_set.add(idx)
        demands[idx] = c["demand"]

    all_customers = linehaul_set | backhaul_set

    routes = solution["routes"]

    # ================================================================
    # Constraint 6 (Eq. 9): Integrality / binary — routes well-formed
    # Each route must start and end at depot, contain only valid
    # customer indices, and not be empty of customers.
    # A route with only backhaul customers is also disallowed
    # (reproduction comment 2), which violates the definition of
    # feasible paths / arcs in A_0, hence constraint 6.
    # ================================================================
    for r_idx, route in enumerate(routes):
        # Route must start and end at depot
        if len(route) < 2:
            lhs = 0.0
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                msg = f"Route {r_idx}: too short (length {len(route)}), must start and end at depot"
                violations.append((6, msg, lhs, rhs, violation_amount))
            continue

        if route[0] != depot:
            lhs = float(route[0])
            rhs = float(depot)
            violation_amount = abs(lhs - rhs)
            msg = f"Route {r_idx}: does not start at depot (starts at {route[0]})"
            violations.append((6, msg, lhs, rhs, violation_amount))

        if route[-1] != depot:
            lhs = float(route[-1])
            rhs = float(depot)
            violation_amount = abs(lhs - rhs)
            msg = f"Route {r_idx}: does not end at depot (ends at {route[-1]})"
            violations.append((6, msg, lhs, rhs, violation_amount))

        # Check for invalid customer indices in interior
        interior = route[1:-1]
        for node in interior:
            if node != depot and node not in all_customers:
                lhs = 1.0
                rhs = 0.0
                violation_amount = abs(lhs - rhs)
                msg = f"Route {r_idx}: contains invalid node {node}"
                violations.append((6, msg, lhs, rhs, violation_amount))

        # Check linehaul-before-backhaul ordering.
        # A feasible route is: depot -> linehaul segment -> (arc to backhaul or depot) -> backhaul segment -> depot
        # Once a backhaul customer is encountered, no linehaul customer may follow.
        seen_backhaul = False
        has_linehaul = False
        for node in interior:
            if node in backhaul_set:
                seen_backhaul = True
            elif node in linehaul_set:
                if seen_backhaul:
                    lhs = 1.0
                    rhs = 0.0
                    violation_amount = abs(lhs - rhs)
                    msg = (f"Route {r_idx}: linehaul customer {node} appears after "
                           f"backhaul customer(s) — violates linehaul-before-backhaul ordering")
                    violations.append((6, msg, lhs, rhs, violation_amount))
                else:
                    has_linehaul = True

        # Routes containing only backhaul customers are NOT allowed (comment 2)
        has_backhaul_only = (not has_linehaul and
                            any(node in backhaul_set for node in interior))
        if has_backhaul_only:
            lhs = 0.0
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            msg = f"Route {r_idx}: contains only backhaul customers (backhaul-only routes not allowed)"
            violations.append((6, msg, lhs, rhs, violation_amount))

        # Check that linehaul load on this route does not exceed Q
        linehaul_load = sum(demands.get(node, 0) for node in interior if node in linehaul_set)
        if linehaul_load > Q + TOL:
            lhs = float(linehaul_load)
            rhs = float(Q)
            violation_amount = lhs - rhs
            msg = (f"Route {r_idx}: linehaul load {linehaul_load} exceeds "
                   f"vehicle capacity {Q}")
            violations.append((6, msg, lhs, rhs, violation_amount))

        # Check that backhaul load on this route does not exceed Q
        backhaul_load = sum(demands.get(node, 0) for node in interior if node in backhaul_set)
        if backhaul_load > Q + TOL:
            lhs = float(backhaul_load)
            rhs = float(Q)
            violation_amount = lhs - rhs
            msg = (f"Route {r_idx}: backhaul load {backhaul_load} exceeds "
                   f"vehicle capacity {Q}")
            violations.append((6, msg, lhs, rhs, violation_amount))

    # ================================================================
    # Constraint 1 (Eq. 4): Each linehaul customer visited exactly once
    # sum_{l in L_i} x_l = 1, for all i in L
    # LHS = number of times customer i appears across all routes
    # RHS = 1
    # ================================================================
    linehaul_visit_count = {i: 0 for i in linehaul_set}
    for route in routes:
        for node in route[1:-1]:
            if node in linehaul_set:
                linehaul_visit_count[node] += 1

    for i in sorted(linehaul_set):
        count = linehaul_visit_count[i]
        lhs = float(count)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            if count == 0:
                msg = f"Linehaul customer {i} is not visited"
            else:
                msg = f"Linehaul customer {i} visited {count} times (should be exactly once)"
            violations.append((1, msg, lhs, rhs, violation_amount))

    # ================================================================
    # Constraint 2 (Eq. 5): Each backhaul customer visited exactly once
    # sum_{l in B_j} y_l = 1, for all j in B
    # LHS = number of times customer j appears across all routes
    # RHS = 1
    # ================================================================
    backhaul_visit_count = {j: 0 for j in backhaul_set}
    for route in routes:
        for node in route[1:-1]:
            if node in backhaul_set:
                backhaul_visit_count[node] += 1

    for j in sorted(backhaul_set):
        count = backhaul_visit_count[j]
        lhs = float(count)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            if count == 0:
                msg = f"Backhaul customer {j} is not visited"
            else:
                msg = f"Backhaul customer {j} visited {count} times (should be exactly once)"
            violations.append((2, msg, lhs, rhs, violation_amount))

    # ================================================================
    # Constraint 3 (Eq. 6): Linehaul-path-end / arc linkage
    # For each i in L:
    #   sum_{l in L_i^E} x_l - sum_{j in B_0} xi_{ij} = 0
    # In route-level terms: the number of routes whose linehaul segment
    # ends at customer i must equal the number of connecting arcs
    # from i (to a backhaul customer or depot).
    # In a well-formed solution each route has exactly one such arc,
    # so this is automatically satisfied if:
    #   (# routes ending linehaul at i) == (# arcs from i to B_0 in solution)
    # Both sides count the same thing from the route structure, so we
    # check that every route has exactly one transition arc from the
    # last linehaul customer.
    # ================================================================
    # Extract the connecting arc (last linehaul -> first backhaul or depot)
    # for each route, and verify consistency.
    linehaul_end_count = {i: 0 for i in linehaul_set}
    arc_from_count = {i: 0 for i in linehaul_set}

    for r_idx, route in enumerate(routes):
        interior = route[1:-1]
        if not interior:
            continue

        # Find linehaul segment and backhaul segment
        lh_segment = []
        bh_segment = []
        seen_bh = False
        for node in interior:
            if node in backhaul_set:
                seen_bh = True
                bh_segment.append(node)
            elif node in linehaul_set:
                if not seen_bh:
                    lh_segment.append(node)
                # If seen_bh and linehaul node appears, it's an ordering violation
                # already caught in constraint 6

        if lh_segment:
            last_lh = lh_segment[-1]
            linehaul_end_count[last_lh] += 1
            # The arc from last_lh goes to first backhaul customer or depot
            arc_from_count[last_lh] += 1

    for i in sorted(linehaul_set):
        lhs = float(linehaul_end_count[i] - arc_from_count[i])
        rhs = 0.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            msg = (f"Linehaul customer {i}: linehaul-path-end count "
                   f"({linehaul_end_count[i]}) != arc-from count ({arc_from_count[i]})")
            violations.append((3, msg, lhs, rhs, violation_amount))

    # ================================================================
    # Constraint 4 (Eq. 7): Backhaul-path-start / arc linkage
    # For each j in B:
    #   sum_{l in B_j^S} y_l - sum_{i in L} xi_{ij} = 0
    # In route-level terms: the number of routes whose backhaul segment
    # starts at customer j must equal the number of connecting arcs
    # from some linehaul customer to j.
    # ================================================================
    backhaul_start_count = {j: 0 for j in backhaul_set}
    arc_to_count = {j: 0 for j in backhaul_set}

    for r_idx, route in enumerate(routes):
        interior = route[1:-1]
        if not interior:
            continue

        # Find first backhaul customer in the route
        bh_segment = []
        for node in interior:
            if node in backhaul_set:
                bh_segment.append(node)

        if bh_segment:
            first_bh = bh_segment[0]
            backhaul_start_count[first_bh] += 1
            arc_to_count[first_bh] += 1

    for j in sorted(backhaul_set):
        lhs = float(backhaul_start_count[j] - arc_to_count[j])
        rhs = 0.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            msg = (f"Backhaul customer {j}: backhaul-path-start count "
                   f"({backhaul_start_count[j]}) != arc-to count ({arc_to_count[j]})")
            violations.append((4, msg, lhs, rhs, violation_amount))

    # ================================================================
    # Constraint 5 (Eq. 8): Exactly M connecting arcs
    # sum_{(i,j) in A_0} xi_{ij} = M
    # A_0 arcs originate from a linehaul customer (i in L) and go to
    # a backhaul customer or depot (j in B_0). Each route with at
    # least one linehaul customer contributes exactly one such arc.
    # Routes without any linehaul customer (e.g., empty [0,0]) do
    # not contribute a connecting arc.
    # ================================================================
    num_connecting_arcs = 0
    for route in routes:
        interior = route[1:-1]
        if any(node in linehaul_set for node in interior):
            num_connecting_arcs += 1
    lhs = float(num_connecting_arcs)
    rhs = float(M)
    violation_amount = abs(lhs - rhs)
    if violation_amount > TOL:
        msg = (f"Number of connecting arcs (A_0) is {num_connecting_arcs}, "
               f"should be exactly {M}")
        violations.append((5, msg, lhs, rhs, violation_amount))

    # ================================================================
    # Constraint 7 (Eq. 3): Objective consistency (Tier C anti-gaming)
    # The reported objective_value must equal the sum of
    # distance_matrix[route[i]][route[i+1]] over all consecutive arcs
    # of all routes. The set-partitioning objective
    #   z = sum_l c_l x_l + sum_l c_l y_l + sum_(i,j) d_ij xi_ij
    # is, at the route level, exactly the total travel distance across
    # all selected routes (since each c_l is a sum of arc costs along
    # the path and the inter-segment arc d_ij is the connector). All
    # variables determining the obj (the routes) are present in the
    # solution, so a full recompute applies.
    # ================================================================
    reported_obj = solution.get("objective_value")
    distance_matrix = instance.get("distance_matrix")
    if reported_obj is not None and distance_matrix is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for route in routes:
                for a, b in zip(route[:-1], route[1:]):
                    true_obj += float(distance_matrix[a][b])
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor; distance
            # matrices here are integer-valued so any real exploit
            # (obj=0, obj=MAX_FLOAT, obj=1e6*ref) is well outside this.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                lhs = reported
                rhs = true_obj
                violation_amount = abs_diff
                msg = (f"Objective consistency violated: reported objective_value="
                       f"{reported} differs from recomputed total route distance="
                       f"{true_obj} (|diff|={abs_diff:.3g}, tol={tol:.3g})")
                violations.append((7, msg, lhs, rhs, violation_amount))

    # ================================================================
    # Build output
    # ================================================================
    violated_constraint_indices = sorted(set(v[0] for v in violations))
    violation_messages = []
    magnitude_list = []

    for v in violations:
        c_idx, msg, lhs_val, rhs_val, raw_excess = v
        violation_messages.append(msg)
        normalizer = max(abs(rhs_val), EPS)
        ratio = raw_excess / normalizer
        magnitude_list.append({
            "constraint": c_idx,
            "lhs": lhs_val,
            "rhs": rhs_val,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": ratio
        })

    feasible = len(violations) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraint_indices,
        "violations": violation_messages,
        "violation_magnitudes": magnitude_list
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for VRPB (Mingozzi et al., 1999)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
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
    result = check_feasibility(instance, solution)
    save_json(result, args.result_path)
    print(f"Feasibility result saved to {args.result_path}")
    print(f"  Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"  Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"    - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
