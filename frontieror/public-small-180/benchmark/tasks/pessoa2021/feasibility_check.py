#!/usr/bin/env python3
"""
Feasibility checker for the Robust Capacitated Vehicle Routing Problem (RCVRP)
with Knapsack Uncertainty, based on Pessoa et al. (2021).

Checks constraints numbered from the formulation (top to bottom):
  Constraint 1 (Eq 15): Each customer visited exactly once
  Constraint 2 (Eq 16): Exactly m vehicles used
  Constraint 3 (Eq 17): Integrality (lambda_r in Z+)
  Constraint 4 (Eq 18): Robust rounded capacity inequalities
  Constraint 5 (Eq 19): New robust capacity inequalities
  Constraint 6 (Eq 20): Strongest combined capacity inequalities
  Constraint 7 (Eq 14): Objective consistency -- the reported
      objective_value must equal the total transportation cost
      recomputed from the routes using the distance matrix
      (depot -> custs[0] -> ... -> custs[-1] -> depot).

For integer route-based solutions, constraints 4-6 reduce to checking that
each route's worst-case demand does not exceed vehicle capacity C.
"""

import argparse
import json
import math

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



def compute_worst_case_demand_card(customer_ids, bar_d, hat_d, Gamma):
    """
    For D^{card}: worst-case demand = sum of nominal demands + sum of
    Gamma largest deviations among customers in the route.
    """
    nominal = sum(bar_d[i] for i in customer_ids)
    devs = sorted([hat_d[i] for i in customer_ids], reverse=True)
    dev_sum = sum(devs[:min(Gamma, len(devs))])
    return nominal + dev_sum


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # If the solver reported infeasible/no solution, return null (not false)
    status = solution.get('status', '')
    obj = solution.get('objective_value')
    if obj is None or (isinstance(status, str) and status.lower() in ('infeasible', 'no_solution')):
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No solution found (solver status: {}).".format(status)],
            "violation_magnitudes": [],
        }

    n = instance['num_customers']
    m = instance['num_vehicles']
    C = instance['vehicle_capacity']
    customers = instance['customers']
    dist = instance['distance_matrix']
    unc = instance['uncertainty_set']
    Gamma = unc['Gamma']

    bar_d = {c['id']: c['nominal_demand'] for c in customers}
    hat_d = {c['id']: c['demand_deviation'] for c in customers}

    routes = solution.get('routes', [])

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 (Eq 15): Each customer i in V0 is visited exactly once
    #   sum_{theta} sum_{r in R_theta} a_i^r * lambda_r = 1, for all i in V0
    # =========================================================================
    visit_count = {}
    for route in routes:
        for cust in route['customers']:
            visit_count[cust] = visit_count.get(cust, 0) + 1

    for i in range(1, n + 1):
        count = visit_count.get(i, 0)
        lhs = float(count)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            violated_constraints.add(1)
            if count == 0:
                violations.append(f"Customer {i} is not visited by any route")
            else:
                violations.append(
                    f"Customer {i} is visited {count} times instead of exactly once"
                )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint 2 (Eq 16): Exactly m vehicles are used
    #   sum_{theta} sum_{r in R_theta} lambda_r = m
    # =========================================================================
    num_routes = len(routes)
    lhs = float(num_routes)
    rhs = float(m)
    violation_amount = abs(lhs - rhs)
    if violation_amount > tol:
        normalizer = max(abs(rhs), eps)
        violated_constraints.add(2)
        violations.append(
            f"Number of routes is {num_routes}, expected exactly {m}"
        )
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # =========================================================================
    # Constraint 3 (Eq 17): Integrality — lambda_r in Z+
    # In route-based solutions each route is used exactly once (lambda=1).
    # We verify that route usage counts are non-negative integers and that
    # customer IDs are valid.
    # =========================================================================
    for r_idx, route in enumerate(routes):
        for cust in route['customers']:
            if not isinstance(cust, int) or cust < 1 or cust > n:
                lhs = float(cust) if isinstance(cust, (int, float)) else 0.0
                rhs = 0.0  # expected valid range [1, n]
                violation_amount = 1.0
                normalizer = max(abs(rhs), eps)
                violated_constraints.add(3)
                violations.append(
                    f"Route {r_idx + 1} contains invalid customer ID {cust} "
                    f"(must be integer in [1, {n}])"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # =========================================================================
    # Constraints 4-6 (Eqs 18-20): Robust capacity inequalities
    #
    # For integer route-based solutions where customers are partitioned among
    # routes, all three capacity inequalities reduce to verifying that each
    # route's worst-case demand does not exceed vehicle capacity C:
    #   max_{d in D} sum_{i in route} d_i <= C
    #
    # For D^{card}: worst_case = sum(bar_d_i) + sum of Gamma largest hat_d_i
    #
    # A violation of route capacity implies violation of Eq (18) for S = route
    # customers, since ceil(worst_case / C) > 1 but only 1 vehicle serves them.
    # Eqs (19) and (20) give tighter bounds but are also violated whenever
    # Eq (18) is violated for an individual route.
    # =========================================================================
    for r_idx, route in enumerate(routes):
        custs = route['customers']
        if not custs:
            continue

        # Guard: only score if every customer is in bar_d (otherwise constraint
        # 3 will already have flagged the invalid ID and a KeyError would
        # break the checker).
        if not all(c in bar_d for c in custs):
            continue

        worst_case = compute_worst_case_demand_card(custs, bar_d, hat_d, Gamma)
        lhs = float(worst_case)
        rhs_cap = float(C)

        # This is a <= constraint: worst_case_demand <= C
        violation_amount = max(0.0, lhs - rhs_cap)
        if violation_amount > tol:
            normalizer = max(abs(rhs_cap), eps)

            # Report under constraint 4 (Eq 18) — the primary capacity inequality
            violated_constraints.add(4)
            violations.append(
                f"Route {r_idx + 1} (customers {custs}): worst-case demand "
                f"{worst_case} exceeds vehicle capacity {C}"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs,
                "rhs": rhs_cap,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

            # The same route also violates Eqs (19) and (20)
            violated_constraints.add(5)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": lhs,
                "rhs": rhs_cap,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

            violated_constraints.add(6)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs,
                "rhs": rhs_cap,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint 7 (Eq 14): Objective consistency.
    # Full recompute -- every variable the obj depends on (the route arcs) is
    # present in the solution, and arc costs c_{ij} are given by the instance
    # distance matrix. Each route is the closed tour
    #   depot(0) -> custs[0] -> ... -> custs[-1] -> depot(0)
    # and the objective is sum over routes of the sum of arc costs.
    # Reject when reported_obj disagrees with the recomputed total.
    # =========================================================================
    try:
        reported = float(obj)
    except (TypeError, ValueError):
        reported = None

    if reported is not None and isinstance(dist, list) and len(dist) > 0:
        depot_id = 0
        n_nodes = len(dist)
        recompute_ok = True
        true_obj = 0.0
        for route in routes:
            custs = route['customers']
            prev = depot_id
            for c in custs:
                if not isinstance(c, int) or c < 0 or c >= n_nodes:
                    # Invalid ID -- constraint 3 already covers it; skip
                    # obj recompute on this route rather than crash.
                    recompute_ok = False
                    break
                true_obj += float(dist[prev][c])
                prev = c
            if not recompute_ok:
                break
            if not (0 <= prev < n_nodes):
                recompute_ok = False
                break
            true_obj += float(dist[prev][depot_id])

        if recompute_ok:
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance, 1e-3 absolute floor.
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                normalizer = max(abs(true_obj), eps)
                violated_constraints.add(7)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum of route arc costs="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": float(abs_diff) / normalizer,
                })

    # =========================================================================
    # Build result
    # =========================================================================
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Robust CVRP (Pessoa et al. 2021)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution"
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result"
    )
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    with open(args.solution_path, 'r') as f:
        solution = json.load(f)
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print("Solution is INFEASIBLE.")
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
