#!/usr/bin/env python3
"""
Feasibility checker for PSCLP and MCLP solutions from:
Cordeau, Furini, and Ljubic (2019) - "Benders decomposition for very large
scale partial set covering and maximal covering location problems."

Constraints numbered top-to-bottom across both formulations:
  PSCLP:
    Constraint 1: sum_{i in I(j)} y_i >= z_j,  for all j in J  (linking)
    Constraint 2: sum_{j in J} d_j z_j >= D                     (demand coverage)
    Constraint 3: y_i in {0,1},  for all i in I                  (binary y)
    Constraint 4: z_j in {0,1},  for all j in J                  (binary z)
  MCLP:
    Constraint 5: sum_{i in I} f_i y_i <= B                      (budget)
    Constraint 6: sum_{i in I(j)} y_i >= z_j,  for all j in J   (linking)
    Constraint 7: y_i in {0,1},  for all i in I                  (binary y)
    Constraint 8: z_j in {0,1},  for all j in J                  (binary z)
  Both:
    Constraint 9: objective-value consistency                    (anti-gaming)
      The self-reported objective_value must equal the objective recomputed
      from `open_facilities`:
        PSCLP objective (1): total opening cost  = sum_{i in I} f_i y_i
        MCLP  objective (6): total covered demand = sum_{j in J} d_j z_j
      Both objectives are fully determined by `open_facilities` (z_j is
      derived from coverage), so a full recompute is possible. This is a
      Tier C defense against candidates that fabricate objective_value while
      keeping the routes/decisions technically feasible. Constraint 9 is only
      ever ADDED to a verdict; constraints 1-8 are checked by the unmodified
      `check_feasibility` and never altered here.

This is the obj-recompute variant of `feasibility_check.py`. `check_feasibility`
is byte-identical to the original; the objective consistency check lives
entirely in `main()`, just before the overall feasibility decision.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'primary_problem_type', 'results')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('results',)
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


def check_feasibility(instance, problem_type, open_facilities):
    """Check feasibility of a solution for a given problem type.

    Returns (violated_constraints, violations, violation_magnitudes).
    """
    tol = 1e-5
    eps = 1e-5

    num_facilities = instance["num_facilities"]
    num_customers = instance["num_customers"]
    customer_demands = instance["customer_demands"]
    facility_cost = instance["facility_cost"]
    cust_to_fac = instance["customer_to_facilities_coverage"]

    violated_constraints = []
    violations = []
    violation_magnitudes = []

    # Build y vector
    y = [0] * num_facilities
    for i in open_facilities:
        y[i] = 1

    # Derive z: z_j = 1 if any facility in I(j) is open, else 0
    open_set = set(open_facilities)
    z = [0] * num_customers
    for j in range(num_customers):
        I_j = cust_to_fac.get(str(j), [])
        for i in I_j:
            if i in open_set:
                z[j] = 1
                break

    if problem_type == "PSCLP":
        D = instance["covering_demand_D"]

        # Constraint 1: sum_{i in I(j)} y_i >= z_j for all j
        # Since z_j is derived as 1 only when sum >= 1, check explicitly
        c1_violated = False
        for j in range(num_customers):
            I_j = cust_to_fac.get(str(j), [])
            lhs = sum(y[i] for i in I_j)
            rhs = z[j]
            violation_amount = rhs - lhs  # >= constraint: violation if rhs > lhs
            if violation_amount > tol:
                if not c1_violated:
                    c1_violated = True
                    violated_constraints.append(1)
                    violations.append(
                        f"Linking constraint violated: customer {j} has z_j={rhs} "
                        f"but sum of y_i over I(j) = {lhs}"
                    )
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

        # Constraint 2: sum_{j in J} d_j z_j >= D
        covered_demand = sum(customer_demands[j] * z[j] for j in range(num_customers))
        violation_amount = D - covered_demand  # >= constraint
        if violation_amount > tol:
            violated_constraints.append(2)
            violations.append(
                f"Demand coverage violated: covered demand = {covered_demand}, "
                f"required D = {D}, shortfall = {violation_amount}"
            )
            normalizer = max(abs(D), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(covered_demand),
                "rhs": float(D),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

        # Constraint 3: y_i in {0,1}
        c3_violated = False
        for i in range(num_facilities):
            if y[i] not in (0, 1):
                if not c3_violated:
                    c3_violated = True
                    violated_constraints.append(3)
                    violations.append(
                        f"Binary constraint on y violated: y_{i} = {y[i]}"
                    )
                val = y[i]
                violation_amount = min(abs(val), abs(val - 1))
                normalizer = max(1.0, eps)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(val),
                    "rhs": float(round(val)),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

        # Constraint 4: z_j in {0,1}
        c4_violated = False
        for j in range(num_customers):
            if z[j] not in (0, 1):
                if not c4_violated:
                    c4_violated = True
                    violated_constraints.append(4)
                    violations.append(
                        f"Binary constraint on z violated: z_{j} = {z[j]}"
                    )
                val = z[j]
                violation_amount = min(abs(val), abs(val - 1))
                normalizer = max(1.0, eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(val),
                    "rhs": float(round(val)),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

        # Also check validity of facility indices
        for i in open_facilities:
            if i < 0 or i >= num_facilities:
                if 3 not in violated_constraints:
                    violated_constraints.append(3)
                    violations.append(
                        f"Invalid facility index: {i} not in [0, {num_facilities-1}]"
                    )

    elif problem_type == "MCLP":
        B = instance["budget_B"]

        # Constraint 5: sum_{i in I} f_i y_i <= B
        total_cost = sum(facility_cost[i] * y[i] for i in range(num_facilities))
        violation_amount = total_cost - B  # <= constraint: violation if lhs > rhs
        if violation_amount > tol:
            violated_constraints.append(5)
            violations.append(
                f"Budget constraint violated: total cost = {total_cost}, "
                f"budget B = {B}, excess = {violation_amount}"
            )
            normalizer = max(abs(B), eps)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(total_cost),
                "rhs": float(B),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

        # Constraint 6: sum_{i in I(j)} y_i >= z_j for all j
        c6_violated = False
        for j in range(num_customers):
            I_j = cust_to_fac.get(str(j), [])
            lhs = sum(y[i] for i in I_j)
            rhs = z[j]
            violation_amount = rhs - lhs
            if violation_amount > tol:
                if not c6_violated:
                    c6_violated = True
                    violated_constraints.append(6)
                    violations.append(
                        f"Linking constraint violated: customer {j} has z_j={rhs} "
                        f"but sum of y_i over I(j) = {lhs}"
                    )
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

        # Constraint 7: y_i in {0,1}
        c7_violated = False
        for i in range(num_facilities):
            if y[i] not in (0, 1):
                if not c7_violated:
                    c7_violated = True
                    violated_constraints.append(7)
                    violations.append(
                        f"Binary constraint on y violated: y_{i} = {y[i]}"
                    )
                val = y[i]
                violation_amount = min(abs(val), abs(val - 1))
                normalizer = max(1.0, eps)
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": float(val),
                    "rhs": float(round(val)),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

        # Constraint 8: z_j in {0,1}
        c8_violated = False
        for j in range(num_customers):
            if z[j] not in (0, 1):
                if not c8_violated:
                    c8_violated = True
                    violated_constraints.append(8)
                    violations.append(
                        f"Binary constraint on z violated: z_{j} = {z[j]}"
                    )
                val = z[j]
                violation_amount = min(abs(val), abs(val - 1))
                normalizer = max(1.0, eps)
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": float(val),
                    "rhs": float(round(val)),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

        # Check validity of facility indices
        for i in open_facilities:
            if i < 0 or i >= num_facilities:
                if 7 not in violated_constraints:
                    violated_constraints.append(7)
                    violations.append(
                        f"Invalid facility index: {i} not in [0, {num_facilities-1}]"
                    )

    feasible = len(violated_constraints) == 0
    return feasible, violated_constraints, violations, violation_magnitudes


def compute_true_objective(instance, problem_type, open_facilities):
    """Recompute the true objective from the solution's `open_facilities`.

    PSCLP objective (1): total opening cost   = sum_{i in I} f_i y_i
    MCLP  objective (6): total covered demand = sum_{j in J} d_j z_j

    Both are full recomputes: `open_facilities` is the only decision variable
    carried in the solution, and z_j is derived from coverage exactly as
    `check_feasibility` derives it. Stray (out-of-range) indices are skipped
    so this recompute can never crash.
    """
    num_facilities = instance["num_facilities"]
    num_customers = instance["num_customers"]
    facility_cost = instance["facility_cost"]
    customer_demands = instance["customer_demands"]
    cust_to_fac = instance["customer_to_facilities_coverage"]

    # y vector / open set restricted to valid facility indices.
    y = [0] * num_facilities
    open_set = set()
    for i in open_facilities:
        if 0 <= i < num_facilities:
            y[i] = 1
            open_set.add(i)

    if problem_type == "PSCLP":
        # total opening cost of the selected facilities
        return float(sum(facility_cost[i] * y[i] for i in range(num_facilities)))

    # MCLP: total covered demand; z_j = 1 if any facility in I(j) is open
    total = 0
    for j in range(num_customers):
        for i in cust_to_fac.get(str(j), []):
            if i in open_set:
                total += customer_demands[j]
                break
    return float(total)


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for PSCLP/MCLP (Cordeau et al. 2019)"
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

    expected_problem_types = set(instance.get("problem_types", []))
    results = solution.get("results")
    primary = solution.get("primary_problem_type")
    result_contract_errors = []
    if not isinstance(results, dict) or set(results) != expected_problem_types:
        result_contract_errors.append(
            "results must exactly cover the instance problem_types: "
            f"expected={sorted(expected_problem_types)}, "
            f"got={sorted(results) if isinstance(results, dict) else type(results).__name__}"
        )
    if primary not in expected_problem_types:
        result_contract_errors.append(
            f"primary_problem_type={primary!r} is not declared by the instance"
        )
    if isinstance(results, dict):
        for problem_type in expected_problem_types:
            result = results.get(problem_type)
            if not isinstance(result, dict):
                result_contract_errors.append(
                    f"results.{problem_type} must be an object"
                )
                continue
            nested_problem_type = result.get("problem_type")
            if nested_problem_type is not None and nested_problem_type != problem_type:
                result_contract_errors.append(
                    f"results.{problem_type}.problem_type must equal {problem_type}"
                )
            facilities = result.get("open_facilities")
            if not isinstance(facilities, list) or not facilities:
                result_contract_errors.append(
                    f"results.{problem_type}.open_facilities must be a nonempty list"
                )
            elif len(facilities) != len(set(facilities)):
                result_contract_errors.append(
                    f"results.{problem_type}.open_facilities contains duplicates"
                )
            nested_objective = result.get("objective_value")
            if (
                isinstance(nested_objective, bool)
                or not isinstance(nested_objective, (int, float))
                or not math.isfinite(float(nested_objective))
            ):
                result_contract_errors.append(
                    f"results.{problem_type}.objective_value must be finite"
                )
    if result_contract_errors:
        with open(args.result_path, "w") as result_handle:
            json.dump(
                {
                    "feasible": False,
                    "violated_constraints": ["solution_schema"],
                    "violations": result_contract_errors,
                    "violation_magnitudes": [],
                },
                result_handle,
                indent=2,
            )
        return

    all_violated_constraints = []
    all_violations = []
    all_violation_magnitudes = []

    # --- Constraint 9: objective-value consistency (Tier C anti-gaming) ----
    # Recompute the objective from `open_facilities` and flag a violation when
    # the self-reported objective_value disagrees. Objectives here are
    # integer-valued (unit facility costs, integer customer demands) and well
    # within float64's exact-integer range, so a tight tolerance is correct:
    # any off-by-one or larger lie fires; genuine values pass exactly.
    OBJ_IDX = 9
    OBJ_TOL = 0.5

    # Recomputed true objective per problem type, reused by the top-level check.
    true_obj_by_ptype = {}

    def flag_obj(reported_raw, true_obj, label):
        """Append a constraint-9 violation if reported_raw disagrees with
        true_obj. A missing / non-numeric objective_value is left to the
        upstream eval pipeline (it rejects those independently)."""
        if reported_raw is None:
            return
        try:
            reported = float(reported_raw)
        except (TypeError, ValueError):
            return
        abs_diff = abs(reported - true_obj)
        if abs_diff > OBJ_TOL:
            all_violated_constraints.append(OBJ_IDX)
            all_violations.append(
                f"Objective consistency violated ({label}): reported "
                f"objective_value={reported} differs from recomputed "
                f"objective={true_obj} (|diff|={abs_diff:.6g}, tol={OBJ_TOL})"
            )
            normalizer = max(abs(true_obj), 1e-5)
            all_violation_magnitudes.append({
                "constraint": OBJ_IDX,
                "lhs": float(reported),
                "rhs": float(true_obj),
                "raw_excess": float(abs_diff),
                "normalizer": float(normalizer),
                "ratio": float(abs_diff / normalizer),
            })

    # Detect solution format
    if "results" in solution:
        # Gurobi format: has results dict with sub-problems
        for ptype, result in solution["results"].items():
            open_facs = result.get("open_facilities", [])
            feasible, vc, vi, vm = check_feasibility(instance, ptype, open_facs)
            all_violated_constraints.extend(vc)
            all_violations.extend(vi)
            all_violation_magnitudes.extend(vm)

            # Objective consistency for this sub-problem's reported value.
            true_obj = compute_true_objective(instance, ptype, open_facs)
            true_obj_by_ptype[ptype] = true_obj
            flag_obj(result.get("objective_value"), true_obj, f"results.{ptype}")

        # Top-level objective_value: the eval pipeline reads THIS field, so it
        # is the primary anti-gaming target. It reports the "primary problem
        # variant". Compare it precisely against the primary variant's
        # recompute when `primary_problem_type` is given; otherwise require it
        # to match at least one variant's recomputed objective.
        top_obj = solution.get("objective_value")
        if top_obj is not None and true_obj_by_ptype:
            if primary in true_obj_by_ptype:
                flag_obj(
                    top_obj, true_obj_by_ptype[primary],
                    f"top-level objective_value (primary={primary})",
                )
            else:
                try:
                    top_val = float(top_obj)
                except (TypeError, ValueError):
                    top_val = None
                if top_val is not None and not any(
                    abs(top_val - t) <= OBJ_TOL
                    for t in true_obj_by_ptype.values()
                ):
                    nearest = min(
                        true_obj_by_ptype.values(),
                        key=lambda t: abs(top_val - t),
                    )
                    flag_obj(
                        top_obj, nearest,
                        "top-level objective_value (primary unknown; "
                        "matched no problem variant)",
                    )
    else:
        # Efficient algorithm format: flat structure
        open_facs = solution.get("open_facilities", [])
        # Determine problem type from instance
        problem_types = instance.get("problem_types", [])
        if "PSCLP" in problem_types:
            ptype = "PSCLP"
        elif "MCLP" in problem_types:
            ptype = "MCLP"
        else:
            ptype = problem_types[0] if problem_types else "PSCLP"

        feasible, vc, vi, vm = check_feasibility(instance, ptype, open_facs)
        all_violated_constraints.extend(vc)
        all_violations.extend(vi)
        all_violation_magnitudes.extend(vm)

        # Objective consistency for the flat solution's reported value.
        true_obj = compute_true_objective(instance, ptype, open_facs)
        flag_obj(
            solution.get("objective_value"), true_obj,
            f"top-level objective_value ({ptype})",
        )

    # Deduplicate constraint indices
    unique_violated = sorted(set(all_violated_constraints))
    overall_feasible = len(unique_violated) == 0

    output = {
        "feasible": overall_feasible,
        "violated_constraints": unique_violated,
        "violations": all_violations,
        "violation_magnitudes": all_violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(output, f, indent=2)

    status = "FEASIBLE" if overall_feasible else "INFEASIBLE"
    print(f"Result: {status}")
    if not overall_feasible:
        print(f"Violated constraints: {unique_violated}")
        for v in all_violations:
            print(f"  - {v}")
    print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
