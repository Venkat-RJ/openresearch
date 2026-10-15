#!/usr/bin/env python3
"""
Feasibility checker for the Two-Dimensional Vector Packing Problem (2DVPP).

Checks a candidate solution against the hard constraints of the set-partitioning
formulation (F) from Wei et al. (2019).

Constraint numbering (top-to-bottom in the formulation sections):
  Constraint 1 — eq(2): each item packed in exactly one bin (= 1)
  Constraint 2 — eq(3): binary domain for lambda_p (structural)
  Constraint 3 — eq(4): covering in master LP (LP-specific, skip)
  Constraint 4 — eq(5): LP relaxation bounds (LP-specific, skip)
  Constraint 5 — eq(7): weight capacity per bin <= W
  Constraint 6 — eq(8): volume capacity per bin <= V
  Constraint 7 — eq(9): incompatible pairs (branching-specific, skip)
  Constraint 8 — eq(10): companion pairs (branching-specific, skip)
  Constraint 9 — eq(11): binary y_i in pricing (pricing-specific, skip)
  Constraint 10 — eq(12): goal cut (valid inequality, skip)
  Constraint 11 — eq(1): objective consistency — reported objective_value
                          must equal the recomputed number of non-empty bins.

Hard constraints checked: 1, 2, 5, 6, 11.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('bins', 'objective_value')
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


def extract_bins(solution):
    """Extract list of bins (each a list of item indices) from solution JSON.
    Handles both efficient_algorithm and gurobi solution formats.
    """
    raw_bins = solution["bins"]
    bins = []
    for b in raw_bins:
        if isinstance(b, dict):
            bins.append(list(b["items"]))
        else:
            bins.append(list(b))
    return bins


def check_feasibility(instance, solution):
    n = instance["n"]
    W = instance["W"]
    V = instance["V"]
    items = instance["items"]

    bins = extract_bins(solution)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 1 — eq(2): sum_p a_{ip} lambda_p = 1  for all i in I
    # Each item must appear in exactly one bin.
    # ------------------------------------------------------------------
    item_count = [0] * n
    for bin_items in bins:
        for i in bin_items:
            if 0 <= i < n:
                item_count[i] += 1

    for i in range(n):
        lhs = float(item_count[i])
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violated_constraints.add(1)
            if item_count[i] == 0:
                violations.append(f"Item {i} is not assigned to any bin")
            else:
                violations.append(
                    f"Item {i} is assigned to {item_count[i]} bins instead of exactly 1"
                )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 2 — eq(3): lambda_p in {0, 1}
    # Structural: each bin is either used or not. We verify that the
    # solution only references valid item indices (non-negative integers
    # within range), which is the operational meaning of this constraint.
    # ------------------------------------------------------------------
    for b_idx, bin_items in enumerate(bins):
        for i in bin_items:
            if not isinstance(i, int) or i < 0 or i >= n:
                lhs = float(i) if isinstance(i, (int, float)) else 0.0
                rhs_lo = 0.0
                rhs_hi = float(n - 1)
                violation_amount = 1.0  # binary violation
                normalizer = max(abs(1.0), EPS)
                ratio = violation_amount / normalizer
                violated_constraints.add(2)
                violations.append(
                    f"Bin {b_idx} contains invalid item index {i} (valid range: 0..{n-1})"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": 1.0,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # ------------------------------------------------------------------
    # Constraint 5 — eq(7): sum_{i in bin} w_i <= W  for each bin
    # ------------------------------------------------------------------
    for b_idx, bin_items in enumerate(bins):
        total_w = sum(items[i]["w"] for i in bin_items if 0 <= i < n)
        lhs = float(total_w)
        rhs = float(W)
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > TOL:
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violated_constraints.add(5)
            violations.append(
                f"Bin {b_idx} exceeds weight capacity: {total_w} > {W}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 6 — eq(8): sum_{i in bin} v_i <= V  for each bin
    # ------------------------------------------------------------------
    for b_idx, bin_items in enumerate(bins):
        total_v = sum(items[i]["v"] for i in bin_items if 0 <= i < n)
        lhs = float(total_v)
        rhs = float(V)
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > TOL:
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violated_constraints.add(6)
            violations.append(
                f"Bin {b_idx} exceeds volume capacity: {total_v} > {V}"
            )
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 11 — eq(1): objective consistency.
    # z = sum_b x_b = number of non-empty bins. The solution carries the
    # full bin partition, so we recompute z exactly from `bins` and reject
    # when the reported objective_value disagrees. Tolerance 0.5 since the
    # true objective is an integer count of bins.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(1 for bin_items in bins if len(bin_items) > 0))
            abs_diff = abs(reported - true_obj)
            tol = 0.5  # integer count of bins; off-by-one or more must fire
            if abs_diff > tol:
                normalizer = max(abs(true_obj), EPS)
                ratio = abs_diff / normalizer
                violated_constraints.add(11)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed number of non-empty bins="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 11,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
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
        description="Feasibility checker for 2DVPP solutions"
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

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    with open(args.solution_path, "r") as f:
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

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE — no constraint violations detected.")
    else:
        print(f"INFEASIBLE — violated constraints: {result['violated_constraints']}")
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
