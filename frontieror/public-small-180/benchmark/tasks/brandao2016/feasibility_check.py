#!/usr/bin/env python3
"""
Feasibility checker for Bin Packing Problem (BPP) solutions based on
Brandao and Pedroso's General Arc-flow Formulation (Equations 17-22).

Constraints numbered top-to-bottom from the formulation:
  Constraint 1 (Eq 18): Flow conservation — each bin capacity <= W, num_bins consistent
  Constraint 2 (Eq 19): Demand >= b_i for items NOT in J (b_i > 1)
  Constraint 3 (Eq 20): Demand  = b_i for items in J (b_i = 1)
  Constraint 4 (Eq 21): Per-arc flow upper bound f_{uvi} <= b_i (per-bin item count <= b_i)
  Constraint 5 (Eq 22): Non-negativity and integrality of flow variables
  Constraint 6 (Eq 17): Objective consistency — reported objective_value must
                        equal the recomputed number of bins z = len(bins).
"""

import argparse
import json
from collections import Counter

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('bins', 'num_bins', 'objective_value')
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



def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    W = instance["parameters"]["W"]
    items_by_type = {}
    for item in instance["items"]:
        items_by_type[item["item_type"]] = {
            "weight": item["weight"],
            "demand": item["demand"],
        }

    # J = {i : b_i = 1}
    J = {t for t, info in items_by_type.items() if info["demand"] == 1}

    bins = solution["bins"]
    num_bins = solution.get("num_bins", len(bins))

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 (Eq 18): Flow conservation
    # =========================================================================
    if abs(num_bins - len(bins)) > tol:
        violated_constraints.add(1)
        lhs = float(len(bins))
        rhs = float(num_bins)
        raw_excess = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        violations.append(
            f"Stated num_bins ({num_bins}) does not match actual number of bins ({len(bins)})"
        )
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": raw_excess / normalizer,
        })

    for b_idx, bin_items in enumerate(bins):
        total_weight = 0.0
        for item_type in bin_items:
            if item_type in items_by_type:
                total_weight += items_by_type[item_type]["weight"]
        lhs = total_weight
        rhs = float(W)
        violation_amount = lhs - rhs
        if violation_amount > tol:
            violated_constraints.add(1)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Bin {b_idx + 1} exceeds capacity: total weight {total_weight} > W={W}"
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
    # Count item usage across all bins
    # =========================================================================
    usage = Counter()
    for bin_items in bins:
        for item_type in bin_items:
            usage[item_type] += 1

    # =========================================================================
    # Constraint 2 (Eq 19): For items NOT in J (b_i > 1), total usage >= b_i
    # =========================================================================
    for item_type, info in items_by_type.items():
        if item_type in J:
            continue
        demand = info["demand"]
        used = usage.get(item_type, 0)
        lhs = float(used)
        rhs = float(demand)
        violation_amount = rhs - lhs
        if violation_amount > tol:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Item type {item_type} (not in J): assigned {used} < demand {demand}"
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
    # Constraint 3 (Eq 20): For items in J (b_i = 1), total usage = b_i
    # =========================================================================
    for item_type, info in items_by_type.items():
        if item_type not in J:
            continue
        demand = info["demand"]
        used = usage.get(item_type, 0)
        lhs = float(used)
        rhs = float(demand)
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(3)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Item type {item_type} (in J): assigned {used} != demand {demand}"
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
    # Constraint 4 (Eq 21): f_{uvi} <= b_i for each item arc
    # =========================================================================
    for b_idx, bin_items in enumerate(bins):
        bin_counter = Counter(bin_items)
        for item_type, count in bin_counter.items():
            if item_type not in items_by_type:
                continue
            demand = items_by_type[item_type]["demand"]
            lhs = float(count)
            rhs = float(demand)
            violation_amount = lhs - rhs
            if violation_amount > tol:
                violated_constraints.add(4)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Bin {b_idx + 1}: item type {item_type} appears {count} times > demand {demand}"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # =========================================================================
    # Constraint 5 (Eq 22): Non-negativity and integrality
    # =========================================================================
    for b_idx, bin_items in enumerate(bins):
        for item_type in bin_items:
            if item_type not in items_by_type:
                violated_constraints.add(5)
                violations.append(
                    f"Bin {b_idx + 1}: invalid item type {item_type} not in instance"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(item_type),
                    "rhs": 0.0,
                    "raw_excess": 1.0,
                    "normalizer": 1.0,
                    "ratio": 1.0,
                })

    # =========================================================================
    # Constraint 6 (Eq 17): Objective consistency
    #   Reported objective_value must equal z = len(bins). The objective in
    #   Brandao & Pedroso's arc-flow formulation is simply the total number
    #   of bins used (an integer count), so we require equality up to 0.5
    #   (any integer mismatch fires).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(len(bins))
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5  # integer count: any mismatch >= 1 fires
            if abs_diff > obj_tol:
                violated_constraints.add(6)
                normalizer = max(abs(true_obj), eps)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed z=len(bins)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for BPP solutions")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
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

    print(f"Feasibility: {result['feasible']}")
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
