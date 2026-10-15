#!/usr/bin/env python3
"""Feasibility checker for the Set Partitioning Problem (Zhang2025).

Hard constraints (numbered top-to-bottom from the formulation):
  Constraint 1: A x = 1_M  (each element covered exactly once)
  Constraint 2: x_j in {0, 1}  (binary decision variables)
  Constraint 3: objective_value matches c^T x (Tier C anti-gaming check)
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'selected_duties')
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

    M = instance["parameters"]["num_trips_M"]
    N = instance["parameters"]["num_duties_N"]
    duties = instance["duties"]
    costs = instance.get("costs")
    selected = solution["selected_duties"]

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # Build x vector (guard against out-of-range indices to avoid IndexError;
    # they're caught by Constraint 2 below).
    x = [0] * N
    for j in selected:
        if 0 <= j < N:
            x[j] = 1

    # ------------------------------------------------------------------
    # Constraint 1: A x = 1_M  (each element i must be covered exactly once)
    # ------------------------------------------------------------------
    coverage = [0] * M
    for j in range(N):
        if x[j] == 1:
            for trip_id in duties[j]:
                coverage[trip_id] += 1

    for i in range(M):
        lhs = float(coverage[i])
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_set.add(1)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            if lhs == 0:
                violations.append(f"Element (trip) {i} is not covered by any selected duty")
            else:
                violations.append(f"Element (trip) {i} is covered by {int(lhs)} duties instead of exactly 1")
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 2: x_j in {0, 1}  (binary)
    # ------------------------------------------------------------------
    for j in selected:
        if j < 0 or j >= N:
            violated_set.add(2)
            violations.append(f"Selected duty index {j} is out of range [0, {N-1}]")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(j),
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": max(abs(0.0), eps),
                "ratio": 1.0 / eps,
            })

    from collections import Counter
    counts = Counter(selected)
    for j, cnt in counts.items():
        if cnt > 1:
            violated_set.add(2)
            lhs = float(cnt)
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(f"Duty {j} selected {cnt} times (must be 0 or 1)")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 3 (NEW): objective_value must equal c^T x.
    # Full recompute: every variable that determines the obj (the chosen
    # duty indices) is in the solution; costs come from the instance.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if costs is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for j in selected:
                if 0 <= j < N:
                    true_obj += float(costs[j])
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_set.add(3)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_j(costs[j]*x_j)="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for Set Partitioning Problem")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
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

    print(f"Feasibility: {result['feasible']} | Violated constraints: {result['violated_constraints']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
