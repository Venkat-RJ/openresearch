#!/usr/bin/env python3
"""
Feasibility checker for the Quadratic Knapsack Problem (QKP).

Checks hard constraints from Caprara, Pisinger, and Toth (1999):
  Constraint 1: sum_{j in N} w_j x_j <= c          (capacity)
  Constraint 2: x_j in {0, 1} for all j in N        (binary / integrality)
  Constraint 3: objective consistency
                reported objective_value must equal recomputed
                  sum_{i in N} sum_{j in N} p_{ij} x_i x_j
                within a small tolerance (Tier C defence against
                self-reported-objective exploits). All variables determining
                the objective (the selected_items vector x) are present in
                the solution, so a full recompute is exact.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'selected_items')
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

    n = instance["n"]
    weights = instance["weights"]
    capacity = instance["capacity"]
    profit_matrix = instance["profit_matrix"]

    # Parse selected_items: could be a binary vector [0,1,0,...] or a list of indices [3,4,7,...]
    raw_items = solution["selected_items"]
    if len(raw_items) == n and all(v in (0, 1, 0.0, 1.0) for v in raw_items):
        # Binary vector format
        x = [float(v) for v in raw_items]
    else:
        # List of selected indices format
        x = [0.0] * n
        for idx in raw_items:
            x[idx] = 1.0

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ---- Constraint 1: capacity constraint ----
    # sum_{j in N} w_j x_j <= c
    lhs_cap = sum(weights[j] * x[j] for j in range(n))
    rhs_cap = float(capacity)
    violation_amount_cap = max(lhs_cap - rhs_cap, 0.0)

    if violation_amount_cap > tol:
        violated_constraints.add(1)
        violations.append(
            f"Capacity constraint violated: total weight {lhs_cap} exceeds capacity {rhs_cap}"
        )
        normalizer = max(abs(rhs_cap), eps)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs_cap,
            "rhs": rhs_cap,
            "raw_excess": violation_amount_cap,
            "normalizer": normalizer,
            "ratio": violation_amount_cap / normalizer,
        })

    # ---- Constraint 2: binary / integrality constraint ----
    # x_j in {0, 1} for all j in N
    for j in range(n):
        diff = min(abs(x[j] - 0.0), abs(x[j] - 1.0))
        if diff > tol:
            violated_constraints.add(2)
            rhs_val = round(x[j])  # nearest integer
            violation_amount = diff
            violations.append(
                f"Integrality violated for item {j}: x[{j}] = {x[j]} is not binary"
            )
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": x[j],
                "rhs": float(rhs_val),
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ---- Constraint 3: objective consistency (Tier C) ----
    # reported objective_value == sum_i sum_j p_{ij} x_i x_j
    # Full recompute is exact: selected_items contains every variable that
    # determines the objective. Tolerance: 0.1% relative with a 0.5 absolute
    # floor, since profit_matrix entries are integers and any honest mismatch
    # is at least 1.
    reported_obj = solution.get("objective_value")
    try:
        reported = float(reported_obj) if reported_obj is not None else None
    except (TypeError, ValueError):
        reported = None
    if reported is not None:
        # Iterate only over the selected items for efficiency (n can be 300+).
        selected = [j for j in range(n) if x[j] > 0.5]
        true_obj = 0.0
        for i in selected:
            row = profit_matrix[i]
            for j in selected:
                true_obj += row[j]
        true_obj = float(true_obj)
        abs_diff = abs(reported - true_obj)
        obj_tol = max(0.5, 1e-3 * abs(true_obj))
        if abs_diff > obj_tol:
            violated_constraints.add(3)
            violations.append(
                f"Objective consistency violated: reported objective_value="
                f"{reported} differs from recomputed sum_i sum_j p_ij x_i x_j="
                f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
            )
            normalizer = max(abs(true_obj), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(reported),
                "rhs": float(true_obj),
                "raw_excess": float(abs_diff),
                "normalizer": float(normalizer),
                "ratio": float(abs_diff / normalizer),
            })

    feasible = len(violated_constraints) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for QKP (Caprara et al. 1999)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write feasibility result JSON file")
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

    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
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
