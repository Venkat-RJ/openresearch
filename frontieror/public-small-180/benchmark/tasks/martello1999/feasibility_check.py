"""
Feasibility checker for the 0-1 Knapsack Problem (KP).
Based on: Martello, Pisinger, and Toth (1999)

Hard constraints (from formulation section, top to bottom):
  Constraint 1 (1b): sum_{j=1}^{n} w_j x_j <= c
  Constraint 2 (1c): x_j in {0, 1}, j = 1, ..., n
  Constraint 3 (obj): reported objective_value must equal
                      sum_{j=1}^{n} p_j x_j (Tier-C anti-exploit check)
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


TOL = 1e-5
EPS = 1e-5


def check_feasibility(instance, solution):
    n = instance["n"]
    capacity = instance["capacity"]
    items = instance["items"]

    # Build lookup from item id to item data
    item_map = {item["id"]: item for item in items}
    valid_ids = set(item_map.keys())

    selected = solution.get("selected_items", [])

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # -------------------------------------------------------------------------
    # Constraint 2 (1c): x_j in {0, 1} — each selected item must be a valid
    # item id, appear at most once, and be within {1, ..., n}.
    # -------------------------------------------------------------------------
    seen = set()
    invalid_items = []
    duplicate_items = []
    for item_id in selected:
        if item_id not in valid_ids:
            invalid_items.append(item_id)
        if item_id in seen:
            duplicate_items.append(item_id)
        seen.add(item_id)

    if invalid_items:
        violated_constraints.add(2)
        violations.append(
            f"Selected items not in instance: {invalid_items}"
        )
        for item_id in invalid_items:
            # violation_amount = 1 per invalid item (binary domain violation)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(item_id),
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": max(abs(0.0), EPS),
                "ratio": 1.0 / max(abs(0.0), EPS),
            })

    if duplicate_items:
        violated_constraints.add(2)
        violations.append(
            f"Duplicate items in selection: {duplicate_items}"
        )
        for item_id in duplicate_items:
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": 2.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": max(abs(1.0), EPS),
                "ratio": 1.0 / max(abs(1.0), EPS),
            })

    # -------------------------------------------------------------------------
    # Constraint 1 (1b): sum_{j=1}^{n} w_j x_j <= c
    # -------------------------------------------------------------------------
    total_weight = 0.0
    for item_id in seen:
        if item_id in item_map:
            total_weight += item_map[item_id]["weight"]

    lhs = total_weight
    rhs = float(capacity)
    violation_amount = lhs - rhs  # for <= constraint: how much LHS exceeds RHS

    if violation_amount > TOL:
        violated_constraints.add(1)
        violations.append(
            f"Capacity exceeded: total weight {lhs} > capacity {rhs}"
        )
        normalizer = max(abs(rhs), EPS)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # -------------------------------------------------------------------------
    # Constraint 3 (obj): reported objective_value must equal
    # sum_{j=1}^{n} p_j x_j. Tier-C defense against LLM exploits that lie
    # about the objective value while passing the constraint checks.
    # -------------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Recompute over unique, valid selected items (matches weight sum)
            true_obj = 0.0
            for item_id in seen:
                if item_id in item_map:
                    true_obj += float(item_map[item_id]["profit"])
            abs_diff = abs(reported - true_obj)
            # Profits are positive integers; an integer mismatch by >=1 fires.
            # Add a tiny relative slack for large magnitudes (true_obj can be
            # ~1e10), so float round-trips never cause false positives.
            tol = max(0.5, 1e-9 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(3)
                violations.append(
                    f"Objective consistency violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_j(p_j*x_j)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # -------------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------------
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
        description="Feasibility checker for 0-1 Knapsack Problem "
                    "(Martello, Pisinger, Toth 1999)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path for the output feasibility result JSON")
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

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
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
