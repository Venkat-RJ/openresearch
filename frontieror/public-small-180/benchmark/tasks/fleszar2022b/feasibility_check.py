"""
Feasibility checker for the Quadratic Multiple Knapsack Problem (QMKP).
Checks constraints from Fleszar (2022) formulation:
  Constraint 1: Capacity  -- sum_i w_i * x_{ik} <= c_k  for all k
  Constraint 2: Assignment -- sum_k x_{ik} <= 1          for all i
  Constraint 3: Binary domain -- x_{ik} in {0, 1}        for all i, k
  Constraint 4: Objective consistency -- reported objective_value must
                equal the objective recomputed from the assignment:
                sum_k ( sum_i v_i x_{ik}
                        + sum_{i<j} v_{ij} x_{ik} x_{jk} )
                (Tier C defense against fabricated objective values.)
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'objective_value')
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
    m = instance["m"]
    weights = instance["weights"]
    capacity = instance["capacity"]

    assignment = solution.get("assignment", {})

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Build x matrix from assignment
    # assignment maps str(item) -> knapsack index
    x = [[0] * m for _ in range(n)]
    for item_str, k in assignment.items():
        i = int(item_str)
        k = int(k)
        if 0 <= i < n and 0 <= k < m:
            x[i][k] = 1

    # ------------------------------------------------------------------
    # Constraint 3: Binary domain x_{ik} in {0, 1}
    # Check that assignment values are valid knapsack indices and items
    # ------------------------------------------------------------------
    for item_str, k_val in assignment.items():
        i = int(item_str)
        k = int(k_val)
        if i < 0 or i >= n:
            violation_amount = 1.0
            rhs = 0.0
            normalizer = max(abs(rhs), EPS)
            violated_constraints.add(3)
            violations.append(
                f"Item index {i} is out of range [0, {n-1}]"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(i),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })
        if k < 0 or k >= m:
            violation_amount = 1.0
            rhs = 0.0
            normalizer = max(abs(rhs), EPS)
            violated_constraints.add(3)
            violations.append(
                f"Knapsack index {k} for item {i} is out of range [0, {m-1}]"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(k),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })
        # x_{ik} must be 0 or 1; since we derive from assignment dict, values
        # are inherently binary. But check that the value is actually 0 or 1.
        val = x[i][k] if (0 <= i < n and 0 <= k < m) else 0
        if val not in (0, 1):
            violation_amount = abs(val - round(val))
            rhs = round(val)
            normalizer = max(abs(rhs), EPS)
            violated_constraints.add(3)
            violations.append(
                f"x[{i},{k}] = {val} is not binary"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(val),
                "rhs": float(rhs),
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint 1: Capacity -- sum_i w_i * x_{ik} <= c_k  for all k
    # ------------------------------------------------------------------
    for k in range(m):
        lhs = sum(weights[i] * x[i][k] for i in range(n))
        rhs = capacity
        violation_amount = lhs - rhs  # for <= constraint
        if violation_amount > TOL:
            normalizer = max(abs(rhs), EPS)
            violated_constraints.add(1)
            items_in_k = [i for i in range(n) if x[i][k] == 1]
            violations.append(
                f"Capacity exceeded on knapsack {k}: "
                f"total weight {lhs} > capacity {rhs} "
                f"(items {items_in_k})"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # ------------------------------------------------------------------
    # Constraint 2: Assignment -- sum_k x_{ik} <= 1  for all i
    # ------------------------------------------------------------------
    for i in range(n):
        lhs = sum(x[i][k] for k in range(m))
        rhs = 1
        violation_amount = lhs - rhs  # for <= constraint
        if violation_amount > TOL:
            normalizer = max(abs(rhs), EPS)
            knapsacks_assigned = [k for k in range(m) if x[i][k] == 1]
            violated_constraints.add(2)
            violations.append(
                f"Item {i} assigned to {int(lhs)} knapsacks "
                f"(knapsacks {knapsacks_assigned}), exceeds limit of 1"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # ------------------------------------------------------------------
    # Constraint 4: Objective consistency (Tier C defense)
    # Recompute the QMKP objective directly from the assignment and
    # reject when the reported objective_value disagrees.
    #   obj = sum_k ( sum_i v_i x_{ik}
    #                 + sum_{i<j} v_{ij} x_{ik} x_{jk} )
    # All objective-determining variables (the assignment) are present
    # in the solution, so a full exact recompute is possible.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    values = instance.get("values")
    pairwise_values = instance.get("pairwise_values")
    if reported_obj is not None and values is not None and pairwise_values is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Items packed in each knapsack, using the same x matrix the
            # constraint checks above are built from (out-of-range item or
            # knapsack indices contribute nothing -- already flagged by
            # constraint 3).
            true_obj = 0.0
            for k in range(m):
                items_in_k = [i for i in range(n) if x[i][k] == 1]
                for i in items_in_k:
                    true_obj += values[i]
                for a in range(len(items_in_k)):
                    for b in range(a + 1, len(items_in_k)):
                        i = items_in_k[a]
                        j = items_in_k[b]
                        true_obj += pairwise_values[i][j]
            true_obj = float(true_obj)
            abs_diff = abs(reported - true_obj)
            # Objective is integer-valued (all parameters are integers and
            # products of binaries), so a mismatch of >= 1 must fire; keep a
            # tiny relative term for very large recomputed magnitudes.
            tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > tol:
                normalizer = max(abs(true_obj), EPS)
                violated_constraints.add(4)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed "
                    f"sum_k(sum_i v_i x_ik + sum_{{i<j}} v_ij x_ik x_jk)="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
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
        description="Feasibility checker for QMKP (Fleszar 2022)"
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


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
