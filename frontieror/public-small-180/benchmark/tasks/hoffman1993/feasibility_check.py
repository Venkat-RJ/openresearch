"""
Feasibility checker for the Set Partitioning Problem with Base Constraints (SPB).

Based on the formulation in:
  Hoffman & Padberg (1993) "Solving Airline Crew Scheduling Problems by Branch-and-Cut"
  Management Science 39(6):657-682

Hard constraints (numbered top-to-bottom from the formulation):
  Constraint (1): Ax = e_m  (each flight leg covered exactly once)
  Constraint (2): d_1 <= Dx <= d_2  (base constraints, if present)
  Constraint (3): x in {0,1}^n  (binary integrality)
  Constraint (4): objective consistency  (Tier C defense): the reported
                  objective_value must equal the recomputed total cost
                  sum_j c_j * x_j. This is a full recompute -- every
                  variable the objective depends on (the binary vector x)
                  is present in the solution, so the true cost is computed
                  exactly and compared against the reported value.
"""

import argparse
import json
import numpy as np

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'selected_rotations', 'variable_values')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('selected_rotations',)
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



def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SPB (Set Partitioning with Base Constraints)"
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

    # Load instance
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    # Load solution
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

    tol = 1e-5
    eps = 1e-5

    # Extract instance data
    m = instance["dimensions"]["num_rows"]
    n = instance["dimensions"]["num_cols"]
    columns = instance["constraint_matrix_A"]["columns"]

    # Build dense A matrix. Some instance files have a `columns` list that
    # is longer than `num_cols` and references row indices ≥ `num_rows` —
    # those are stale data beyond the model gurobi_code actually builds (it
    # only uses the first `num_cols` columns clipped to `num_rows` rows).
    # Mirror gurobi_code's slicing here so the checker validates the same
    # sub-problem the solver optimized.
    A = np.zeros((m, n), dtype=float)
    for j, col_rows in enumerate(columns):
        if j >= n:
            break
        for i in col_rows:
            if 0 <= i < m:
                A[i, j] = 1.0

    # Extract solution vector x
    x = np.zeros(n, dtype=float)
    variable_values = solution.get("variable_values", {})
    for j_str, val in variable_values.items():
        j = int(j_str)
        if 0 <= j < n:
            x[j] = val

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # =========================================================================
    # Constraint (1): Ax = e_m
    # Each flight leg must be covered exactly once.
    # For each row i: sum_j a_{ij} * x_j = 1
    # This is an equality constraint: violation_amount = |LHS - RHS|
    # =========================================================================
    Ax = A @ x
    for i in range(m):
        lhs = float(Ax[i])
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraint_set.add(1)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            if lhs < rhs:
                violations.append(
                    f"Flight leg {i} is not covered: covered {lhs:.4f} times instead of exactly 1"
                )
            else:
                violations.append(
                    f"Flight leg {i} is over-covered: covered {lhs:.4f} times instead of exactly 1"
                )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # =========================================================================
    # Constraint (2): d_1 <= Dx <= d_2 (base constraints)
    # Only present when has_base_constraints is True.
    # This is a pair of inequalities for each base constraint b:
    #   d_1[b] <= (Dx)[b]  (>= constraint: violation = max(0, d_1[b] - (Dx)[b]))
    #   (Dx)[b] <= d_2[b]  (<= constraint: violation = max(0, (Dx)[b] - d_2[b]))
    # =========================================================================
    has_base = instance.get("has_base_constraints", False)
    base_constraints = instance.get("base_constraints", None)
    if has_base and base_constraints is not None:
        d_rows = base_constraints["D_matrix"]["rows"]
        d1 = base_constraints["lower_bounds_d1"]
        d2 = base_constraints["upper_bounds_d2"]
        num_base = base_constraints["num_constraints"]

        D = np.array(d_rows, dtype=float)  # shape (num_base, n)
        Dx = D @ x

        for b in range(num_base):
            lhs_val = float(Dx[b])

            # Lower bound: d_1[b] <= (Dx)[b]  =>  equivalent to (Dx)[b] >= d_1[b]
            # This is a >= constraint: violation = max(0, RHS - LHS)
            lower_rhs = float(d1[b])
            lower_violation = max(0.0, lower_rhs - lhs_val)
            if lower_violation > tol:
                violated_constraint_set.add(2)
                normalizer = max(abs(lower_rhs), eps)
                ratio = lower_violation / normalizer
                violations.append(
                    f"Base constraint {b} lower bound violated: Dx[{b}]={lhs_val:.4f} < d1[{b}]={lower_rhs:.4f}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs_val,
                    "rhs": lower_rhs,
                    "raw_excess": lower_violation,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

            # Upper bound: (Dx)[b] <= d_2[b]
            # This is a <= constraint: violation = max(0, LHS - RHS)
            upper_rhs = float(d2[b])
            upper_violation = max(0.0, lhs_val - upper_rhs)
            if upper_violation > tol:
                violated_constraint_set.add(2)
                normalizer = max(abs(upper_rhs), eps)
                ratio = upper_violation / normalizer
                violations.append(
                    f"Base constraint {b} upper bound violated: Dx[{b}]={lhs_val:.4f} > d2[{b}]={upper_rhs:.4f}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs_val,
                    "rhs": upper_rhs,
                    "raw_excess": upper_violation,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # =========================================================================
    # Constraint (3): x in {0,1}^n (binary integrality)
    # For each variable j: x_j must be 0 or 1.
    # violation_amount = min(|x_j - 0|, |x_j - 1|) = min(x_j, 1 - x_j)
    #   if x_j is not close to 0 or 1.
    # RHS for normalization: closest integer (0 or 1). Use eps if RHS=0.
    # =========================================================================
    for j in range(n):
        dist_to_0 = abs(x[j])
        dist_to_1 = abs(x[j] - 1.0)
        violation_amount = min(dist_to_0, dist_to_1)
        if violation_amount > tol:
            violated_constraint_set.add(3)
            closest_int = 0.0 if dist_to_0 <= dist_to_1 else 1.0
            rhs = closest_int
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Variable x[{j}]={x[j]:.6f} is not binary (closest integer: {int(closest_int)})"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(x[j]),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # Also check out-of-range: x_j < 0 or x_j > 1
    for j in range(n):
        if x[j] < -tol:
            violated_constraint_set.add(3)
            violation_amount = abs(x[j])
            rhs = 0.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Variable x[{j}]={x[j]:.6f} is negative (out of [0,1] range)"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(x[j]),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })
        elif x[j] > 1.0 + tol:
            violated_constraint_set.add(3)
            violation_amount = x[j] - 1.0
            rhs = 1.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Variable x[{j}]={x[j]:.6f} exceeds 1 (out of [0,1] range)"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(x[j]),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # selected_rotations: list[int] of rotation indices must be in [0, n) and unique.
    selected_rotations = solution.get("selected_rotations", [])
    seen_rotations = set()
    for r in selected_rotations:
        if not isinstance(r, int) or r < 0 or r >= n:
            violated_constraint_set.add(3)
            violations.append(
                f"Invalid rotation index in selected_rotations: {r} (valid range 0..{n-1})"
            )
            violation_magnitudes.append({
                "constraint": 3, "lhs": float(r if isinstance(r, (int, float)) else -1),
                "rhs": 0.0, "raw_excess": 1.0, "normalizer": max(1.0, eps), "ratio": 1.0,
            })
        elif r in seen_rotations:
            violated_constraint_set.add(3)
            violations.append(
                f"Duplicate rotation index in selected_rotations: {r} (binary domain violated)"
            )
            violation_magnitudes.append({
                "constraint": 3, "lhs": 2.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
        else:
            seen_rotations.add(r)

    # =========================================================================
    # Constraint (4): Objective consistency (Tier C defense)
    # The objective is  min sum_j c_j * x_j  (SPB / SPP). Every variable the
    # objective depends on -- the binary selection vector x -- is present in
    # the solution (`variable_values`), so a FULL recompute is possible: we
    # recompute the true total cost from the same x the constraint checks
    # above used, and reject when the reported objective_value disagrees.
    # This catches LLM score-gaming exploits that report fabricated objective
    # values (e.g. 0.0 or sys.float_info.max) while the routes pass.
    # =========================================================================
    cost_vector = instance.get("cost_vector")
    reported_obj = solution.get("objective_value")
    if cost_vector is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            n_cost = min(n, len(cost_vector))
            true_obj = float(sum(float(cost_vector[j]) * float(x[j]) for j in range(n_cost)))
            abs_diff = abs(reported - true_obj)
            # Cost coefficients are integers and x is binary, so the recompute
            # is an exact integer; an integer mismatch by >= 1 must fire. The
            # tiny relative term only absorbs float rounding on huge values.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraint_set.add(4)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Objective consistency violated: reported objective_value={reported} "
                    f"differs from recomputed sum_j(c_j*x_j)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # Build result
    feasible = len(violated_constraint_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraint_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {feasible}")
    if not feasible:
        print(f"Violated constraints: {sorted(violated_constraint_set)}")
        for v in violations:
            print(f"  - {v}")
    print(f"Result written to: {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
