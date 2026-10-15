#!/usr/bin/env python3
"""
Feasibility checker for Cardinality Constrained Mean-Variance Portfolio Optimization.

Constraints (from math_model.txt):
  (C1) r^T x >= b          : portfolio return must be at least b
  (C2) e^T x = 1           : portfolio weights sum to 1
  (C3) sum delta(x_i) <= k : at most k assets have nonzero weights
  (C4) Objective consistency (Tier C exploit defense): recompute
       f(x) = (1/2) x^T Q x from the solution variables and reject when
       the reported objective_value disagrees. All variables on which the
       objective depends (the vector x) are carried in the solution, so a
       full recompute is exact.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'x')
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
    k = instance["k"]
    b = instance["b"]
    r = instance["r"]
    Q = instance["Q"]
    x = solution["x"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # --- Constraint 1: r^T x >= b ---
    lhs1 = sum(r[i] * x[i] for i in range(n))
    rhs1 = b
    violation_amount1 = rhs1 - lhs1  # for >= constraint: how much RHS exceeds LHS
    if violation_amount1 > tol:
        violated_constraints.add(1)
        violations.append(
            f"Constraint 1 violated: portfolio return r^T x = {lhs1:.10f} < b = {rhs1:.10f}"
        )
        normalizer1 = max(abs(rhs1), eps)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs1,
            "rhs": rhs1,
            "raw_excess": violation_amount1,
            "normalizer": normalizer1,
            "ratio": violation_amount1 / normalizer1,
        })

    # --- Constraint 2: e^T x = 1 ---
    lhs2 = sum(x[i] for i in range(n))
    rhs2 = 1.0
    violation_amount2 = abs(lhs2 - rhs2)
    if violation_amount2 > tol:
        violated_constraints.add(2)
        violations.append(
            f"Constraint 2 violated: sum of weights = {lhs2:.10f}, should equal 1.0"
        )
        normalizer2 = max(abs(rhs2), eps)
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": lhs2,
            "rhs": rhs2,
            "raw_excess": violation_amount2,
            "normalizer": normalizer2,
            "ratio": violation_amount2 / normalizer2,
        })

    # --- Constraint 3: sum delta(x_i) <= k ---
    # delta(x_i) = 1 if x_i != 0, 0 otherwise. Use tol for zero check.
    num_nonzero = sum(1 for i in range(n) if abs(x[i]) > tol)
    lhs3 = float(num_nonzero)
    rhs3 = float(k)
    violation_amount3 = lhs3 - rhs3  # for <= constraint: how much LHS exceeds RHS
    if violation_amount3 > tol:
        violated_constraints.add(3)
        violations.append(
            f"Constraint 3 violated: {num_nonzero} nonzero assets exceeds cardinality limit k = {k}"
        )
        normalizer3 = max(abs(rhs3), eps)
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs3,
            "rhs": rhs3,
            "raw_excess": violation_amount3,
            "normalizer": normalizer3,
            "ratio": violation_amount3 / normalizer3,
        })

    # --- Constraint 4: Objective consistency (Tier C exploit defense) ---
    # Recompute f(x) = (1/2) * x^T Q x from the solution variables and
    # reject when the reported value disagrees.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and not (math.isnan(reported) or math.isinf(reported)):
            # Compute x^T Q x via the symmetric quadratic form.
            qx = [0.0] * n
            for i in range(n):
                Qi = Q[i]
                acc = 0.0
                for j in range(n):
                    acc += Qi[j] * x[j]
                qx[i] = acc
            quad = 0.0
            for i in range(n):
                quad += x[i] * qx[i]
            true_obj = 0.5 * quad
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-6 absolute floor. The Gurobi
            # objective for n=50/100 portfolios is typically O(1e-4)-O(1e-2)
            # (a half-variance), so an absolute floor far below 1e-3 is
            # required to catch sub-permille exploits while tolerating
            # double-precision noise from the n^2 quadratic accumulation.
            tol_obj = max(1e-6, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_constraints.add(4)
                violations.append(
                    f"Constraint 4 violated: reported objective_value={reported} differs from "
                    f"recomputed (1/2) x^T Q x = {true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )
                normalizer4 = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer4,
                    "ratio": abs_diff / normalizer4,
                })
        elif reported is not None:
            # NaN/Inf reported obj -- always a lie since the true obj is
            # bounded by (1/2) * max-eigenvalue(Q) * ||x||^2 and x sums to 1.
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4 violated: reported objective_value={reported_obj} is NaN or Inf"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": reported,
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": eps,
                "ratio": float("inf"),
            })

    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for CCMV portfolio optimization")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON file")
    parser.add_argument("--result_path", required=True, help="Path to output result JSON file")
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
        for v in result["violations"]:
            print(f"  - {v}")
    print(f"Result saved to {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
