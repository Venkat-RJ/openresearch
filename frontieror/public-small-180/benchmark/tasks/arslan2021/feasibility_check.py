#!/usr/bin/env python3
"""
Feasibility checker for the benchmark's LP-relaxed Two-Stage Robust Knapsack.

Based on: Arslan & Detienne (2022), "Decomposition-based approaches for a class of
two-stage robust binary optimization problems", INFORMS Journal on Computing 34(2).

The benchmark evaluates the LP-relaxed recourse variant of the min-max-min problem:
  min_{x in {0,1}^I} sum_i (f_i - p_bar_i) x_i
    + max_{xi in Xi} min_{(y,r) in Y(x)} sum_i [(p_hat_i xi_i - f_i) y_i - p_hat_i xi_i r_i]

The ONLY first-stage decision variable is x in {0,1}^I. The second-stage
(y, r) and the dual variables (u0, v) from the deterministic-equivalent
single-level reformulation (Eq. 10-14) are NOT part of the original
decision space. Any binary x is feasible (relatively complete recourse:
Y(x) always contains (y=0, r=0)).

Checked constraints:
  Constraint 1: x_i in {0,1}  for all i                      [original domain]
  Constraint 2: reported objective_value equals the exact fixed-x value of
                the deterministic-equivalent LP relaxation.
"""

import json
import math
import argparse


def compute_relaxed_fixed_x_objective(instance, x):
    """Solve the benchmark's LP-relaxed recourse problem for fixed x."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    items = instance["items"]
    item_count = len(items)
    params = instance["problem_parameters"]
    capacity = float(params["knapsack_capacity"])
    gamma = float(params["uncertainty_budget"])
    r_offset = item_count
    u_index = 2 * item_count
    v_offset = u_index + 1
    variable_count = 3 * item_count + 1

    objective = np.zeros(variable_count, dtype=float)
    first_stage = 0.0
    for i, item in enumerate(items):
        outsource = float(item["outsource_penalty"])
        nominal = float(item["nominal_profit"])
        first_stage += (outsource - nominal) * x[i]
        objective[i] = -outsource
        objective[v_offset + i] = 1.0
    objective[u_index] = gamma

    rows = []
    columns = []
    values = []
    rhs = []

    def add_constraint(coefficients, upper):
        row = len(rhs)
        for column, value in coefficients.items():
            rows.append(row)
            columns.append(column)
            values.append(value)
        rhs.append(float(upper))

    for i, item in enumerate(items):
        degradation = float(item["max_degradation"])
        add_constraint({
            i: degradation,
            r_offset + i: -degradation,
            u_index: -1.0,
            v_offset + i: -1.0,
        }, 0.0)
        add_constraint({r_offset + i: 1.0, i: -1.0}, 0.0)
    capacity_coefficients = {}
    for i, item in enumerate(items):
        capacity_coefficients[i] = float(item["weight"])
        capacity_coefficients[r_offset + i] = float(item["repair_capacity"])
    add_constraint(capacity_coefficients, capacity)

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(rhs), variable_count),
    ).tocsr()
    bounds = (
        [(0.0, float(x[i])) for i in range(item_count)]
        + [(0.0, 1.0) for _ in range(item_count)]
        + [(0.0, None)]
        + [(0.0, None) for _ in range(item_count)]
    )
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.asarray(rhs),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(
            f"fixed-x relaxed recourse LP failed: {result.message}"
        )
    return first_stage + float(result.fun)

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



def check_feasibility(instance, solution):
    """Check x and the exact objective of the LP-relaxed benchmark.

    The only decision variable in the original problem is the first-stage
    x in {0,1}^I. Relatively complete recourse holds (Y(x) contains
    (y=0, r=0) for any x), so any binary x is feasible.

    Additionally, the reported objective_value is checked against an
    analytic envelope [L_Z, U_Z] computed from x and the instance data
    to defend against fabricated/lying objective reports (Tier C
    obj-consistency check).
    """
    tol = 1e-5
    eps = 1e-5

    items = instance["items"]
    num_items = len(items)

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # ---- Parse x (list or dict format) ----
    raw_x = solution.get("x")
    if isinstance(raw_x, list):
        try:
            x = [float(v) for v in raw_x]
        except (TypeError, ValueError):
            x = None
    elif isinstance(raw_x, dict):
        try:
            x = [float(raw_x[str(i)]) for i in range(num_items)]
        except (KeyError, TypeError, ValueError):
            x = None
    else:
        x = None

    # ==================================================================
    # Constraint 1: First-stage binary domain  x_i in {0,1}  for all i
    # ==================================================================
    if x is None:
        violated_constraints.add(1)
        violations.append("Solution missing first-stage decision variable 'x'.")
    else:
        for i in range(num_items):
            val = x[i]
            nearest = round(val)
            if nearest not in (0, 1) or abs(val - nearest) > tol:
                violated_constraints.add(1)
                nearest_binary = 0.0 if val < 0.5 else 1.0
                va = abs(val - nearest_binary)
                violations.append(
                    f"Binary constraint violated for x_{i}: x_{i} = {val}"
                )
                normalizer = max(abs(nearest_binary), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": val,
                    "rhs": nearest_binary,
                    "raw_excess": va,
                    "normalizer": normalizer,
                    "ratio": va / normalizer,
                })

    # ==================================================================
    # Constraint 2: exact objective for the benchmark's LP-relaxed recourse.
    # ==================================================================
    if x is not None and 1 not in violated_constraints:
        expected = compute_relaxed_fixed_x_objective(instance, x)
        reported = float(solution["objective_value"])
        difference = abs(reported - expected)
        tol_obj = max(1e-4, 1e-8 * abs(expected))
        if difference > tol_obj:
            violated_constraints.add(2)
            violations.append(
                f"Objective consistency violated: objective_value={reported} "
                f"does not match the exact fixed-x LP-relaxation value "
                f"{expected} (|diff|={difference:.4g}, tol={tol_obj:.4g})"
            )
            normalizer = max(abs(expected), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": reported,
                "rhs": expected,
                "raw_excess": difference,
                "normalizer": normalizer,
                "ratio": difference / normalizer,
            })

    return {
        "feasible": len(violated_constraints) == 0,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Two-Stage Robust Knapsack"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write JSON feasibility result")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
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
        print("Solution is feasible.")
    else:
        print(f"Solution is INFEASIBLE. "
              f"Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
