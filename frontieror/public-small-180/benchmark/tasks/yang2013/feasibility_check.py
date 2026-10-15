#!/usr/bin/env python3
"""
Feasibility checker for the 0-1 Quadratic Knapsack Problem (QKP).

Mathematical formulation (Yang et al., 2013):
    max  sum_{i=1}^{n} sum_{j=1}^{i} p_{ij} * x_i * x_j
    s.t.
        Constraint 1: sum_{j=1}^{n} w_j * x_j <= c
        Constraint 2: x_j in {0, 1}, j = 1, ..., n
        Constraint 3: Binary domain — auxiliary vars in {0,1} (auto)
        Constraint 4: Integer domain — auxiliary vars integral (auto)
        Constraint 5: Objective consistency — reported objective_value must
                      equal the recomputed sum_{j<=i} p_{ij} * x_i * x_j.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'solution')
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
    c = instance["capacity"]
    w = instance["weights"]

    # Accept either the full solution dict or a bare solution vector.
    if isinstance(solution, dict):
        if "solution_vector" in solution:
            solution_vector = solution["solution_vector"]
        elif "solution" in solution:
            solution_vector = solution["solution"]
        else:
            raise ValueError("No solution vector found in solution dict")
        reported_obj = solution.get("objective_value")
    else:
        solution_vector = solution
        reported_obj = None

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # --- Constraint 1: sum_{j=1}^{n} w_j * x_j <= c ---
    lhs = sum(w[j] * solution_vector[j] for j in range(n))
    rhs = c
    violation_amount = lhs - rhs  # For <= constraint: how much LHS exceeds RHS
    if violation_amount > tol:
        violated_constraints.add(1)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violations.append(
            f"Knapsack capacity exceeded: total weight {lhs} > capacity {rhs}"
        )
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })

    # --- Constraint 2: x_j in {0, 1} for all j ---
    non_binary_items = []
    for j in range(n):
        val = solution_vector[j]
        # Check if val is 0 or 1 (within tolerance)
        dist_to_0 = abs(val - 0)
        dist_to_1 = abs(val - 1)
        violation_amount = min(dist_to_0, dist_to_1)
        if violation_amount > tol:
            non_binary_items.append(j)
            rhs_val = round(val)  # nearest binary value
            normalizer = max(abs(rhs_val), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    if non_binary_items:
        violated_constraints.add(2)
        violations.append(
            f"Non-binary variables at indices: {non_binary_items}"
        )


        # =====================================================================
        # =====================================================================
        # Constraint 3: Binary domain — variables must be 0 or 1
    _domain_check_vars_binary = []
    _domain_check_vars_integer = []

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 3: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > tol or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > tol:
                        violated_constraints.add(3)
                        violations.append(
                            f"Constraint 3 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        violation_magnitudes.append({
                            "constraint": 3,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    # Constraint 4: Integer domain — variables must be integral
    for var_name, var_dict in _domain_check_vars_integer:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                frac = abs(v - round(v))
                if frac > tol:
                    violated_constraints.add(4)
                    violations.append(
                        f"Constraint 4 (integer domain): {var_name}[{key}] = {v} is not integer")
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": v,
                        "rhs": round(v),
                        "raw_excess": float(frac),
                        "normalizer": max(abs(round(v)), eps),
                        "ratio": float(frac / max(abs(round(v)), eps)),
                    })

    # --- Constraint 5: Objective consistency ---
    # Recompute the QKP objective from the solution vector and the symmetric
    # profit matrix and compare against the reported objective_value. This is
    # a Tier C defense against LLM exploits that fabricate objective values.
    profit_matrix = instance.get("profit_matrix")
    if profit_matrix is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Round x to nearest binary value so the obj recompute is robust
            # to tiny floating-point noise even if constraint 2 fires.
            x_bin = [1 if abs(solution_vector[j] - 1) <= 0.5 else 0
                     for j in range(n)]
            true_obj = 0.0
            for i in range(n):
                if not x_bin[i]:
                    continue
                row = profit_matrix[i]
                # j ranges from 0..i  (j <= i in 1-indexed math == j <= i in 0-indexed)
                for j in range(i + 1):
                    if x_bin[j]:
                        true_obj += float(row[j])
            abs_diff = abs(reported - true_obj)
            # QKP profits are non-negative integers, so true_obj is integral:
            # any integer disagreement (>= 1) must fire. Use 0.5 absolute floor
            # with a 1e-6 relative slack for safety on very large objectives.
            tol_obj = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_constraints.add(5)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_{{j<=i}}"
                    f"(p_ij * x_i * x_j)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
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
        description="Feasibility checker for the 0-1 Quadratic Knapsack Problem"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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

    # Pass the full solution dict so the obj-recompute check sees both the
    # vector and the reported objective_value.
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
