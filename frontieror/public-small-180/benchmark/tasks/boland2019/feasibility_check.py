#!/usr/bin/env python3
"""
Feasibility checker for BO-SSCFLP solutions from:
Gadegaard, Nielsen, and Ehrgott (2019),
"Bi-objective branch-and-cut algorithms based on LP relaxation and bound sets"

Constraints numbered top-to-bottom from the formulation in math_model.txt:
  Constraint 1: Ax <= b
      Maps to SSCFLP constraints (inferred from code, not specified in paper):
        (a) Single-source assignment: sum_i x[i,j] = 1  for all j
        (b) Capacity: sum_j d[j]*x[i,j] <= s[i]*y[i]  for all i
        (c) Linking: x[i,j] <= y[i]  for all i,j
  Constraint 2: x in {0,1}^n  (binary integrality)
  Constraint 3: Objective consistency -- reported objective_value must equal
      the minimum over non-dominated solutions of (z1+z2)/2, where each
      (z1, z2) is recomputed from open_facilities and assignments using the
      instance fixed_costs / assignment_costs. Guards against fabricated
      objective_value exploits (Tier C defense).
  Constraints 4-11: Subproblem constraints (weighted sum, fathoming LPs) -- skipped.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'pareto_front', 'solutions')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('pareto_front',)
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

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    nI = instance["num_facilities"]
    nJ = instance["num_customers"]
    d = instance["demands"]
    s = instance["capacities"]
    f1 = instance["fixed_costs_obj1"]
    f2 = instance["fixed_costs_obj2"]
    c1 = instance["assignment_costs_obj1"]
    c2 = instance["assignment_costs_obj2"]

    # Extract solutions with variable details
    solutions = solution.get("solutions", [])

    if not solutions:
        # No detailed solution data (e.g. efficient algorithm output with only
        # pareto_front points). Cannot verify structural constraints.
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": [
                "No detailed solution variables available (open_facilities, "
                "assignments missing); feasibility cannot be verified"
            ],
            "violation_magnitudes": []
        }

    for sol_idx, sol in enumerate(solutions):
        open_fac = sol.get("open_facilities", [])
        assignments = sol.get("assignments", {})

        # Reconstruct y[i] and x[i][j]
        y = [0] * nI
        for i in open_fac:
            if 0 <= i < nI:
                y[i] = 1

        x = [[0] * nJ for _ in range(nI)]
        for j_str, i_val in assignments.items():
            j = int(j_str)
            i = int(i_val)
            if 0 <= i < nI and 0 <= j < nJ:
                x[i][j] = 1

        # ==================================================================
        # Constraint 1: Ax <= b
        # Encompasses all SSCFLP structural constraints.
        # ==================================================================

        # (a) Single-source assignment: sum_i x[i][j] = 1 for all j (= constraint)
        for j in range(nJ):
            lhs = sum(x[i][j] for i in range(nI))
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                violated_constraints.add(1)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Solution {sol_idx}: Customer {j} assigned to "
                    f"{lhs} facilities (expected exactly 1)"
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer)
                })

        # (b) Capacity: sum_j d[j]*x[i][j] <= s[i]*y[i] for all i
        for i in range(nI):
            lhs = sum(d[j] * x[i][j] for j in range(nJ))
            rhs = float(s[i] * y[i])
            violation_amount = lhs - rhs  # <= constraint: violation if lhs > rhs
            if violation_amount > tol:
                violated_constraints.add(1)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Solution {sol_idx}: Capacity exceeded at facility {i}: "
                    f"load {lhs} > capacity {rhs}"
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer)
                })

        # (c) Linking: x[i][j] <= y[i] for all i, j
        for i in range(nI):
            for j in range(nJ):
                lhs = float(x[i][j])
                rhs = float(y[i])
                violation_amount = lhs - rhs  # <= constraint
                if violation_amount > tol:
                    violated_constraints.add(1)
                    normalizer = max(abs(rhs), eps)
                    violations.append(
                        f"Solution {sol_idx}: Customer {j} assigned to "
                        f"closed facility {i} (x[{i},{j}]={lhs}, y[{i}]={rhs})"
                    )
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": float(lhs),
                        "rhs": float(rhs),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(violation_amount / normalizer)
                    })

        # ==================================================================
        # Constraint 2: x in {0,1}^n (binary integrality)
        # ==================================================================
        # Since we reconstruct from open_facilities/assignments, values are
        # inherently 0 or 1. Check that assignment targets are valid indices
        # and that every customer is assigned (covered by constraint 1a above).
        for j_str, i_val in assignments.items():
            j = int(j_str)
            i = int(i_val)
            if i < 0 or i >= nI:
                violated_constraints.add(2)
                rhs_val = float(nI - 1)
                raw = float(abs(i) - rhs_val) if abs(i) > rhs_val else 0.0
                normalizer = max(abs(rhs_val), eps)
                violations.append(
                    f"Solution {sol_idx}: Customer {j} assigned to "
                    f"invalid facility index {i} (valid: 0..{nI-1})"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(i),
                    "rhs": rhs_val,
                    "raw_excess": raw,
                    "normalizer": normalizer,
                    "ratio": raw / normalizer
                })
            if j < 0 or j >= nJ:
                violated_constraints.add(2)
                rhs_val = float(nJ - 1)
                raw = float(abs(j) - rhs_val) if abs(j) > rhs_val else 0.0
                normalizer = max(abs(rhs_val), eps)
                violations.append(
                    f"Solution {sol_idx}: Invalid customer index {j} "
                    f"(valid: 0..{nJ-1})"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(j),
                    "rhs": rhs_val,
                    "raw_excess": raw,
                    "normalizer": normalizer,
                    "ratio": raw / normalizer
                })

        # Check that open_facilities indices are valid
        for i in open_fac:
            if i < 0 or i >= nI:
                violated_constraints.add(2)
                rhs_val = float(nI - 1)
                raw = float(abs(i) - rhs_val) if abs(i) > rhs_val else 0.0
                normalizer = max(abs(rhs_val), eps)
                violations.append(
                    f"Solution {sol_idx}: Invalid facility index {i} "
                    f"in open_facilities (valid: 0..{nI-1})"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(i),
                    "rhs": rhs_val,
                    "raw_excess": raw,
                    "normalizer": normalizer,
                    "ratio": raw / normalizer
                })

    # ==================================================================
    # Constraint 3: Objective consistency (Tier C defense against
    # fabricated objective_value exploits). The reported objective_value
    # is documented as the minimum over non-dominated solutions of
    # (z1+z2)/2; recompute it from open_facilities and assignments and
    # reject if it disagrees with what the program reported.
    # ==================================================================
    reported_obj_raw = solution.get("objective_value")
    try:
        reported_obj = float(reported_obj_raw) if reported_obj_raw is not None else None
    except (TypeError, ValueError):
        reported_obj = None

    if reported_obj is not None:
        true_half_min = None
        for sol_idx, sol in enumerate(solutions):
            open_fac = sol.get("open_facilities", []) or []
            assignments = sol.get("assignments", {}) or {}

            # Recompute z1, z2 from variables using instance data.
            z1_rec = 0
            z2_rec = 0
            for i in open_fac:
                if 0 <= i < nI:
                    z1_rec += f1[i]
                    z2_rec += f2[i]
            for j_str, i_val in assignments.items():
                try:
                    j = int(j_str)
                    i = int(i_val)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < nI and 0 <= j < nJ:
                    z1_rec += c1[i][j]
                    z2_rec += c2[i][j]

            half = (z1_rec + z2_rec) / 2.0
            if true_half_min is None or half < true_half_min:
                true_half_min = half

        if true_half_min is not None:
            abs_diff = abs(reported_obj - true_half_min)
            # Objectives are integer per the paper (outcome space Z^2),
            # so (z1+z2)/2 lives on a half-integer grid. A 0.5 absolute
            # floor plus 0.1% relative tolerance lets legitimate rounding
            # through while catching any nontrivial fabrication.
            obj_tol = max(0.5, 1e-3 * abs(true_half_min))
            if abs_diff > obj_tol:
                violated_constraints.add(3)
                normalizer = max(abs(true_half_min), eps)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported_obj} differs from recomputed "
                    f"min_s (z1_s+z2_s)/2 = {true_half_min} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(reported_obj),
                    "rhs": float(true_half_min),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer)
                })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for BO-SSCFLP solutions"
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

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: No hard constraint violations detected.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
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
