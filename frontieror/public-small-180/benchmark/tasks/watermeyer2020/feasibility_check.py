#!/usr/bin/env python3
"""
Feasibility checker for RCPSP/max-pi solutions.

Checks the four hard constraints of Problem P1 from Watermeyer & Zimmermann (2020):
  Constraint 1: S_j - S_i >= delta_ij  for all (i,j) in E   (temporal)
  Constraint 2: sum_{i in V} r_{ik}^c(S_i) <= R_k  for all k in R  (resource capacity)
  Constraint 3: S_0 = 0
  Constraint 4: S_i in Z_{>=0}  for all i in V  (non-negative integer start times)
  Constraint 5: objective_value == S_{n+1}  (Tier C obj-recompute defense)
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'start_times')
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



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_r_u(S_i, p_i, Pi_k):
    """r^u_{ik}(S_i) = |]S_i, S_i + p_i] ∩ Pi_k|

    Count periods in Pi_k that are strictly after S_i and at most S_i + p_i.
    """
    if p_i == 0:
        return 0
    lo = S_i + 1
    hi = S_i + p_i
    count = 0
    for t in Pi_k:
        if t > hi:
            break
        if t >= lo:
            count += 1
    return count


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    # Extract start times from solution (handle both formats)
    start_times = solution.get("start_times") or solution.get("schedule")
    if start_times is None:
        # No schedule provided (e.g., solver reported infeasible)
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["No schedule provided in solution (solver status: {})".format(
                solution.get("status", "unknown"))],
            "violation_magnitudes": []
        }

    # Convert keys to int
    S = {int(k): v for k, v in start_times.items()}

    num_activities = instance["num_activities"]
    activities = instance["activities"]
    temporal_constraints = instance["temporal_constraints"]
    resources = instance["resources"]

    # Parse processing times and resource demands
    p = {}
    r_d = {}
    for i_str, act_data in activities.items():
        i = int(i_str)
        p[i] = act_data["processing_time"]
        r_d[i] = {}
        for k_str, demand in act_data["resource_demands"].items():
            r_d[i][int(k_str)] = demand

    # Parse resource data
    R_cap = {}
    Pi = {}
    for k_str, res_data in resources.items():
        k = int(k_str)
        R_cap[k] = res_data["capacity"]
        Pi[k] = sorted(res_data["periods"])

    # =========================================================================
    # Constraint 1: S_j - S_i >= delta_ij  for all (i,j) in E
    # This is a >= constraint: LHS = S_j - S_i, RHS = delta_ij
    # violation_amount = max(0, RHS - LHS) = max(0, delta_ij - (S_j - S_i))
    # =========================================================================
    constraint_1_violations = []
    for tc in temporal_constraints:
        i = tc["i"]
        j = tc["j"]
        delta = tc["delta"]

        lhs = S[j] - S[i]
        rhs = delta
        violation_amount = max(0.0, rhs - lhs)

        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            constraint_1_violations.append({
                "i": i, "j": j, "delta": delta,
                "lhs": float(lhs), "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio)
            })

    if constraint_1_violations:
        violated_constraints_set.add(1)
        msgs = []
        for v in constraint_1_violations:
            msgs.append(
                f"Temporal constraint violated: S_{v['j']} - S_{v['i']} = {v['lhs']} "
                f"< delta_{v['i']},{v['j']} = {v['rhs']}"
            )
        violations.append(
            "Constraint 1 (temporal): {} violation(s) - {}".format(
                len(constraint_1_violations),
                "; ".join(msgs)
            )
        )
        for v in constraint_1_violations:
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": v["lhs"],
                "rhs": v["rhs"],
                "raw_excess": v["raw_excess"],
                "normalizer": v["normalizer"],
                "ratio": v["ratio"]
            })

    # =========================================================================
    # Constraint 2: sum_{i in V} r_{ik}^c(S_i) <= R_k  for all k in R
    # This is a <= constraint: LHS = total consumption, RHS = R_k
    # violation_amount = max(0, LHS - RHS)
    # =========================================================================
    constraint_2_violations = []
    for k_str in resources:
        k = int(k_str)
        total_consumption = 0.0
        for i in range(num_activities):
            demand = r_d[i].get(k, 0)
            if demand == 0:
                continue
            r_u = compute_r_u(S[i], p[i], Pi[k])
            total_consumption += r_u * demand

        lhs = total_consumption
        rhs = float(R_cap[k])
        violation_amount = max(0.0, lhs - rhs)

        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            constraint_2_violations.append({
                "k": k,
                "lhs": float(lhs), "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio)
            })

    if constraint_2_violations:
        violated_constraints_set.add(2)
        msgs = []
        for v in constraint_2_violations:
            msgs.append(
                f"Resource {v['k']}: total consumption {v['lhs']} > capacity {v['rhs']}"
            )
        violations.append(
            "Constraint 2 (resource capacity): {} violation(s) - {}".format(
                len(constraint_2_violations),
                "; ".join(msgs)
            )
        )
        for v in constraint_2_violations:
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": v["lhs"],
                "rhs": v["rhs"],
                "raw_excess": v["raw_excess"],
                "normalizer": v["normalizer"],
                "ratio": v["ratio"]
            })

    # =========================================================================
    # Constraint 3: S_0 = 0
    # This is an = constraint: LHS = S_0, RHS = 0
    # violation_amount = |S_0 - 0| = |S_0|
    # =========================================================================
    lhs_3 = float(S.get(0, 0))
    rhs_3 = 0.0
    violation_amount_3 = abs(lhs_3 - rhs_3)

    if violation_amount_3 > tol:
        violated_constraints_set.add(3)
        normalizer_3 = max(abs(rhs_3), eps)
        ratio_3 = violation_amount_3 / normalizer_3
        violations.append(
            f"Constraint 3 (S_0 = 0): S_0 = {lhs_3} != 0"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs_3,
            "rhs": rhs_3,
            "raw_excess": float(violation_amount_3),
            "normalizer": float(normalizer_3),
            "ratio": float(ratio_3)
        })

    # =========================================================================
    # Constraint 4: S_i in Z_{>=0}  for all i in V
    # Two sub-checks: non-negativity (>= 0) and integrality
    # For non-negativity: LHS = S_i, RHS = 0, >= constraint
    #   violation_amount = max(0, 0 - S_i) = max(0, -S_i)
    # For integrality: violation_amount = |S_i - round(S_i)|
    # =========================================================================
    constraint_4_violations = []
    for i in range(num_activities):
        s_i = S.get(i)
        if s_i is None:
            continue

        # Non-negativity check
        lhs_val = float(s_i)
        rhs_val = 0.0
        neg_violation = max(0.0, rhs_val - lhs_val)  # >= constraint
        if neg_violation > tol:
            normalizer = max(abs(rhs_val), eps)
            ratio = neg_violation / normalizer
            constraint_4_violations.append({
                "activity": i,
                "type": "non-negative",
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": float(neg_violation),
                "normalizer": float(normalizer),
                "ratio": float(ratio)
            })

        # Integrality check
        int_violation = abs(lhs_val - round(lhs_val))
        if int_violation > tol:
            # For integrality: treat as = constraint with RHS = round(S_i)
            rhs_int = float(round(lhs_val))
            normalizer = max(abs(rhs_int), eps)
            ratio = int_violation / normalizer
            constraint_4_violations.append({
                "activity": i,
                "type": "integrality",
                "lhs": lhs_val,
                "rhs": rhs_int,
                "raw_excess": float(int_violation),
                "normalizer": float(normalizer),
                "ratio": float(ratio)
            })

    if constraint_4_violations:
        violated_constraints_set.add(4)
        msgs = []
        for v in constraint_4_violations:
            if v["type"] == "non-negative":
                msgs.append(f"Activity {v['activity']}: S_{v['activity']} = {v['lhs']} < 0")
            else:
                msgs.append(f"Activity {v['activity']}: S_{v['activity']} = {v['lhs']} is not integer")
        violations.append(
            "Constraint 4 (non-negative integers): {} violation(s) - {}".format(
                len(constraint_4_violations),
                "; ".join(msgs)
            )
        )
        for v in constraint_4_violations:
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": v["lhs"],
                "rhs": v["rhs"],
                "raw_excess": v["raw_excess"],
                "normalizer": v["normalizer"],
                "ratio": v["ratio"]
            })

    # =========================================================================
    # Constraint 5: objective_value == S_{n+1}  (Tier C obj-recompute defense)
    # Full recompute: project duration is exactly the start time of the
    # project-end activity (index n+1). All variables needed are in the
    # solution. Treat as = constraint with tol = 0.5 since the objective is
    # an integer (project duration), so any integer mismatch by >=1 fires.
    # =========================================================================
    n = instance["n"]
    end_idx = n + 1
    reported_obj = solution.get("objective_value")
    if reported_obj is not None and end_idx in S:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(S[end_idx])
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5  # integer project duration; any |diff| >= 1 fires
            if abs_diff > obj_tol:
                violated_constraints_set.add(5)
                normalizer_5 = max(abs(true_obj), eps)
                ratio_5 = abs_diff / normalizer_5
                violations.append(
                    f"Constraint 5 (objective consistency): reported "
                    f"objective_value={reported} differs from recomputed "
                    f"S_{end_idx}={true_obj} (|diff|={abs_diff:.3g}, "
                    f"tol={obj_tol})"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer_5),
                    "ratio": float(ratio_5)
                })

    # =========================================================================
    # Build result
    # =========================================================================
    feasible = len(violated_constraints_set) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for RCPSP/max-pi solutions "
                    "(Watermeyer & Zimmermann 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result.")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {result['feasible']}")
    if not result['feasible']:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result['violations']:
            print(f"  {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
