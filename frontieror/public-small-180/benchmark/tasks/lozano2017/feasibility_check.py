"""
Feasibility checker for the Bilevel Mixed-Integer Linear Program (BMILP)
from Lozano & Smith (2017).

Checks the three hard constraints of the BMILP formulation (14a)-(14d):
  Constraint 1 (14b): A1*x + B1*y <= b1  (upper-level constraints)
  Constraint 2 (14c): submitted complete follower response y is feasible and
                       its reported value is internally consistent.
  Constraint 3 (14d): x in H^x  (x >= 0, all integer)
  Constraint 4      : objective_value consistency (Tier C anti-gaming check)
                       reported objective_value must equal the recomputed
                       leader objective phi^l(x,y) = c*x + d1*y of (14a).
                       d1 (leader's coefficients on follower variables) is
                       explicitly fixed to zero by the released instance
                       contract, matching the Gurobi reference solver.
"""

import json
import math
import argparse
import numpy as np

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'x', 'y')
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



def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for BMILP (Lozano & Smith 2017)")
    parser.add_argument("--instance_path", required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to candidate solution JSON file")
    parser.add_argument("--result_path", required=True,
                        help="Path to write feasibility result JSON file")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)
    with open(args.solution_path) as f:
        sol = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
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

    # Instance data
    n1 = data["n1"]
    n2 = data["n2"]
    m1 = data["m1"]
    m2 = data["m2"]
    A1 = np.array(data["A1"], dtype=float)
    B1 = np.array(data["B1"], dtype=float)
    b1 = np.array(data["b1"], dtype=float)
    A2 = np.array(data["A2"], dtype=float)
    B2 = np.array(data["B2"], dtype=float)
    b2 = np.array(data["b2"], dtype=float)
    d2 = np.array(data["d2"], dtype=float)
    integer_follower = set(data["integer_follower_vars"])  # 1-indexed

    # Solution vectors
    x = np.array(sol["x"], dtype=float)
    y = np.array(sol["y"], dtype=float)

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # ------------------------------------------------------------------
    # Constraint 1 (14b): A1*x + B1*y <= b1, for j = 1, ..., m1
    # ------------------------------------------------------------------
    for j in range(m1):
        lhs_val = float(A1[j] @ x + B1[j] @ y)
        rhs_val = float(b1[j])
        excess = lhs_val - rhs_val  # <= constraint
        if excess > tol:
            violated_set.add(1)
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Upper-level constraint {j+1}: "
                f"A1[{j+1}]*x + B1[{j+1}]*y = {lhs_val:.6f} > b1[{j+1}] = {rhs_val:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": excess,
                "normalizer": normalizer,
                "ratio": excess / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint 2 (14c): Follower optimality
    #   y in argmax{d2*y^f | A2*x + B2*y^f <= b2; y^f in H^y}
    #
    # Sub-checks:
    #   2a. Follower feasibility: A2*x + B2*y <= b2
    #   2b. y >= 0
    #   2c. Integrality of follower integer variables
    #   2d. Follower optimality (d2*y equals optimal follower objective)
    # ------------------------------------------------------------------

    # 2a: Follower feasibility — A2*x + B2*y <= b2
    follower_feasible = True
    for j in range(m2):
        lhs_val = float(A2[j] @ x + B2[j] @ y)
        rhs_val = float(b2[j])
        excess = lhs_val - rhs_val
        if excess > tol:
            follower_feasible = False
            violated_set.add(2)
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Follower feasibility constraint {j+1}: "
                f"A2[{j+1}]*x + B2[{j+1}]*y = {lhs_val:.6f} > b2[{j+1}] = {rhs_val:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": excess,
                "normalizer": normalizer,
                "ratio": excess / normalizer,
            })

    # 2b: y >= 0
    for k in range(n2):
        if y[k] < -tol:
            violated_set.add(2)
            lhs_val = float(y[k])
            rhs_val = 0.0
            excess = -lhs_val  # >= constraint: violation = RHS - LHS
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Follower non-negativity: y[{k+1}] = {lhs_val:.6f} < 0"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": excess,
                "normalizer": normalizer,
                "ratio": excess / normalizer,
            })

    # 2c: Integrality of integer follower variables
    for j_idx in sorted(integer_follower):  # 1-indexed
        val = y[j_idx - 1]
        frac = abs(val - round(val))
        if frac > tol:
            violated_set.add(2)
            lhs_val = float(val)
            rhs_val = float(round(val))
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Follower integrality: y[{j_idx}] = {val:.6f} "
                f"not integer (frac = {frac:.6f})"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": frac,
                "normalizer": normalizer,
                "ratio": frac / normalizer,
            })

    # 2d: Complete follower witness objective consistency.  The checker does
    # not run an optimizer; benchmark quality is evaluated from the submitted
    # complete solution and its reported objectives.
    if follower_feasible:
        computed = float(d2 @ y)
        reported_follower = float(sol.get("follower_objective_value", computed))
        gap = abs(reported_follower - computed)
        if gap > tol:
            violated_set.add(2)
            normalizer = max(abs(computed), eps)
            violations.append(
                f"Follower objective mismatch: reported={reported_follower:.6f}, "
                f"d2*y={computed:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 2, "lhs": reported_follower, "rhs": computed,
                "raw_excess": gap, "normalizer": normalizer,
                "ratio": gap / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint 3 (14d): x in H^x  (x >= 0, all integer)
    # ------------------------------------------------------------------

    # 3a: x >= 0
    for k in range(n1):
        if x[k] < -tol:
            violated_set.add(3)
            lhs_val = float(x[k])
            rhs_val = 0.0
            excess = -lhs_val
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Leader non-negativity: x[{k+1}] = {lhs_val:.6f} < 0"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": excess,
                "normalizer": normalizer,
                "ratio": excess / normalizer,
            })

    # 3b: All leader variables must be integer (Assumption 3)
    for k in range(n1):
        frac = abs(x[k] - round(x[k]))
        if frac > tol:
            violated_set.add(3)
            lhs_val = float(x[k])
            rhs_val = float(round(x[k]))
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Leader integrality: x[{k+1}] = {x[k]:.6f} "
                f"not integer (frac = {frac:.6f})"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": frac,
                "normalizer": normalizer,
                "ratio": frac / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint 4: Objective consistency  (Tier C anti-gaming defense)
    #
    #   The reported objective_value must equal the recomputed leader
    #   objective phi^l(x, y) = c*x + d1*y of (14a).  d1 (the leader's
    #   coefficients on the follower variables) is explicitly fixed to zero
    #   by the released BMILP instance contract.
    #
    #   This catches LLM-evolved candidates that pass every constraint
    #   above but lie about objective_value (e.g. report 0 or
    #   sys.float_info.max) to game the eval score.
    # ------------------------------------------------------------------
    c = np.array(data["c"], dtype=float)
    if data.get("leader_follower_objective_coefficients_fixed_zero") is not True:
        violations.append("instance contract does not explicitly fix d1=0")
        violated_set.add(4)
    d1 = np.zeros(n2, dtype=float)
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(c @ x + d1 @ y)
            # 0.1% relative tolerance with a 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if math.isfinite(reported):
                abs_diff = abs(reported - true_obj)
                is_violation = abs_diff > obj_tol
            else:
                # non-finite reported value (inf / nan) is always a lie
                abs_diff = abs(true_obj) + 1.0  # finite sentinel for reporting
                is_violation = True
            if is_violation:
                violated_set.add(4)
                normalizer = max(abs(true_obj), eps)
                violations.append(
                    f"Objective consistency: reported objective_value = "
                    f"{reported} differs from recomputed leader objective "
                    f"c*x + d1*y = {true_obj:.6f} "
                    f"(|diff| = {abs_diff:.6f} > tol = {obj_tol:.6f})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported if math.isfinite(reported) else true_obj,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # ------------------------------------------------------------------
    # Build and write result
    # ------------------------------------------------------------------
    violated_constraints = sorted(violated_set)
    result = {
        "feasible": len(violated_constraints) == 0,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

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
