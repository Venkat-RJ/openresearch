"""
Feasibility checker for the Generalized Set Partitioning Problem
for Aircrew Rostering (Gamache, Soumis, Marquis, Desrosiers, 1999).

Checks the following hard constraints from the mathematical model:
  Constraint (2): Pairing coverage — sum_k sum_{r in R_k} a_{pr} x_r + e_p = b_p
  Constraint (3): Roster assignment — exactly one roster per employee (sum_{r in R_k} x_r = 1)
  Constraint (4): Non-negativity of slack — e_p >= 0
  Constraint (5): Binary domain — x_r in {0, 1}
  Constraint (6): Objective consistency — the reported objective_value must
                  equal the recomputed objective sum_p c_p e_p (Tier C
                  defense against fabricated objective values).

This file is the obj-recompute-augmented variant of feasibility_check.py.
The original is kept untouched; constraints (2)-(5) are byte-identical to it.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'rosters', 'uncovered_pairings')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('rosters',)
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



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # Handle infeasible / no-solution cases: return null feasibility
    if (solution.get("objective_value") is None
            or solution.get("status") in ("no_solution", "infeasible")):
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": [f"No solution to check (status={solution.get('status', 'unknown')})"],
            "violation_magnitudes": [],
        }

    pairings = instance["pairings"]
    crew_members = instance["crew_members"]
    n_pairings = len(pairings)
    n_crew = len(crew_members)

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # Build the roster mapping: employee_id -> list of assigned pairing ids
    rosters = solution.get("rosters", {})

    # Determine slack variables (e_p) from the solution.
    # e_p = b_p - sum_k a_{pr} x_r  (the number of uncovered copies)
    # We can reconstruct this from the solution's uncovered_pairings field,
    # or compute it directly from rosters.

    # First, compute pairing coverage from rosters
    pairing_coverage = [0] * n_pairings
    for emp_key, roster in rosters.items():
        assigned_pairings = roster.get("pairings", [])
        for p_id in assigned_pairings:
            if 0 <= p_id < n_pairings:
                pairing_coverage[p_id] += 1

    # Build coverage requirement
    coverage_req = [p["coverage_requirement"] for p in pairings]

    # Compute slack: e_p = b_p - coverage_p
    slack = [coverage_req[p] - pairing_coverage[p] for p in range(n_pairings)]

    # =========================================================================
    # Constraint (2): Pairing coverage
    #   sum_k sum_{r in R_k} a_{pr} x_r + e_p = b_p,  for p = 1, ..., n
    #
    # With e_p = b_p - coverage_p, this is automatically satisfied by definition.
    # However, we must also check that coverage does not EXCEED b_p
    # (since e_p >= 0 is a separate constraint, overcoverage means e_p < 0
    # which would violate constraint (4); but the equality in (2) itself
    # can be violated if coverage > b_p because then e_p would need to be
    # negative to satisfy the equality, which is infeasible).
    #
    # We check: coverage_p + e_p == b_p.  Since we defined e_p = b_p - coverage_p,
    # the equality always holds. The real check is whether coverage_p <= b_p
    # (no overcoverage) which ties into constraint (4).
    #
    # Additionally, a pairing could be assigned to an employee who has it
    # assigned more than once (duplicate), which would be caught here.
    #
    # We check: for each pairing p, coverage_p + max(0, e_p) == b_p.
    # Violation happens when coverage_p != b_p - max(0, e_p), i.e., when
    # coverage exceeds b_p.
    # =========================================================================
    for p in range(n_pairings):
        b_p = coverage_req[p]
        cov_p = pairing_coverage[p]
        # e_p is the slack (must be >= 0, checked in constraint 4)
        # The equality constraint (2) says: cov_p + e_p = b_p
        # With e_p defined as b_p - cov_p, check if this creates issues:
        # If cov_p > b_p, then e_p < 0, violating (4) but (2) holds by definition.
        # If cov_p < b_p, then e_p > 0, which is fine — pairing is undercovered.
        # If cov_p == b_p, then e_p == 0, perfect coverage.
        #
        # The actual constraint (2) check: LHS = cov_p + e_p, RHS = b_p.
        # By our definition of e_p = b_p - cov_p, LHS = cov_p + (b_p - cov_p) = b_p = RHS.
        # So constraint (2) is always satisfied by construction.
        # The real violation scenario is overcoverage (cov_p > b_p), which
        # manifests as a violation of constraint (4) (e_p < 0).
        #
        # However, we should still check for the case where the solution has
        # explicit uncovered_pairings that are inconsistent with the roster data.
        # We treat constraint (2) as violated if coverage is inconsistent.
        lhs = cov_p + max(0, slack[p])
        rhs = b_p
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraint_set.add(2)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint (2) violated for pairing {p}: "
                f"coverage={cov_p}, slack={slack[p]}, "
                f"LHS={lhs}, RHS(b_p)={rhs}"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (3): Roster assignment
    #   sum_{r in R_k} x_r = 1,  for k = 1, ..., m
    #
    # Each employee must have exactly one roster assigned.
    # In the solution, each employee key in "rosters" represents one selected
    # roster (x_r = 1 for that roster). We check that every employee from
    # k = 0, ..., m-1 has exactly one roster entry.
    # =========================================================================
    for k in range(n_crew):
        key = str(k)
        if key in rosters:
            # Employee has a roster — count = 1 (x_r = 1 for this roster)
            lhs = 1.0
        else:
            # Employee has no roster entry — sum x_r = 0
            lhs = 0.0
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraint_set.add(3)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint (3) violated for employee {k}: "
                f"no roster assigned (sum x_r = {lhs}, expected {rhs})"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (4): Non-negativity of slack variables
    #   e_p >= 0,  for p = 1, ..., n
    #
    # e_p = b_p - coverage_p.  If coverage exceeds the requirement, e_p < 0.
    # =========================================================================
    for p in range(n_pairings):
        e_p = slack[p]
        # Constraint: e_p >= 0,  i.e., LHS = e_p, RHS = 0, operator >=
        # Violation when RHS > LHS, i.e., 0 > e_p, i.e., e_p < 0
        if e_p < -tol:
            violation_amount = -e_p  # how much RHS (0) exceeds LHS (e_p)
            violated_constraint_set.add(4)
            lhs_val = e_p
            rhs_val = 0.0
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Constraint (4) violated for pairing {p}: "
                f"e_p={e_p:.6f} < 0 (overcovered by {-e_p} copies, "
                f"coverage={pairing_coverage[p]}, b_p={coverage_req[p]})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (5): Binary domain
    #   x_r in {0, 1}
    #
    # In the solution representation, each employee has at most one roster
    # entry (representing x_r = 1 for that roster, 0 for all others).
    # The solution format inherently enforces binarity — each roster is
    # either selected (present in the dict) or not.
    #
    # We verify that no employee appears with multiple roster entries
    # (which would mean x_r = 1 for more than one roster of the same employee,
    # violating both constraint (3) and (5)).
    #
    # We also check that the roster variables are valid (no fractional values).
    # Since the JSON format uses one roster entry per employee, the binary
    # constraint is satisfied by construction. We validate the structure.
    # =========================================================================
    # Check for duplicate employee IDs in rosters (structural check)
    employee_ids_in_rosters = []
    for emp_key, roster in rosters.items():
        emp_id = roster.get("employee_id", int(emp_key))
        employee_ids_in_rosters.append(emp_id)

    from collections import Counter
    emp_counts = Counter(employee_ids_in_rosters)
    for emp_id, count in emp_counts.items():
        if count > 1:
            # Multiple rosters for same employee — x_r = 1 for multiple r in R_k
            lhs = float(count)
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                violated_constraint_set.add(5)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Constraint (5) violated for employee {emp_id}: "
                    f"{count} rosters selected (must be binary, "
                    f"at most 1 roster with x_r=1)"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # =========================================================================
    # Constraint (6): Objective consistency (Tier C — objective recomputation)
    #   Objective (1):  min  sum_{p=1}^{n} c_p e_p
    #   where c_p = duration of pairing p (minutes, = pairing.duration_minutes)
    #   and   e_p = number of uncovered copies of pairing p = b_p - coverage_p.
    #
    # FULL RECOMPUTE applies: every variable the objective depends on is
    # available. The selected rosters (x_r) fully determine pairing coverage,
    # hence every slack e_p; and the instance carries c_p (duration_minutes)
    # and b_p (coverage_requirement). So the true objective is computed
    # exactly from the same `slack` array used by constraints (2)-(4):
    #
    #   true_obj = sum_p duration_minutes[p] * max(0, e_p)
    #
    # (e_p is clamped at 0 because the slack variable is non-negative by
    # constraint (4); overcoverage is already reported there separately.)
    #
    # The check rejects any solution whose self-reported objective_value
    # disagrees with this recomputed value — closing the score-gaming
    # exploit where a candidate returns a fabricated objective (e.g. 0 or
    # sys.float_info.max) while the routes are technically feasible.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            durations = [p.get("duration_minutes", 0) for p in pairings]
            true_obj = float(sum(
                durations[p] * max(0, slack[p]) for p in range(n_pairings)
            ))
            abs_diff = abs(reported - true_obj)
            # true_obj is an integer total of minutes; the 0.5 absolute floor
            # catches any integer-level mismatch (>= 1 minute), while the
            # tiny relative term tolerates float rounding on large values.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraint_set.add(6)
                normalizer = max(abs(true_obj), eps)
                violations.append(
                    f"Constraint (6) violated: reported objective_value="
                    f"{reported} differs from recomputed objective "
                    f"sum_p c_p*e_p={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # =========================================================================
    # Build result
    # =========================================================================
    feasible = len(violated_constraint_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraint_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Aircrew Rostering "
                    "(Gamache et al. 1999)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
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

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE — no constraint violations detected.")
    else:
        print(f"INFEASIBLE — {len(result['violated_constraints'])} "
              f"constraint(s) violated: {result['violated_constraints']}")
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
