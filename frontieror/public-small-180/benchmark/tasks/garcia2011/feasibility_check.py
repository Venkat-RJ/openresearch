#!/usr/bin/env python3
"""
Feasibility checker for the p-median problem (CF + CR formulation).

Based on: García, Labbé, Marín (2011) - "Solving Large p-Median Problems
with a Radius Formulation", INFORMS Journal on Computing 23(4):546-556.

Checks constraints numbered (1)-(5) as listed top-to-bottom in math_model.txt:
  (1) sum_j x_{ij} = 1           for all i        [each customer assigned once]
  (2) x_{ij} <= x_{jj}           for all i!=j      [assign only to open facilities]
  (3) sum_j x_{jj} = p                             [exactly p facilities (CF)]
  (4) sum_i y_i = p                                [exactly p medians (CR)]
  (5) z_{ik} + sum_{j: c_{ij}<D_{ik}} y_j >= 1    [CR covering]
  (6) objective consistency: reported objective_value must equal the
      recomputed total allocation cost sum_{i,j} c_{ij} x_{ij}  [Tier C
      defense against fabricated objective_value reports]

Also verifies that reported assignment costs match the actual rounded
Euclidean distances from the instance data.

NOTE: this is the obj-recompute ("_new") variant of feasibility_check.py.
It is a strict superset of the original: it adds only constraint (6) and
never alters the verdict of constraints (1)-(3).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignments', 'objective_value', 'opened_facilities')
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



def compute_cost_matrix(coords):
    """Compute rounded Euclidean distance matrix."""
    n = len(coords)
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            dist = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
            c[i][j] = dist
            c[j][i] = dist
    return c


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n"]
    p = instance["p"]
    coords = instance["node_coordinates"]
    c = compute_cost_matrix(coords)

    opened = set(solution["opened_facilities"])
    assignments = solution["assignments"]

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # Build assignment map: customer -> (facility, reported_cost)
    assign_map = {}
    for a in assignments:
        cust = a["customer"]
        fac = a["facility"]
        cost = a["cost"]
        if cust in assign_map:
            assign_map[cust].append((fac, cost))
        else:
            assign_map[cust] = [(fac, cost)]

    # ------------------------------------------------------------------
    # Constraint (1): sum_j x_{ij} = 1 for all i = 1..n
    # Each customer must be assigned to exactly one facility.
    # ------------------------------------------------------------------
    for i in range(n):
        num_assignments = len(assign_map.get(i, []))
        # LHS = num_assignments, RHS = 1 (equality)
        violation_amount = abs(num_assignments - 1)
        if violation_amount > tol:
            violated_set.add(1)
            rhs = 1.0
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint (1): Customer {i} has {num_assignments} assignment(s) instead of exactly 1"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(num_assignments),
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": float(violation_amount) / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint (2): x_{ij} <= x_{jj} for all i != j
    # A customer can only be assigned to an open facility.
    # ------------------------------------------------------------------
    for i in range(n):
        for fac, cost in assign_map.get(i, []):
            if i == fac:
                continue  # constraint only for i != j
            if fac not in opened:
                # x_{ij} = 1 but x_{jj} = 0, so LHS=1, RHS=0 for x_{ij} <= x_{jj}
                violated_set.add(2)
                lhs_val = 1.0
                rhs_val = 0.0
                violation_amount = lhs_val - rhs_val  # 1.0
                normalizer = max(abs(rhs_val), eps)
                violations.append(
                    f"Constraint (2): Customer {i} assigned to facility {fac} which is not open"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint (3): sum_j x_{jj} = p  (CF: exactly p facilities)
    # ------------------------------------------------------------------
    num_opened = len(opened)
    violation_amount_3 = abs(num_opened - p)
    if violation_amount_3 > tol:
        violated_set.add(3)
        rhs = float(p)
        normalizer = max(abs(rhs), eps)
        violations.append(
            f"Constraint (3): {num_opened} facilities opened instead of p={p}"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": float(num_opened),
            "rhs": rhs,
            "raw_excess": float(violation_amount_3),
            "normalizer": normalizer,
            "ratio": float(violation_amount_3) / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraints (4) and (5) SKIPPED: these belong to the CR (canonical
    # representation / radius) reformulation.
    #   (4) sum_i y_i = p — redundant with (3) under CF
    #   (5) z_{ik} + sum_{j: c_{ij}<D_{ik}} y_j >= 1 — uses cumulative cost
    #       indicator z_{ik} from the CR reformulation. A generated algorithm
    #       operating on the p-median problem description produces opened
    #       facilities + assignments (not z_{ik}); the CR covering constraint
    #       is equivalent to CF and automatically satisfied when (1)-(3) hold.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Additional: verify reported assignment costs match actual distances
    # (not a numbered constraint, but important for solution validity)
    # ------------------------------------------------------------------
    for a in assignments:
        i = a["customer"]
        j = a["facility"]
        reported = a["cost"]
        if 0 <= i < n and 0 <= j < n:
            actual = c[i][j]
            if abs(reported - actual) > tol:
                # Report under constraint (2) context since it relates to
                # the allocation variables, but use a distinct message.
                violated_set.add(2)
                diff = abs(reported - actual)
                normalizer = max(abs(actual), eps)
                violations.append(
                    f"Constraint (2): Customer {i} -> facility {j}: "
                    f"reported cost {reported} != actual distance {actual}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(reported),
                    "rhs": float(actual),
                    "raw_excess": diff,
                    "normalizer": normalizer,
                    "ratio": diff / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint (6): Objective consistency (Tier C obj-recompute defense)
    #
    # The CF/CR objective is the total allocation cost
    #     obj = sum_{i,j} c_{ij} x_{ij}
    #         = sum over assignments of c[customer][facility].
    # Every variable that determines the objective (the assignment
    # decisions x_{ij}) is present in the solution, so this is a FULL
    # recompute: the objective is reconstructed exactly from the instance
    # cost matrix and compared to the program-reported objective_value.
    #
    # The recompute uses the actual rounded-Euclidean cost matrix `c`
    # (NOT the per-assignment `cost` fields the program reports), so a
    # candidate that lies about objective_value is caught even if it also
    # tampers with the reported per-assignment costs. This rejects
    # fabricated objective_value reports (e.g. obj=0 or
    # obj=sys.float_info.max) that otherwise pass constraints (1)-(3).
    #
    # The check is skipped (no violation recorded) when the objective is
    # not deterministically recomputable from the solution -- i.e. there
    # are no assignments, an assignment carries non-integer / out-of-range
    # indices, or objective_value is missing / non-numeric. In every such
    # case constraints (1)/(2) already flag the solution, so skipping here
    # only avoids a spurious second violation.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    obj_computable = len(assignments) > 0
    true_obj = 0.0
    for a in assignments:
        i = a.get("customer")
        j = a.get("facility")
        if not (isinstance(i, int) and not isinstance(i, bool)
                and isinstance(j, int) and not isinstance(j, bool)
                and 0 <= i < n and 0 <= j < n):
            obj_computable = False
            break
        true_obj += c[i][j]

    if obj_computable and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            abs_diff = abs(reported - true_obj)
            # Costs are integer rounded Euclidean distances, so the true
            # objective is an integer sum: a 0.5 absolute floor makes any
            # off-by-one mismatch fire, with a 1e-6 relative term so very
            # large objectives keep a sane (non-flaky) tolerance.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_set.add(6)
                normalizer = max(abs(true_obj), eps)
                violations.append(
                    f"Constraint (6): reported objective_value {reported} "
                    f"differs from recomputed total allocation cost "
                    f"sum_ij c_ij*x_ij = {true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": float(abs_diff) / normalizer,
                })

    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for p-median (Garcia-Quiles et al. 2011)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to candidate solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write feasibility result JSON file")
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
        print(f"FEASIBLE - no constraint violations found")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
