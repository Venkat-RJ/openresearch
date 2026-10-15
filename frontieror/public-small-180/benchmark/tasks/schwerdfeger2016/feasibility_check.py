"""
Feasibility checker for P || NSSWD (Workload Balancing on Identical Parallel Machines).

Checks the hard constraints from the mathematical formulation:
  Constraint 1 (Eq. 3): sum_j p_j * x_{ij} = C_i,  for each machine i
  Constraint 2 (Eq. 4): sum_i x_{ij} = 1,           for each job j
  Constraint 3 (Eq. 5): x_{ij} in {0, 1},           for all i, j
  Constraint 4 (Eq. 2): objective_value consistency with assignment
                        (Tier C: full recompute of NSSWD from x_{ij})
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'machine_completion_times', 'objective_value')
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

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    n = instance["num_jobs"]
    m = instance["num_machines"]
    p = instance["processing_times"]  # list of length n

    assignment = solution.get("assignment", {})
    machine_completion_times = solution.get("machine_completion_times", {})

    # Build assignment matrix: for each job, which machines is it assigned to?
    job_assignment_count = [0] * n
    job_assignments = [[] for _ in range(n)]  # which machines each job is on

    for machine_key, job_list in assignment.items():
        for j in job_list:
            if 0 <= j < n:
                job_assignment_count[j] += 1
                job_assignments[j].append(machine_key)

    # ------------------------------------------------------------------
    # Constraint 1 (Eq. 3): sum_j p_j * x_{ij} = C_i for each machine i
    # ------------------------------------------------------------------
    for machine_key, job_list in assignment.items():
        computed_completion = sum(p[j] for j in job_list if 0 <= j < n)
        reported_completion = machine_completion_times.get(machine_key, 0)
        if isinstance(reported_completion, str):
            reported_completion = float(reported_completion)

        lhs = computed_completion
        rhs = reported_completion
        violation_amount = abs(lhs - rhs)

        if violation_amount > tol:
            violated_constraints.add(1)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint 1 violated on machine {machine_key}: "
                f"sum(p_j) = {lhs}, reported C_i = {rhs}, diff = {violation_amount:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ------------------------------------------------------------------
    # Constraint 2 (Eq. 4): sum_i x_{ij} = 1 for each job j
    # ------------------------------------------------------------------
    for j in range(n):
        count = job_assignment_count[j]
        lhs = count
        rhs = 1
        violation_amount = abs(lhs - rhs)

        if violation_amount > tol:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            if count == 0:
                msg = f"Constraint 2 violated: job {j} is not assigned to any machine"
            else:
                machines_str = ", ".join(str(mk) for mk in job_assignments[j])
                msg = (
                    f"Constraint 2 violated: job {j} is assigned to "
                    f"{count} machines ({machines_str}) instead of exactly 1"
                )
            violations.append(msg)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ------------------------------------------------------------------
    # Constraint 3 (Eq. 5): x_{ij} in {0, 1} — domain/integrality
    # Check that all job indices in assignments are valid (0..n-1)
    # and that the solution uses exactly m machines.
    # ------------------------------------------------------------------
    for machine_key, job_list in assignment.items():
        for j in job_list:
            if not isinstance(j, int) or j < 0 or j >= n:
                violated_constraints.add(3)
                lhs = float(j) if isinstance(j, (int, float)) else 0.0
                rhs = 0.0  # expected: valid index in [0, n-1]
                violation_amount = 1.0  # binary domain violation
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 3 violated: invalid job index {j} "
                    f"on machine {machine_key} (valid range: 0..{n-1})"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(lhs) if isinstance(j, (int, float)) else 0.0,
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })

    # Check number of machines matches
    num_machines_in_sol = len(assignment)
    if num_machines_in_sol != m:
        violated_constraints.add(3)
        lhs = float(num_machines_in_sol)
        rhs = float(m)
        violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violations.append(
            f"Constraint 3 violated: solution has {num_machines_in_sol} machines "
            f"but instance requires {m}"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })

    # ------------------------------------------------------------------
    # Constraint 4 (Eq. 2): objective_value consistency (Tier C)
    # Full recompute of NSSWD = (1/mu) * sqrt(sum_i (C_i - mu)^2) from the
    # assignment. The solution carries every variable the objective
    # depends on (the per-machine job lists), so an exact recomputation
    # is possible. Reject when the reported value disagrees beyond a
    # tight tolerance: this catches obj=0 / obj=MAX_FLOAT exploits.
    # ------------------------------------------------------------------
    reported_obj_raw = solution.get("objective_value")
    if reported_obj_raw is not None and m > 0 and sum(p) > 0:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and math.isfinite(reported_obj):
            mu = sum(p) / m
            # Use the same C_i the assignment implies (ignore reported C_i:
            # constraint 1 already validates that link).
            C = [sum(p[j] for j in job_list if 0 <= j < n)
                 for job_list in assignment.values()]
            sq_dev = sum((c - mu) ** 2 for c in C)
            true_obj = math.sqrt(sq_dev) / mu if mu > 0 else 0.0
            abs_diff = abs(reported_obj - true_obj)
            # Tight tolerance: Gurobi-reported obj matches recompute to
            # exact float on the reference solutions, so 1e-8 absolute
            # floor with 1e-6 relative is safe against FP noise while
            # still catching obj=0 exploits on small-obj instances
            # (true_obj ~5e-4 here, so diff ~5e-4 >> 1e-8).
            obj_tol = max(1e-8, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(4)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint 4 violated: reported objective_value="
                    f"{reported_obj} differs from recomputed NSSWD="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported_obj),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })
        elif reported_obj is not None:
            # Non-finite (inf / nan) reported objective: also a violation.
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4 violated: reported objective_value is "
                f"non-finite ({reported_obj_raw})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float("inf") if reported_obj == float("inf") else 0.0,
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": eps,
                "ratio": float("inf"),
            })

    # Build result
    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for P || NSSWD"
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

    if result["feasible"]:
        print("FEASIBLE: All hard constraints satisfied.")
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
