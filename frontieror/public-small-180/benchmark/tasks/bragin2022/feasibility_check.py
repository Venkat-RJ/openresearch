#!/usr/bin/env python3
"""
Feasibility checker for the Generalized Assignment Problem (GAP)
from Bragin & Tucker (2022).

Constraints (numbered top-to-bottom from the formulation):
  Constraint 1: sum_i x[i][j] = 1 for all j  (each job assigned to exactly one machine)
  Constraint 2: sum_j a[i][j] * x[i][j] <= b[i] for all i  (machine capacity)
  Constraint 3: x[i][j] in {0, 1}  (binary integrality)
  Constraint 4: reported objective_value matches sum_{i,j} c[i][j] * x[i][j]
                (Tier C obj-consistency check — full recompute, since the
                solution carries every variable the objective depends on)
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignments', 'objective_value')
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


TOL = 1e-5
EPS = 1e-5


def check_feasibility(instance, solution):
    num_machines = instance["num_machines"]
    num_jobs = instance["num_jobs"]
    cost = instance["cost_matrix"]            # c[i][j]
    resource = instance["resource_matrix"]    # a[i][j]
    capacity = instance["capacities"]         # b[i]

    assignments = solution.get("assignments", {})

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Build assignment matrix from solution
    # assignments maps str(job_index) -> machine_index
    x = [[0] * num_jobs for _ in range(num_machines)]
    assigned_jobs = set()
    for job_str, machine in assignments.items():
        j = int(job_str)
        i = int(machine)
        if 0 <= i < num_machines and 0 <= j < num_jobs:
            x[i][j] = 1
            assigned_jobs.add(j)

    # -------------------------------------------------------------------------
    # Constraint 1: sum_i x[i][j] = 1 for all j (assignment equality)
    # -------------------------------------------------------------------------
    unassigned_jobs = []
    multi_assigned_jobs = []
    for j in range(num_jobs):
        lhs = sum(x[i][j] for i in range(num_machines))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            violated_constraints.add(1)
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })
            if lhs == 0:
                unassigned_jobs.append(j)
            else:
                multi_assigned_jobs.append(j)

    if unassigned_jobs:
        violations.append(
            f"Constraint 1 violated: jobs {unassigned_jobs} are not assigned to any machine"
        )
    if multi_assigned_jobs:
        violations.append(
            f"Constraint 1 violated: jobs {multi_assigned_jobs} are assigned to multiple machines"
        )

    # -------------------------------------------------------------------------
    # Constraint 2: sum_j a[i][j] * x[i][j] <= b[i] for all i (capacity)
    # -------------------------------------------------------------------------
    capacity_violated_machines = []
    for i in range(num_machines):
        lhs = sum(resource[i][j] * x[i][j] for j in range(num_jobs))
        rhs = float(capacity[i])
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > TOL:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })
            capacity_violated_machines.append(
                f"machine {i} (usage {lhs} > capacity {rhs})"
            )

    if capacity_violated_machines:
        violations.append(
            f"Constraint 2 violated: capacity exceeded on {', '.join(capacity_violated_machines)}"
        )

    # -------------------------------------------------------------------------
    # Constraint 3: x[i][j] in {0, 1} (binary integrality)
    # -------------------------------------------------------------------------
    non_binary_vars = []
    for job_str, machine in assignments.items():
        j = int(job_str)
        i = int(machine)
        # Check that machine index is valid
        if i < 0 or i >= num_machines:
            violated_constraints.add(3)
            rhs_val = float(num_machines - 1)
            normalizer = max(abs(rhs_val), EPS)
            violation_amount = abs(i - max(0, min(i, num_machines - 1)))
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(i),
                "rhs": rhs_val,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": float(violation_amount) / normalizer
            })
            non_binary_vars.append(f"x[{i}][{j}] has invalid machine index")
        if j < 0 or j >= num_jobs:
            violated_constraints.add(3)
            rhs_val = float(num_jobs - 1)
            normalizer = max(abs(rhs_val), EPS)
            violation_amount = abs(j - max(0, min(j, num_jobs - 1)))
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(j),
                "rhs": rhs_val,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": float(violation_amount) / normalizer
            })
            non_binary_vars.append(f"x[{i}][{j}] has invalid job index")

    if non_binary_vars:
        violations.append(
            f"Constraint 3 violated: {', '.join(non_binary_vars)}"
        )

    # -------------------------------------------------------------------------
    # Constraint 4: objective consistency (Tier C defense).
    # Recompute true_obj = sum_{i,j} c[i][j] * x[i][j] from the solution
    # variables and compare to the reported objective_value. The GAP solution
    # carries every obj-determining variable (the full assignment dict), so a
    # full recompute applies. Tolerance: max(1e-3 absolute, 1e-3 relative)
    # plus a 0.5 integer-floor since cost coefficients are integers in this
    # benchmark and any genuine LP/MIP-precision deviation stays well below
    # half a unit, while LLM exploits typically lie by orders of magnitude.
    # -------------------------------------------------------------------------
    reported_raw = solution.get("objective_value")
    if reported_raw is not None:
        try:
            reported = float(reported_raw)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(
                cost[i][j] * x[i][j]
                for i in range(num_machines) for j in range(num_jobs)
            ))
            abs_diff = abs(reported - true_obj)
            tol = max(0.5, 1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(4)
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff) / float(normalizer),
                })
                violations.append(
                    f"Constraint 4 violated: reported objective_value={reported} "
                    f"differs from recomputed sum_(i,j) c[i][j]*x[i][j]={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )

    # -------------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------------
    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(description="Check feasibility of a GAP solution")
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

    feasible = result["feasible"]
    print(f"Feasibility: {feasible}")
    if not feasible:
        print(f"Violated constraints: {result['violated_constraints']}")
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
