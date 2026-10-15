#!/usr/bin/env python3
"""
Feasibility checker for the Mixed No-Idle Permutation Flowshop Scheduling
Problem (MNPFSP) from Pan & Ruiz (2014).

Checks all hard constraints from the MILP formulation against a candidate solution.

Constraints:
  1: each job occupies exactly one position
  2: each position is occupied by exactly one job (permutation length = n)
  3: C_{1,k} >= sum_j X_{j,k} * p_{1,j}
  4: C_{i,k} >= C_{i-1,k} + sum_j X_{j,k} * p_{i,j}
  5: no-idle equality / regular >= sequence-precedence
  6: non-negativity C_{i,k} >= 0
  7: valid job indices (X binary)
  8: objective consistency -- reported objective_value matches makespan
     C_{m-1, n-1} computed from solution's completion_times (Tier C
     defense against LLM score-gaming exploits).
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('completion_times', 'objective_value', 'sequence')
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



def compute_completion_times(perm, p, no_idle, n_seq, m):
    """
    Compute minimum feasible completion times for a given permutation in MNPFSP.

    For no-idle machines, the start time is chosen so that the no-idle equality
    and machine-precedence constraints are both satisfied.
    """
    C = [[0.0] * n_seq for _ in range(m)]

    # Machine 0: no predecessor machine, so C[0][k] = cumulative processing
    C[0][0] = p[0][perm[0]]
    for k in range(1, n_seq):
        C[0][k] = C[0][k - 1] + p[0][perm[k]]

    # Machines 1..m-1
    for i in range(1, m):
        if no_idle[i]:
            # No-idle: C[i][k] = S_i + cumP[k], need S_i >= C[i-1][k] - cumP[k-1] for all k
            # where cumP[k] = sum_{l=0}^{k} p[i][perm[l]]
            cumsum = 0.0  # tracks cumP[k-1]
            S_i = C[i - 1][0]  # k=0: S_i >= C[i-1][0]
            for k in range(1, n_seq):
                cumsum += p[i][perm[k - 1]]
                needed = C[i - 1][k] - cumsum
                if needed > S_i:
                    S_i = needed
            C[i][0] = S_i + p[i][perm[0]]
            for k in range(1, n_seq):
                C[i][k] = C[i][k - 1] + p[i][perm[k]]
        else:
            # Regular machine
            C[i][0] = C[i - 1][0] + p[i][perm[0]]
            for k in range(1, n_seq):
                C[i][k] = max(C[i - 1][k], C[i][k - 1]) + p[i][perm[k]]

    return C


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n_jobs"]
    m = instance["n_machines"]
    no_idle = instance["no_idle_flags"]
    pt = instance["processing_times"]  # pt[j][i]

    # p[i][j] = processing time of job j on machine i
    p = [[pt[j][i] for j in range(n)] for i in range(m)]

    # Extract permutation
    if "permutation" in solution:
        perm = solution["permutation"]
    elif "sequence" in solution:
        perm = solution["sequence"]
    else:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["No permutation or sequence found in solution"],
            "violation_magnitudes": [],
        }

    # Validate job indices before using them as array indices
    for k, j in enumerate(perm):
        if not isinstance(j, int) or j < 0 or j >= n:
            return {
                "feasible": False,
                "violated_constraints": [7],
                "violations": [
                    f"Constraint 7 violated: invalid job index {j} at position {k}"
                ],
                "violation_magnitudes": [],
            }

    # Completion times are required in the solution
    if "completion_times" not in solution:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["No completion_times found in solution"],
            "violation_magnitudes": [],
        }
    C = []
    for i in range(m):
        C.append(solution["completion_times"][f"machine_{i}"])

    violations = []
    violation_magnitudes = []
    violated_set = set()
    n_seq = len(perm)

    # ------------------------------------------------------------------
    # Constraint 1: Each job occupies exactly one position
    #   sum_{k} X_{j,k} = 1 for all j
    # ------------------------------------------------------------------
    job_counts = {}
    for j in perm:
        job_counts[j] = job_counts.get(j, 0) + 1

    for j in range(n):
        count = job_counts.get(j, 0)
        if count != 1:
            lhs = float(count)
            rhs = 1.0
            va = abs(lhs - rhs)
            if va > tol:
                violated_set.add(1)
                normalizer = max(abs(rhs), eps)
                if count == 0:
                    violations.append(f"Job {j} not assigned to any position")
                else:
                    violations.append(
                        f"Job {j} assigned to {count} positions instead of 1"
                    )
                violation_magnitudes.append(
                    {
                        "constraint": 1,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": va,
                        "normalizer": normalizer,
                        "ratio": va / normalizer,
                    }
                )

    # ------------------------------------------------------------------
    # Constraint 2: Each position occupied by exactly one job
    #   sum_{j} X_{j,k} = 1 for all k
    # ------------------------------------------------------------------
    if n_seq != n:
        lhs = float(n_seq)
        rhs = float(n)
        va = abs(lhs - rhs)
        if va > tol:
            violated_set.add(2)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Permutation length {n_seq} does not match number of jobs {n}"
            )
            violation_magnitudes.append(
                {
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": va,
                    "normalizer": normalizer,
                    "ratio": va / normalizer,
                }
            )

    # ------------------------------------------------------------------
    # Constraint 3: C_{1,k} >= sum_j X_{j,k} * p_{1,j}  for k=1..n
    #   i.e. C[0][k] >= p[0][perm[k]]
    # ------------------------------------------------------------------
    for k in range(n_seq):
        job_k = perm[k]
        lhs_val = C[0][k]
        rhs_val = float(p[0][job_k])
        va = rhs_val - lhs_val  # >= constraint
        if va > tol:
            violated_set.add(3)
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Constraint 3 violated at position {k}: "
                f"C[0][{k}]={lhs_val:.4f} < p[0][job {job_k}]={rhs_val:.4f}"
            )
            violation_magnitudes.append(
                {
                    "constraint": 3,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": va,
                    "normalizer": normalizer,
                    "ratio": va / normalizer,
                }
            )

    # ------------------------------------------------------------------
    # Constraint 4: C_{i,k} >= C_{i-1,k} + sum_j X_{j,k} * p_{i,j}
    #   for i=2..m, k=1..n  (0-indexed: i=1..m-1, k=0..n-1)
    # ------------------------------------------------------------------
    for i in range(1, m):
        for k in range(n_seq):
            job_k = perm[k]
            lhs_val = C[i][k]
            rhs_val = C[i - 1][k] + p[i][job_k]
            va = rhs_val - lhs_val
            if va > tol:
                violated_set.add(4)
                normalizer = max(abs(rhs_val), eps)
                violations.append(
                    f"Constraint 4 violated at machine {i}, position {k}: "
                    f"C[{i}][{k}]={lhs_val:.4f} < "
                    f"C[{i-1}][{k}]+p[{i}][job {job_k}]={rhs_val:.4f}"
                )
                violation_magnitudes.append(
                    {
                        "constraint": 4,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": va,
                        "normalizer": normalizer,
                        "ratio": va / normalizer,
                    }
                )

    # ------------------------------------------------------------------
    # Constraint 5: Sequence precedence for k=2..n, i=1..m
    #   (0-indexed: k=1..n-1, i=0..m-1)
    #   No-idle (i in M'):  C[i][k] = C[i][k-1] + p[i][perm[k]]
    #   Regular (i not in M'): C[i][k] >= C[i][k-1] + p[i][perm[k]]
    # ------------------------------------------------------------------
    for i in range(m):
        for k in range(1, n_seq):
            job_k = perm[k]
            lhs_val = C[i][k]
            rhs_val = C[i][k - 1] + p[i][job_k]
            if no_idle[i]:
                # Equality constraint
                va = abs(lhs_val - rhs_val)
                if va > tol:
                    violated_set.add(5)
                    normalizer = max(abs(rhs_val), eps)
                    violations.append(
                        f"Constraint 5 (no-idle =) violated at machine {i}, "
                        f"position {k}: C[{i}][{k}]={lhs_val:.4f} != "
                        f"C[{i}][{k-1}]+p[{i}][job {job_k}]={rhs_val:.4f}"
                    )
                    violation_magnitudes.append(
                        {
                            "constraint": 5,
                            "lhs": lhs_val,
                            "rhs": rhs_val,
                            "raw_excess": va,
                            "normalizer": normalizer,
                            "ratio": va / normalizer,
                        }
                    )
            else:
                # >= inequality
                va = rhs_val - lhs_val
                if va > tol:
                    violated_set.add(5)
                    normalizer = max(abs(rhs_val), eps)
                    violations.append(
                        f"Constraint 5 (regular >=) violated at machine {i}, "
                        f"position {k}: C[{i}][{k}]={lhs_val:.4f} < "
                        f"C[{i}][{k-1}]+p[{i}][job {job_k}]={rhs_val:.4f}"
                    )
                    violation_magnitudes.append(
                        {
                            "constraint": 5,
                            "lhs": lhs_val,
                            "rhs": rhs_val,
                            "raw_excess": va,
                            "normalizer": normalizer,
                            "ratio": va / normalizer,
                        }
                    )

    # ------------------------------------------------------------------
    # Constraint 6: Non-negativity  C_{i,k} >= 0
    # ------------------------------------------------------------------
    for i in range(m):
        for k in range(n_seq):
            lhs_val = C[i][k]
            rhs_val = 0.0
            va = rhs_val - lhs_val
            if va > tol:
                violated_set.add(6)
                normalizer = eps  # |RHS| = 0
                violations.append(
                    f"Constraint 6 violated: C[{i}][{k}]={lhs_val:.4f} < 0"
                )
                violation_magnitudes.append(
                    {
                        "constraint": 6,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": va,
                        "normalizer": normalizer,
                        "ratio": va / normalizer,
                    }
                )

    # ------------------------------------------------------------------
    # Constraint 7: Binary / valid job indices  X_{j,k} in {0,1}
    # ------------------------------------------------------------------
    for k, j in enumerate(perm):
        if not isinstance(j, int) or j < 0 or j >= n:
            violated_set.add(7)
            violations.append(
                f"Constraint 7 violated: invalid job index {j} at position {k}"
            )
            violation_magnitudes.append(
                {
                    "constraint": 7,
                    "lhs": float(j) if isinstance(j, (int, float)) else 0.0,
                    "rhs": float(n - 1),
                    "raw_excess": 1.0,
                    "normalizer": max(float(n - 1), eps),
                    "ratio": 1.0 / max(float(n - 1), eps),
                }
            )

    # ------------------------------------------------------------------
    # Constraint 8: Objective consistency -- reported objective_value must
    #   match the makespan C_{m,n} = C[m-1][n_seq-1] derived from the
    #   solution's own completion_times. This is the Tier C defense
    #   against LLM exploits that fabricate objective_value (e.g.
    #   obj=0 on a minimization problem) while keeping the schedule
    #   itself nominally feasible.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None and m > 0 and n_seq > 0:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            try:
                true_obj = float(C[m - 1][n_seq - 1])
            except (IndexError, TypeError, ValueError):
                true_obj = None
            if true_obj is not None:
                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with 1e-3 absolute floor.
                obj_tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    violated_set.add(8)
                    normalizer = max(abs(true_obj), eps)
                    violations.append(
                        f"Constraint 8 violated: reported objective_value="
                        f"{reported} differs from makespan C[{m-1}][{n_seq-1}]="
                        f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                    )
                    violation_magnitudes.append(
                        {
                            "constraint": 8,
                            "lhs": reported,
                            "rhs": true_obj,
                            "raw_excess": abs_diff,
                            "normalizer": normalizer,
                            "ratio": abs_diff / normalizer,
                        }
                    )

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MNPFSP solutions"
    )
    parser.add_argument(
        "--instance_path", required=True, help="Path to instance JSON file"
    )
    parser.add_argument(
        "--solution_path", required=True, help="Path to candidate solution JSON file"
    )
    parser.add_argument(
        "--result_path", required=True, help="Path to write feasibility result JSON"
    )
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


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
