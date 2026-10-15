#!/usr/bin/env python3
"""
Feasibility checker for Pm||sum w_j T_j (Kowalczyk et al., 2024).

Checks a candidate schedule against the BDDF formulation (Eq. 6), the
model that gurobi_code.py actually implements. The solver exports only
the chosen start time per job (schedule = {job_id: start_time}); BDDF
flow equations (6c)/(6d)/(6f) cannot be verified from edge flows, so we
enforce the equivalent schedule-level necessary conditions:
  Constraint 1 (6b): sum_{e in A^1 : p_B(e)=j} x_e = 1 => each job is
      scheduled exactly once.
  Constraint 2 (machine capacity; implied by BDD construction + (6d)):
      at most m jobs processing simultaneously at any time in [0, T].
  Constraint 3 (BDD node domain + (6e)): 0 <= start_j <= T - p_j so
      every scheduled job completes within the planning horizon T.
  Constraint 4 (objective consistency; Tier C anti-gaming defense):
      the reported objective_value must equal the total weighted
      tardiness recomputed from the exported schedule,
      obj = sum_{j in J} w_j * max(0, start_j + p_j - d_j)  (Eq. 6a / 1a).
      All objective-determining variables (the start times) are present
      in the solution, so this is a full recompute, not a lower bound.

This file is a side-by-side obj-recompute variant of feasibility_check.py;
the original is left untouched (see docs/data_review_fix/
checker_add_obj_recompute.md). Constraints 1-3 are byte-for-byte identical
to the original; constraint 4 is the only addition.
"""

import json
import argparse
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'schedule')
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



def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_planning_horizon(jobs, m):
    """T = ceil((sum p_j - p_max) / m) + p_max"""
    P = sum(j['p'] for j in jobs)
    p_max = max(j['p'] for j in jobs)
    return math.ceil((P - p_max) / m) + p_max


def extract_start_times(instance, solution):
    """
    Extract {job_id: start_time} from the solution, handling both formats.

    Efficient format: schedule is {machine_str: [job_id_list]}, jobs run
    sequentially on each machine starting from time 0.

    Gurobi format: schedule is {job_id_str: start_time}.
    """
    schedule = solution.get("schedule", {})
    if not schedule:
        return {}

    # Detect format: if values are lists, it's efficient format;
    # if values are ints/floats, it's Gurobi format.
    first_val = next(iter(schedule.values()))

    jobs_by_id = {j['id']: j for j in instance['jobs']}

    if isinstance(first_val, list):
        # Efficient format: machine -> ordered job list
        start_times = {}
        for machine_str, job_list in schedule.items():
            t = 0
            for jid in job_list:
                start_times[jid] = t
                t += jobs_by_id[jid]['p']
        return start_times
    else:
        # Gurobi format: job_id_str -> start_time
        return {int(k): int(v) for k, v in schedule.items()}


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    jobs = instance['jobs']
    m = instance['m']
    n = instance['n']
    T = instance.get('T', compute_planning_horizon(jobs, m))
    jobs_by_id = {j['id']: j for j in jobs}
    job_ids = set(j['id'] for j in jobs)

    start_times = extract_start_times(instance, solution)

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # ----------------------------------------------------------------
    # Constraint 1 (BDDF 6b): sum_{e in A^1 : p_B(e)=j} x_e = 1, j in J
    #   => each job is scheduled exactly once in the exported schedule.
    # ----------------------------------------------------------------
    scheduled_ids = set(start_times.keys())

    # Jobs not scheduled
    missing = job_ids - scheduled_ids
    for jid in sorted(missing):
        violated_constraints.add(1)
        violations.append(f"Job {jid} is not scheduled (assigned 0 times, expected 1)")
        lhs = 0.0
        rhs = 1.0
        raw_excess = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": raw_excess / normalizer,
        })

    # Jobs scheduled that don't exist
    extra = scheduled_ids - job_ids
    for jid in sorted(extra):
        violated_constraints.add(1)
        violations.append(f"Job {jid} is scheduled but does not exist in the instance")
        lhs = 1.0
        rhs = 0.0
        raw_excess = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": raw_excess / normalizer,
        })

    # Check for duplicate job assignments (efficient format can have duplicates)
    schedule = solution.get("schedule", {})
    first_val = next(iter(schedule.values()), None)
    if isinstance(first_val, list):
        all_assigned = []
        for machine_str, job_list in schedule.items():
            all_assigned.extend(job_list)
        from collections import Counter
        counts = Counter(all_assigned)
        for jid, cnt in counts.items():
            if cnt > 1 and jid in job_ids:
                violated_constraints.add(1)
                violations.append(
                    f"Job {jid} is scheduled {cnt} times (expected 1)"
                )
                lhs = float(cnt)
                rhs = 1.0
                raw_excess = abs(lhs - rhs)
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

    # ----------------------------------------------------------------
    # Constraint 2 (BDDF machine capacity; implied by BDD construction
    #   and eq (6d) sum_{e in delta^-(1)} x_e = m): at most m jobs
    #   processing simultaneously at any time in [0, T].
    # ----------------------------------------------------------------
    # For each time period t (1-indexed), count how many jobs are processing.
    # Job j starting at period s (1-indexed, time s-1) is processing during
    # periods s, s+1, ..., s+p_j-1. A job with start_time st (0-indexed)
    # starts at period st+1.
    for t in range(1, T + 1):
        count = 0
        for jid, st in start_times.items():
            if jid not in jobs_by_id:
                continue
            p_j = jobs_by_id[jid]['p']
            period_start = st + 1  # TIF period
            # Job processes during periods period_start .. period_start + p_j - 1
            if period_start <= t <= period_start + p_j - 1:
                count += 1

        lhs = float(count)
        rhs = float(m)
        violation_amount = max(0.0, lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(2)
            violations.append(
                f"At time period {t}, {count} jobs are processing (capacity is {m})"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ----------------------------------------------------------------
    # Constraint 3 (BDDF node-domain / eq (6e) x_e in {0,1} restricts
    #   high edges to valid BDD nodes): for every scheduled job j the
    #   start time t satisfies 0 <= t <= T - p_j (job fits in horizon).
    # ----------------------------------------------------------------
    for jid, st in start_times.items():
        if jid not in jobs_by_id:
            continue
        p_j = jobs_by_id[jid]['p']
        period = st + 1  # TIF period (1-indexed)

        # Check lower bound: period >= 1 (start_time >= 0)
        if period < 1:
            violated_constraints.add(3)
            violations.append(
                f"Job {jid} starts at time {st} (period {period}), "
                f"which is before the valid range [1, {T - p_j + 1}]"
            )
            lhs = float(period)
            rhs = 1.0
            raw_excess = max(0.0, rhs - lhs)
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": raw_excess / normalizer,
            })

        # Check upper bound: period <= T - p_j + 1 (job completes within horizon)
        max_period = T - p_j + 1
        if period > max_period:
            violated_constraints.add(3)
            violations.append(
                f"Job {jid} starts at period {period} but must start by period "
                f"{max_period} to complete within horizon T={T} "
                f"(completion time {st + p_j} > {T})"
            )
            lhs = float(period)
            rhs = float(max_period)
            raw_excess = max(0.0, lhs - rhs)
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": raw_excess / normalizer,
            })

    # ----------------------------------------------------------------
    # Constraint 4 (objective consistency; Tier C anti-gaming defense):
    #   The reported objective_value must equal the total weighted
    #   tardiness recomputed from the exported schedule. A job j with
    #   0-indexed start time st completes at time st + p_j (BDDF cost
    #   c_e = w_j * max{0, t + p_j - d_j}; equivalently TIF c~_{jt} with
    #   period = st + 1 yields completion st + p_j). The objective (6a):
    #     obj = sum_{j in J} w_j * max(0, start_j + p_j - d_j)
    #   Every objective-determining variable (the start times) is present
    #   in the solution, so this is a FULL recompute. We compare the
    #   reported value to the recomputed value within a tight tolerance.
    #   The objective is integer-valued (p, d, w, start times are all
    #   integers), so a tolerance of 0.5 fires on any genuine mismatch
    #   of >= 1 while tolerating float round-trip noise.
    # ----------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for jid, st in start_times.items():
                if jid not in jobs_by_id:
                    # Non-existent jobs are flagged by constraint 1; they
                    # contribute nothing to a well-defined objective.
                    continue
                job = jobs_by_id[jid]
                completion = st + job['p']
                tardiness = max(0, completion - job['d'])
                true_obj += job['w'] * tardiness
            true_obj = float(true_obj)
            abs_diff = abs(reported - true_obj)
            # 0.5 absolute floor (integer objective) with a 1e-6 relative
            # term as a safety net for very large objectives.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(4)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed total weighted tardiness "
                    f"sum_j w_j*max(0, start_j+p_j-d_j)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                rhs = true_obj
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported,
                    "rhs": rhs,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # ----------------------------------------------------------------
    # Build result
    # ----------------------------------------------------------------
    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Pm||sum w_j T_j (Kowalczyk et al., 2024)"
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the data instance')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the JSON file containing the candidate solution')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write the JSON file containing the feasibility result')
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

    print(f"Feasibility result written to {args.result_path}")
    if result['feasible']:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
