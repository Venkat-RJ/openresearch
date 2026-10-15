#!/usr/bin/env python3
"""
Feasibility checker for P || sum w_j T_j
(Identical parallel machine scheduling minimizing total weighted tardiness)

Based on the Arc-Time-Indexed Formulation (ATIF) from:
"An Improved Branch-Cut-and-Price Algorithm for Parallel Machine Scheduling Problems"
Oliveira & Pessoa (2019/2020), INFORMS Journal on Computing

Checks constraints (1b)-(1g) from the ATIF formulation, numbered 1-6 top to bottom,
plus an added objective-consistency check (constraint 7) defending against
LLM score-gaming exploits that lie about objective_value.

Constraint 1 (1b): Machine capacity — at most m jobs active simultaneously
Constraint 2 (1c): Job assignment — each job j in J scheduled exactly once
Constraint 3 (1d): Flow conservation — completion = start + processing time;
                    predecessor consistency for arc-based solutions
Constraint 4 (1e): Idle flow conservation — total processing fits within m*T
Constraint 5 (1f): Integrality — completion times are non-negative integers,
                    C_j >= p_j
Constraint 6 (1g): Domain bounds — start >= 0, completion <= T
Constraint 7 (obj, eq 1a): Objective consistency — reported objective_value
                    must equal the recomputed total weighted tardiness
                    sum_j w_j * max(0, C_j - d_j) within tolerance. Full
                    recompute is possible because every job's completion
                    time is in the solution and w_j, d_j are deterministic
                    instance data.
"""

import argparse
import json
import math
from collections import Counter


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # ---- Extract instance data ----
    jobs_data = instance["jobs"]
    n = instance["parameters"]["n"]
    m = instance["parameters"]["m"]

    p = {}
    w = {}
    d = {}
    for job in jobs_data:
        p[job["id"]] = job["processing_time"]
        w[job["id"]] = job["weight"]
        d[job["id"]] = job["due_date"]

    # Compute time horizon T = floor((sum p_j - p_max) / m) + p_max
    total_p = sum(j["processing_time"] for j in jobs_data)
    p_max = max(j["processing_time"] for j in jobs_data)
    T = (total_p - p_max) // m + p_max
    # Use instance-provided T if available
    if "time_horizon" in instance:
        T = instance["time_horizon"]

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # ---- Validate and extract the complete public schedule certificate ----
    if not isinstance(solution, dict):
        return {
            "feasible": False,
            "violated_constraints": ["solution_schema"],
            "violations": ["solution must be a JSON object"],
            "violation_magnitudes": [],
        }
    schedule = solution.get("schedule")
    reported_raw = solution.get("objective_value")
    schema_errors = []
    if not isinstance(schedule, list):
        schema_errors.append("schedule must be a list")
        schedule = []
    if (
        isinstance(reported_raw, bool)
        or not isinstance(reported_raw, (int, float))
        or not math.isfinite(float(reported_raw))
    ):
        schema_errors.append("objective_value must be a finite JSON number")

    required = {
        "job", "machine_id", "prev_job", "start_time", "completion_time",
        "weighted_tardiness",
    }
    for index, entry in enumerate(schedule):
        if not isinstance(entry, dict):
            schema_errors.append(f"schedule[{index}] must be an object")
            continue
        missing_fields = sorted(required - set(entry))
        if missing_fields:
            schema_errors.append(
                f"schedule[{index}] is missing fields: {missing_fields}"
            )
            continue
        if (
            isinstance(entry["job"], bool)
            or not isinstance(entry["job"], int)
            or isinstance(entry["machine_id"], bool)
            or not isinstance(entry["machine_id"], int)
            or isinstance(entry["prev_job"], bool)
            or not isinstance(entry["prev_job"], int)
        ):
            schema_errors.append(
                f"schedule[{index}] job, machine_id, and prev_job must be integers"
            )
        for field in ("start_time", "completion_time", "weighted_tardiness"):
            value = entry[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                schema_errors.append(
                    f"schedule[{index}].{field} must be a finite number"
                )
    if schema_errors:
        return {
            "feasible": False,
            "violated_constraints": ["solution_schema"],
            "violations": schema_errors,
            "violation_magnitudes": [],
        }

    # Build per-job data
    job_entries = {}
    job_ids_list = []
    for entry in schedule:
        jid = entry["job"]
        job_ids_list.append(jid)
        C_j = entry["completion_time"]
        S_j = entry["start_time"]
        job_entries[jid] = {
            "completion_time": C_j,
            "start_time": S_j,
            "prev_job": entry["prev_job"],
            "machine_id": entry["machine_id"],
            "weighted_tardiness": entry["weighted_tardiness"],
        }

    # ================================================================
    # Constraint 1 (1b): Machine capacity
    # ================================================================
    if job_entries:
        events = []
        for jid, info in job_entries.items():
            events.append((info["start_time"], +1))
            events.append((info["completion_time"], -1))
        events.sort(key=lambda e: (e[0], e[1]))

        current = 0
        max_concurrent = 0
        max_time = None
        for t_val, delta in events:
            current += delta
            if current > max_concurrent:
                max_concurrent = current
                max_time = t_val

        if max_concurrent > m + tol:
            violation_amount = float(max_concurrent - m)
            rhs = float(m)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Machine capacity exceeded: {max_concurrent} jobs active "
                f"simultaneously at time {max_time}, limit is {m}"
            )
            violated_constraints.add(1)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(max_concurrent),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

        # The certificate names the machine explicitly.  Validate each
        # machine's sequence, which is stronger than merely checking aggregate
        # concurrency and makes the assignment independently reproducible.
        by_machine = {k: [] for k in range(m)}
        for jid, info in job_entries.items():
            machine_id = info["machine_id"]
            if machine_id not in by_machine:
                violations.append(
                    f"Job {jid}: machine_id {machine_id} is outside 0..{m - 1}"
                )
                violated_constraints.add(1)
                continue
            by_machine[machine_id].append(jid)
        for machine_id, sequence in by_machine.items():
            sequence.sort(key=lambda jid: (job_entries[jid]["start_time"], jid))
            previous = 0
            previous_completion = 0.0
            for jid in sequence:
                info = job_entries[jid]
                if info["start_time"] < previous_completion - tol:
                    violations.append(
                        f"Machine {machine_id}: job {jid} starts at "
                        f"{info['start_time']} before the preceding job finishes "
                        f"at {previous_completion}"
                    )
                    violated_constraints.add(1)
                if info["prev_job"] != previous:
                    violations.append(
                        f"Machine {machine_id}: job {jid} reports prev_job="
                        f"{info['prev_job']}, expected {previous}"
                    )
                    violated_constraints.add(3)
                previous = jid
                previous_completion = info["completion_time"]

    # ================================================================
    # Constraint 2 (1c): Each job j in J processed exactly once
    # ================================================================
    expected_jobs = set(range(1, n + 1))
    scheduled_jobs = set(job_entries.keys())
    missing = sorted(expected_jobs - scheduled_jobs)
    extra = sorted(scheduled_jobs - expected_jobs)
    job_counts = Counter(job_ids_list)
    duplicates = sorted(j for j, c in job_counts.items() if c > 1)

    if missing:
        violations.append(
            f"Jobs not scheduled (each must appear exactly once): {missing}"
        )
        violated_constraints.add(2)
        for jid in missing:
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })

    if duplicates:
        for jid in duplicates:
            cnt = job_counts[jid]
            violations.append(f"Job {jid} scheduled {cnt} times instead of 1")
            violated_constraints.add(2)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(cnt),
                "rhs": 1.0,
                "raw_excess": float(cnt - 1),
                "normalizer": 1.0,
                "ratio": float(cnt - 1),
            })

    if extra:
        violations.append(f"Unknown jobs in schedule: {extra}")
        violated_constraints.add(2)
        for jid in extra:
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": 1.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": eps,
                "ratio": 1.0 / eps,
            })

    # ================================================================
    # Constraint 3 (1d): Flow conservation for jobs
    # ================================================================
    for jid in sorted(job_entries.keys()):
        if jid not in p:
            continue
        info = job_entries[jid]
        C_j = info["completion_time"]
        S_j = info["start_time"]
        expected_C = S_j + p[jid]
        diff = abs(C_j - expected_C)
        if diff > tol:
            rhs_val = float(expected_C)
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Job {jid}: completion_time {C_j} != start_time {S_j} "
                f"+ processing_time {p[jid]} = {expected_C}"
            )
            violated_constraints.add(3)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(C_j),
                "rhs": rhs_val,
                "raw_excess": diff,
                "normalizer": normalizer,
                "ratio": diff / normalizer,
            })

    # ================================================================
    # Constraint 4 (1e): Idle flow conservation
    # ================================================================
    scheduled_total_p = sum(p.get(jid, 0) for jid in job_entries.keys())
    capacity = m * T
    if scheduled_total_p > capacity + tol:
        violation_amount = float(scheduled_total_p - capacity)
        rhs_val = float(capacity)
        normalizer = max(abs(rhs_val), eps)
        violations.append(
            f"Total processing time {scheduled_total_p} exceeds total "
            f"machine capacity m*T = {m}*{T} = {capacity}"
        )
        violated_constraints.add(4)
        violation_magnitudes.append({
            "constraint": 4,
            "lhs": float(scheduled_total_p),
            "rhs": rhs_val,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # ================================================================
    # Constraint 5 (1f): Integrality of arc variables
    # ================================================================
    for jid in sorted(job_entries.keys()):
        if jid not in p:
            continue
        info = job_entries[jid]
        C_j = info["completion_time"]

        if abs(C_j - round(C_j)) > tol:
            rounded = round(C_j)
            diff = abs(C_j - rounded)
            normalizer = max(abs(float(rounded)), eps)
            violations.append(
                f"Job {jid}: completion_time {C_j} is not an integer"
            )
            violated_constraints.add(5)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(C_j),
                "rhs": float(rounded),
                "raw_excess": diff,
                "normalizer": normalizer,
                "ratio": diff / normalizer,
            })

        if C_j < p[jid] - tol:
            violation_amount = float(p[jid] - C_j)
            rhs_val = float(p[jid])
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Job {jid}: completion_time {C_j} < processing_time {p[jid]}"
            )
            violated_constraints.add(5)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(C_j),
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ================================================================
    # Constraint 6 (1g): Domain bounds for idle arcs / time bounds
    # ================================================================
    for jid in sorted(job_entries.keys()):
        if jid not in p:
            continue
        info = job_entries[jid]
        C_j = info["completion_time"]
        S_j = info["start_time"]

        if S_j < -tol:
            violation_amount = abs(S_j)
            rhs_val = 0.0
            normalizer = max(abs(rhs_val), eps)
            violations.append(f"Job {jid}: start_time {S_j} < 0")
            violated_constraints.add(6)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": float(S_j),
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

        if C_j > T + tol:
            violation_amount = float(C_j - T)
            rhs_val = float(T)
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Job {jid}: completion_time {C_j} > time horizon T={T}"
            )
            violated_constraints.add(6)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": float(C_j),
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ================================================================
    # Constraint 7 (obj, eq 1a): Objective consistency.
    #   The objective is total weighted tardiness:
    #     z = sum_{j in J} w_j * max(0, C_j - d_j)
    #   Every job's completion time is in the schedule, and w_j, d_j are
    #   deterministic instance data, so we full-recompute z and compare to
    #   the reported objective_value. This rejects LLM exploits that lie
    #   about objective_value while returning a constraint-feasible schedule.
    # ================================================================
    if job_entries:
        reported = float(reported_raw)
        if reported is not None:
            true_obj = 0.0
            for jid, info in job_entries.items():
                if jid not in w or jid not in d:
                    continue  # job id outside instance — handled by constraint 2
                C_j = info["completion_time"]
                tardiness = max(0.0, float(C_j) - float(d[jid]))
                expected_contribution = float(w[jid]) * tardiness
                true_obj += expected_contribution
                contribution_diff = abs(
                    float(info["weighted_tardiness"]) - expected_contribution
                )
                if contribution_diff > tol:
                    violations.append(
                        f"Job {jid}: weighted_tardiness="
                        f"{info['weighted_tardiness']} does not match "
                        f"{expected_contribution}"
                    )
                    violated_constraints.add(7)
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-5, 1e-8 * abs(true_obj))
            if abs_diff > obj_tol:
                rhs_val = float(true_obj)
                normalizer = max(abs(rhs_val), eps)
                violations.append(
                    f"Objective consistency violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_j w_j*max(0,C_j-d_j)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violated_constraints.add(7)
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": float(reported),
                    "rhs": rhs_val,
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": float(abs_diff) / normalizer,
                })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    }


def _write_error_result(result_path, message):
    result = {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": [message],
        "violation_magnitudes": [],
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for P || sum w_j T_j schedules."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to instance JSON file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to solution JSON file.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write feasibility result JSON.")
    args = parser.parse_args()

    try:
        with open(args.instance_path) as f:
            instance = json.load(f)
        with open(args.solution_path) as f:
            solution = json.load(f)
        result = check_feasibility(instance, solution)
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as exc:
        result = _write_error_result(
            args.result_path,
            f"checker could not parse or validate solution: {type(exc).__name__}: {exc}",
        )

    print(f"Feasibility: {result['feasible']}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
