"""Check full job/operation coverage and recompute objective independently."""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path


def audit(instance, solution):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Schedule times and objective must be numbers")
        if not math.isfinite(value):
            raise ValueError("Nonfinite schedule value")
        return value

    reported_jobs = solution["schedule"]
    ids = [job["job_id"] for job in reported_jobs]
    expected_ids = {job["job_id"] for job in instance["jobs"]}
    require(len(ids) == len(set(ids)), "Duplicate jobs in solution")
    require(set(ids) == expected_ids, "Job coverage differs from instance")
    by_id = {job["job_id"]: job for job in reported_jobs}
    machine_intervals = {}
    objective, operation_count = 0, 0
    for job in instance["jobs"]:
        output = by_id[job["job_id"]]
        require(
            len(output["operations"]) == len(job["operations"]),
            f"Job {job['job_id']}: incorrect operation count",
        )
        previous_end = job["release_date"]
        for index, (operation, actual) in enumerate(
            zip(job["operations"], output["operations"], strict=True)
        ):
            start = number(actual["start_time"])
            end = start + operation["processing_time"]
            require(
                actual["machine"] == operation["machine"], "Machine assignment differs"
            )
            require(
                start >= 0 and start + 1e-5 >= previous_end,
                "Release/precedence violation",
            )
            machine_intervals.setdefault(operation["machine"], []).append((start, end))
            previous_end = end
            operation_count += 1
        late = max(0, previous_end - job["due_date"])
        require(
            abs(number(output["completion_time"]) - previous_end) <= 1e-5,
            "Completion time differs from actual operation end",
        )
        require(
            abs(number(output["tardiness"]) - late) <= 1e-5,
            "Tardiness differs from actual lateness",
        )
        objective += job["weight"] * late
    for machine, intervals in machine_intervals.items():
        intervals.sort()
        for previous, current in pairwise(intervals):
            require(previous[1] <= current[0] + 1e-5, f"Machine {machine}: overlap")
    require(
        abs(number(solution["objective_value"]) - objective) <= 1e-5,
        "Reported objective differs from recomputed weighted tardiness",
    )
    return {
        "feasible": not errors,
        "recomputed_objective": objective,
        "jobs_checked": len(expected_ids),
        "operations_checked": operation_count,
        "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance_path", required=True, type=Path)
    parser.add_argument("--solution_path", required=True, type=Path)
    parser.add_argument("--result_path", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(
            json.loads(args.instance_path.read_text()),
            json.loads(args.solution_path.read_text()),
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result = {"feasible": False, "errors": [str(error)]}
    args.result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
