"""Plain CP-SAT model for the benchmark's total weighted tardiness objective.

Intervals, precedence and NoOverlap follow the official OR-Tools job-shop guide:
https://developers.google.com/optimization/scheduling/job_shop
The guide minimizes makespan. This model instead minimizes weighted tardiness.
"""

import argparse
import json
from pathlib import Path

import ortools
from dispatch_baseline import solve as dispatch
from ortools.sat.python import cp_model


def solve(instance, time_limit, seed):
    jobs = instance["jobs"]
    for job in jobs:
        values = [job["weight"], job["release_date"], job["due_date"]]
        values.extend(operation["processing_time"] for operation in job["operations"])
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(
                "This pilot model requires nonnegative integer instance data"
            )
    horizon = max(job["release_date"] for job in jobs) + sum(
        operation["processing_time"] for job in jobs for operation in job["operations"]
    )
    model = cp_model.CpModel()
    intervals = [[] for _ in range(instance["num_machines"])]
    starts, ends, tardiness = {}, {}, {}
    hint = {job["job_id"]: job for job in dispatch(instance)["schedule"]}

    for job in jobs:
        job_id = job["job_id"]
        previous_end = None
        for index, operation in enumerate(job["operations"]):
            key = job_id, index
            start = model.new_int_var(0, horizon, f"start_{job_id}_{index}")
            end = model.new_int_var(0, horizon, f"end_{job_id}_{index}")
            interval = model.new_interval_var(
                start, operation["processing_time"], end, f"interval_{job_id}_{index}"
            )
            intervals[operation["machine"]].append(interval)
            if previous_end is None:
                model.add(start >= job["release_date"])
            else:
                model.add(start >= previous_end)
            hint_start = hint[job_id]["operations"][index]["start_time"]
            model.add_hint(start, hint_start)
            model.add_hint(end, hint_start + operation["processing_time"])
            starts[key], ends[key] = start, end
            previous_end = end
        late = model.new_int_var(0, horizon, f"tardiness_{job_id}")
        model.add_max_equality(late, [0, previous_end - job["due_date"]])
        model.add_hint(late, hint[job_id]["tardiness"])
        tardiness[job_id] = late

    for machine_intervals in intervals:
        model.add_no_overlap(machine_intervals)
    model.minimize(sum(job["weight"] * tardiness[job["job_id"]] for job in jobs))
    validation_error = model.validate()
    if validation_error:
        raise ValueError(validation_error)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No schedule: {solver.status_name(status)}")

    schedule, objective = [], 0
    for job in jobs:
        job_id = job["job_id"]
        completion = solver.value(ends[job_id, len(job["operations"]) - 1])
        late = max(0, completion - job["due_date"])
        objective += job["weight"] * late
        schedule.append(
            {
                "job_id": job_id,
                "completion_time": completion,
                "tardiness": late,
                "operations": [
                    {
                        "machine": operation["machine"],
                        "start_time": solver.value(starts[job_id, index]),
                    }
                    for index, operation in enumerate(job["operations"])
                ],
            }
        )
    if abs(objective - solver.objective_value) > 1e-5:
        raise RuntimeError("Recomputed and solver objectives disagree")
    metadata = {
        "ortools_version": ortools.__version__,
        "status": solver.status_name(status),
        "objective_value": objective,
        "best_objective_bound": solver.best_objective_bound,
        "solver_wall_time_seconds": solver.wall_time,
        "solver_time_limit_seconds": time_limit,
        "num_search_workers": 1,
        "random_seed": seed,
        "hint": "deterministic earliest-start scheduling with due-date tie break",
    }
    return {"objective_value": objective, "schedule": schedule}, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance_path", required=True, type=Path)
    parser.add_argument("--solution_path", required=True, type=Path)
    parser.add_argument("--metadata_path", required=True, type=Path)
    parser.add_argument("--time_limit", type=float, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error("--time_limit must be positive")
    result, metadata = solve(
        json.loads(args.instance_path.read_text()), args.time_limit, args.seed
    )
    args.solution_path.write_text(json.dumps(result, indent=2) + "\n")
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))
