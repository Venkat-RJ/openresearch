"""Deterministic earliest-start scheduling with a due-date tie break."""

import argparse
import json
from pathlib import Path


def solve(instance):
    jobs = instance["jobs"]
    next_operation = [0] * len(jobs)
    job_ready = [job["release_date"] for job in jobs]
    machine_ready = [0] * instance["num_machines"]
    starts = [[] for _ in jobs]

    for _ in range(sum(len(job["operations"]) for job in jobs)):
        candidates = []
        for index, job in enumerate(jobs):
            operation_index = next_operation[index]
            if operation_index == len(job["operations"]):
                continue
            operation = job["operations"][operation_index]
            earliest = max(job_ready[index], machine_ready[operation["machine"]])
            candidates.append(
                (earliest, job["due_date"], -job["weight"], job["job_id"], index)
            )
        start, _, _, _, index = min(candidates)
        operation = jobs[index]["operations"][next_operation[index]]
        end = start + operation["processing_time"]
        starts[index].append(start)
        next_operation[index] += 1
        job_ready[index] = end
        machine_ready[operation["machine"]] = end

    schedule = []
    objective = 0
    for job, job_starts, completion in zip(jobs, starts, job_ready, strict=True):
        tardiness = max(0, completion - job["due_date"])
        objective += job["weight"] * tardiness
        schedule.append(
            {
                "job_id": job["job_id"],
                "completion_time": completion,
                "tardiness": tardiness,
                "operations": [
                    {"machine": operation["machine"], "start_time": start}
                    for operation, start in zip(
                        job["operations"], job_starts, strict=True
                    )
                ],
            }
        )
    return {"objective_value": objective, "schedule": schedule}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance_path", required=True, type=Path)
    parser.add_argument("--solution_path", required=True, type=Path)
    args = parser.parse_args()
    result = solve(json.loads(args.instance_path.read_text()))
    args.solution_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"objective_value": result["objective_value"]}))
