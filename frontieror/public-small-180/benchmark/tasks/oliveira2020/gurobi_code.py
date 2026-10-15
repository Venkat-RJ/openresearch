#!/usr/bin/env python3
"""Solve P||sum(w_j T_j) and emit a complete schedule certificate.

The paper presents an arc-time-indexed formulation. The released solver uses
an equivalent compact disjunctive MILP because the literal arc expansion is
unnecessarily large for the public instances. Both formulations have the same
schedule semantics checked by ``feasibility_check.py``.
"""

import argparse
import json
import os as _os
import sys as _sys

import gurobipy as gp
from gurobipy import GRB
import time


_GUROBI_CODE_START_TIME = time.time()
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(
    _os.path.join(_repo, "scripts", "utils")
):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, "scripts", "utils")):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):
        del log_path


def compute_time_horizon(jobs, machine_count):
    total = sum(job["processing_time"] for job in jobs)
    largest = max(job["processing_time"] for job in jobs)
    return (total - largest) // machine_count + largest


def greedy_start(jobs, machine_count, horizon):
    """Return a deterministic feasible MIP start when list scheduling fits."""
    ordered = sorted(
        jobs,
        key=lambda job: (
            job["due_date"],
            job["processing_time"] / max(job["weight"], 1),
            job["id"],
        ),
    )
    machine_jobs = [[] for _ in range(machine_count)]
    loads = [0] * machine_count
    for job in ordered:
        machine = min(range(machine_count), key=lambda k: (loads[k], k))
        machine_jobs[machine].append(job["id"])
        loads[machine] += job["processing_time"]
    if max(loads, default=0) > horizon:
        return None
    return machine_jobs


def solve(instance_path, solution_path, time_limit):
    with open(instance_path) as handle:
        data = json.load(handle)

    jobs = sorted(data["jobs"], key=lambda job: job["id"])
    job_ids = [job["id"] for job in jobs]
    p = {job["id"]: int(job["processing_time"]) for job in jobs}
    w = {job["id"]: int(job["weight"]) for job in jobs}
    due = {job["id"]: int(job["due_date"]) for job in jobs}
    machine_count = int(data["parameters"]["m"])
    horizon = int(data.get("time_horizon", compute_time_horizon(jobs, machine_count)))

    model = gp.Model("parallel_machine_weighted_tardiness")
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 1
    model.Params.MIPFocus = 1

    assigned = model.addVars(job_ids, range(machine_count), vtype=GRB.BINARY, name="a")
    start = {
        j: model.addVar(lb=0.0, ub=horizon - p[j], vtype=GRB.CONTINUOUS, name=f"S_{j}")
        for j in job_ids
    }
    tardiness = {
        j: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Tard_{j}")
        for j in job_ids
    }

    before = {}
    for pos, i in enumerate(job_ids):
        for j in job_ids[pos + 1:]:
            for k in range(machine_count):
                before[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{k}")
                before[j, i, k] = model.addVar(vtype=GRB.BINARY, name=f"b_{j}_{i}_{k}")

    for j in job_ids:
        model.addConstr(assigned.sum(j, "*") == 1, name=f"assign_{j}")
        model.addConstr(tardiness[j] >= start[j] + p[j] - due[j], name=f"tardy_{j}")

    big_m = float(horizon)
    for pos, i in enumerate(job_ids):
        for j in job_ids[pos + 1:]:
            for k in range(machine_count):
                bij = before[i, j, k]
                bji = before[j, i, k]
                model.addConstr(bij <= assigned[i, k])
                model.addConstr(bij <= assigned[j, k])
                model.addConstr(bji <= assigned[i, k])
                model.addConstr(bji <= assigned[j, k])
                model.addConstr(
                    bij + bji >= assigned[i, k] + assigned[j, k] - 1
                )
                model.addConstr(bij + bji <= 1)
                model.addConstr(start[j] >= start[i] + p[i] - big_m * (1 - bij))
                model.addConstr(start[i] >= start[j] + p[j] - big_m * (1 - bji))

    model.setObjective(
        gp.quicksum(w[j] * tardiness[j] for j in job_ids),
        GRB.MINIMIZE,
    )

    initial = greedy_start(jobs, machine_count, horizon)
    if initial is not None:
        initial_machine = {}
        initial_start = {}
        for k, sequence in enumerate(initial):
            clock = 0
            for j in sequence:
                initial_machine[j] = k
                initial_start[j] = clock
                clock += p[j]
        for j in job_ids:
            start[j].Start = initial_start[j]
            tardiness[j].Start = max(0, initial_start[j] + p[j] - due[j])
            for k in range(machine_count):
                assigned[j, k].Start = int(initial_machine[j] == k)
        for (i, j, k), variable in before.items():
            variable.Start = int(
                initial_machine[i] == k
                and initial_machine[j] == k
                and initial_start[i] < initial_start[j]
            )

    model.optimize()

    result = {
        "instance_id": data.get("instance_id", ""),
        "solver": "Gurobi_compact_disjunctive_MILP",
        "status": int(model.Status),
        "solver_status": {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "time_limit",
            GRB.INFEASIBLE: "infeasible",
        }.get(model.Status, "other"),
        "n": len(job_ids),
        "m": machine_count,
        "T": horizon,
        "objective_value": None,
        "schedule": [],
    }

    if model.SolCount > 0:
        machine_of = {
            j: max(range(machine_count), key=lambda k: assigned[j, k].X)
            for j in job_ids
        }
        machine_sequences = {k: [] for k in range(machine_count)}
        for j in job_ids:
            machine_sequences[machine_of[j]].append(j)
        for k, sequence in machine_sequences.items():
            sequence.sort(key=lambda j: (start[j].X, j))
            previous = 0
            for j in sequence:
                start_value = round(start[j].X)
                completion = start_value + p[j]
                result["schedule"].append({
                    "machine_id": k,
                    "prev_job": previous,
                    "job": j,
                    "start_time": start_value,
                    "completion_time": completion,
                    "weighted_tardiness": w[j] * max(0, completion - due[j]),
                })
                previous = j
        result["schedule"].sort(
            key=lambda row: (row["start_time"], row["machine_id"], row["job"])
        )
        result["objective_value"] = sum(
            row["weighted_tardiness"] for row in result["schedule"]
        )
        result["best_bound"] = float(model.ObjBound)
        result["optimality_gap"] = float(model.MIPGap)

    with open(solution_path, "w") as handle:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve P||sum(w_j T_j).")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=3600)
    parser.add_argument(
        "--no-time-limit", action="store_true",
        help="Run Gurobi without a solver time limit.",
    )
    parser.add_argument("--log_path", default=None)
    arguments = parser.parse_args()
    install_gurobi_logger(arguments.log_path)
    solve(
        arguments.instance_path,
        arguments.solution_path,
        None if arguments.no_time_limit else arguments.time_limit,
    )
