#!/usr/bin/env python3
"""Solve the released finite-completion OCRSP set-covering model."""

import argparse
import json
import os as _os
import sys as _sys
import time

import gurobipy as gp
from gurobipy import GRB

from release_contract import (
    RELEASED_VARIANT,
    ReleasedInstance,
    completion_options,
)


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


def solve(data, time_limit):
    inst = ReleasedInstance(data)
    libraries = {
        duty_id: completion_options(inst, duty)
        for duty_id, duty in sorted(inst.duties.items())
    }

    model = gp.Model("released_OCRSP")
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 1

    chosen = {}
    for duty_id, options in libraries.items():
        for index, option in enumerate(options):
            chosen[duty_id, index] = model.addVar(
                vtype=GRB.BINARY,
                obj=option["cost"],
                name=f"x_{duty_id}_{index}",
            )
    canceled = {
        task_id: model.addVar(
            vtype=GRB.BINARY,
            obj=(
                inst.params["cost_cancel_A_to_A"]
                if task["task_type"] == "A_to_A"
                else inst.params["cost_cancel_A_to_B"]
            ),
            name=f"y_{task_id}",
        )
        for task_id, task in inst.tasks.items()
    }

    for duty_id, options in libraries.items():
        model.addConstr(
            gp.quicksum(chosen[duty_id, index] for index in range(len(options))) == 1,
            name=f"assign_{duty_id}",
        )
    for task_id in inst.tasks:
        model.addConstr(
            gp.quicksum(
                chosen[duty_id, index]
                for duty_id, options in libraries.items()
                for index, option in enumerate(options)
                if task_id in option["covered_tasks"]
            ) + canceled[task_id] >= 1,
            name=f"cover_{task_id}",
        )

    # The first option is always the canonical empty/fallback completion.
    for duty_id, options in libraries.items():
        for index in range(len(options)):
            chosen[duty_id, index].Start = int(index == 0)
    for task_id in canceled:
        canceled[task_id].Start = 1

    model.ModelSense = GRB.MINIMIZE
    model.optimize()

    result = {
        "instance_id": data.get("instance_id", "unknown"),
        "released_problem_variant": RELEASED_VARIANT,
        "solver": "Gurobi_finite_completion_set_covering",
        "status": "infeasible",
        "objective_value": None,
        "canceled_tasks": [],
        "duty_assignments": {},
    }
    if model.SolCount == 0:
        return result

    result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
    result["best_bound"] = float(model.ObjBound)
    result["optimality_gap"] = float(model.MIPGap)
    result["canceled_tasks"] = sorted(
        task_id for task_id, variable in canceled.items() if variable.X > 0.5
    )
    for duty_id, options in libraries.items():
        index = max(range(len(options)), key=lambda k: chosen[duty_id, k].X)
        option = dict(options[index])
        option["completion_index"] = index
        result["duty_assignments"][duty_id] = option

    cancellation_cost = sum(
        inst.params["cost_cancel_A_to_A"]
        if inst.tasks[task_id]["task_type"] == "A_to_A"
        else inst.params["cost_cancel_A_to_B"]
        for task_id in result["canceled_tasks"]
    )
    result["objective_value"] = cancellation_cost + sum(
        assignment["cost"] for assignment in result["duty_assignments"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Solve the released OCRSP model.")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", default=None)
    arguments = parser.parse_args()
    install_gurobi_logger(arguments.log_path)
    with open(arguments.instance_path) as handle:
        data = json.load(handle)
    result = solve(data, arguments.time_limit)
    with open(arguments.solution_path, "w") as handle:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, handle, indent=2)
    print(
        f"status={result['status']} objective={result['objective_value']} "
        f"assignments={len(result['duty_assignments'])}"
    )


if __name__ == "__main__":
    main()
