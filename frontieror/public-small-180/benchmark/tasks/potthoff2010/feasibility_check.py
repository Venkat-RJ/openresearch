#!/usr/bin/env python3
"""Fail-closed checker for the released OCRSP completion certificate."""

import argparse
import json
import math

from release_contract import (
    RELEASED_VARIANT,
    ReleasedInstance,
    completion_cost,
    validate_completion,
)


def rejection(category, messages):
    return {
        "feasible": False,
        "violated_constraints": [category],
        "violations": messages,
        "violation_magnitudes": [],
    }


def check_feasibility(data, solution):
    if not isinstance(solution, dict):
        return rejection("solution_schema", ["solution must be a JSON object"])
    required = {
        "released_problem_variant",
        "objective_value",
        "canceled_tasks",
        "duty_assignments",
    }
    missing = sorted(required - set(solution))
    if missing:
        return rejection(
            "solution_schema", [f"missing required solution fields: {missing}"]
        )
    if solution["released_problem_variant"] != RELEASED_VARIANT:
        return rejection(
            "solution_schema",
            [f"released_problem_variant must be {RELEASED_VARIANT}"],
        )
    objective = solution["objective_value"]
    if (
        isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
    ):
        return rejection("solution_schema", ["objective_value must be finite"])
    if not isinstance(solution["canceled_tasks"], list) or any(
        not isinstance(task_id, str) for task_id in solution["canceled_tasks"]
    ):
        return rejection(
            "solution_schema", ["canceled_tasks must be a list of identifiers"]
        )
    if not isinstance(solution["duty_assignments"], dict):
        return rejection("solution_schema", ["duty_assignments must be an object"])

    try:
        inst = ReleasedInstance(data)
    except Exception as error:
        return rejection("instance_contract", [str(error)])

    violations = []
    violated = set()
    canceled_list = solution["canceled_tasks"]
    canceled = set(canceled_list)
    if len(canceled) != len(canceled_list):
        violations.append("canceled_tasks contains duplicates")
        violated.add(4)
    unknown_canceled = sorted(canceled - set(inst.tasks))
    if unknown_canceled:
        violations.append(f"unknown/non-reschedulable canceled tasks: {unknown_canceled}")
        violated.add(4)

    assignments = solution["duty_assignments"]
    expected_duties = set(inst.duties)
    actual_duties = set(assignments)
    if actual_duties != expected_duties:
        missing_duties = sorted(expected_duties - actual_duties)
        extra_duties = sorted(actual_duties - expected_duties)
        violations.append(
            f"duty assignment keys are incomplete: missing={missing_duties}, "
            f"unexpected={extra_duties}"
        )
        violated.add(3)

    coverage = {task_id: 0 for task_id in inst.tasks}
    completion_cost_sum = 0.0
    required_assignment_fields = {
        "completion_index",
        "cost",
        "covered_tasks",
        "uses_taxi",
        "release_at_relief_point",
        "return_station",
        "return_time_minutes",
    }
    for duty_id in sorted(actual_duties & expected_duties):
        assignment = assignments[duty_id]
        if not isinstance(assignment, dict):
            violations.append(f"assignment for {duty_id} must be an object")
            violated.add("solution_schema")
            continue
        missing_fields = sorted(required_assignment_fields - set(assignment))
        if missing_fields:
            violations.append(
                f"assignment for {duty_id} is missing fields {missing_fields}"
            )
            violated.add("solution_schema")
            continue
        if (
            isinstance(assignment["completion_index"], bool)
            or not isinstance(assignment["completion_index"], int)
            or assignment["completion_index"] < 0
        ):
            violations.append(f"{duty_id}: completion_index must be nonnegative integer")
            violated.add("solution_schema")
        declared_cost = assignment["cost"]
        if (
            isinstance(declared_cost, bool)
            or not isinstance(declared_cost, (int, float))
            or not math.isfinite(float(declared_cost))
        ):
            violations.append(f"{duty_id}: cost must be a finite number")
            violated.add("solution_schema")
            continue

        derived, error = validate_completion(
            inst,
            inst.duties[duty_id],
            assignment["covered_tasks"],
            assignment["uses_taxi"],
            assignment["release_at_relief_point"],
        )
        if error is not None:
            violations.append(f"{duty_id}: {error}")
            violated.add("completion_feasibility")
            continue
        if assignment["return_station"] != derived["return_station"]:
            violations.append(
                f"{duty_id}: return_station={assignment['return_station']} does not "
                f"match derived {derived['return_station']}"
            )
            violated.add("completion_feasibility")
        if abs(float(assignment["return_time_minutes"]) - derived["return_time_minutes"]) > 1e-6:
            violations.append(
                f"{duty_id}: return_time_minutes does not match derived return time"
            )
            violated.add("completion_feasibility")

        expected_cost = completion_cost(
            inst,
            inst.duties[duty_id],
            assignment["covered_tasks"],
            assignment["uses_taxi"],
            assignment["release_at_relief_point"],
        )
        if abs(float(declared_cost) - expected_cost) > 1e-6:
            violations.append(
                f"{duty_id}: cost={declared_cost} does not match derived {expected_cost}"
            )
            violated.add(5)
        completion_cost_sum += expected_cost
        for task_id in assignment["covered_tasks"]:
            if task_id in coverage:
                coverage[task_id] += 1

    uncovered = sorted(
        task_id
        for task_id, count in coverage.items()
        if count == 0 and task_id not in canceled
    )
    if uncovered:
        violations.append(f"tasks neither covered nor canceled: {uncovered}")
        violated.add(2)

    cancellation_cost = 0.0
    for task_id in canceled & set(inst.tasks):
        task = inst.tasks[task_id]
        cancellation_cost += (
            inst.params["cost_cancel_A_to_A"]
            if task["task_type"] == "A_to_A"
            else inst.params["cost_cancel_A_to_B"]
        )
    expected_objective = completion_cost_sum + cancellation_cost
    difference = abs(float(objective) - expected_objective)
    tolerance = max(1e-6, 1e-9 * abs(expected_objective))
    if difference > tolerance:
        violations.append(
            f"objective_value={objective} does not match exact completion plus "
            f"cancellation cost {expected_objective}"
        )
        violated.add(5)

    return {
        "feasible": not violations,
        "violated_constraints": sorted(violated, key=str),
        "violations": violations,
        "violation_magnitudes": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Check released OCRSP certificate.")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--result_path", required=True)
    arguments = parser.parse_args()
    try:
        with open(arguments.instance_path) as handle:
            data = json.load(handle)
        with open(arguments.solution_path) as handle:
            solution = json.load(handle)
        result = check_feasibility(data, solution)
    except Exception as error:
        result = rejection(
            "malformed_solution",
            [f"checker rejected malformed input: {type(error).__name__}: {error}"],
        )
    with open(arguments.result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Feasible: {result['feasible']}")
    for message in result["violations"]:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
