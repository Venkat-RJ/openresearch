#!/usr/bin/env python3
"""Solver-free checker for Bertsimas et al. (2024) MCFND."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solverless_witness import require_linked_variables, validate_complete_witness


def rejection(kind, messages):
    return {"feasible": False, "violated_constraints": [kind],
            "violations": messages, "violation_magnitudes": []}


def check_feasibility(instance, solution, instance_path):
    errors = []
    if not isinstance(solution, dict):
        return rejection("solution_schema", ["solution must be a JSON object"])
    for field in ("objective_value", "z", "model_variables"):
        if field not in solution or solution[field] is None:
            errors.append(f"missing required solution field: {field}")
    if errors:
        return rejection("solution_schema", errors)

    edge_count = int(instance["num_edges"])
    z_raw = solution["z"]
    if not isinstance(z_raw, dict) or set(z_raw) != {str(edge) for edge in range(edge_count)}:
        errors.append("z must contain every edge index exactly once")
        z_raw = z_raw if isinstance(z_raw, dict) else {}
    z = []
    for edge in range(edge_count):
        raw = z_raw.get(str(edge), 0.0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors.append(f"z[{edge}] must be binary")
            value = 0.0
        else:
            value = float(raw)
            if not math.isfinite(value) or abs(value - round(value)) > 1e-5 or round(value) not in (0, 1):
                errors.append(f"z[{edge}] must be binary")
        z.append(value)
    if errors:
        return rejection("solution_schema", errors[:20])

    violations = []
    if sum(z) > float(instance["c_0"]) + 1e-5:
        violations.append(f"cardinality violated: sum(z)={sum(z)} > c_0={instance['c_0']}")
    objective, witness_errors, diagnostics = validate_complete_witness(
        instance_path, solution, Path(__file__).resolve().parent
    )
    witness_errors.extend(require_linked_variables(
        solution, {f"z[{edge}]": z[edge] for edge in range(edge_count)}
    ))
    violations.extend(witness_errors)

    reported = solution["objective_value"]
    if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(float(reported)):
        violations.append("objective_value must be a finite number")
    elif objective is not None:
        difference = abs(float(reported) - objective)
        tolerance = max(1e-3, 1e-7 * max(abs(float(reported)), abs(objective)))
        if difference > tolerance:
            violations.append(
                f"objective_value={reported} differs from complete-witness objective="
                f"{objective:.12g} by {difference:.6g} (tol={tolerance:.6g})"
            )
    return {
        "feasible": not violations,
        "violated_constraints": [] if not violations else [1, 2, 3, 4, 5],
        "violations": violations,
        "violation_magnitudes": [] if not violations else [diagnostics],
    }


def main():
    parser = argparse.ArgumentParser(description="Solver-free MCFND witness checker")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()
    with open(args.instance_path) as handle:
        instance = json.load(handle)
    with open(args.solution_path) as handle:
        solution = json.load(handle)
    result = check_feasibility(instance, solution, args.instance_path)
    with open(args.result_path, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        result_path = None
        for index, argument in enumerate(sys.argv):
            if argument == "--result_path" and index + 1 < len(sys.argv):
                result_path = sys.argv[index + 1]
            elif argument.startswith("--result_path="):
                result_path = argument.split("=", 1)[1]
        if result_path is None:
            raise
        with open(result_path, "w") as handle:
            json.dump(rejection("malformed_solution", [f"{type(error).__name__}: {str(error)[:500]}"]), handle, indent=2)
