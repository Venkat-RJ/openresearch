#!/usr/bin/env python3
"""Solver-free complete-witness checker for Riley et al. (2025)."""

import argparse
from collections import defaultdict
import json
import math
import re
import sys

TOL = 1e-5
VARIABLE = re.compile(r"^([A-Za-z_]+)\[([^]]+)\]$")


def rejection(kind, messages):
    return {"feasible": False, "violated_constraints": [kind],
            "violations": messages[:30], "violation_magnitudes": []}


def check_feasibility(data, solution):
    if not isinstance(solution, dict):
        return rejection("solution_schema", ["solution must be a JSON object"])
    missing = [key for key in ("objective_value", "y", "model_variables") if key not in solution]
    if missing:
        return rejection("solution_schema", ["missing required field(s): " + ", ".join(missing)])
    witness = solution["model_variables"]
    if not isinstance(witness, dict) or not witness:
        return rejection("solution_schema", ["model_variables must be a nonempty sparse object"])

    I_n, J_n = int(data["num_demand_sites"]), int(data["num_production_sites"])
    S_n, T_n = int(data["num_scenarios"]), int(data["num_time_periods"])
    M_n, K_n = int(data["num_module_sizes"]), int(data["units_per_module_size"])
    shapes = {
        "y": (M_n, K_n), "z": (M_n, S_n), "x": (I_n, J_n, T_n, S_n),
        "q": (I_n, T_n, S_n), "v": (J_n, M_n, T_n, S_n),
        "v_depot": (M_n, T_n, S_n), "w": (J_n, J_n, M_n, T_n, S_n),
        "w_fd": (J_n, M_n, T_n, S_n), "w_td": (J_n, M_n, T_n, S_n),
    }
    values = {name: {} for name in shapes}
    errors = []
    for name, raw in witness.items():
        match = VARIABLE.match(str(name))
        if not match or match.group(1) not in shapes:
            errors.append(f"unknown model variable {name!r}")
            continue
        family = match.group(1)
        try:
            key = tuple(int(part) for part in match.group(2).split(","))
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"invalid model variable {name!r}")
            continue
        if len(key) != len(shapes[family]) or any(i < 0 or i >= bound for i, bound in zip(key, shapes[family])):
            errors.append(f"out-of-range model variable {name!r}")
            continue
        if family == "w" and key[0] == key[1]:
            errors.append(f"invalid self-relocation variable {name!r}")
            continue
        if not math.isfinite(value) or value < -TOL:
            errors.append(f"invalid value for {name}: {raw}")
            continue
        if family in ("x", "q"):
            if value > 1 + TOL:
                errors.append(f"upper bound violated for {name}: {value}")
        elif abs(value - round(value)) > TOL or (family == "y" and value > 1 + TOL):
            errors.append(f"integrality/domain violated for {name}: {value}")
        values[family][key] = value
    if errors:
        return rejection("solution_schema", errors)

    get = lambda family, *key: values[family].get(tuple(key), 0.0)
    public_y = solution["y"]
    expected_keys = {f"y_{m}_{k}" for m in range(M_n) for k in range(K_n)}
    if not isinstance(public_y, dict) or set(public_y) != expected_keys:
        return rejection("solution_schema", ["y must cover every module-size/unit pair"])
    violations = []
    y = {}
    for m in range(M_n):
        for k in range(K_n):
            raw = float(public_y[f"y_{m}_{k}"])
            if abs(raw - round(raw)) > TOL or round(raw) not in (0, 1):
                violations.append(f"public y_{m}_{k} must be binary")
            y[m, k] = round(raw)
            if abs(get("y", m, k) - raw) > TOL:
                violations.append(f"public y_{m}_{k} disagrees with complete witness")
            if k and y[m, k] > y[m, k - 1]:
                violations.append(f"symmetry constraint violated at y[{m},{k}]")

    for m in range(M_n):
        count = sum(y[m, k] for k in range(K_n))
        for s in range(S_n):
            if abs(get("z", m, s) - count) > TOL:
                violations.append(f"z definition violated at ({m},{s})")

    demand_sum = defaultdict(float)
    production = defaultdict(float)
    objective = 0.0
    scenarios = data["scenarios"]
    probabilities = [float(scenario["probability"]) for scenario in scenarios]
    demand = [scenario["demands"] for scenario in scenarios]
    for (i, j, t, s), value in values["x"].items():
        demand_sum[i, t, s] += value
        amount = float(demand[s][i][t]) * value
        production[j, t, s] += amount
        objective += probabilities[s] * float(data["production_costs"][i][j]) * amount
    for (i, t, s), value in values["q"].items():
        demand_sum[i, t, s] += value
        objective += probabilities[s] * float(data["penalty_costs"][i][t]) * float(demand[s][i][t]) * value
    for i in range(I_n):
        for t in range(T_n):
            for s in range(S_n):
                if abs(demand_sum[i, t, s] - 1.0) > TOL:
                    violations.append(f"demand balance violated at ({i},{t},{s})")
                    if len(violations) >= 30:
                        return rejection("constraints", violations)

    depot_net, site_net = defaultdict(float), defaultdict(float)
    for (j, m, t, s), value in values["w_fd"].items():
        depot_net[m, t, s] -= value
        site_net[j, m, t, s] += value
    for (j, m, t, s), value in values["w_td"].items():
        depot_net[m, t, s] += value
        site_net[j, m, t, s] -= value
    relocation = data["relocation_costs"]
    for (j, jp, m, t, s), value in values["w"].items():
        site_net[j, m, t, s] -= value
        site_net[jp, m, t, s] += value
        objective += probabilities[s] * float(relocation[j][jp][m]) * value
    for m in range(M_n):
        for s in range(S_n):
            previous = get("z", m, s)
            for t in range(T_n):
                current = get("v_depot", m, t, s)
                if abs(current - previous - depot_net[m, t, s]) > TOL:
                    violations.append(f"depot balance violated at ({m},{t},{s})")
                previous = current
    for j in range(J_n):
        for m in range(M_n):
            for s in range(S_n):
                previous = 0.0
                for t in range(T_n):
                    current = get("v", j, m, t, s)
                    if abs(current - previous - site_net[j, m, t, s]) > TOL:
                        violations.append(f"site balance violated at ({j},{m},{t},{s})")
                    previous = current

    capacities = [float(entry["capacity"]) for entry in data["module_sizes"]]
    for j in range(J_n):
        for t in range(T_n):
            for s in range(S_n):
                used = production[j, t, s]
                available = sum(get("v", j, m, t, s) * capacities[m] for m in range(M_n))
                tolerance = TOL + 1e-7 * max(1.0, abs(used), abs(available))
                if used > available + tolerance:
                    violations.append(f"module capacity violated at ({j},{t},{s})")
                if used > float(data["u_max"]) + tolerance:
                    violations.append(f"global site capacity violated at ({j},{t},{s})")

    capital = {(int(x["module_size_id"]), int(x["unit_id"])): float(x["capital_cost"]) for x in data["module_units"]}
    objective += sum(capital[m, k] * y[m, k] for m in range(M_n) for k in range(K_n))
    reported = solution["objective_value"]
    if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(float(reported)):
        violations.append("objective_value must be finite")
    else:
        difference = abs(float(reported) - objective)
        tolerance = max(1e-3, 1e-6 * max(1.0, abs(objective)))
        if difference > tolerance:
            violations.append(f"objective mismatch: reported={reported}, recomputed={objective}, difference={difference}")
    return {"feasible": not violations, "violated_constraints": [] if not violations else [12, 13, 14, 15, 16, 17, 18],
            "violations": violations[:30], "violation_magnitudes": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()
    with open(args.instance_path) as handle:
        instance = json.load(handle)
    with open(args.solution_path) as handle:
        solution = json.load(handle)
    result = check_feasibility(instance, solution)
    with open(args.result_path, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        path = None
        for index, arg in enumerate(sys.argv):
            if arg == "--result_path" and index + 1 < len(sys.argv):
                path = sys.argv[index + 1]
            elif arg.startswith("--result_path="):
                path = arg.split("=", 1)[1]
        if path is None:
            raise
        with open(path, "w") as handle:
            json.dump(rejection("malformed_solution", [f"{type(error).__name__}: {error}"]), handle, indent=2)
