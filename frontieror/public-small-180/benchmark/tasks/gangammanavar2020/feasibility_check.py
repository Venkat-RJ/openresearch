#!/usr/bin/env python3
"""
Feasibility checker for the Two-Stage Stochastic Linear Program (2-SLP)
for the Multilocation Transshipment Problem with Random Cost Coefficients.

Based on: Gangammanavar, Liu, and Sen (2020)

Constraints (numbered top to bottom from the math model formulation):
  Constraint 1: x in X := {x | Ax <= b}  =>  s_i >= 0 for all i
  Constraint 2: D y = xi(omega) - C x     =>  per-scenario equality constraints:
      (2a) f_i + sum_{j!=i} t_{ij} + e_i = s_i        (stock allocation)
      (2b) f_i + sum_{j!=i} t_{ji} + r_i = d_i(omega)  (demand satisfaction)
      (2c) e_i + q_i = s_i                              (inventory-replenishment)
      (2d) sum_i(r_i + q_i) = sum_i d_i(omega)          (system balance)
  Constraint 3: y >= 0  =>  e_i, f_i, q_i, r_i, t_{ij} >= 0
  Constraint 4: Objective consistency. The reported objective_value must lie
      inside an envelope derived from the (e, r, t) decision variables
      reported in the solution. Holding/penalty costs are deterministic and
      computed exactly; transshipment costs on deterministic links use
      c_mean; transshipment costs on random links contribute in
      [0, c_mean + K*c_std] (lower clipped at 0 to mirror the Gurobi
      sampler's max(., 0) clipping; upper widened by K=10 sigmas to cover
      any plausible cost realization). Reporting an objective outside this
      envelope is a Tier C exploit (e.g. obj=0 or obj=sys.float_info.max
      with a Gurobi-feasible route).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 's', 'scenarios')
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
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []

    N = instance["N"]
    second = instance["second_stage"]
    demand_info = second["demand_distribution"]
    scenarios_data = demand_info["scenarios"]
    num_scenarios = demand_info["num_scenarios"]

    s = solution.get("s")
    if s is None:
        violations.append("No first-stage solution (s) provided")
        return violations, violation_magnitudes

    # ------------------------------------------------------------------
    # Constraint 1: s_i >= 0 for all i  (first-stage feasibility)
    # This is a >= constraint: LHS = s_i, RHS = 0
    # violation_amount = max(RHS - LHS, 0) = max(-s_i, 0)
    # ------------------------------------------------------------------
    for i in range(N):
        lhs = s[i]
        rhs = 0.0
        violation_amount = max(rhs - lhs, 0.0)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint 1 violated: s[{i}] = {lhs:.6f} < 0"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # If solution has no scenario-level data, we can only check constraint 1
    scenario_solutions = solution.get("scenarios")
    if scenario_solutions is None:
        # Build result from what we checked
        violated_indices = sorted(set(v["constraint"] for v in violation_magnitudes))
        feasible = len(violated_indices) == 0
        return {
            "feasible": feasible,
            "violated_constraints": violated_indices,
            "violations": violations,
            "violation_magnitudes": violation_magnitudes,
        }

    # ------------------------------------------------------------------
    # Constraint 2: D y = xi(omega) - C x  (second-stage equalities)
    # These are = constraints, so violation_amount = |LHS - RHS|
    # ------------------------------------------------------------------
    for sc_idx, sc_sol in enumerate(scenario_solutions):
        e = sc_sol["e"]
        f = sc_sol["f"]
        q = sc_sol["q"]
        r = sc_sol["r"]
        t = sc_sol["t"]
        d_demand = scenarios_data[sc_idx]

        # (2a) Stock allocation: f_i + sum_{j!=i} t_{ij} + e_i = s_i
        for i in range(N):
            lhs = f[i] + sum(t[i][j] for j in range(N) if j != i) + e[i]
            rhs = s[i]
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 2 violated (stock allocation): scenario {sc_idx}, "
                    f"retailer {i}: LHS={lhs:.6f} != RHS={rhs:.6f}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

        # (2b) Demand satisfaction: f_i + sum_{j!=i} t_{ji} + r_i = d_i
        for i in range(N):
            lhs = f[i] + sum(t[j][i] for j in range(N) if j != i) + r[i]
            rhs = d_demand[i]
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 2 violated (demand satisfaction): scenario {sc_idx}, "
                    f"retailer {i}: LHS={lhs:.6f} != RHS={rhs:.6f}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

        # (2c) Inventory-replenishment: e_i + q_i = s_i
        for i in range(N):
            lhs = e[i] + q[i]
            rhs = s[i]
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 2 violated (inventory-replenishment): scenario {sc_idx}, "
                    f"retailer {i}: LHS={lhs:.6f} != RHS={rhs:.6f}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

        # (2d) System balance: sum_i(r_i + q_i) = sum_i d_i
        lhs = sum(r[i] + q[i] for i in range(N))
        rhs = sum(d_demand)
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint 2 violated (system balance): scenario {sc_idx}: "
                f"LHS={lhs:.6f} != RHS={rhs:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 3: y >= 0  (non-negativity of second-stage variables)
    # This is a >= constraint: LHS = var, RHS = 0
    # violation_amount = max(-var, 0)
    # ------------------------------------------------------------------
    for sc_idx, sc_sol in enumerate(scenario_solutions):
        e = sc_sol["e"]
        f = sc_sol["f"]
        q = sc_sol["q"]
        r = sc_sol["r"]
        t = sc_sol["t"]

        var_map = {"e": e, "f": f, "q": q, "r": r}
        for var_name, var_vals in var_map.items():
            for i in range(N):
                lhs = var_vals[i]
                rhs = 0.0
                violation_amount = max(rhs - lhs, 0.0)
                if violation_amount > tol:
                    normalizer = max(abs(rhs), eps)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint 3 violated: scenario {sc_idx}, "
                        f"{var_name}[{i}] = {lhs:.6f} < 0"
                    )
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": violation_amount,
                        "normalizer": normalizer,
                        "ratio": ratio,
                    })

        for i in range(N):
            for j in range(N):
                if i != j:
                    lhs = t[i][j]
                    rhs = 0.0
                    violation_amount = max(rhs - lhs, 0.0)
                    if violation_amount > tol:
                        normalizer = max(abs(rhs), eps)
                        ratio = violation_amount / normalizer
                        violations.append(
                            f"Constraint 3 violated: scenario {sc_idx}, "
                            f"t[{i}][{j}] = {lhs:.6f} < 0"
                        )
                        violation_magnitudes.append({
                            "constraint": 3,
                            "lhs": lhs,
                            "rhs": rhs,
                            "raw_excess": violation_amount,
                            "normalizer": normalizer,
                            "ratio": ratio,
                        })

    # ------------------------------------------------------------------
    # Constraint 4: exact objective consistency.
    #
    # Objective formula (paper Eq. C.1):
    #   obj = sum_omega prob[omega] * (
    #           sum_i h_i * e_i(omega)
    #         + sum_i p_i * r_i(omega)
    #         + sum_{i!=j} c_{ij}(omega) * t_{ij}(omega)
    #         )
    #
    # gurobi_code.py generates the random-link scenario costs from the stored
    # instance seed with NumPy RandomState.  Replaying that deterministic
    # generation makes every coefficient available to the checker, so the full
    # objective is exactly determined by the serialized scenario decisions.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and not math.isnan(reported):
            h_costs = second["holding_costs"]
            p_costs = second["penalty_costs"]
            c_mean = second["transshipment_costs_mean"]
            c_std = second["transshipment_costs_std"]
            probabilities = demand_info["probabilities"]
            import numpy as np
            random_links = [
                (int(link[0]), int(link[1]))
                for link in second.get("random_cost_links", [])
            ]
            rng = np.random.RandomState(instance.get("seed", 42))
            scenario_costs = []
            for _ in range(num_scenarios):
                costs = [row[:] for row in c_mean]
                for i, j in random_links:
                    costs[i][j] = max(
                        float(rng.normal(c_mean[i][j], c_std[i][j])),
                        0.0,
                    )
                scenario_costs.append(costs)

            true_obj = 0.0
            for sc_idx, sc_sol in enumerate(scenario_solutions):
                e = sc_sol["e"]
                r = sc_sol["r"]
                t = sc_sol["t"]
                prob = probabilities[sc_idx]

                h_term = sum(h_costs[i] * e[i] for i in range(N))
                p_term = sum(p_costs[i] * r[i] for i in range(N))

                transshipment_term = 0.0
                for i in range(N):
                    row_t = t[i]
                    row_cost = scenario_costs[sc_idx][i]
                    for j in range(N):
                        if i == j:
                            continue
                        transshipment_term += row_cost[j] * row_t[j]

                true_obj += prob * (h_term + p_term + transshipment_term)

            obj_tol = max(1e-3, 1e-8 * max(abs(true_obj), 1.0))
            difference = abs(reported - true_obj)
            if difference > obj_tol:
                normalizer = max(abs(true_obj), eps)
                violations.append(
                    f"Constraint 4 violated: reported objective_value="
                    f"{reported:.12g} differs from exactly recomputed value="
                    f"{true_obj:.12g} (difference={difference:.6g}, "
                    f"tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": difference,
                    "normalizer": normalizer,
                    "ratio": difference / normalizer,
                })

    violated_indices = sorted(set(v["constraint"] for v in violation_magnitudes))
    feasible = len(violated_indices) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for 2-SLP Transshipment Problem "
                    "(Gangammanavar, Liu, Sen 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if result["violated_constraints"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
