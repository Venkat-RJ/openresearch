"""
Feasibility checker for the Capacitated Lot Sizing Problem with Setup Times (CLST).
Regular formulation (CL) from de Araujo et al. (2015).

Checks constraints (2)-(5) from the mathematical formulation, plus a
Tier C objective-consistency check (constraint 6) that recomputes the
reported objective_value from the solution variables.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('inventory', 'objective_value', 'production', 'setup')
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


def expand_param(raw, num_items, num_periods):
    """Expand scalar-per-item to per-item-per-period if needed."""
    if isinstance(raw[0], list):
        return raw
    return [[raw[i]] * num_periods for i in range(num_items)]


def check_feasibility(instance, solution):
    # 1e-5 was too tight for Gurobi LP numerical noise: when binary y=0 the
    # constraint x <= UB*y = 0 was tripped by tiny x values (~3e-5). LP
    # tolerance is typically 1e-6..1e-5, so accept up to 1e-4 absolute slack.
    tol = 1e-4
    eps = 1e-5

    nI = instance["num_items"]
    nT = instance["num_periods"]

    d = instance["demand"]
    st = expand_param(instance["setup_time"], nI, nT)
    vt = expand_param(instance["variable_time"], nI, nT)
    cap = instance["capacity"]

    # Compute sd[i][t] = sum of demand from period t to end
    sd = [[0] * nT for _ in range(nI)]
    for i in range(nI):
        sd[i][nT - 1] = d[i][nT - 1]
        for t in range(nT - 2, -1, -1):
            sd[i][t] = sd[i][t + 1] + d[i][t]

    # Extract solution variables
    x = solution["production"]  # x[i][t], 0-indexed
    y = solution["setup"]       # y[i][t], 0-indexed

    # inventory[i] has nT+1 entries: [s_{i0}, s_{i,end of period 0}, ..., s_{i,end of period nT-1}]
    inv = solution["inventory"]

    # Some solutions store initial_inventory separately
    if "initial_inventory" in solution:
        s0 = solution["initial_inventory"]
    else:
        s0 = [inv[i][0] for i in range(nI)]

    # Build full inventory array: s[i][0] = s_{i0}, s[i][t+1] = end-of-period t
    s = []
    for i in range(nI):
        if len(inv[i]) == nT + 1:
            s.append(inv[i])
        else:
            # If inventory has nT entries starting from end-of-period 0
            s.append([s0[i]] + inv[i])

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint (2): Demand balance
    #   s_{i,t-1} + x_{it} = d_{it} + s_{it}  for all i in I, t in T
    #
    # In 0-indexed code: s[i][t] + x[i][t] = d[i][t] + s[i][t+1]
    # where s[i][0] = s_{i0} and s[i][t+1] = end-of-period t inventory
    # =========================================================================
    for i in range(nI):
        for t in range(nT):
            lhs = s[i][t] + x[i][t]
            rhs = d[i][t] + s[i][t + 1]
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints.add(2)
                violations.append(
                    f"Demand balance violated for item {i}, period {t}: "
                    f"s_prev({s[i][t]}) + x({x[i][t]}) = {lhs} != d({d[i][t]}) + s({s[i][t+1]}) = {rhs}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # =========================================================================
    # Constraint (3): Capacity
    #   sum_i (st_{it} * y_{it} + vt_{it} * x_{it}) <= cap_t  for all t in T
    # =========================================================================
    for t in range(nT):
        lhs = sum(st[i][t] * y[i][t] + vt[i][t] * x[i][t] for i in range(nI))
        rhs = cap[t]
        violation_amount = lhs - rhs  # <= constraint: violation if lhs > rhs
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints.add(3)
            violations.append(
                f"Capacity exceeded in period {t}: usage={lhs:.4f} > capacity={rhs}"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # =========================================================================
    # Constraint (4): Setup forcing / variable upper bound
    #   x_{it} <= min((cap_t - st_{it}) / vt_{it}, sd_{it|T|}) * y_{it}
    #   for all i in I, t in T
    # =========================================================================
    for i in range(nI):
        for t in range(nT):
            if vt[i][t] > 0:
                ub_cap = (cap[t] - st[i][t]) / vt[i][t]
            else:
                ub_cap = float("inf")
            ub_val = max(min(ub_cap, sd[i][t]), 0) * y[i][t]
            lhs = x[i][t]
            rhs = ub_val
            violation_amount = lhs - rhs  # <= constraint
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints.add(4)
                violations.append(
                    f"Setup forcing violated for item {i}, period {t}: "
                    f"x={lhs:.4f} > UB*y={rhs:.4f} (y={y[i][t]})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # =========================================================================
    # Constraint (5): Variable domains and nonnegativity
    #   y_{it} in {0,1}
    #   x_{it} >= 0
    #   s_{it} >= 0 for all t
    #   s_{i0} >= 0
    #   s_{i|T|} = 0 (ending inventory is zero)
    # =========================================================================

    # (5a) y_{it} in {0,1}
    for i in range(nI):
        for t in range(nT):
            val = y[i][t]
            if val not in (0, 1) and abs(val - round(val)) > tol:
                lhs = abs(val - round(val))
                rhs = 0.0
                violation_amount = lhs
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints.add(5)
                violations.append(
                    f"Binary constraint violated for y[{i}][{t}]: value={val}"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # (5b) x_{it} >= 0
    for i in range(nI):
        for t in range(nT):
            if x[i][t] < -tol:
                lhs = 0.0  # RHS of >= constraint is 0
                rhs_val = 0.0
                violation_amount = -x[i][t]  # how much below 0
                normalizer = max(abs(rhs_val), eps)
                ratio = violation_amount / normalizer
                violated_constraints.add(5)
                violations.append(
                    f"Non-negativity violated for x[{i}][{t}]: value={x[i][t]:.6f}"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": x[i][t],
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # (5c) s_{it} >= 0 for all t (including s_{i0})
    for i in range(nI):
        for t in range(nT + 1):
            if s[i][t] < -tol:
                violation_amount = -s[i][t]
                rhs_val = 0.0
                normalizer = max(abs(rhs_val), eps)
                ratio = violation_amount / normalizer
                violated_constraints.add(5)
                violations.append(
                    f"Non-negativity violated for s[{i}][{t}]: value={s[i][t]:.6f}"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": s[i][t],
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # (5d) s_{i|T|} = 0 (ending inventory must be zero)
    for i in range(nI):
        ending_inv = s[i][nT]
        violation_amount = abs(ending_inv)
        if violation_amount > tol:
            rhs_val = 0.0
            normalizer = max(abs(rhs_val), eps)
            ratio = violation_amount / normalizer
            violated_constraints.add(5)
            violations.append(
                f"Ending inventory not zero for item {i}: s[{i}][{nT}]={ending_inv:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": ending_inv,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # =========================================================================
    # Constraint (6): Objective consistency (Tier C anti-exploit check)
    #   reported objective_value must equal the recomputed sum
    #     sum_i fc_i * s_{i0}
    #     + sum_i sum_t (sc_{it} * y_{it} + vc_{it} * x_{it} + hc_{it} * s_{it})
    #   from the solution variables, within a small tolerance.
    #
    # All variables that determine the objective (x, y, inventory including
    # s_{i0}) are present in the solution schema, so a full recompute applies.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            sc = expand_param(instance["setup_cost"], nI, nT)
            vc = expand_param(instance["variable_cost"], nI, nT)
            hc = expand_param(instance["holding_cost"], nI, nT)
            fc = instance["initial_inventory_cost"]

            true_obj = 0.0
            for i in range(nI):
                true_obj += float(fc[i]) * float(s[i][0])
                for t in range(nT):
                    true_obj += (
                        float(sc[i][t]) * float(y[i][t])
                        + float(vc[i][t]) * float(x[i][t])
                        + float(hc[i][t]) * float(s[i][t + 1])
                    )

            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with a generous 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violated_constraints.add(6)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed "
                    f"sum_i fc_i*s_i0 + sum_{{i,t}}(sc*y + vc*x + hc*s)="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })
        elif reported is not None:
            # Non-finite reported value (inf/nan) — flag as obj inconsistency.
            violated_constraints.add(6)
            violations.append(
                f"Objective consistency violated: reported objective_value="
                f"{reported_obj} is not finite"
            )
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": reported,
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": eps,
                "ratio": float("inf"),
            })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CLST (de Araujo et al. 2015)"
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

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
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
