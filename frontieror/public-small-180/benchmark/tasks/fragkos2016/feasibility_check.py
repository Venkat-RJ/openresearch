"""
Feasibility checker for the Capacitated Lot-Sizing Problem with Setup Times (CLST).
Reference: Fragkos, Degraeve, and De Reyck (2016), INFORMS Journal on Computing 28(3): 465-482.

Constraints (numbered top-to-bottom from the formulation section of math_model.txt):
  Constraint 1: Flow balance (equality)
      s_{it} + x_{it} = d_{it} + s_{i,t+1}   for all i in I, t in T
  Constraint 2: Setup-production link (<=)
      x_{it} <= M_{it} * y_{it}               for all i in I, t in T
  Constraint 3: Capacity (<=)
      sum_i st_{it}*y_{it} + sum_i vt_{it}*x_{it} <= cap_t   for all t in T
  Constraint 4: Nonnegativity, integrality, and boundary conditions
      x_{it} >= 0, s_{it} >= 0, s_{i,m+1} = 0, y_{it} in {0,1}
  Constraint 5: Objective consistency (Tier C obj-recompute defense)
      reported objective_value must equal the recomputed total cost
        sum_{i,t} sc_{it}*y_{it} + vc_{it}*x_{it} + hc_{it}*s_{it}
      within tolerance.  All objective-determining variables (x, s, y) are
      present in the solution, so a full recompute is possible.

This is the obj-recompute variant of feasibility_check.py (Tier C defense
against LLM score-gaming).  The original file is left untouched; the only
behavioural difference here is the addition of Constraint 5.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 's', 'x', 'y')
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


TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_feasibility(instance, solution):
    n = instance["n_items"]
    m = instance["n_periods"]
    d = instance["demands"]
    cap = instance["capacity"]
    st = instance["setup_times"]
    vt = instance["variable_times"]
    big_M = instance["big_M"]
    init_inv = instance["initial_inventory"]

    x = solution["x"]
    y = solution["y"]

    # If solution is null/infeasible, report immediately
    if x is None or y is None:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution is null/infeasible (no decision variables provided)."],
            "violation_magnitudes": []
        }

    # math_model.txt note 2: s_{i,1} is a decision variable, not a fixed input.
    # Reconstructing s from x and init_inv fabricates s_{i,1} = init_inv[i] and
    # auto-satisfies flow balance (eq 2), so missing 's' must be rejected rather
    # than synthesised.
    s = solution.get("s")
    if s is None:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution is missing required field 's' (inventory)."],
            "violation_magnitudes": []
        }

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # --- Constraint 1: Flow balance (equality) ---
    # s_{it} + x_{it} = d_{it} + s_{i,t+1}  for all i, t
    for i in range(n):
        for t in range(m):
            lhs = s[i][t] + x[i][t]
            rhs = d[i][t] + s[i][t + 1]
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(1)
                violations.append(
                    f"Constraint 1: Flow balance violated for item {i}, period {t}: "
                    f"s[{i}][{t}] + x[{i}][{t}] = {lhs:.6f} != d[{i}][{t}] + s[{i}][{t+1}] = {rhs:.6f}"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # --- Constraint 2: Setup-production link (<=) ---
    # x_{it} <= M_{it} * y_{it}
    for i in range(n):
        for t in range(m):
            lhs = x[i][t]
            rhs = big_M[i][t] * y[i][t]
            violation_amount = lhs - rhs
            if violation_amount > TOL:
                violated_constraints.add(2)
                violations.append(
                    f"Constraint 2: Setup link violated for item {i}, period {t}: "
                    f"x[{i}][{t}] = {lhs:.6f} > M[{i}][{t}]*y[{i}][{t}] = {rhs:.6f}"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # --- Constraint 3: Capacity (<=) ---
    # sum_i st_{it}*y_{it} + sum_i vt_{it}*x_{it} <= cap_t
    for t in range(m):
        lhs = sum(st[i][t] * y[i][t] + vt[i][t] * x[i][t] for i in range(n))
        rhs = cap[t]
        violation_amount = lhs - rhs
        if violation_amount > TOL:
            violated_constraints.add(3)
            violations.append(
                f"Constraint 3: Capacity violated in period {t}: "
                f"usage = {lhs:.6f} > capacity = {rhs:.6f}"
            )
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # --- Constraint 4: Nonnegativity, integrality, and boundary ---
    # x_{it} >= 0
    for i in range(n):
        for t in range(m):
            if x[i][t] < -TOL:
                violated_constraints.add(4)
                lhs_val = 0.0
                rhs_val = x[i][t]
                violation_amount = -x[i][t]
                violations.append(
                    f"Constraint 4: Nonnegativity violated: x[{i}][{t}] = {x[i][t]:.6f} < 0"
                )
                normalizer = max(abs(rhs_val), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": rhs_val,
                    "rhs": lhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # s_{it} >= 0 for t in 0..m
    for i in range(n):
        for t in range(m + 1):
            if s[i][t] < -TOL:
                violated_constraints.add(4)
                lhs_val = 0.0
                rhs_val = s[i][t]
                violation_amount = -s[i][t]
                violations.append(
                    f"Constraint 4: Nonnegativity violated: s[{i}][{t}] = {s[i][t]:.6f} < 0"
                )
                normalizer = max(abs(rhs_val), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": rhs_val,
                    "rhs": lhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # s_{i,m+1} = 0 (ending inventory)
    for i in range(n):
        violation_amount = abs(s[i][m])
        if violation_amount > TOL:
            violated_constraints.add(4)
            lhs_val = s[i][m]
            rhs_val = 0.0
            violations.append(
                f"Constraint 4: Ending inventory violated: s[{i}][{m}] = {s[i][m]:.6f} != 0"
            )
            normalizer = max(abs(rhs_val), EPS)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # y_{it} in {0, 1}
    for i in range(n):
        for t in range(m):
            val = y[i][t]
            dist_to_binary = min(abs(val - 0), abs(val - 1))
            if dist_to_binary > TOL:
                violated_constraints.add(4)
                # Treat as equality to nearest binary: violation = distance
                nearest = round(val)
                lhs_val = val
                rhs_val = float(nearest)
                violation_amount = dist_to_binary
                violations.append(
                    f"Constraint 4: Integrality violated: y[{i}][{t}] = {val:.6f} not in {{0, 1}}"
                )
                normalizer = max(abs(rhs_val), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # --- Constraint 5: Objective consistency (Tier C obj-recompute) ---
    # The reported objective_value must agree with the objective recomputed
    # directly from the solution variables:
    #   obj = sum_{i,t} sc_{it}*y_{it} + vc_{it}*x_{it} + hc_{it}*s_{it}
    # All objective-determining variables (x, s, y) are present in the
    # solution, so this is an exact full recompute (no lower bound needed).
    # Holding cost sums s_{it} over t in T (periods 0..m-1); the boundary
    # entry s_{i,m} is excluded, consistent with the paper's Eq. (1) and
    # the Gurobi reference implementation (gurobi_code.py).
    sc = instance.get("setup_costs")
    vc = instance.get("variable_costs")
    hc = instance.get("holding_costs")
    reported_obj = solution.get("objective_value")
    if sc is not None and vc is not None and hc is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            try:
                true_obj = float(sum(
                    sc[i][t] * y[i][t] + vc[i][t] * x[i][t] + hc[i][t] * s[i][t]
                    for i in range(n) for t in range(m)
                ))
            except (IndexError, TypeError):
                true_obj = None
            if true_obj is not None:
                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with a 1e-3 absolute floor.
                tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > tol:
                    violated_constraints.add(5)
                    violations.append(
                        f"Constraint 5: Objective consistency violated: reported "
                        f"objective_value = {reported} differs from recomputed "
                        f"sum_(i,t)[sc*y + vc*x + hc*s] = {true_obj} "
                        f"(|diff| = {abs_diff:.6g}, tol = {tol:.6g})"
                    )
                    normalizer = max(abs(true_obj), EPS)
                    violation_magnitudes.append({
                        "constraint": 5,
                        "lhs": reported,
                        "rhs": true_obj,
                        "raw_excess": abs_diff,
                        "normalizer": normalizer,
                        "ratio": abs_diff / normalizer
                    })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CLST (Fragkos et al. 2016)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
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
    print(f"  Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"  Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"    - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
