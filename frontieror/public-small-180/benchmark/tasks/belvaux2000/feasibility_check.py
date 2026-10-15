"""
Feasibility checker for the Con-NT Small Bucket Lot-Sizing Model.

From: Belvaux & Wolsey (2000), "bc-prod: A Specialized Branch-and-Cut System
for Lot-Sizing Problems", Management Science 46(5):724-738.

Checks all hard constraints from the mathematical formulation (math_model.txt),
numbered strictly top-to-bottom in the formulation section.

Constraints in the formulation section (top to bottom):
  Constraint 1:  Flow balance for intermediate products (multilevel only)
  Constraint 2:  Flow conservation for final products (generic)
  Constraint 3:  Capacity / variable upper bound (generic)
  Constraint 4:  Minimum production (generic, only if L^{ik} > 0)
  Constraint 5:  Start-up / switch-off linking (generic)
  Constraint 6:  Variable domains (generic): x, s, r >= 0; y, z, w in {0,1}
  Constraint 7:  Single set-up per machine per period (additional linking)
  Constraint 8:  Machine capacity constraint (additional linking)
  -- Example 1 (set1ch) constraints --
  Constraint 9:  Flow balance for set1ch (eq 6)
  Constraint 10: Capacity for set1ch (eq 7)
  Constraint 11: Machine capacity for set1ch (eq 8)
  Constraint 12: Variable domains for set1ch
  -- Example 2 (Con-NT) constraints --
  Constraint 13: Flow balance (eq 9)
  Constraint 14: Capacity with startup (eq 10)
  Constraint 15: Startup definition (eq 11)
  Constraint 16: One item per machine per period (eq 12)
  Constraint 17: Startup bound z <= y (eq 13)
  Constraint 18: Linking constraint (eq 14)
  Constraint 19: Variable domains for Con-NT: s, r, x >= 0; y, z in {0,1}; s0=r0=0
  Constraint 20: Objective consistency (Tier C anti-exploit check) --
                 reported objective_value must equal recomputed
                 sum_{i,k,t}(f*y + g*z) + sum_{i,t}(h_i*s + e_i*r).

The instances are Con-NT (Example 2), so constraints 13-19 are the active ones.
Constraints 1-5 are generic templates; constraints 13-18 are the Con-NT specializations.
Constraint 6 variable domains are also checked via Constraint 19's specialization.
"""

import json
import argparse
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('backlog', 'objective_value', 'production', 'setup', 'startup', 'stock')
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
    with open(path, "r") as f:
        return json.load(f)


def get_val(dct, key, default=0.0):
    """Get a value from a sparse dict (solution), defaulting to 0."""
    return dct.get(key, default)


def normalize_variable_keys(values, prefix):
    """Accept public schema keys and legacy Gurobi-prefixed keys."""
    normalized = {}
    marker = f"{prefix}_"
    for raw_key, value in values.items():
        key = raw_key[len(marker):] if raw_key.startswith(marker) else raw_key
        if key in normalized and normalized[key] != value:
            raise ValueError(
                f"conflicting values for {key!r} via prefixed/unprefixed keys"
            )
        normalized[key] = value
    return normalized


def record_violation(violations_list, magnitudes_list, constraint_idx, lhs, rhs, message):
    """Record a constraint violation with normalized magnitude."""
    raw_excess = abs(lhs - rhs)
    normalizer = max(abs(rhs), EPS)
    ratio = raw_excess / normalizer
    violations_list.append(message)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(raw_excess),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def check_feasibility(instance, solution):
    NI = instance["parameters"]["NI"]
    NK = instance["parameters"]["NK"]
    NT = instance["parameters"]["NT"]

    C = instance["machine_data"]["capacity_C"]
    sigma = instance["machine_data"]["startup_time_sigma"]
    demand = instance["demand"]
    s0 = instance["initial_conditions"]["initial_stock_s0"]
    r0 = instance["initial_conditions"]["initial_backlog_r0"]

    f_setup = instance["costs"]["setup_cost_f"]
    g_startup = instance["costs"]["startup_cost_g"]
    h_hold = instance["costs"]["holding_cost_h"]
    e_back = instance["costs"]["backlogging_cost_e"]

    prod = normalize_variable_keys(solution.get("production", {}), "x")
    stock = normalize_variable_keys(solution.get("stock", {}), "s")
    backlog = normalize_variable_keys(solution.get("backlog", {}), "r")
    setup = normalize_variable_keys(solution.get("setup", {}), "y")
    startup = normalize_variable_keys(solution.get("startup", {}), "z")

    # Helper to extract variable values from sparse solution dicts
    def x(i, k, t):
        return get_val(prod, f"{i}_{k}_{t}")

    def s(i, t):
        return get_val(stock, f"{i}_{t}")

    def r(i, t):
        return get_val(backlog, f"{i}_{t}")

    def y(i, k, t):
        return get_val(setup, f"{i}_{k}_{t}")

    def z(i, k, t):
        return get_val(startup, f"{i}_{k}_{t}")

    violations = []
    magnitudes = []
    violated_constraints = set()

    # =========================================================================
    # Constraint 13 (paper eq 9): Flow balance for final products (Con-NT)
    #   s_{i,t-1} - r_{i,t-1} + sum_k x_{ikt} = d_{it} + s_{it} - r_{it}
    #   for all i, t
    # This is an equality constraint.
    # =========================================================================
    for i in range(NI):
        for t in range(NT):
            s_prev = s0[i] if t == 0 else s(i, t - 1)
            r_prev = r0[i] if t == 0 else r(i, t - 1)

            lhs = s_prev - r_prev + sum(x(i, k, t) for k in range(NK))
            rhs = demand[i][t] + s(i, t) - r(i, t)

            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(13)
                record_violation(
                    violations, magnitudes, 13, lhs, rhs,
                    f"Constraint 13 (flow balance eq 9) violated for item {i}, period {t}: "
                    f"LHS={lhs:.6f}, RHS={rhs:.6f}, diff={violation_amount:.6f}"
                )

    # =========================================================================
    # Constraint 14 (paper eq 10): Capacity with startup
    #   x_{ikt} + sigma_k * z_{ikt} <= C_k * y_{ikt}
    #   for all i, k, t
    # This is a <= constraint.
    # =========================================================================
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                lhs = x(i, k, t) + sigma[k] * z(i, k, t)
                rhs = C[k] * y(i, k, t)

                violation_amount = lhs - rhs
                if violation_amount > TOL:
                    violated_constraints.add(14)
                    record_violation(
                        violations, magnitudes, 14, lhs, rhs,
                        f"Constraint 14 (capacity eq 10) violated for item {i}, machine {k}, period {t}: "
                        f"x={x(i,k,t):.4f} + sigma*z={sigma[k]*z(i,k,t):.4f} = {lhs:.4f} > "
                        f"C*y={rhs:.4f}"
                    )

    # =========================================================================
    # Constraint 15 (paper eq 11): Startup definition
    #   z_{ikt} >= y_{ikt} - y_{ik,t-1}
    #   for all i, k, t
    # This is a >= constraint: z >= y_curr - y_prev
    # Equivalently: violation if y_curr - y_prev - z > 0
    # =========================================================================
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                y_prev = 0 if t == 0 else y(i, k, t - 1)
                lhs = z(i, k, t)
                rhs = y(i, k, t) - y_prev

                violation_amount = rhs - lhs  # >= constraint: RHS exceeds LHS
                if violation_amount > TOL:
                    violated_constraints.add(15)
                    record_violation(
                        violations, magnitudes, 15, lhs, rhs,
                        f"Constraint 15 (startup def eq 11) violated for item {i}, machine {k}, period {t}: "
                        f"z={z(i,k,t):.4f} < y_curr - y_prev = {y(i,k,t):.4f} - {y_prev:.4f} = {rhs:.4f}"
                    )

    # =========================================================================
    # Constraint 16 (paper eq 12): One item per machine per period
    #   sum_i y_{ikt} <= 1   for all k, t
    # This is a <= constraint.
    # =========================================================================
    for k in range(NK):
        for t in range(NT):
            lhs = sum(y(i, k, t) for i in range(NI))
            rhs = 1.0

            violation_amount = lhs - rhs
            if violation_amount > TOL:
                violated_constraints.add(16)
                record_violation(
                    violations, magnitudes, 16, lhs, rhs,
                    f"Constraint 16 (one setup eq 12) violated for machine {k}, period {t}: "
                    f"sum_i y_ikt = {lhs:.4f} > 1"
                )

    # =========================================================================
    # Constraint 17 (paper eq 13): Startup bound
    #   z_{ikt} <= y_{ikt}   for all i, k, t
    # This is a <= constraint.
    # =========================================================================
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                lhs = z(i, k, t)
                rhs = y(i, k, t)

                violation_amount = lhs - rhs
                if violation_amount > TOL:
                    violated_constraints.add(17)
                    record_violation(
                        violations, magnitudes, 17, lhs, rhs,
                        f"Constraint 17 (z<=y eq 13) violated for item {i}, machine {k}, period {t}: "
                        f"z={z(i,k,t):.4f} > y={y(i,k,t):.4f}"
                    )

    # =========================================================================
    # Constraint 18 (paper eq 14): Linking constraint
    #   y_{ik,t-1} + z_{ikt} + sum_{j!=i} (y_{jkt} - z_{jkt}) <= 1
    #   for all i, k, t
    # This is a <= constraint.
    # =========================================================================
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                y_prev_i = 0 if t == 0 else y(i, k, t - 1)
                lhs = y_prev_i + z(i, k, t)
                for j in range(NI):
                    if j != i:
                        lhs += y(j, k, t) - z(j, k, t)
                rhs = 1.0

                violation_amount = lhs - rhs
                if violation_amount > TOL:
                    violated_constraints.add(18)
                    record_violation(
                        violations, magnitudes, 18, lhs, rhs,
                        f"Constraint 18 (linking eq 14) violated for item {i}, machine {k}, period {t}: "
                        f"LHS={lhs:.4f} > 1"
                    )

    # =========================================================================
    # Constraint 19: Variable domains for Con-NT
    #   s, r, x >= 0; y, z in {0, 1}; s^i_0 = r^i_0 = 0
    # =========================================================================

    # Check non-negativity of continuous variables (x, s, r >= 0)
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                val = x(i, k, t)
                if val < -TOL:
                    violated_constraints.add(19)
                    record_violation(
                        violations, magnitudes, 19, val, 0.0,
                        f"Constraint 19 (domain) violated: x_{i}_{k}_{t} = {val:.6f} < 0"
                    )

    for i in range(NI):
        for t in range(NT):
            val_s = s(i, t)
            if val_s < -TOL:
                violated_constraints.add(19)
                record_violation(
                    violations, magnitudes, 19, val_s, 0.0,
                    f"Constraint 19 (domain) violated: s_{i}_{t} = {val_s:.6f} < 0"
                )
            val_r = r(i, t)
            if val_r < -TOL:
                violated_constraints.add(19)
                record_violation(
                    violations, magnitudes, 19, val_r, 0.0,
                    f"Constraint 19 (domain) violated: r_{i}_{t} = {val_r:.6f} < 0"
                )

    # Check binary variables y, z in {0, 1}
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                val_y = y(i, k, t)
                if abs(val_y - round(val_y)) > TOL or round(val_y) not in (0, 1):
                    violated_constraints.add(19)
                    record_violation(
                        violations, magnitudes, 19, val_y,
                        round(val_y) if round(val_y) in (0, 1) else 0.0,
                        f"Constraint 19 (domain) violated: y_{i}_{k}_{t} = {val_y:.6f} is not binary"
                    )
                val_z = z(i, k, t)
                if abs(val_z - round(val_z)) > TOL or round(val_z) not in (0, 1):
                    violated_constraints.add(19)
                    record_violation(
                        violations, magnitudes, 19, val_z,
                        round(val_z) if round(val_z) in (0, 1) else 0.0,
                        f"Constraint 19 (domain) violated: z_{i}_{k}_{t} = {val_z:.6f} is not binary"
                    )

    # =========================================================================
    # Constraint 20: Objective consistency (Tier C anti-exploit check)
    #   reported objective_value must equal recomputed
    #     sum_{i,k,t} (f * y_{ikt} + g * z_{ikt})
    #       + sum_{i,t} (h_i * s_{it} + e_i * r_{it})
    # All variables (y, z, s, r) appearing in the obj formula are present in
    # the solution, so a full recompute is exact (modulo float rounding from
    # solution serialization). Tolerance: 0.1% relative with 1e-3 absolute
    # floor, matching the recipe in docs/data_review_fix/checker_add_obj_recompute.md.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            true_obj = 0.0
            for i in range(NI):
                for k in range(NK):
                    for t in range(NT):
                        true_obj += f_setup * y(i, k, t) + g_startup * z(i, k, t)
                for t in range(NT):
                    true_obj += h_hold[i] * s(i, t) + e_back[i] * r(i, t)
            abs_diff = abs(reported - true_obj)
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(20)
                record_violation(
                    violations, magnitudes, 20, reported, true_obj,
                    f"Constraint 20 (objective consistency) violated: "
                    f"reported objective_value={reported} differs from recomputed "
                    f"sum_(i,k,t)(f*y+g*z) + sum_(i,t)(h_i*s+e_i*r)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
        elif reported is not None:
            # Non-finite reported value (e.g. inf, nan) -- always inconsistent.
            violated_constraints.add(20)
            record_violation(
                violations, magnitudes, 20, reported, 0.0,
                f"Constraint 20 (objective consistency) violated: "
                f"reported objective_value={reported} is not a finite number"
            )

    feasible = len(violated_constraints) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Con-NT lot-sizing model (Belvaux & Wolsey 2000)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON candidate solution file")
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

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    n_violations = len(result["violated_constraints"])
    print(f"{status} ({n_violations} violated constraint(s))")
    if not result["feasible"]:
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
