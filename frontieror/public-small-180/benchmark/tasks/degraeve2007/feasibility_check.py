"""
Feasibility checker for the Capacitated Lot Sizing Problem with Setup Times (CLST).
Based on Degraeve and Jans (2007).

Constraints (numbered top-to-bottom from the formulation):
  1. Flow balance for period 1:    si_i + x_{i,1} = d_{i,1} + s_{i,1}
  2. Flow balance for periods 2..m: s_{i,t-1} + x_{it} = d_{it} + s_{it}
  3. Setup forcing / prod upper bound: x_{it} <= min((cap_t - st_{it})/vt_{it}, sd_{itm}) * y_{it}
  4. Capacity per period: sum_i (st_{it}*y_{it} + vt_{it}*x_{it}) <= cap_t
  5. Variable domains: y_{it} in {0,1}, x_{it} >= 0, s_{it} >= 0, si_i >= 0
  6. Objective consistency: reported objective_value must equal the objective
     recomputed from the solution variables:
       obj = sum_i fc_i*si_i + sum_i sum_t (sc_it*y_it + vc_it*x_it + hc_it*s_it)

NOTE: this is the obj-recompute ("Tier C") variant of feasibility_check.py.
Constraints 1..5 are byte-identical in logic to the original checker; only
constraint 6 (objective consistency) is added. See
docs/data_review_fix/checker_add_obj_recompute.md.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'variables')
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



def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["dimensions"]["num_products"]
    m = instance["dimensions"]["num_periods"]
    params = instance["parameters"]

    d = params["demand"]
    st = params["setup_time"]
    vt = params["variable_production_time"]
    cap = params["capacity"]

    variables = solution["variables"]
    x = variables["production"]
    y = variables["setup"]
    s = variables["inventory"]
    si = variables["initial_inventory"]

    # Precompute cumulative demand sd[i][t] = sum d[i][l] for l=t..m-1
    sd = [[0.0] * m for _ in range(n)]
    for i in range(n):
        sd[i][m - 1] = d[i][m - 1]
        for t in range(m - 2, -1, -1):
            sd[i][t] = sd[i][t + 1] + d[i][t]

    violations = []
    violation_magnitudes = []
    violated_set = set()

    def record(constraint_idx, msg, lhs, rhs, violation_amount):
        violated_set.add(constraint_idx)
        violations.append(msg)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # --- Constraint 1: Flow balance for period 1 (equality) ---
    # si_i + x_{i,1} = d_{i,1} + s_{i,1}   for all i
    for i in range(n):
        lhs = si[i] + x[i][0]
        rhs = d[i][0] + s[i][0]
        va = abs(lhs - rhs)
        if va > tol:
            record(1, f"Flow balance violated for product {i+1} in period 1: "
                   f"LHS={lhs}, RHS={rhs}, diff={va}", lhs, rhs, va)

    # --- Constraint 2: Flow balance for periods 2..m (equality) ---
    # s_{i,t-1} + x_{it} = d_{it} + s_{it}  for all i, t>=2
    for i in range(n):
        for t in range(1, m):
            lhs = s[i][t - 1] + x[i][t]
            rhs = d[i][t] + s[i][t]
            va = abs(lhs - rhs)
            if va > tol:
                record(2, f"Flow balance violated for product {i+1} in period {t+1}: "
                       f"LHS={lhs}, RHS={rhs}, diff={va}", lhs, rhs, va)

    # --- Constraint 3: Setup forcing / production upper bound (<=) ---
    # x_{it} <= min((cap_t - st_{it})/vt_{it}, sd_{itm}) * y_{it}
    for i in range(n):
        for t in range(m):
            if vt[i][t] > 0:
                ub_cap = (cap[t] - st[i][t]) / vt[i][t]
            else:
                ub_cap = 1e15
            ub = min(ub_cap, sd[i][t]) * y[i][t]
            ub = max(ub, 0.0) * y[i][t] if y[i][t] > 0.5 else 0.0
            # Recompute cleanly
            raw_ub = max(min(ub_cap, sd[i][t]), 0.0)
            rhs_val = raw_ub * y[i][t]
            lhs_val = x[i][t]
            va = lhs_val - rhs_val
            if va > tol:
                record(3, f"Setup forcing violated for product {i+1} in period {t+1}: "
                       f"x={lhs_val}, UB*y={rhs_val}, excess={va}",
                       lhs_val, rhs_val, va)

    # --- Constraint 4: Capacity constraint (<=) ---
    # sum_i (st_{it}*y_{it} + vt_{it}*x_{it}) <= cap_t
    for t in range(m):
        lhs_val = sum(st[i][t] * y[i][t] + vt[i][t] * x[i][t] for i in range(n))
        rhs_val = cap[t]
        va = lhs_val - rhs_val
        if va > tol:
            record(4, f"Capacity exceeded in period {t+1}: "
                   f"usage={lhs_val}, capacity={rhs_val}, excess={va}",
                   lhs_val, rhs_val, va)

    # --- Constraint 5: Variable domains ---
    # y_{it} in {0,1}, x_{it} >= 0, s_{it} >= 0, si_i >= 0
    for i in range(n):
        # si_i >= 0
        if si[i] < -tol:
            va = -si[i]
            record(5, f"Initial inventory for product {i+1} is negative: si={si[i]}",
                   si[i], 0.0, va)
        for t in range(m):
            # y binary
            y_val = y[i][t]
            if abs(y_val - round(y_val)) > tol or round(y_val) not in (0, 1):
                va = abs(y_val - round(y_val))
                record(5, f"Setup variable for product {i+1}, period {t+1} is not binary: y={y_val}",
                       y_val, round(y_val), va)
            # x >= 0
            if x[i][t] < -tol:
                va = -x[i][t]
                record(5, f"Production for product {i+1}, period {t+1} is negative: x={x[i][t]}",
                       x[i][t], 0.0, va)
            # s >= 0
            if s[i][t] < -tol:
                va = -s[i][t]
                record(5, f"Inventory for product {i+1}, period {t+1} is negative: s={s[i][t]}",
                       s[i][t], 0.0, va)

    # --- Constraint 6: Objective consistency (full recompute) ---
    # The CLST objective is fully determined by the solution variables
    # (production x, setup y, inventory s, initial_inventory si), so the
    # reported objective_value can be recomputed exactly and compared:
    #   obj = sum_i fc_i*si_i
    #       + sum_i sum_t (sc_it*y_it + vc_it*x_it + hc_it*s_it)
    # This rejects fabricated objective values (obj-score-gaming exploits).
    hc = params["holding_cost"]
    sc = params["setup_cost"]
    vc = params["variable_production_cost"]
    fc = params["initial_inventory_cost"]
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for i in range(n):
                true_obj += fc[i] * si[i]
                for t in range(m):
                    true_obj += (sc[i][t] * y[i][t]
                                 + vc[i][t] * x[i][t]
                                 + hc[i][t] * s[i][t])
            abs_diff = abs(reported - true_obj)
            # 1e-6 relative tolerance with 1e-3 absolute floor: tight enough
            # to flag fabricated values while absorbing float accumulation
            # over the n*m product/period terms.
            tol_obj = max(1e-3, 1e-6 * abs(true_obj))
            if abs_diff > tol_obj:
                record(6, f"Objective consistency violated: reported "
                       f"objective_value={reported} differs from recomputed "
                       f"objective sum_i fc_i*si_i + sum_it("
                       f"sc_it*y_it + vc_it*x_it + hc_it*s_it)={true_obj} "
                       f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})",
                       reported, true_obj, abs_diff)

    violated_constraints = sorted(violated_set)
    feasible = len(violated_constraints) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CLST (Degraeve & Jans, 2007)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the candidate solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)
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
        print("FEASIBLE: All constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
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
