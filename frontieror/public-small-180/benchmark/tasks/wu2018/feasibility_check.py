"""
Feasibility checker for the Capacitated Multi-Item Lot Sizing Problem
with Nonidentical Machines (CMLS-NM).

Reference: Wu, Liang, and Zhang (2018), LSNM Formulation (Section 3.1).

Checks constraints (2)-(7) from the mathematical formulation, plus
constraint 1 (binary domain) and constraint 8 (objective-value
consistency: reported objective_value must match the recomputed cost
sum_{j,t,m in M_j} sc^m_j Y^m_{jt}
+ sum_{j,t,m in M_j} pc^m_j X^m_{jt}
+ sum_{j,t} hc_j I_{jt}).
"""

import argparse
import json
import math
import re

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('inventory', 'objective_value', 'production', 'setups')
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


def parse_instance(data):
    """Parse instance JSON into structured data (0-indexed)."""
    num_items = data["num_items"]
    num_periods = data["num_periods"]
    num_machines = data["num_machines"]

    J = list(range(num_items))
    T = list(range(num_periods))
    M = list(range(num_machines))

    # Machine capabilities: 1-based in JSON -> 0-based
    M_j = {}
    for j in J:
        M_j[j] = [m - 1 for m in data["machine_capabilities"][j]]

    F_m = {m: [] for m in M}
    for j in J:
        for m in M_j[j]:
            F_m[m].append(j)

    mm_j = {j: data["max_machines_per_item_per_period"][j] for j in J}

    d = {}
    for j in J:
        for t in T:
            d[j, t] = data["demand"][j][t]

    pt = {}
    for j in J:
        for m in M_j[j]:
            pt[j, m] = data["production_time"][f"item_{j+1}_machine_{m+1}"]

    hc = {j: data["holding_cost"][j] for j in J}

    sc = {}
    for j in J:
        for m in M_j[j]:
            sc[j, m] = data["setup_cost"][f"item_{j+1}_machine_{m+1}"]

    st = {}
    for j in J:
        for m in M_j[j]:
            st[j, m] = data["setup_time"][f"item_{j+1}_machine_{m+1}"]

    pc = {}
    for j in J:
        for m in M_j[j]:
            pc[j, m] = data["production_cost"][f"item_{j+1}_machine_{m+1}"]

    Cap = {}
    for m in M:
        for t in T:
            Cap[m, t] = data["capacity"][m][t]

    # Initial inventory I_{j,0}: given parameter (default 0 if absent).
    init_list = data.get("initial_inventory", [0.0] * num_items)
    I0 = {j: init_list[j] for j in J}

    return J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, Cap, I0


def parse_solution(sol, J, T, M_j):
    """
    Parse solution JSON into X, I, Y dicts (0-indexed).
    Missing variables default to 0.
    """
    # Parse production X[j, m, t]
    X = {}
    for j in J:
        for m in M_j[j]:
            for t in T:
                X[j, m, t] = 0.0
    if "production" in sol:
        for key, val in sol["production"].items():
            # Key format: X_item{j+1}_machine{m+1}_period{t+1}
            match = re.match(r"X_item(\d+)_machine(\d+)_period(\d+)", key)
            if match:
                j = int(match.group(1)) - 1
                m = int(match.group(2)) - 1
                t = int(match.group(3)) - 1
                X[j, m, t] = float(val)

    # Parse inventory I[j, t]
    I = {}
    for j in J:
        for t in T:
            I[j, t] = 0.0
    if "inventory" in sol:
        for key, val in sol["inventory"].items():
            match = re.match(r"I_item(\d+)_period(\d+)", key)
            if match:
                j = int(match.group(1)) - 1
                t = int(match.group(2)) - 1
                I[j, t] = float(val)

    # Parse setups Y[j, m, t]
    Y = {}
    for j in J:
        for m in M_j[j]:
            for t in T:
                Y[j, m, t] = 0
    if "setups" in sol:
        for key, val in sol["setups"].items():
            match = re.match(r"Y_item(\d+)_machine(\d+)_period(\d+)", key)
            if match:
                j = int(match.group(1)) - 1
                m = int(match.group(2)) - 1
                t = int(match.group(3)) - 1
                Y[j, m, t] = int(round(float(val)))

    return X, I, Y


def check_feasibility(J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, Cap, I0,
                       X, I, Y):
    """
    Check all hard constraints (2)-(7).
    Returns (violated_constraints, violations, violation_magnitudes).
    """
    tol = 1e-5
    eps = 1e-5
    num_T = len(T)

    violated_set = set()
    violations = []
    magnitudes = []

    def record_violation(constraint_idx, message, lhs_val, rhs_val, viol_amount):
        violated_set.add(constraint_idx)
        violations.append(message)
        normalizer = max(abs(rhs_val), eps)
        magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs_val,
            "rhs": rhs_val,
            "raw_excess": viol_amount,
            "normalizer": normalizer,
            "ratio": viol_amount / normalizer
        })

    # ---------------------------------------------------------------
    # Constraint (2): Demand satisfaction (equality)
    #   sum_{m in M_j} X^m_{jt} + I_{j,t-1} - I_{jt} = d_{jt}
    #   I_{j,0} is a given parameter (read from instance as I0).
    # ---------------------------------------------------------------
    for j in J:
        for t in T:
            lhs = sum(X[j, m, t] for m in M_j[j])
            if t > 0:
                lhs += I[j, t - 1]
            else:
                lhs += I0[j]
            lhs -= I[j, t]
            rhs = d[j, t]
            viol = abs(lhs - rhs)
            if viol > tol:
                record_violation(
                    2,
                    f"Demand violation: item {j+1}, period {t+1}: "
                    f"LHS={lhs:.6f}, demand={rhs}",
                    lhs, rhs, viol
                )

    # ---------------------------------------------------------------
    # Constraint (3): Capacity constraints (<=)
    #   sum_{j in F_m} pt^m_j * X^m_{jt} + st^m_j * Y^m_{jt} <= C^m_t
    # ---------------------------------------------------------------
    for m in M:
        for t in T:
            lhs = sum(pt[j, m] * X[j, m, t] + st[j, m] * Y[j, m, t]
                      for j in F_m[m])
            rhs = Cap[m, t]
            viol = lhs - rhs
            if viol > tol:
                record_violation(
                    3,
                    f"Capacity exceeded: machine {m+1}, period {t+1}: "
                    f"usage={lhs:.6f}, capacity={rhs:.6f}",
                    lhs, rhs, viol
                )

    # ---------------------------------------------------------------
    # Constraint (4): Setup forcing / Big-M (<=)
    #   X^m_{jt} <= BM^m_{jt} * Y^m_{jt}
    #   BM^m_{jt} = min(sum_{q=t}^{|T|} d_{jq}, (C^m_t - st^m_j)/pt^m_j)
    # ---------------------------------------------------------------
    for j in J:
        for m in M_j[j]:
            for t in T:
                cum_demand = sum(d[j, q] for q in range(t, num_T))
                if pt[j, m] > 1e-12:
                    cap_bound = max((Cap[m, t] - st[j, m]) / pt[j, m], 0.0)
                else:
                    cap_bound = cum_demand
                BM = min(cum_demand, cap_bound)
                lhs = X[j, m, t]
                rhs = BM * Y[j, m, t]
                viol = lhs - rhs
                if viol > tol:
                    record_violation(
                        4,
                        f"Setup forcing violated: item {j+1}, machine {m+1}, "
                        f"period {t+1}: X={lhs:.6f}, BM*Y={rhs:.6f}",
                        lhs, rhs, viol
                    )

    # ---------------------------------------------------------------
    # Constraint (5): Max machines per item per period (<=)
    #   sum_{m in M_j} Y^m_{jt} <= mm_j
    # ---------------------------------------------------------------
    for j in J:
        for t in T:
            lhs = sum(Y[j, m, t] for m in M_j[j])
            rhs = mm_j[j]
            viol = lhs - rhs
            if viol > tol:
                record_violation(
                    5,
                    f"Max machines exceeded: item {j+1}, period {t+1}: "
                    f"setups={lhs}, limit={rhs}",
                    float(lhs), float(rhs), float(viol)
                )

    # ---------------------------------------------------------------
    # Constraint (6): Nonnegativity (>=)
    #   X^m_{jt} >= 0, I_{jt} >= 0
    # ---------------------------------------------------------------
    for j in J:
        for m in M_j[j]:
            for t in T:
                if X[j, m, t] < -tol:
                    viol = -X[j, m, t]
                    record_violation(
                        6,
                        f"Negative production: item {j+1}, machine {m+1}, "
                        f"period {t+1}: X={X[j, m, t]:.6f}",
                        X[j, m, t], 0.0, viol
                    )
    for j in J:
        for t in T:
            if I[j, t] < -tol:
                viol = -I[j, t]
                record_violation(
                    6,
                    f"Negative inventory: item {j+1}, period {t+1}: "
                    f"I={I[j, t]:.6f}",
                    I[j, t], 0.0, viol
                )

    # ---------------------------------------------------------------
    # Constraint (7): Binary setup variables
    #   Y^m_{jt} in {0, 1}
    # ---------------------------------------------------------------
    for j in J:
        for m in M_j[j]:
            for t in T:
                y_val = Y[j, m, t]
                viol = min(abs(y_val - 0), abs(y_val - 1))
                if viol > tol:
                    record_violation(
                        7,
                        f"Non-binary setup: item {j+1}, machine {m+1}, "
                        f"period {t+1}: Y={y_val}",
                        float(y_val),
                        round(y_val),
                        viol
                    )

    return violated_set, violations, magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CMLS-NM (Wu et al. 2018)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON.")
    args = parser.parse_args()

    # Load data
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

    # Parse instance
    J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, Cap, I0 = parse_instance(instance)

    # Parse solution
    X, I, Y = parse_solution(solution, J, T, M_j)

    # Check feasibility
    violated_set, violations, magnitudes = check_feasibility(
        J, T, M, M_j, F_m, mm_j, d, pt, hc, sc, st, pc, Cap, I0, X, I, Y
    )


    # =====================================================================
    # =====================================================================
    # Constraint 1: Binary domain — variables must be 0 or 1
    tol = 1e-5
    eps = 1e-5
    _domain_check_vars_binary = [("Y", Y)]
    _domain_check_vars_integer = []

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 1: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > tol or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > tol:
                        violated_set.add(1)
                        violations.append(
                            f"Constraint 1 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        magnitudes.append({
                            "constraint": 1,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    # =====================================================================
    # Constraint 8: Objective-value consistency (Tier C defense)
    #   Recompute the LSNM objective from the solution variables and
    #   reject when the reported objective_value disagrees beyond a
    #   0.1% relative / 1e-3 absolute tolerance.
    #   true_obj = sum_{j,t,m in M_j} sc^m_j * Y^m_{jt}
    #            + sum_{j,t,m in M_j} pc^m_j * X^m_{jt}
    #            + sum_{j,t}        hc_j   * I_{jt}
    # =====================================================================
    reported_obj_raw = solution.get("objective_value")
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and math.isfinite(reported_obj):
            setup_cost_total = 0.0
            production_cost_total = 0.0
            for j in J:
                for m in M_j[j]:
                    for t in T:
                        setup_cost_total += sc[j, m] * Y[j, m, t]
                        production_cost_total += pc[j, m] * X[j, m, t]
            holding_cost_total = 0.0
            for j in J:
                for t in T:
                    holding_cost_total += hc[j] * I[j, t]
            true_obj = float(setup_cost_total
                             + production_cost_total
                             + holding_cost_total)
            abs_diff = abs(reported_obj - true_obj)
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_set.add(8)
                violations.append(
                    f"Constraint 8 (objective consistency): "
                    f"reported objective_value={reported_obj} differs from "
                    f"recomputed setup+production+holding cost={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol_obj:.3g})"
                )
                magnitudes.append({
                    "constraint": 8,
                    "lhs": reported_obj,
                    "rhs": true_obj,
                    "raw_excess": float(abs_diff),
                    "normalizer": max(abs(true_obj), eps),
                    "ratio": float(abs_diff) / max(abs(true_obj), eps),
                })

    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": magnitudes
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {feasible}")
    if not feasible:
        print(f"Violated constraints: {sorted(violated_set)}")
        print(f"Number of violations: {len(violations)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
