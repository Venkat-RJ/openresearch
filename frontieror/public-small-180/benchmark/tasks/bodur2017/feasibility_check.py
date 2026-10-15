"""
Feasibility checker for the Stochastic Capacitated Facility Location (CAP) problem.
Based on formulation (15) from Bodur, Dash, Günlük, Luedtke (2017).

Constraints numbered top-to-bottom from the formulation section:
  Constraint 1 (15b): sum_i y^k_{ij} >= lambda^k_j,  for all j in J, k in K
  Constraint 2 (15c): sum_j y^k_{ij} <= s_i * x_i,   for all i in I, k in K
  Constraint 3 (15d): sum_i s_i * x_i >= max_k sum_j lambda^k_j
  Constraint 4 (15e): x in {0,1}^|I|,  y >= 0
  Constraint 5 (Tier C obj-consistency): reported objective_value must equal the
      recomputed objective sum_i f_i*x_i + sum_k p_k*sum_{i,j} q_{ij}*y^k_{ij}.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'open_facilities', 'x', 'y')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('open_facilities', 'y')
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



def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []

    I = instance["num_facilities"]
    J = instance["num_customers"]
    K = instance["num_scenarios"]

    facilities = instance["facilities"]
    scenarios = instance["scenarios"]

    s_cap = [facilities[i]["capacity"] for i in range(I)]
    lam = [[scenarios[k]["demands"][j] for j in range(J)] for k in range(K)]

    # Parse x values
    x_sol = solution.get("x", {})
    x = [0.0] * I
    for key, val in x_sol.items():
        x[int(key)] = val

    # Parse y values (may not exist for efficient solutions)
    y_raw = solution.get("y", None)
    has_y = y_raw is not None and len(y_raw) > 0

    # Build y[k][i][j] sparse structure
    y = {}
    if has_y:
        for k_str, scen_data in y_raw.items():
            k_idx = int(k_str)
            y[k_idx] = {}
            for i_str, row_data in scen_data.items():
                i_idx = int(i_str)
                y[k_idx][i_idx] = {}
                for j_str, val in row_data.items():
                    y[k_idx][i_idx][int(j_str)] = val

    # -------------------------------------------------------------------------
    # Constraint 1 (15b): sum_i y^k_{ij} >= lambda^k_j, for all j, k
    # This is a >= constraint. violation_amount = max(0, RHS - LHS)
    # -------------------------------------------------------------------------
    if has_y:
        for k in range(K):
            for j in range(J):
                lhs = 0.0
                scen_y = y.get(k, {})
                for i in range(I):
                    lhs += scen_y.get(i, {}).get(j, 0.0)
                rhs = lam[k][j]
                violation_amount = rhs - lhs
                if violation_amount > tol:
                    normalizer = max(abs(rhs), eps)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint 1 (15b): Demand not met for customer {j}, "
                        f"scenario {k}: sum_i y[{k}][i][{j}] = {lhs:.6f} < "
                        f"lambda[{k}][{j}] = {rhs:.6f}"
                    )
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": violation_amount,
                        "normalizer": normalizer,
                        "ratio": ratio,
                    })

    # -------------------------------------------------------------------------
    # Constraint 2 (15c): sum_j y^k_{ij} <= s_i * x_i, for all i, k
    # This is a <= constraint. violation_amount = max(0, LHS - RHS)
    # -------------------------------------------------------------------------
    if has_y:
        for k in range(K):
            for i in range(I):
                lhs = 0.0
                scen_y = y.get(k, {})
                fac_y = scen_y.get(i, {})
                for j_idx, val in fac_y.items():
                    lhs += val
                rhs = s_cap[i] * x[i]
                violation_amount = lhs - rhs
                if violation_amount > tol:
                    normalizer = max(abs(rhs), eps)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint 2 (15c): Capacity exceeded for facility {i}, "
                        f"scenario {k}: sum_j y[{k}][{i}][j] = {lhs:.6f} > "
                        f"s[{i}]*x[{i}] = {rhs:.6f}"
                    )
                    violation_magnitudes.append({
                        "constraint": 2,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": violation_amount,
                        "normalizer": normalizer,
                        "ratio": ratio,
                    })

    # -------------------------------------------------------------------------
    # Constraint 3 (15d): sum_i s_i * x_i >= max_k sum_j lambda^k_j
    # This is a >= constraint. violation_amount = max(0, RHS - LHS)
    # -------------------------------------------------------------------------
    lhs_3 = sum(s_cap[i] * x[i] for i in range(I))
    max_total_demand = max(sum(lam[k][j] for j in range(J)) for k in range(K))
    rhs_3 = max_total_demand
    violation_amount_3 = rhs_3 - lhs_3
    if violation_amount_3 > tol:
        normalizer = max(abs(rhs_3), eps)
        ratio = violation_amount_3 / normalizer
        violations.append(
            f"Constraint 3 (15d): Total open capacity insufficient: "
            f"sum_i s_i*x_i = {lhs_3:.6f} < max_k sum_j lambda^k_j = {rhs_3:.6f}"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs_3,
            "rhs": rhs_3,
            "raw_excess": violation_amount_3,
            "normalizer": normalizer,
            "ratio": ratio,
        })

    # -------------------------------------------------------------------------
    # Constraint 4 (15e): x in {0,1}^|I|, y >= 0
    # For x binary: violation_amount = min(|x_i|, |x_i - 1|) if not 0 or 1
    # For y non-negative: violation_amount = max(0, -y^k_{ij})
    # -------------------------------------------------------------------------
    # Check x binary
    for i in range(I):
        xi = x[i]
        dist_to_binary = min(abs(xi), abs(xi - 1.0))
        if dist_to_binary > tol:
            rhs = round(xi)  # nearest binary value
            violation_amount = dist_to_binary
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint 4 (15e): x[{i}] = {xi} is not binary"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": xi,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # Check y >= 0
    if has_y:
        for k_idx, scen_data in y.items():
            for i_idx, row_data in scen_data.items():
                for j_idx, val in row_data.items():
                    if val < -tol:
                        violation_amount = -val
                        rhs = 0.0
                        normalizer = eps
                        ratio = violation_amount / normalizer
                        violations.append(
                            f"Constraint 4 (15e): y[{k_idx}][{i_idx}][{j_idx}] "
                            f"= {val:.6f} < 0"
                        )
                        violation_magnitudes.append({
                            "constraint": 4,
                            "lhs": val,
                            "rhs": rhs,
                            "raw_excess": violation_amount,
                            "normalizer": normalizer,
                            "ratio": ratio,
                        })

    # -------------------------------------------------------------------------
    # Constraint 5 (Tier C): objective consistency.
    # Recompute obj = sum_i f_i*x_i + sum_k p_k * sum_{i,j} q_{ij} * y[k,i,j]
    # and reject when reported objective_value disagrees by more than tolerance.
    # Only applied when y is present (efficient checker mode without y cannot
    # verify the second-stage transportation cost).
    # -------------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if has_y and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            f_cost = [facilities[i]["opening_cost"] for i in range(I)]
            p_k = [scenarios[k]["probability"] for k in range(K)]
            transport = instance["transportation_costs"]

            opening_part = sum(f_cost[i] * float(x[i]) for i in range(I))
            transport_part = 0.0
            for k_idx, scen_data in y.items():
                if k_idx < 0 or k_idx >= K:
                    continue
                p = p_k[k_idx]
                for i_idx, row_data in scen_data.items():
                    if i_idx < 0 or i_idx >= I:
                        continue
                    q_row = transport[i_idx]
                    for j_idx, val in row_data.items():
                        if j_idx < 0 or j_idx >= J:
                            continue
                        transport_part += p * q_row[j_idx] * float(val)
            true_obj = opening_part + transport_part

            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint 5 (obj): reported objective_value={reported} "
                    f"differs from recomputed sum_i f_i*x_i + sum_k p_k*"
                    f"sum_{{i,j}} q_{{ij}}*y[k,i,j]={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })
        elif reported is not None and not math.isfinite(reported):
            # Non-finite (inf, nan, sys.float_info.max-ish overflow path):
            # always flag as obj-inconsistent regardless of y.
            violations.append(
                f"Constraint 5 (obj): reported objective_value={reported} is "
                f"not a finite number."
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": reported,
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": eps,
                "ratio": float("inf"),
            })

    # Build result
    violated_constraints = sorted(set(vm["constraint"] for vm in violation_magnitudes))
    feasible = len(violated_constraints) == 0

    # Aggregate violation messages per constraint
    aggregated_violations = []
    for c in violated_constraints:
        msgs = [v for v, vm in zip(violations, violation_magnitudes) if vm["constraint"] == c]
        if len(msgs) <= 3:
            aggregated_violations.append("; ".join(msgs))
        else:
            aggregated_violations.append(
                f"{msgs[0]}; ... ({len(msgs)} total violations for constraint {c})"
            )

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": aggregated_violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Stochastic CAP (Bodur et al. 2017, formulation 15)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
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

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    n_violations = len(result["violation_magnitudes"])
    print(f"{status} ({n_violations} violation(s) across "
          f"{len(result['violated_constraints'])} constraint(s))")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
