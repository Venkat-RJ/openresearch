#!/usr/bin/env python3
"""
Feasibility checker for scenario reduction solutions from:
"Optimal Scenario Reduction for One- and Two-Stage Robust Optimization"
by Marc Goerigk and Mohammad Khosravi (2022)

Constraints are numbered top-to-bottom across all formulation blocks in math_model.txt:

  Cont formulation (3)-(9):
    Constraint 1  (tag 4):  t*c^i_j <= sum_k sum_ell mu_{ik}*lambda_{k,ell}*c^ell_j  for all i,j
    Constraint 2  (tag 5):  sum_{i} lambda_{ki} = 1                                   for all k
    Constraint 3  (tag 6):  sum_{k} mu_{ik} = 1                                       for all i
    Constraint 4  (tag 7):  t >= 0
    Constraint 5  (tag 8):  lambda_{ki} >= 0                                           for all k,i
    Constraint 6  (tag 9):  mu_{ik} >= 0                                               for all i,k

  IP-mu formulation:
    Constraint 7  (tag 10): t*c^i_j <= sum_ell lambda_{k,ell}*c^ell_j + M*(1-mu_{ik}) for all i,j,k
    Constraint 8  (tag 5):  sum_{i} lambda_{ki} = 1                                   for all k
    Constraint 9  (tag 6):  sum_{k} mu_{ik} = 1                                       for all i
    Constraint 10 (tag 7):  t >= 0
    Constraint 11 (tag 8):  lambda_{ki} >= 0                                           for all k,i
    Constraint 12:          mu_{ik} in {0,1}                                           for all i,k

  IP-lambda formulation (11)-(18):
    Constraint 13 (tag 12): t*c^i_j <= sum_ell mu_{i,ell}*c^ell_j                     for all i,j
    Constraint 14 (tag 13): mu_{i,ell} <= lambda_ell                                   for all i,ell
    Constraint 15 (tag 14): sum_{i} lambda_i = K
    Constraint 16 (tag 15): sum_ell mu_{i,ell} = 1                                     for all i
    Constraint 17 (tag 16): lambda_i in {0,1}                                          for all i
    Constraint 18 (tag 17): mu_{i,ell} >= 0                                            for all i,ell
    Constraint 19 (tag 18): t >= 0

  Two-stage IP (31)-(38):
    Constraint 20 (tag 32): t*c^i_j <= sum_ell mu_{i,ell}*c^ell_j                     for all i,j
    Constraint 21 (tag 33): mu_{i,ell} <= lambda_ell                                   for all i,ell
    Constraint 22 (tag 34): sum_{i} lambda_i = K
    Constraint 23 (tag 35): sum_ell mu_{i,ell} = 1                                     for all i
    Constraint 24 (tag 36): lambda_i in {0,1}                                          for all i
    Constraint 25 (tag 37): mu_{i,ell} in {0,1}                                        for all i,ell
    Constraint 26 (tag 38): t >= 0

  Simplified two-stage:
    Constraint 27 (tag 39): t <= sum_ell d_{i,ell}*mu_{i,ell}                          for all i

  Objective consistency:
    Constraint 28:          objective_value == max_{K} t  (across one_stage_IP_mu and two_stage_IP)

Note: Solutions only store summary values (t, approximation_ratio,
selected_scenarios), not full variable values (lambda, mu).  The selected
scenarios are nevertheless a complete certificate for the simplified two-stage
model, so its achieved t is recomputed exactly.  The one-stage entries remain
value-only records because the released schema omits lambda and mu.
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'one_stage_IP_mu', 'two_stage_IP')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('one_stage_IP_mu', 'two_stage_IP')
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



def compute_d_matrix(scenarios, N, n):
    """Compute d_{i,ell} = min_{j in [n]} c^ell_j / c^i_j."""
    d = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for ell in range(N):
            min_ratio = float('inf')
            for j in range(n):
                if scenarios[i][j] > 0:
                    ratio = scenarios[ell][j] / scenarios[i][j]
                    min_ratio = min(min_ratio, ratio)
                else:
                    if scenarios[ell][j] == 0:
                        min_ratio = min(min_ratio, 1.0)
            if min_ratio == float('inf'):
                min_ratio = 0.0
            d[i][ell] = min_ratio
    return d


def add_violation(violated_set, violations_list, magnitudes_list,
                  constraint_idx, msg, lhs, rhs, raw_excess, tol, eps):
    """Record a violation if raw_excess > tol."""
    if raw_excess > tol:
        normalizer = max(abs(rhs), eps)
        violated_set.add(constraint_idx)
        violations_list.append(msg)
        magnitudes_list.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": raw_excess / normalizer,
        })


def check_feasibility(instance, solution):
    scenarios = instance["scenarios"]
    K_values = instance["K_values"]
    N = len(scenarios)
    n = len(scenarios[0])

    tol = 1e-5
    eps = 1e-5

    violated_set = set()
    violations_list = []
    magnitudes_list = []

    expected_k_keys = {str(int(K)) for K in K_values}

    # Both method blocks are part of the released solution interface.  Missing
    # or extra K entries used to let a candidate report only its most convenient
    # subproblem while still passing the top-level objective check.
    for method_name in ("one_stage_IP_mu", "two_stage_IP"):
        method_results = solution.get(method_name, {})
        actual_k_keys = set(method_results) if isinstance(method_results, dict) else set()
        if actual_k_keys != expected_k_keys:
            violated_set.add(28)
            violations_list.append(
                f"{method_name} must contain exactly K_values; "
                f"missing={sorted(expected_k_keys - actual_k_keys)}, "
                f"unexpected={sorted(actual_k_keys - expected_k_keys)}"
            )
            magnitudes_list.append({
                "constraint": 28,
                "lhs": float(len(actual_k_keys)),
                "rhs": float(len(expected_k_keys)),
                "raw_excess": float(len(actual_k_keys ^ expected_k_keys)),
                "normalizer": float(max(len(expected_k_keys), 1)),
                "ratio": float(len(actual_k_keys ^ expected_k_keys))
                         / float(max(len(expected_k_keys), 1)),
            })

    has_ip_mu = "one_stage_IP_mu" in solution
    has_two_stage = "two_stage_IP" in solution

    # =========================================================================
    # One-stage IP-mu formulation: constraints 7-12.  The release now stores
    # the complete mu assignment and sparse lambda weights, so t is checked
    # against the actual decision witness instead of being trusted as a scalar.
    # =========================================================================
    if has_ip_mu:
        for K_str, result in solution["one_stage_IP_mu"].items():
            if K_str not in expected_k_keys or not isinstance(result, dict):
                continue
            K = int(K_str)
            t = result.get("t")
            ratio = result.get("approximation_ratio")
            t_is_valid = (
                isinstance(t, (int, float))
                and not isinstance(t, bool)
                and t == t
                and t not in (float("inf"), float("-inf"))
                and t > 0
            )
            expected_ratio = 1.0 / t if t_is_valid else float("inf")
            ratio_tol = max(1e-6, 1e-6 * abs(expected_ratio))
            if (
                not isinstance(ratio, (int, float))
                or isinstance(ratio, bool)
                or not t_is_valid
                or not (ratio > 0)
                or abs(float(ratio) - expected_ratio) > ratio_tol
            ):
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    10,
                    f"IP-mu K={K}: approximation_ratio={ratio} does not "
                    f"equal 1/t={expected_ratio}",
                    lhs=float(ratio) if isinstance(ratio, (int, float)) else 0.0,
                    rhs=expected_ratio if expected_ratio != float("inf") else 0.0,
                    raw_excess=max(
                        ratio_tol * 2,
                        abs(float(ratio) - expected_ratio)
                        if isinstance(ratio, (int, float))
                        and expected_ratio != float("inf") else 1.0,
                    ),
                    tol=ratio_tol, eps=eps,
                )

            if not t_is_valid:
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    10,
                    f"IP-mu K={K}: t={t} must be a finite positive number",
                    lhs=0.0, rhs=0.0, raw_excess=1.0,
                    tol=tol, eps=eps,
                )
                continue

            raw_assignment = result.get("assignment")
            raw_lambda = result.get("lambda_weights")
            assignment = {}
            assignment_ok = isinstance(raw_assignment, dict)
            if assignment_ok:
                for raw_i, raw_k in raw_assignment.items():
                    try:
                        i = int(raw_i)
                        k = int(raw_k)
                    except (TypeError, ValueError, OverflowError):
                        assignment_ok = False
                        break
                    if (
                        isinstance(raw_k, bool)
                        or i in assignment
                        or i < 0 or i >= N
                        or k < 0 or k >= K
                    ):
                        assignment_ok = False
                        break
                    assignment[i] = k
            assignment_ok = assignment_ok and set(assignment) == set(range(N))
            if not assignment_ok:
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    9,
                    f"IP-mu K={K}: assignment must map every scenario 0..{N-1} "
                    f"exactly once to an aggregate in 0..{K-1}",
                    lhs=float(len(assignment)), rhs=float(N), raw_excess=1.0,
                    tol=tol, eps=eps,
                )

            lambda_rows = {}
            lambda_ok = isinstance(raw_lambda, dict)
            if lambda_ok:
                for raw_k, raw_weights in raw_lambda.items():
                    try:
                        k = int(raw_k)
                    except (TypeError, ValueError, OverflowError):
                        lambda_ok = False
                        break
                    if k in lambda_rows or k < 0 or k >= K or not isinstance(raw_weights, dict):
                        lambda_ok = False
                        break
                    row = {}
                    for raw_i, raw_weight in raw_weights.items():
                        try:
                            i = int(raw_i)
                            weight = float(raw_weight)
                        except (TypeError, ValueError, OverflowError):
                            lambda_ok = False
                            break
                        if (
                            isinstance(raw_weight, bool)
                            or i in row
                            or i < 0 or i >= N
                            or weight != weight
                            or weight in (float("inf"), float("-inf"))
                            or weight < -tol
                        ):
                            lambda_ok = False
                            break
                        row[i] = max(0.0, weight)
                    if not lambda_ok:
                        break
                    lambda_rows[k] = row
            lambda_ok = lambda_ok and set(lambda_rows) == set(range(K))
            if lambda_ok:
                lambda_ok = all(
                    abs(sum(lambda_rows[k].values()) - 1.0) <= 5e-6
                    for k in range(K)
                )
            if not lambda_ok:
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    8,
                    f"IP-mu K={K}: lambda_weights must contain K nonnegative "
                    "convex-combination rows over valid scenario indices",
                    lhs=float(len(lambda_rows)), rhs=float(K), raw_excess=1.0,
                    tol=tol, eps=eps,
                )

            if assignment_ok and lambda_ok:
                aggregates = [
                    [
                        sum(lambda_rows[k].get(ell, 0.0) * scenarios[ell][j]
                            for ell in range(N))
                        for j in range(n)
                    ]
                    for k in range(K)
                ]
                big_m = max(max(row) for row in scenarios)
                max_excess = 0.0
                max_detail = None
                active_ratios = []
                for i in range(N):
                    assigned_k = assignment[i]
                    for k in range(K):
                        for j in range(n):
                            lhs = float(t) * scenarios[i][j]
                            rhs = aggregates[k][j] + big_m * (k != assigned_k)
                            excess = lhs - rhs
                            if excess > max_excess:
                                max_excess = excess
                                max_detail = (i, j, k, lhs, rhs)
                    for j in range(n):
                        coefficient = scenarios[i][j]
                        if coefficient > 0:
                            active_ratios.append(aggregates[assigned_k][j] / coefficient)

                if max_excess > tol:
                    i, j, k, lhs, rhs = max_detail
                    add_violation(
                        violated_set, violations_list, magnitudes_list,
                        7,
                        f"IP-mu K={K}: constraint (10) fails most at "
                        f"i={i}, j={j}, k={k}",
                        lhs=lhs, rhs=rhs, raw_excess=max_excess,
                        tol=tol, eps=eps,
                    )

                # For fixed lambda and mu, maximizing t gives the minimum
                # active component ratio.  Equality prevents a candidate from
                # attaching an arbitrary scalar to an otherwise valid witness.
                achieved_t = min(active_ratios)
                t_tol = max(1e-6, 1e-6 * abs(achieved_t))
                difference = abs(float(t) - achieved_t)
                if difference > t_tol:
                    add_violation(
                        violated_set, violations_list, magnitudes_list,
                        7,
                        f"IP-mu K={K}: reported t={t} does not match "
                        f"witness-derived value {achieved_t}",
                        lhs=float(t), rhs=achieved_t, raw_excess=difference,
                        tol=t_tol, eps=eps,
                    )

    # =========================================================================
    # Two-stage IP formulation: constraints 20-27
    # Stored: t, selected_scenarios per K
    # Verifiable:
    #   Constraint 22 (tag 34): sum lambda_i = K  (i.e. len(selected) == K)
    #   Constraint 24 (tag 36): lambda_i in {0,1} (valid indices, no duplicates)
    #   Constraint 26 (tag 38): t >= 0
    #   Constraint 27 (tag 39): t <= min_i max_{ell in selected} d_{i,ell}
    # =========================================================================
    if has_two_stage:
        d = compute_d_matrix(scenarios, N, n)

        for K_str, result in solution["two_stage_IP"].items():
            K = int(K_str)
            t = result["t"]
            selected = result.get("selected_scenarios", [])

            ratio = result["approximation_ratio"]
            expected_ratio = 1.0 / t if t > 0 else float("inf")
            ratio_tol = max(1e-6, 1e-6 * abs(expected_ratio))
            if (
                not isinstance(ratio, (int, float))
                or isinstance(ratio, bool)
                or not isinstance(t, (int, float))
                or isinstance(t, bool)
                or not (t > 0)
                or not (ratio > 0)
                or abs(float(ratio) - expected_ratio) > ratio_tol
            ):
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    26,
                    f"Two-stage K={K}: approximation_ratio={ratio} does not "
                    f"equal 1/t={expected_ratio}",
                    lhs=float(ratio) if isinstance(ratio, (int, float)) else 0.0,
                    rhs=expected_ratio if expected_ratio != float("inf") else 0.0,
                    raw_excess=max(
                        ratio_tol * 2,
                        abs(float(ratio) - expected_ratio)
                        if isinstance(ratio, (int, float))
                        and expected_ratio != float("inf") else 1.0,
                    ),
                    tol=ratio_tol, eps=eps,
                )

            # ------ Constraint 22 (tag 34): sum lambda_i = K ------
            actual_K = len(selected)
            if actual_K != K:
                diff = abs(actual_K - K)
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    22,
                    f"Two-stage K={K}: |selected_scenarios|={actual_K} != K={K}",
                    lhs=float(actual_K), rhs=float(K),
                    raw_excess=float(diff),
                    tol=tol, eps=eps,
                )

            # ------ Constraint 24 (tag 36): lambda_i in {0,1} ------
            # Check indices are valid integers in [0, N-1] with no duplicates
            for idx in selected:
                if not isinstance(idx, int) or idx < 0 or idx >= N:
                    add_violation(
                        violated_set, violations_list, magnitudes_list,
                        24,
                        f"Two-stage K={K}: invalid scenario index {idx} "
                        f"(must be integer in [0, {N-1}])",
                        lhs=float(idx) if isinstance(idx, (int, float)) else 0.0,
                        rhs=float(N - 1),
                        raw_excess=1.0,
                        tol=tol, eps=eps,
                    )

            if len(selected) != len(set(selected)):
                num_dupes = len(selected) - len(set(selected))
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    24,
                    f"Two-stage K={K}: duplicate indices in selected_scenarios",
                    lhs=float(len(selected)),
                    rhs=float(len(set(selected))),
                    raw_excess=float(num_dupes),
                    tol=tol, eps=eps,
                )

            # ------ Constraint 26 (tag 38): t >= 0 ------
            if t < -tol:
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    26,
                    f"Two-stage K={K}: t={t} violates t >= 0",
                    lhs=0.0, rhs=t, raw_excess=abs(t),
                    tol=tol, eps=eps,
                )

            # ------ Constraint 27 (tag 39) ------
            # Once lambda (the selected scenarios) is fixed, continuous mu may
            # put all mass on the best selected scenario independently for each
            # original scenario.  Hence the exact achieved value is
            # min_i max_{ell in selected} d[i,ell].  Requiring equality makes t a
            # derived value instead of an unchecked scalar.
            selected_is_valid = (
                len(selected) == K
                and len(selected) == len(set(selected))
                and all(isinstance(idx, int) and not isinstance(idx, bool)
                        and 0 <= idx < N for idx in selected)
            )
            if selected_is_valid:
                achieved_t = min(max(d[i][ell] for ell in selected) for i in range(N))
                t_tol = max(1e-6, 1e-6 * abs(achieved_t))
                abs_diff = abs(float(t) - achieved_t)
                if abs_diff > t_tol:
                    add_violation(
                        violated_set, violations_list, magnitudes_list,
                        27,
                        f"Two-stage K={K}: reported t={t} does not match "
                        f"selected-scenario value {achieved_t}",
                        lhs=float(t), rhs=achieved_t, raw_excess=abs_diff,
                        tol=t_tol, eps=eps,
                    )

    # =========================================================================
    # Constraint 28: Objective consistency (Tier C anti-exploit check)
    # objective_value must equal the maximum t across all reported per-K
    # entries in one_stage_IP_mu and two_stage_IP. The solution_schema defines
    # objective_value as the "best scaling factor achieved across all reduction
    # methods and target sizes" -- since t is maximized, "best" == max(t).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    t_values = []
    if has_ip_mu:
        for _K_str, result in solution["one_stage_IP_mu"].items():
            t_val = result.get("t")
            if isinstance(t_val, (int, float)):
                t_values.append(float(t_val))
    if has_two_stage:
        for _K_str, result in solution["two_stage_IP"].items():
            t_val = result.get("t")
            if isinstance(t_val, (int, float)):
                t_values.append(float(t_val))

    if reported_obj is not None and t_values:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = max(t_values)
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-6, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                add_violation(
                    violated_set, violations_list, magnitudes_list,
                    28,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed max_K(t)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})",
                    lhs=reported, rhs=true_obj, raw_excess=abs_diff,
                    tol=obj_tol, eps=eps,
                )

    feasible = len(violated_set) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations_list,
        "violation_magnitudes": magnitudes_list,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for scenario reduction solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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
