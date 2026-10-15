#!/usr/bin/env python3
"""
Feasibility checker for the Stochastic Capacitated Facility Location (SFL) problem.
Rahmaniani et al. (2020), Appendix D.2, Equations (D.4)-(D.7).

Constraint numbering (top to bottom in formulation section):
  Constraint 1: (D.5) sum_i x_{ij}^s >= d_j^s       for all j in M, s in S  [demand satisfaction, >=]
  Constraint 2: (D.6) sum_j x_{ij}^s <= u_i * y_i    for all i in N, s in S  [capacity, <=]
  Constraint 3: (D.7) sum_i u_i * y_i >= max_s sum_j d_j^s                   [complete recourse, >=]
  Constraint 4: (D.4) Objective consistency: reported objective_value must equal
                f^T y + sum_s p_s * min_x { sum_{i,j} c_{ij} x_{ij}^s : D.5, D.6, x>=0 }
                (Tier C defense against fabricated objective values.)

Since solutions only contain y (facility opening decisions) and not x (flow variables),
constraints 1 and 2 are checked by solving an LP per scenario to determine if feasible
x exists given y. If demand cannot be fully met, constraint 1 (D.5) is reported violated.
Constraint 2 (D.6) is verified explicitly on the LP-optimal x.
Constraint 3 (D.7) is checked directly from y.

Constraint 4 recomputes the true objective by solving a cost-minimization transportation
LP per scenario (the recourse problem) and comparing the expected total cost against the
reported objective_value.
"""

import json
import argparse
import numpy as np
from scipy.optimize import linprog

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'y')
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



def check_feasibility(data, sol):
    tol = 1e-5
    eps = 1e-5

    N = data["num_facilities"]
    M = data["num_customers"]
    S = data["num_scenarios"]
    cap = data["facilities"]["capacities"]
    demands = data["scenarios"]["demands"]
    y = sol["y"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 3 (D.7): sum_i u_i * y_i >= max_s sum_j d_j^s
    # Type: >= constraint. Violation = RHS - LHS when RHS > LHS.
    # ------------------------------------------------------------------
    D_max = max(sum(demands[s][j] for j in range(M)) for s in range(S))
    lhs_3 = sum(cap[i] * y[i] for i in range(N))
    rhs_3 = float(D_max)
    viol_3 = rhs_3 - lhs_3
    if viol_3 > tol:
        violated_constraints.add(3)
        violations.append(
            f"Constraint 3 (D.7) violated: sum(u_i*y_i)={lhs_3:.6f} < "
            f"max_s(sum_j d_j^s)={rhs_3:.6f}"
        )
        normalizer = max(abs(rhs_3), eps)
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs_3,
            "rhs": rhs_3,
            "raw_excess": viol_3,
            "normalizer": normalizer,
            "ratio": viol_3 / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraints 1 (D.5) and 2 (D.6): checked via LP per scenario.
    #
    # For each scenario s we solve:
    #   min  sum_j slack_j
    #   s.t. sum_i x_{ij} + slack_j >= d_j^s   for all j   (D.5 relaxed)
    #        sum_j x_{ij}          <= u_i*y_i   for all i   (D.6 enforced)
    #        x >= 0, slack >= 0
    #
    # If total slack > 0, constraint 1 (D.5) is violated for that scenario.
    # Constraint 2 (D.6) is then verified explicitly on the found x.
    # ------------------------------------------------------------------
    feasibility_ok_per_scenario = [True] * S
    for s in range(S):
        d_s = demands[s]
        n_x = N * M
        n_slack = M
        n_vars = n_x + n_slack

        # Objective: minimize total demand shortfall
        c_obj = np.zeros(n_vars)
        c_obj[n_x:] = 1.0

        A_ub_rows = []
        b_ub_vals = []

        # D.5 (>=): sum_i x_{ij} + slack_j >= d_j^s
        #   => -sum_i x_{ij} - slack_j <= -d_j^s
        for j in range(M):
            row = np.zeros(n_vars)
            for i in range(N):
                row[i * M + j] = -1.0
            row[n_x + j] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(-float(d_s[j]))

        # D.6 (<=): sum_j x_{ij} <= u_i * y_i
        for i in range(N):
            row = np.zeros(n_vars)
            for j in range(M):
                row[i * M + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(float(cap[i] * y[i]))

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds = [(0, None)] * n_vars

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            # Should not happen with slack variables, but handle defensively
            violated_constraints.add(1)
            violations.append(
                f"Constraint 1 (D.5) violated: LP solver failed for scenario {s}"
            )
            feasibility_ok_per_scenario[s] = False
            continue

        # Check D.5 violations via slack values
        if res.fun > tol:
            violated_constraints.add(1)
            feasibility_ok_per_scenario[s] = False
            slack = res.x[n_x:]
            shortfall_details = []
            for j in range(M):
                if slack[j] > tol:
                    # LHS = sum_i x_{ij}^s = d_j^s - slack_j (actual flow)
                    lhs_val = float(d_s[j]) - slack[j]
                    rhs_val = float(d_s[j])
                    viol_amt = slack[j]
                    normalizer = max(abs(rhs_val), eps)
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": viol_amt,
                        "normalizer": normalizer,
                        "ratio": viol_amt / normalizer,
                    })
                    shortfall_details.append(f"customer {j} short by {slack[j]:.6f}")
            violations.append(
                f"Constraint 1 (D.5) violated in scenario {s}: "
                f"total shortfall={res.fun:.6f} ({'; '.join(shortfall_details)})"
            )

        # Explicitly verify D.6 on the LP-optimal x
        x_vals = res.x[:n_x].reshape(N, M)
        for i in range(N):
            flow_i = float(np.sum(x_vals[i, :]))
            cap_rhs = float(cap[i] * y[i])
            viol_6 = flow_i - cap_rhs  # <= constraint: violation when LHS > RHS
            if viol_6 > tol:
                violated_constraints.add(2)
                normalizer = max(abs(cap_rhs), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": flow_i,
                    "rhs": cap_rhs,
                    "raw_excess": viol_6,
                    "normalizer": normalizer,
                    "ratio": viol_6 / normalizer,
                })
                violations.append(
                    f"Constraint 2 (D.6) violated: facility {i}, scenario {s}, "
                    f"flow={flow_i:.6f} > capacity={cap_rhs:.6f}"
                )

    # ------------------------------------------------------------------
    # Constraint 4: Objective consistency (Tier C anti-exploit defense).
    #
    # Recompute the optimal objective for the given y by solving the recourse
    # problem (a transportation LP) per scenario, then compare to the reported
    # objective_value. The full-recompute formula:
    #
    #   true_obj = sum_i f_i * y_i
    #            + sum_s p_s * min_{x>=0} { sum_{i,j} c_{ij} x_{ij}^s
    #                                       : sum_i x_{ij}^s >= d_j^s (D.5),
    #                                         sum_j x_{ij}^s <= u_i*y_i (D.6) }
    #
    # Skip the check if any structural constraint (1, 2, or 3) already
    # marks the solution infeasible -- in that case the obj is undefined,
    # and the existing violations already cause feasible=False.
    # ------------------------------------------------------------------
    structural_violation = bool(violated_constraints & {1, 2, 3})
    reported_obj_raw = sol.get("objective_value")
    try:
        reported_obj = float(reported_obj_raw) if reported_obj_raw is not None else None
    except (TypeError, ValueError):
        reported_obj = None

    if reported_obj is not None and not structural_violation:
        fixed_costs = data["facilities"]["fixed_costs"]
        routing_costs = data["routing_costs"]
        probs = data["scenarios"]["probabilities"]

        setup_cost = float(sum(fixed_costs[i] * y[i] for i in range(N)))

        # Build the cost vector for the routing LP (same A_ub structure as above
        # but without slack variables, since structural feasibility is already
        # confirmed for every scenario).
        n_x = N * M
        c_route = np.empty(n_x)
        for i in range(N):
            for j in range(M):
                c_route[i * M + j] = float(routing_costs[i][j])

        # D.6 capacity rows are scenario-independent (depend only on y).
        cap_rows = np.zeros((N, n_x))
        cap_rhs_arr = np.empty(N)
        for i in range(N):
            for j in range(M):
                cap_rows[i, i * M + j] = 1.0
            cap_rhs_arr[i] = float(cap[i] * y[i])

        # D.5 demand rows depend on scenario; build template once.
        demand_template = np.zeros((M, n_x))
        for j in range(M):
            for i in range(N):
                demand_template[j, i * M + j] = -1.0

        recompute_failed_for_scenario = False
        expected_recourse = 0.0
        for s in range(S):
            d_s = demands[s]
            demand_rhs = np.array([-float(d_s[j]) for j in range(M)])
            A_ub_s = np.vstack([demand_template, cap_rows])
            b_ub_s = np.concatenate([demand_rhs, cap_rhs_arr])
            res = linprog(
                c_route,
                A_ub=A_ub_s,
                b_ub=b_ub_s,
                bounds=[(0, None)] * n_x,
                method="highs",
            )
            if not res.success or res.fun is None:
                recompute_failed_for_scenario = True
                break
            expected_recourse += float(probs[s]) * float(res.fun)

        if not recompute_failed_for_scenario:
            true_obj = setup_cost + expected_recourse
            abs_diff = abs(reported_obj - true_obj)
            # 0.1% relative tolerance with 1.0 absolute floor (obj is on the
            # order of 1e3 to 1e5 cost units; floating-point recompute noise
            # is well under 1).
            tol_obj = max(1.0, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_constraints.add(4)
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported_obj),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": float(abs_diff) / normalizer,
                })
                violations.append(
                    f"Constraint 4 (D.4) violated: reported objective_value="
                    f"{reported_obj} differs from recomputed setup+E[recourse]="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    vc_sorted = sorted(violated_constraints)
    return {
        "feasible": len(vc_sorted) == 0,
        "violated_constraints": vc_sorted,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if violation_magnitudes else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SFL (Rahmaniani et al. 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)
    with open(args.solution_path) as f:
        sol = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
        if _frontieror_contract_result is not None:
            with open(args.result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return

    result = check_feasibility(data, sol)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
