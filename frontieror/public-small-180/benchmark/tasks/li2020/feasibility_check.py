#!/usr/bin/env python3
"""
Feasibility checker for the bi-level partial interdiction problem on
capacitated nested hierarchical facilities.

Reference: Forghani, Dehghanian, Salari, Ghiami (2020)
Computers and Operations Research 114 (2020) 104831

Checks all hard constraints from the mathematical formulation.
Constraint numbering (top to bottom in formulation):
  Constraint 1 = eq (2):  Exactly one interdiction level per facility
  Constraint 2 = eq (3):  Interdiction budget
  Constraint 3 = eq (4):  Binary integrality of x_{jk}
  Constraint 4 = eq (6):  Type I demand balance
  Constraint 5 = eq (7):  Type II demand balance
  Constraint 6 = eq (8):  Referral type II balance
  Constraint 7 = eq (9):  Type I capacity at level I facilities
  Constraint 8 = eq (10): Type I capacity at level II facilities
  Constraint 9 = eq (11): Type II capacity at level II facilities
  Constraint 10 = eq (12): Non-negativity of continuous variables
  Constraint 11 = obj (5): Objective consistency -- the reported
                  objective_value must equal the recomputed lower-level
                  LP optimum H(z) (defender's minimum cost) for the
                  reported interdiction pattern. The solution only
                  carries the upper-level interdiction pattern; the
                  defender's reassignment/outsourcing variables are,
                  by definition (eq 5), the optimal lower-level LP
                  solution -- which this checker already solves exactly
                  via linprog. This is therefore a FULL recompute, not
                  a lower bound. Defends against score-gaming exploits
                  that report a fabricated objective_value.
"""

import argparse
import json
import math
import numpy as np
from scipy.optimize import linprog

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('interdiction_pattern', 'objective_value')
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



tol = 1e-5
eps = 1e-5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_interdiction_pattern(sol, nS1, nS2):
    """Parse interdiction pattern from either efficient or gurobi solution format.

    Returns a list of length nS1+nS2 with the interdiction level for each facility.
    """
    nFac = nS1 + nS2
    pat = sol["interdiction_pattern"]

    # Detect format
    sample_key = next(iter(pat))
    if sample_key.startswith("S"):
        # Gurobi format: "S1_0", "S1_1", ..., "S2_0", ...
        levels = [0] * nFac
        for key, val in pat.items():
            parts = key.split("_")
            fac_type = parts[0]  # "S1" or "S2"
            fac_idx = int(parts[1])
            if fac_type == "S1":
                levels[fac_idx] = int(val)
            else:
                levels[nS1 + fac_idx] = int(val)
    else:
        # Efficient format: "0", "1", ..., "9"
        levels = [0] * nFac
        for key, val in pat.items():
            levels[int(key)] = int(val)

    return levels


def add_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs, violation_amount):
    """Record a constraint violation."""
    normalizer = max(abs(rhs), eps)
    ratio = violation_amount / normalizer
    violations_list.append((constraint_idx, message))
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def check_feasibility(inst, sol):
    gp_ = inst["global_parameters"]
    ip = inst["instance_parameters"]
    inter = inst["interdiction_parameters"]

    theta = gp_["theta"]
    sigma = gp_["sigma"]
    alpha = gp_["alpha"]
    beta = gp_["beta"]
    gamma_ = gp_["gamma"]
    alpha_p = gp_["alpha_prime"]
    beta_p = gp_["beta_prime"]
    gamma_p = gp_["gamma_prime"]

    nI = ip["num_customers"]
    nS1 = ip["num_level1_facilities"]
    nS2 = ip["num_level2_facilities"]
    k_max = ip["k_max"]
    B = ip["total_interdiction_budget"]
    nFac = nS1 + nS2

    # Interdiction costs
    h1 = {0: 0.0}
    for ks, v in inter["interdiction_cost_level1_h1"].items():
        h1[int(ks)] = v
    h2 = {0: 0.0}
    for ks, v in inter["interdiction_cost_level2_h2"].items():
        h2[int(ks)] = v

    # Capacity reduction ratios
    d1 = {0: 0.0}
    for ks, v in inter["capacity_reduction_level1_d1"].items():
        d1[int(ks)] = v
    d2 = {0: 0.0}
    for ks, v in inter["capacity_reduction_level2_d2"].items():
        d2[int(ks)] = v

    # Data arrays
    z = np.array(inst["customers"]["demands"])
    c1_s1 = np.array(inst["level1_facilities"]["capacity_type1"])
    c1_s2 = np.array(inst["level2_facilities"]["capacity_type1"])
    c2_s2 = np.array(inst["level2_facilities"]["capacity_type2"])

    l_cs1 = np.array(inst["distances"]["customer_to_level1_facility"])
    l_cs2 = np.array(inst["distances"]["customer_to_level2_facility"])
    l_s1s2 = np.array(inst["distances"]["level1_to_level2_facility"])

    # S2-to-S2 distances
    s2x = np.array(inst["level2_facilities"]["x_coordinates"])
    s2y = np.array(inst["level2_facilities"]["y_coordinates"])
    l_s2s2 = np.sqrt(
        np.subtract.outer(s2x, s2x) ** 2 + np.subtract.outer(s2y, s2y) ** 2
    )
    l_fac_s2 = np.vstack([l_s1s2, l_s2s2])

    # Parse interdiction pattern
    pattern = parse_interdiction_pattern(sol, nS1, nS2)

    violations_list = []
    magnitudes_list = []

    # =========================================================
    # Constraint 1 (eq 2): sum_k x_{jk} = 1 for all j
    # Each facility must have exactly one interdiction level.
    # Since we parsed pattern as a single level per facility,
    # we check that each level is in {0, ..., k_max}.
    # =========================================================
    for j in range(nFac):
        k = pattern[j]
        if k < 0 or k > k_max:
            lhs_val = k
            rhs_val = k_max
            viol_amt = abs(k - (k_max if k > k_max else 0))
            fac_label = f"S1_{j}" if j < nS1 else f"S2_{j - nS1}"
            add_violation(violations_list, magnitudes_list, 1,
                          f"Facility {fac_label} has invalid interdiction level {k} (must be 0..{k_max})",
                          lhs_val, rhs_val, viol_amt)

    # Also check that every facility has exactly one level assigned
    pat_dict = sol["interdiction_pattern"]
    expected_keys_count = nFac
    sample_key = next(iter(pat_dict))
    if sample_key.startswith("S"):
        actual_count = len(pat_dict)
    else:
        actual_count = len(pat_dict)
    if actual_count != expected_keys_count:
        viol_amt = abs(actual_count - expected_keys_count)
        add_violation(violations_list, magnitudes_list, 1,
                      f"Expected {expected_keys_count} facility entries in interdiction pattern, got {actual_count}",
                      float(actual_count), float(expected_keys_count), float(viol_amt))

    # =========================================================
    # Constraint 2 (eq 3): Budget constraint
    # sum_{j in S1} h1_k * x_{jk} + sum_{j in S2} h2_k * x_{jk} <= B
    # =========================================================
    total_cost = 0.0
    for j in range(nS1):
        total_cost += h1.get(pattern[j], 0.0)
    for j in range(nS2):
        total_cost += h2.get(pattern[nS1 + j], 0.0)

    lhs_budget = total_cost
    rhs_budget = B
    viol_budget = max(0.0, lhs_budget - rhs_budget)
    if viol_budget > tol:
        add_violation(violations_list, magnitudes_list, 2,
                      f"Interdiction budget exceeded: cost {lhs_budget:.2f} > budget {rhs_budget:.2f}",
                      lhs_budget, rhs_budget, viol_budget)

    # =========================================================
    # Constraint 3 (eq 4): Binary integrality x_{jk} in {0,1}
    # Since pattern stores the chosen level as an integer, this
    # is inherently satisfied. We verify values are integers.
    # =========================================================
    for j in range(nFac):
        k = pattern[j]
        if not isinstance(k, int) and abs(k - round(k)) > tol:
            viol_amt = abs(k - round(k))
            fac_label = f"S1_{j}" if j < nS1 else f"S2_{j - nS1}"
            add_violation(violations_list, magnitudes_list, 3,
                          f"Facility {fac_label} interdiction level {k} is not integer",
                          float(k), float(round(k)), viol_amt)

    # =========================================================
    # Constraints 4-10 (eqs 6-12): Lower-level constraints
    # Solve the lower-level LP and check constraints from solution.
    # =========================================================
    # Remaining capacities after interdiction
    rem_c1_s1 = np.array([
        c1_s1[j] * (1 - d1.get(pattern[j], 0.0))
        for j in range(nS1)
    ])
    rem_c1_s2 = np.array([
        c1_s2[j] * (1 - d2.get(pattern[nS1 + j], 0.0))
        for j in range(nS2)
    ])
    rem_c2_s2 = np.array([
        c2_s2[j] * (1 - d2.get(pattern[nS1 + j], 0.0))
        for j in range(nS2)
    ])

    # Variable layout (same as efficient_algorithm.py)
    off_u1 = 0
    off_u2 = nI * nFac
    off_u3 = off_u2 + nI * nS2
    off_o1 = off_u3 + nFac * nS2
    off_o2 = off_o1 + nI
    off_o3 = off_o2 + nI
    n_vars = off_o3 + nFac

    # Cost vector (eq 5)
    c_vec = np.zeros(n_vars)
    for i in range(nI):
        for j in range(nS1):
            c_vec[off_u1 + i * nFac + j] = l_cs1[i][j] * alpha
        for j in range(nS2):
            c_vec[off_u1 + i * nFac + nS1 + j] = l_cs2[i][j] * beta
    for i in range(nI):
        for j in range(nS2):
            c_vec[off_u2 + i * nS2 + j] = l_cs2[i][j] * beta
    for j in range(nFac):
        for f in range(nS2):
            c_vec[off_u3 + j * nS2 + f] = l_fac_s2[j][f] * gamma_
    for i in range(nI):
        c_vec[off_o1 + i] = alpha_p + sigma * gamma_p
    for i in range(nI):
        c_vec[off_o2 + i] = beta_p
    for j in range(nFac):
        c_vec[off_o3 + j] = gamma_p

    # Equality constraints: (6), (7), (8)
    n_eq = nI + nI + nFac
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)
    row = 0

    # (6): theta*z_i = sum_j u1[i,j] + o1[i]
    for i in range(nI):
        for j in range(nFac):
            A_eq[row, off_u1 + i * nFac + j] = 1.0
        A_eq[row, off_o1 + i] = 1.0
        b_eq[row] = theta * z[i]
        row += 1

    # (7): (1-theta)*z_i = sum_{j in S2} u2[i,j] + o2[i]
    for i in range(nI):
        for j in range(nS2):
            A_eq[row, off_u2 + i * nS2 + j] = 1.0
        A_eq[row, off_o2 + i] = 1.0
        b_eq[row] = (1 - theta) * z[i]
        row += 1

    # (8): sigma*sum_i u1[i,j] = sum_f u3[j,f] + o3[j]
    for j in range(nFac):
        for i in range(nI):
            A_eq[row, off_u1 + i * nFac + j] = sigma
        for f in range(nS2):
            A_eq[row, off_u3 + j * nS2 + f] = -1.0
        A_eq[row, off_o3 + j] = -1.0
        b_eq[row] = 0.0
        row += 1

    # Inequality constraints: (9), (10), (11)
    n_ineq = nS1 + nS2 + nS2
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    row = 0

    # (9): sum_i u1[i,j] <= rem_c1_s1[j]
    for j in range(nS1):
        for i in range(nI):
            A_ub[row, off_u1 + i * nFac + j] = 1.0
        b_ub[row] = rem_c1_s1[j]
        row += 1

    # (10): sum_i u1[i,nS1+j] <= rem_c1_s2[j]
    for j in range(nS2):
        for i in range(nI):
            A_ub[row, off_u1 + i * nFac + nS1 + j] = 1.0
        b_ub[row] = rem_c1_s2[j]
        row += 1

    # (11): sum_i u2[i,j] + sum_f u3[f,j] <= rem_c2_s2[j]
    for j in range(nS2):
        for i in range(nI):
            A_ub[row, off_u2 + i * nS2 + j] = 1.0
        for f in range(nFac):
            A_ub[row, off_u3 + f * nS2 + j] = 1.0
        b_ub[row] = rem_c2_s2[j]
        row += 1

    bounds = [(0, None)] * n_vars
    result = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if not result.success:
        # LP infeasible means lower-level constraints can't all be satisfied
        add_violation(violations_list, magnitudes_list, 4,
                      "Lower-level LP is infeasible: type I demand balance (eq 6) cannot be satisfied",
                      0.0, 0.0, 1.0)
        add_violation(violations_list, magnitudes_list, 5,
                      "Lower-level LP is infeasible: type II demand balance (eq 7) cannot be satisfied",
                      0.0, 0.0, 1.0)
    else:
        x_sol = result.x

        # ---- Constraint 4 (eq 6): theta*z_i = sum_j u1[i,j] + o1[i] ----
        for i in range(nI):
            lhs = sum(x_sol[off_u1 + i * nFac + j] for j in range(nFac)) + x_sol[off_o1 + i]
            rhs = theta * z[i]
            viol = abs(lhs - rhs)
            if viol > tol:
                add_violation(violations_list, magnitudes_list, 4,
                              f"Type I demand balance violated for customer {i}: LHS={lhs:.6f}, RHS={rhs:.6f}",
                              lhs, rhs, viol)

        # ---- Constraint 5 (eq 7): (1-theta)*z_i = sum_{j in S2} u2[i,j] + o2[i] ----
        for i in range(nI):
            lhs = sum(x_sol[off_u2 + i * nS2 + j] for j in range(nS2)) + x_sol[off_o2 + i]
            rhs = (1 - theta) * z[i]
            viol = abs(lhs - rhs)
            if viol > tol:
                add_violation(violations_list, magnitudes_list, 5,
                              f"Type II demand balance violated for customer {i}: LHS={lhs:.6f}, RHS={rhs:.6f}",
                              lhs, rhs, viol)

        # ---- Constraint 6 (eq 8): sigma*sum_i u1[i,j] = sum_f u3[j,f] + o3[j] ----
        for j in range(nFac):
            lhs = sigma * sum(x_sol[off_u1 + i * nFac + j] for i in range(nI))
            rhs = sum(x_sol[off_u3 + j * nS2 + f] for f in range(nS2)) + x_sol[off_o3 + j]
            viol = abs(lhs - rhs)
            if viol > tol:
                fac_label = f"S1_{j}" if j < nS1 else f"S2_{j - nS1}"
                add_violation(violations_list, magnitudes_list, 6,
                              f"Referral balance violated for facility {fac_label}: LHS={lhs:.6f}, RHS={rhs:.6f}",
                              lhs, rhs, viol)

        # ---- Constraint 7 (eq 9): sum_i u1[i,j] <= rem_c1_s1[j] for j in S1 ----
        for j in range(nS1):
            lhs = sum(x_sol[off_u1 + i * nFac + j] for i in range(nI))
            rhs = rem_c1_s1[j]
            viol = max(0.0, lhs - rhs)
            if viol > tol:
                add_violation(violations_list, magnitudes_list, 7,
                              f"Type I capacity exceeded at S1 facility {j}: allocated={lhs:.2f}, capacity={rhs:.2f}",
                              lhs, rhs, viol)

        # ---- Constraint 8 (eq 10): sum_i u1[i,nS1+j] <= rem_c1_s2[j] for j in S2 ----
        for j in range(nS2):
            lhs = sum(x_sol[off_u1 + i * nFac + nS1 + j] for i in range(nI))
            rhs = rem_c1_s2[j]
            viol = max(0.0, lhs - rhs)
            if viol > tol:
                add_violation(violations_list, magnitudes_list, 8,
                              f"Type I capacity exceeded at S2 facility {j}: allocated={lhs:.2f}, capacity={rhs:.2f}",
                              lhs, rhs, viol)

        # ---- Constraint 9 (eq 11): sum_i u2[i,j] + sum_f u3[f,j] <= rem_c2_s2[j] ----
        for j in range(nS2):
            lhs = (sum(x_sol[off_u2 + i * nS2 + j] for i in range(nI))
                   + sum(x_sol[off_u3 + f * nS2 + j] for f in range(nFac)))
            rhs = rem_c2_s2[j]
            viol = max(0.0, lhs - rhs)
            if viol > tol:
                add_violation(violations_list, magnitudes_list, 9,
                              f"Type II capacity exceeded at S2 facility {j}: allocated={lhs:.2f}, capacity={rhs:.2f}",
                              lhs, rhs, viol)

        # ---- Constraint 10 (eq 12): Non-negativity ----
        min_val = np.min(x_sol)
        if min_val < -tol:
            viol = abs(min_val)
            idx = np.argmin(x_sol)
            add_violation(violations_list, magnitudes_list, 10,
                          f"Non-negativity violated: variable index {idx} has value {min_val:.6e}",
                          float(min_val), 0.0, viol)

        # ---- Constraint 11 (obj, eq 5): Objective consistency ----
        # The reported objective_value is, by definition (eq 5), the
        # defender's minimum total cost H(z) for the chosen interdiction
        # pattern. The lower-level problem is an LP whose optimum we just
        # computed exactly above -- result.fun. The defender's
        # reassignment/outsourcing variables are NOT in the solution
        # (only the interdiction pattern is), but they are determined
        # deterministically as the optimal lower-level LP solution, so
        # this is a FULL recompute. Reject when the reported value
        # disagrees with the recomputed optimum (defends against
        # score-gaming exploits that fabricate objective_value).
        reported_obj = sol.get("objective_value")
        if reported_obj is not None:
            try:
                reported = float(reported_obj)
            except (TypeError, ValueError):
                reported = None
            if reported is not None and math.isfinite(reported):
                true_obj = float(result.fun)
                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with a 1e-3 absolute floor.
                obj_tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    add_violation(violations_list, magnitudes_list, 11,
                                  f"Objective consistency violated: reported objective_value="
                                  f"{reported} differs from recomputed lower-level LP optimum "
                                  f"H(z)={true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                                  reported, true_obj, abs_diff)
            elif reported is not None:
                # Non-finite reported objective (inf / nan) is never a
                # valid cost -- treat as an objective inconsistency.
                true_obj = float(result.fun)
                add_violation(violations_list, magnitudes_list, 11,
                              f"Objective consistency violated: reported objective_value="
                              f"{reported} is not a finite number; recomputed lower-level "
                              f"LP optimum H(z)={true_obj}",
                              0.0, true_obj, abs(true_obj))

    # =========================================================
    # Build output
    # =========================================================
    violated_indices = sorted(set(v[0] for v in violations_list))
    violation_messages = []
    for ci in violated_indices:
        msgs = [v[1] for v in violations_list if v[0] == ci]
        if len(msgs) <= 3:
            violation_messages.extend(msgs)
        else:
            violation_messages.append(msgs[0])
            violation_messages.append(f"... and {len(msgs) - 1} more violations for constraint {ci}")

    feasible = len(violated_indices) == 0

    output = {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": magnitudes_list if not feasible else [],
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for bi-level partial interdiction on hierarchical facilities")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    inst = load_json(args.instance_path)
    sol = load_json(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return

    result = check_feasibility(inst, sol)

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
