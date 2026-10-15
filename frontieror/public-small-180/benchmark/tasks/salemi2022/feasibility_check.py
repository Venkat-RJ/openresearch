"""
Feasibility checker for the two-stage stochastic Unit Commitment Problem (UCP)
from Salemi and Davarnia (2022).

Constraints (numbered top-to-bottom from the formulation):
  1  (1b) Start-up cost linearisation
  2  (1c) Logical relation  y - ybar = x_j - x_{j-1}
  3  (1d) Minimum up-time
  4  (1e) Minimum down-time
  5  (1f) Ramp-up
  6  (1g) Ramp-down
  7  (1h) Generation capacity bounds  m*x <= p <= pbar <= M*x
  8  (1i) Demand satisfaction
  9  (1j) Spinning reserve
 10  (1k) Binary variable domains
 11  (1l) Non-negativity of q, p, pbar
 12  (Tier C obj recompute) Objective consistency:
       reported objective_value must equal
       sum_{i,j}(c_f^i x_j^i + q_j^i) + (1/S) sum_{i,j,s} c_g^i p_j^{i,s}
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'p', 'p_bar', 'q', 'x', 'y', 'y_bar')
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


# Bumped from 1e-5 to 1e-3 to absorb Gurobi numerical noise on large
# instances. 1e-5 was rejecting valid OPT solutions; 1e-3 keeps catch
# for genuine infeasibilities.
TOL = 1e-3
EPS = 1e-3


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalizer(rhs):
    return max(abs(rhs), EPS)


def record(violations_list, magnitudes_list, constraint_idx, msg, lhs, rhs, viol_amount):
    norm = normalizer(rhs)
    violations_list.append((constraint_idx, msg))
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": lhs,
        "rhs": rhs,
        "raw_excess": viol_amount,
        "normalizer": norm,
        "ratio": viol_amount / norm,
    })


def check_feasibility(instance, solution):
    """Return (feasible, violated_constraints, violations, violation_magnitudes)."""

    violations = []   # list of (constraint_idx, message)
    magnitudes = []   # list of dicts

    # --- Unpack instance ---
    n_gen = instance["n_generators"]
    n_scen = instance["n_scenarios"]
    T = instance["T"]
    gens = instance["generators"]
    demand = instance["demand_scenarios"]   # [s][j]
    reserve = instance["reserve_scenarios"] # [s][j]

    # --- Unpack solution ---
    x = solution.get("x")
    y = solution.get("y")
    y_bar = solution.get("y_bar")
    q = solution.get("q")
    p = solution.get("p")         # p[i][s][j]  (gurobi format)
    p_bar = solution.get("p_bar") # p_bar[i][s][j]

    # If the solution is empty / null (e.g. INFEASIBLE solver status), nothing to check
    if x is None:
        return True, [], [], []   # No solution to check; vacuously feasible

    # Derive y, y_bar from x if not provided (using constraint 1c definition)
    if y is None:
        y = []
        for i in range(n_gen):
            yi = []
            for j in range(T):
                x_prev = x[i][j - 1] if j > 0 else 0
                diff = x[i][j] - x_prev
                yi.append(max(diff, 0))
            y.append(yi)
    if y_bar is None:
        y_bar = []
        for i in range(n_gen):
            ybi = []
            for j in range(T):
                x_prev = x[i][j - 1] if j > 0 else 0
                diff = x[i][j] - x_prev
                ybi.append(max(-diff, 0))
            y_bar.append(ybi)

    # Derive q from x if not provided (set to minimum feasible value for checking)
    if q is None:
        q = []
        for i in range(n_gen):
            K = gens[i]["startup_costs_K"]
            qi = []
            for j in range(T):
                max_q = 0.0
                # Paper (1b): k in {1,...,j-1} for 1-indexed j; with 0-indexed
                # j the constraint set is k in {1,...,j} and is empty at j=0.
                max_k = min(j, len(K))
                for k in range(1, max_k + 1):
                    past_sum = sum(x[i][j - h] for h in range(1, k + 1) if j - h >= 0)
                    val = K[k - 1] * (x[i][j] - past_sum)
                    if val > max_q:
                        max_q = val
                qi.append(max(max_q, 0.0))
            q.append(qi)

    # =====================================================================
    # Constraint 1 (1b): Start-up cost linearisation
    #   q_j^i >= K_k^i * (x_j^i - sum_{h=1}^{k} x_{j-h}^i)
    #   for k in {1,...,j-1} (paper, 1-indexed). With 0-indexed j (j' = j-1)
    #   that becomes k in {1,...,j'} and is empty at j'=0 (paper's first
    #   period). gurobi_code.py:88 uses the same convention.
    # =====================================================================
    for i in range(n_gen):
        K = gens[i]["startup_costs_K"]
        for j in range(T):
            max_k = min(j, len(K))
            for k in range(1, max_k + 1):
                past_sum = sum(x[i][j - h] for h in range(1, k + 1) if j - h >= 0)
                rhs_val = K[k - 1] * (x[i][j] - past_sum)
                lhs_val = q[i][j]
                # Constraint: lhs >= rhs  =>  violation if rhs - lhs > tol
                viol = rhs_val - lhs_val
                if viol > TOL:
                    record(violations, magnitudes, 1,
                           f"Startup cost violated: gen {i+1}, period {j+1}, k={k}: "
                           f"q={lhs_val:.4f} < K*(...) = {rhs_val:.4f}",
                           lhs_val, rhs_val, viol)

    # =====================================================================
    # Constraint 2 (1c): Logical relation
    #   y_j^i - ybar_j^i = x_j^i - x_{j-1}^i
    # =====================================================================
    for i in range(n_gen):
        for j in range(T):
            x_prev = x[i][j - 1] if j > 0 else 0
            lhs_val = y[i][j] - y_bar[i][j]
            rhs_val = x[i][j] - x_prev
            viol = abs(lhs_val - rhs_val)
            if viol > TOL:
                record(violations, magnitudes, 2,
                       f"Logical relation violated: gen {i+1}, period {j+1}: "
                       f"y-ybar={lhs_val} != x_j-x_{{j-1}}={rhs_val}",
                       lhs_val, rhs_val, viol)

    # =====================================================================
    # Constraint 3 (1d): Minimum up-time
    #   sum_{j'=j-L+1}^{j} y_{j'}^i <= x_j^i
    #   for j in {L^i,...,T} (1-indexed) => j in {L-1,...,T-1} (0-indexed)
    # =====================================================================
    for i in range(n_gen):
        L = gens[i]["L"]
        for j in range(L - 1, T):
            lhs_val = sum(y[i][jp] for jp in range(j - L + 1, j + 1))
            rhs_val = x[i][j]
            viol = lhs_val - rhs_val
            if viol > TOL:
                record(violations, magnitudes, 3,
                       f"Min up-time violated: gen {i+1}, period {j+1}: "
                       f"sum(y)={lhs_val} > x={rhs_val}",
                       lhs_val, rhs_val, viol)

    # =====================================================================
    # Constraint 4 (1e): Minimum down-time
    #   sum_{j'=j-ell+1}^{j} ybar_{j'}^i <= 1 - x_j^i
    #   for j in {ell^i,...,T} (1-indexed) => j in {ell-1,...,T-1} (0-indexed)
    # =====================================================================
    for i in range(n_gen):
        ell = gens[i]["ell"]
        for j in range(ell - 1, T):
            lhs_val = sum(y_bar[i][jp] for jp in range(j - ell + 1, j + 1))
            rhs_val = 1 - x[i][j]
            viol = lhs_val - rhs_val
            if viol > TOL:
                record(violations, magnitudes, 4,
                       f"Min down-time violated: gen {i+1}, period {j+1}: "
                       f"sum(ybar)={lhs_val} > 1-x={rhs_val}",
                       lhs_val, rhs_val, viol)

    # =====================================================================
    # Constraints 5-9 require p, p_bar (per scenario)
    # =====================================================================
    if p is not None and p_bar is not None:
        for s in range(n_scen):
            for i in range(n_gen):
                RU = gens[i]["RU"]
                SU = gens[i]["SU"]
                RD = gens[i]["RD"]
                SD = gens[i]["SD"]
                m_cap = gens[i]["m"]
                M_cap = gens[i]["M"]

                for j in range(T):
                    x_prev = x[i][j - 1] if j > 0 else 0
                    p_ij = p[i][s][j]
                    p_prev = p[i][s][j - 1] if j > 0 else 0.0
                    pbar_ij = p_bar[i][s][j]

                    # =================================================
                    # Constraint 5 (1f): Ramp-up
                    #   p_j - p_{j-1} <= RU * x_{j-1} + SU * y_j
                    # =================================================
                    lhs_val = p_ij - p_prev
                    rhs_val = RU * x_prev + SU * y[i][j]
                    viol = lhs_val - rhs_val
                    if viol > TOL:
                        record(violations, magnitudes, 5,
                               f"Ramp-up violated: gen {i+1}, period {j+1}, "
                               f"scenario {s+1}: LHS={lhs_val:.4f} > RHS={rhs_val:.4f}",
                               lhs_val, rhs_val, viol)

                    # =================================================
                    # Constraint 6 (1g): Ramp-down
                    #   p_{j-1} - p_j <= RD * x_j + SD * ybar_j
                    # =================================================
                    lhs_val = p_prev - p_ij
                    rhs_val = RD * x[i][j] + SD * y_bar[i][j]
                    viol = lhs_val - rhs_val
                    if viol > TOL:
                        record(violations, magnitudes, 6,
                               f"Ramp-down violated: gen {i+1}, period {j+1}, "
                               f"scenario {s+1}: LHS={lhs_val:.4f} > RHS={rhs_val:.4f}",
                               lhs_val, rhs_val, viol)

                    # =================================================
                    # Constraint 7 (1h): Capacity bounds
                    #   m*x <= p <= pbar <= M*x
                    # =================================================
                    # 7a: m*x <= p  (p >= m*x)
                    lhs_val_7a = m_cap * x[i][j]
                    viol_7a = lhs_val_7a - p_ij
                    if viol_7a > TOL:
                        record(violations, magnitudes, 7,
                               f"Capacity lower bound violated: gen {i+1}, "
                               f"period {j+1}, scenario {s+1}: "
                               f"m*x={lhs_val_7a:.4f} > p={p_ij:.4f}",
                               p_ij, lhs_val_7a, viol_7a)

                    # 7b: p <= pbar
                    viol_7b = p_ij - pbar_ij
                    if viol_7b > TOL:
                        record(violations, magnitudes, 7,
                               f"p > pbar violated: gen {i+1}, period {j+1}, "
                               f"scenario {s+1}: p={p_ij:.4f} > pbar={pbar_ij:.4f}",
                               p_ij, pbar_ij, viol_7b)

                    # 7c: pbar <= M*x
                    rhs_val_7c = M_cap * x[i][j]
                    viol_7c = pbar_ij - rhs_val_7c
                    if viol_7c > TOL:
                        record(violations, magnitudes, 7,
                               f"Capacity upper bound violated: gen {i+1}, "
                               f"period {j+1}, scenario {s+1}: "
                               f"pbar={pbar_ij:.4f} > M*x={rhs_val_7c:.4f}",
                               pbar_ij, rhs_val_7c, viol_7c)

            # =================================================
            # Constraint 8 (1i): Demand satisfaction
            #   sum_i p_j^i >= D_j
            # =================================================
            for j in range(T):
                lhs_val = sum(p[i][s][j] for i in range(n_gen))
                rhs_val = demand[s][j]
                viol = rhs_val - lhs_val  # >= constraint: violation if rhs > lhs
                if viol > TOL:
                    record(violations, magnitudes, 8,
                           f"Demand not met: period {j+1}, scenario {s+1}: "
                           f"sum(p)={lhs_val:.4f} < D={rhs_val:.4f}",
                           lhs_val, rhs_val, viol)

            # =================================================
            # Constraint 9 (1j): Spinning reserve
            #   sum_i pbar_j^i >= D_j + R_j
            # =================================================
            for j in range(T):
                lhs_val = sum(p_bar[i][s][j] for i in range(n_gen))
                rhs_val = demand[s][j] + reserve[s][j]
                viol = rhs_val - lhs_val
                if viol > TOL:
                    record(violations, magnitudes, 9,
                           f"Spinning reserve not met: period {j+1}, scenario {s+1}: "
                           f"sum(pbar)={lhs_val:.4f} < D+R={rhs_val:.4f}",
                           lhs_val, rhs_val, viol)

    # =====================================================================
    # Constraint 10 (1k): Binary variable domains
    #   x, y, ybar in {0, 1}
    # =====================================================================
    for i in range(n_gen):
        for j in range(T):
            for var_name, var_val in [("x", x[i][j]), ("y", y[i][j]),
                                       ("ybar", y_bar[i][j])]:
                dist = min(abs(var_val - 0), abs(var_val - 1))
                if dist > TOL:
                    record(violations, magnitudes, 10,
                           f"Binary domain violated: gen {i+1}, period {j+1}: "
                           f"{var_name}={var_val}",
                           var_val, round(var_val), dist)

    # =====================================================================
    # Constraint 11 (1l): Non-negativity  q, p, pbar >= 0
    # =====================================================================
    for i in range(n_gen):
        for j in range(T):
            if q[i][j] < -TOL:
                record(violations, magnitudes, 11,
                       f"Non-negativity violated: gen {i+1}, period {j+1}: "
                       f"q={q[i][j]:.6f} < 0",
                       q[i][j], 0.0, -q[i][j])

    if p is not None and p_bar is not None:
        for i in range(n_gen):
            for s in range(n_scen):
                for j in range(T):
                    if p[i][s][j] < -TOL:
                        record(violations, magnitudes, 11,
                               f"Non-negativity violated: gen {i+1}, "
                               f"scenario {s+1}, period {j+1}: "
                               f"p={p[i][s][j]:.6f} < 0",
                               p[i][s][j], 0.0, -p[i][s][j])
                    if p_bar[i][s][j] < -TOL:
                        record(violations, magnitudes, 11,
                               f"Non-negativity violated: gen {i+1}, "
                               f"scenario {s+1}, period {j+1}: "
                               f"pbar={p_bar[i][s][j]:.6f} < 0",
                               p_bar[i][s][j], 0.0, -p_bar[i][s][j])

    # =====================================================================
    # Constraint 12 (Tier C): Objective consistency
    #   Recompute the expected total cost from the solution variables
    #   (formula 1a; second-stage component averaged across scenarios)
    #   and reject solutions whose reported objective_value disagrees.
    #   Defends against LLM exploits that lie about objective_value
    #   (e.g. obj=0 or obj=sys.float_info.max on otherwise feasible
    #   commitment+dispatch decisions).
    # =====================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None and p is not None:
        try:
            reported = float(reported_obj)
            if not math.isfinite(reported):
                raise ValueError("non-finite")
            ok = True
        except (TypeError, ValueError):
            ok = False
        if ok:
            # First-stage cost: sum_{i,j} (c_f^i * x_j^i + q_j^i)
            first_stage = 0.0
            for i in range(n_gen):
                cf = gens[i]["c_f"]
                for j in range(T):
                    first_stage += cf * x[i][j] + q[i][j]
            # Expected second-stage production cost:
            #   (1/n_scen) * sum_{i,j,s} c_g^i * p[i][s][j]
            second_stage = 0.0
            if n_scen > 0:
                acc = 0.0
                for i in range(n_gen):
                    cg = gens[i]["c_g"]
                    for s in range(n_scen):
                        for j in range(T):
                            acc += cg * p[i][s][j]
                second_stage = acc / n_scen
            true_obj = first_stage + second_stage
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with a 1e-3 absolute floor.
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                record(violations, magnitudes, 12,
                       f"Objective consistency violated: reported "
                       f"objective_value={reported} differs from recomputed "
                       f"sum_{{i,j}}(c_f*x+q) + (1/S)*sum_{{i,j,s}}(c_g*p)="
                       f"{true_obj} (|diff|={abs_diff:.3g}, tol={tol_obj:.3g})",
                       reported, true_obj, abs_diff)

    # --- Aggregate results ---
    violated_indices = sorted(set(v[0] for v in violations))
    violation_msgs = []
    for idx in violated_indices:
        msgs = [v[1] for v in violations if v[0] == idx]
        # Aggregate: take first message and count
        if len(msgs) == 1:
            violation_msgs.append(msgs[0])
        else:
            violation_msgs.append(f"{msgs[0]} (and {len(msgs)-1} more)")

    feasible = len(violated_indices) == 0
    return feasible, violated_indices, violation_msgs, magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the stochastic UCP (Salemi & Davarnia 2022)."
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

    feasible, violated_constraints, violation_msgs, magnitudes = \
        check_feasibility(instance, solution)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violation_msgs,
        "violation_magnitudes": magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {feasible}")
    if violated_constraints:
        print(f"Violated constraints: {violated_constraints}")
        for msg in violation_msgs:
            print(f"  - {msg}")
    print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
