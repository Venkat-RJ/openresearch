"""
Feasibility checker for the Stochastic Distributed Operating Room Scheduling (SDORS) problem.

Checks first-stage hard constraints (1b)-(1h) and scenario-specific DE
constraints (3c)-(3e) from the mathematical formulation in Guo et al. (2021).

Constraint numbering (top to bottom in the formulation):
  1: (1b) Mandatory patients scheduled exactly once
  2: (1c) Non-mandatory patients: scheduled once or postponed (x + w = 1)
  3: (1d) Symmetry-breaking: OR opening order (y_{hdr} <= y_{hd,r-1})
  4: (1e) Symmetry-breaking: cancellation cost ordering across ORs
  5: (1f) OR can only be opened if hospital suite is open (y_{hdr} <= u_{hd})
  6: (1g) Patient assigned only to opened OR (x_{hdpr} <= y_{hdr})
  7: (1h) All first-stage variables are binary {0,1}
  8: (3c) Scenario OR time limit: sum_p T^s_p z^s_{hdpr} <= B_{hd} y_{hdr}
  9: (3d) Scenario linking: z^s_{hdpr} <= x_{hdpr}
 10: (3e) Scenario z^s variables are binary {0,1}
 11: Objective consistency (Tier C anti-score-gaming defense): the reported
     objective_value must lie within [lower bound, plausibility cap] that
     are recomputed from the first-stage variables u, y, x, w.  See the
     extended comment in check_feasibility() for the exact bound.

This file is the obj-recompute-augmented sibling of feasibility_check.py.
It is byte-for-byte identical to the original for constraints 1-10; the
ONLY addition is constraint 11.  The original feasibility_check.py is
left untouched.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'u', 'w', 'x', 'y')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('w',)
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
    with open(path, 'r') as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # Parse instance
    num_patients = instance["num_patients"]
    num_hospitals = instance["num_hospitals"]
    num_days = instance["num_days"]
    num_ors = instance["num_ors_per_hospital"]

    P = list(range(num_patients))
    H = list(range(num_hospitals))
    D = list(range(num_days))
    R = list(range(num_ors))

    patients = instance["patients"]
    P_mandatory = [pt["patient_id"] for pt in patients if pt["is_mandatory"]]
    P_nonmandatory = [pt["patient_id"] for pt in patients if not pt["is_mandatory"]]
    c_cancel = {pt["patient_id"]: pt["c_cancel"] for pt in patients}

    # Second-stage data for (3c)-(3e) checks
    num_scenarios = instance.get("num_scenarios", len(instance.get("scenarios", [])))
    S = list(range(num_scenarios))
    T_s = {}
    for sc in instance.get("scenarios", []):
        T_s[sc["scenario_id"]] = {p: sc["surgery_durations_minutes"][p] for p in P}
    B = {}
    for h_data in instance.get("hospitals", []):
        B[h_data["hospital_id"]] = {dd["day_id"]: dd["B_hd"] for dd in h_data["days"]}

    # Parse solution variables (missing keys default to 0)
    sol_u = solution.get("u", {})
    sol_y = solution.get("y", {})
    sol_x = solution.get("x", {})
    sol_w = solution.get("w", {})
    sol_z = solution.get("z", {})

    def get_u(h, d):
        return float(sol_u.get(f"{h},{d}", 0))

    def get_y(h, d, r):
        return float(sol_y.get(f"{h},{d},{r}", 0))

    def get_x(h, d, p, r):
        return float(sol_x.get(f"{h},{d},{p},{r}", 0))

    def get_w(p):
        return float(sol_w.get(str(p), 0))

    def get_z(s, h, d, p, r):
        return float(sol_z.get(f"{s},{h},{d},{p},{r}", 0))

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
            "ratio": violation_amount / normalizer
        })

    # ------------------------------------------------------------------
    # Constraint 1: (1b) Mandatory patients scheduled exactly once
    #   sum_{h,d,r} x_{hdpr} = 1  for all p in P'
    # ------------------------------------------------------------------
    for p in P_mandatory:
        lhs = sum(get_x(h, d, p, r) for h in H for d in D for r in R)
        rhs = 1.0
        viol = abs(lhs - rhs)
        if viol > tol:
            record(1, f"Constraint 1 (1b): Mandatory patient {p} scheduled {lhs} times (should be 1)", lhs, rhs, viol)

    # ------------------------------------------------------------------
    # Constraint 2: (1c) Non-mandatory patients: x + w = 1
    #   sum_{h,d,r} x_{hdpr} + w_p = 1  for all p in P \ P'
    # ------------------------------------------------------------------
    for p in P_nonmandatory:
        lhs = sum(get_x(h, d, p, r) for h in H for d in D for r in R) + get_w(p)
        rhs = 1.0
        viol = abs(lhs - rhs)
        if viol > tol:
            record(2, f"Constraint 2 (1c): Non-mandatory patient {p} has sum(x)+w={lhs} (should be 1)", lhs, rhs, viol)

    # ------------------------------------------------------------------
    # Constraint 3 and 4 SKIPPED: (1d) and (1e) are SYMMETRY-BREAKING
    # constraints (the paper itself labels them as such, since ORs within a
    # hospital-day are homogeneous). They are NOT business rules — a
    # symmetric-but-valid alternative assignment (different OR indices) should
    # not be rejected. The original problem only requires each assignment to
    # respect capacity and open-OR-open-suite linking, which are checked in
    # Constraints 5, 6, 8.

    # ------------------------------------------------------------------
    # Constraint 5: (1f) OR requires open suite: y_{hdr} <= u_{hd}
    # ------------------------------------------------------------------
    for h in H:
        for d in D:
            for r in R:
                lhs = get_y(h, d, r)
                rhs = get_u(h, d)
                viol = lhs - rhs
                if viol > tol:
                    record(5, f"Constraint 5 (1f): y[{h},{d},{r}]={lhs} > u[{h},{d}]={rhs}", lhs, rhs, viol)

    # ------------------------------------------------------------------
    # Constraint 6: (1g) Patient assigned to opened OR: x_{hdpr} <= y_{hdr}
    # ------------------------------------------------------------------
    for h in H:
        for d in D:
            for p in P:
                for r in R:
                    lhs = get_x(h, d, p, r)
                    rhs = get_y(h, d, r)
                    viol = lhs - rhs
                    if viol > tol:
                        record(6, f"Constraint 6 (1g): x[{h},{d},{p},{r}]={lhs} > y[{h},{d},{r}]={rhs}", lhs, rhs, viol)

    # ------------------------------------------------------------------
    # Constraint 7: (1h) Binary domain: all variables in {0, 1}
    # ------------------------------------------------------------------
    # Check u
    for h in H:
        for d in D:
            val = get_u(h, d)
            if abs(val - round(val)) > tol or round(val) not in (0, 1):
                record(7, f"Constraint 7 (1h): u[{h},{d}]={val} not binary", val, round(val), abs(val - round(val)))

    # Check y
    for h in H:
        for d in D:
            for r in R:
                val = get_y(h, d, r)
                if abs(val - round(val)) > tol or round(val) not in (0, 1):
                    record(7, f"Constraint 7 (1h): y[{h},{d},{r}]={val} not binary", val, round(val), abs(val - round(val)))

    # Check x
    for h in H:
        for d in D:
            for p in P:
                for r in R:
                    val = get_x(h, d, p, r)
                    if abs(val - round(val)) > tol or round(val) not in (0, 1):
                        record(7, f"Constraint 7 (1h): x[{h},{d},{p},{r}]={val} not binary", val, round(val), abs(val - round(val)))

    # Check w
    for p in P:
        val = get_w(p)
        if abs(val - round(val)) > tol or round(val) not in (0, 1):
            record(7, f"Constraint 7 (1h): w[{p}]={val} not binary", val, round(val), abs(val - round(val)))

    # ------------------------------------------------------------------
    # Constraint 8: (3c) Scenario OR time limit
    #   sum_p T^s_p * z^s_{hdpr} <= B_{hd} * y_{hdr}  for all s,h,d,r
    # ------------------------------------------------------------------
    for s in S:
        for h in H:
            for d in D:
                for r in R:
                    lhs = sum(T_s[s][p] * get_z(s, h, d, p, r) for p in P)
                    rhs = B[h][d] * get_y(h, d, r)
                    viol = lhs - rhs
                    if viol > tol:
                        record(8, f"Constraint 8 (3c): scenario {s}, h={h}, d={d}, r={r}: sum T*z={lhs} > B*y={rhs}", lhs, rhs, viol)

    # ------------------------------------------------------------------
    # Constraint 9: (3d) Scenario linking: z^s_{hdpr} <= x_{hdpr}
    # ------------------------------------------------------------------
    for s in S:
        for h in H:
            for d in D:
                for p in P:
                    for r in R:
                        lhs = get_z(s, h, d, p, r)
                        rhs = get_x(h, d, p, r)
                        viol = lhs - rhs
                        if viol > tol:
                            record(9, f"Constraint 9 (3d): z[{s},{h},{d},{p},{r}]={lhs} > x[{h},{d},{p},{r}]={rhs}", lhs, rhs, viol)

    # ------------------------------------------------------------------
    # Constraint 10: (3e) Scenario z^s binary in {0,1}
    # ------------------------------------------------------------------
    for s in S:
        for h in H:
            for d in D:
                for p in P:
                    for r in R:
                        val = get_z(s, h, d, p, r)
                        if abs(val - round(val)) > tol or round(val) not in (0, 1):
                            record(10, f"Constraint 10 (3e): z[{s},{h},{d},{p},{r}]={val} not binary", val, round(val), abs(val - round(val)))

    # ------------------------------------------------------------------
    # Constraint 11: Objective consistency (Tier C anti-score-gaming defense)
    #
    #   The SAA Deterministic-Equivalent objective (3a) is
    #     obj =  sum_{h,d}     G_hd * u_hd                      (suite opening)
    #          + sum_{h,d,r}   F_hd * y_hdr                     (OR opening)
    #          + sum_{h,d,p,r} c_sched_dp * x_hdpr              (sched. benefit, <= 0)
    #          + sum_{p in P\P'} c_unsched_p * w_p              (postponement penalty)
    #          + (1/|S|) sum_{s,h,d,p,r} c_cancel_p*(x_hdpr - z^s_hdpr)  (recourse)
    #
    #   The first four terms ("obj_det") are FULLY determined by the
    #   solution variables u, y, x, w that the program writes.  The recourse
    #   term depends on the SECOND-STAGE variables z^s, which are NOT carried
    #   in the solution (solution_schema.json lists only u, y, x, w; every
    #   Gurobi reference and every historical LLM solution omits z).  A full
    #   objective recompute is therefore impossible -- this is a LOWER-BOUND
    #   plus plausibility-envelope check (check_style = "lower_bound"):
    #
    #     * recompute_lb = obj_det + T5_min, where T5_min is the OPTIMAL
    #       (cost-minimising) expected recourse for the given fixed first
    #       stage.  The recourse problem Q(x,y,T^s) decomposes per
    #       (scenario, OR) into an independent 0/1 knapsack -- keep the
    #       maximum-c_cancel subset of the OR's assigned patients whose total
    #       scenario duration fits B_hd*y_hdr, cancel the rest -- so T5_min is
    #       computed exactly here.  Because *any* feasible second stage costs
    #       >= the optimal recourse, recompute_lb is a rigorous lower bound on
    #       the true objective.  A reported value strictly below recompute_lb
    #       is provably a lie (claiming a better-than-achievable cost is the
    #       score-gaming exploit for this minimisation problem).
    #
    #     * obj_upper = recompute_lb + max(1.0, 0.10*|recompute_lb|) is a
    #       plausibility cap.  The recourse is a trivial per-OR minimisation
    #       that any competent solver optimises to within a fraction of a
    #       percent of optimal (observed slack < 1.1% of |obj| across every
    #       Gurobi reference and every historical LLM solution).  A reported
    #       value far above recompute_lb -- e.g. the obj=0 or
    #       obj=sys.float_info.max fabrications -- is therefore rejected.
    #
    #   The whole block is wrapped in try/except: if anything in the
    #   recompute fails on a malformed solution, the objective check is
    #   silently skipped (no constraint-11 violation) so the checker never
    #   crashes -- a malformed solution is already caught by constraints 1-10.
    # ------------------------------------------------------------------
    try:
        reported_obj = solution.get("objective_value")
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None

        if reported is not None and math.isfinite(reported):
            # --- instance-side cost tables -------------------------------
            c_unsched = {pt["patient_id"]: float(pt.get("c_unsched", 0.0)) for pt in patients}
            # c_sched_per_day is keyed by 1-based day_label; map to 0-based day_id.
            day_label_of = {}
            for h_data in instance.get("hospitals", []):
                for dd in h_data["days"]:
                    day_label_of[dd["day_id"]] = dd.get("day_label", dd["day_id"] + 1)
            c_sched = {}
            for pt in patients:
                pid = pt["patient_id"]
                per_day = pt.get("c_sched_per_day", {}) or {}
                for d in D:
                    label = day_label_of.get(d, d + 1)
                    val = per_day.get(str(label), per_day.get(str(d + 1)))
                    if val is not None:
                        c_sched[(pid, d)] = float(val)
            G = {}
            F = {}
            for h_data in instance.get("hospitals", []):
                h_id = h_data["hospital_id"]
                for dd in h_data["days"]:
                    G[(h_id, dd["day_id"])] = float(dd["G_hd"])
                    F[(h_id, dd["day_id"])] = float(dd["F_hd"])

            # --- deterministic objective terms (exact, from u, y, x, w) --
            t_suite = sum(G.get((h, d), 0.0) * get_u(h, d) for h in H for d in D)
            t_or = sum(F.get((h, d), 0.0) * get_y(h, d, r)
                       for h in H for d in D for r in R)
            t_sched = sum(c_sched.get((p, d), 0.0) * get_x(h, d, p, r)
                          for h in H for d in D for p in P for r in R)
            mand_set = set(P_mandatory)
            t_unsched = sum(c_unsched.get(p, 0.0) * get_w(p)
                            for p in P if p not in mand_set)
            obj_det = t_suite + t_or + t_sched + t_unsched

            # --- optimal expected recourse T5_min ------------------------
            # Per (h,d,r): assigned patient set is identical across scenarios
            # (x has no scenario index); only durations T^s_p change.
            def knapsack_max_kept(items, cap):
                """Max sum of values of a subset whose total weight <= cap.

                Exact Pareto-frontier dynamic programming works with the
                instance's numeric durations directly and avoids the previous
                fractional relaxation for ORs containing more than 16 patients.
                """
                if not items:
                    return 0.0
                total_w = sum(w for w, v in items)
                if total_w <= cap + 1e-6:
                    return sum(v for w, v in items)
                if cap <= 1e-9:
                    return 0.0
                frontier = [(0.0, 0.0)]
                for weight, value in items:
                    additions = [
                        (used + weight, kept + value)
                        for used, kept in frontier
                        if used + weight <= cap + 1e-6
                    ]
                    merged = sorted(frontier + additions,
                                    key=lambda state: (state[0], -state[1]))
                    pruned = []
                    best_value = -float("inf")
                    for used, kept in merged:
                        if kept > best_value + 1e-9:
                            pruned.append((used, kept))
                            best_value = kept
                    frontier = pruned
                    if len(frontier) > 200000:
                        raise RuntimeError(
                            "exact recourse frontier exceeded 200000 states"
                        )
                return max(kept for used, kept in frontier)

            t5_sum = 0.0
            for h in H:
                for d in D:
                    for r in R:
                        assigned = [p for p in P if get_x(h, d, p, r) > 0.5]
                        if not assigned:
                            continue
                        cap = B.get(h, {}).get(d, 0.0) * get_y(h, d, r)
                        total_cc = sum(c_cancel.get(p, 0.0) for p in assigned)
                        for s in S:
                            durs = T_s.get(s)
                            if durs is None:
                                # missing scenario contributes 0 (>= 0 is a
                                # valid lower bound for that scenario's cost)
                                continue
                            items = [(float(durs[p]), float(c_cancel.get(p, 0.0)))
                                     for p in assigned]
                            kept = knapsack_max_kept(items, cap)
                            t5_sum += (total_cc - kept)
            t5_min = (t5_sum / num_scenarios) if num_scenarios > 0 else 0.0

            recompute_lb = obj_det + t5_min
            # Old references omit z and two time-limited incumbents retain a
            # recourse value up to 0.31% above the optimal completion.  Keep a
            # narrowly bounded 0.5% compatibility window; the previous 10%
            # window let ordinary +/-2% score mutations through.
            tol_low = max(1e-3, 1e-8 * abs(recompute_lb))
            obj_upper = recompute_lb + max(1e-3, 0.005 * abs(recompute_lb))

            if reported < recompute_lb - tol_low:
                shortfall = recompute_lb - reported
                record(
                    11,
                    f"Constraint 11 (objective consistency): reported "
                    f"objective_value={reported} is below the recomputed "
                    f"lower bound={recompute_lb} (obj_det={obj_det} + optimal "
                    f"expected recourse T5_min={t5_min}); a feasible solution "
                    f"can never achieve a cost lower than its optimal recourse "
                    f"(shortfall={shortfall:.6g}, tol={tol_low:.6g})",
                    reported, recompute_lb, shortfall,
                )
            elif reported > obj_upper:
                excess = reported - obj_upper
                record(
                    11,
                    f"Constraint 11 (objective consistency): reported "
                    f"objective_value={reported} exceeds the plausibility cap "
                    f"obj_upper={obj_upper} (recomputed lower bound="
                    f"{recompute_lb}, +0.5% legacy-incumbent margin); the "
                    f"reported cost is too far "
                    f"above what the first-stage decisions u,y,x,w can yield "
                    f"(excess={excess:.6g})",
                    reported, obj_upper, excess,
                )
    except Exception as error:
        # Objective validation is part of the solution contract.  Failing open
        # here would turn malformed or resource-exhausting witnesses into a
        # score-gaming bypass.
        record(
            11,
            f"Constraint 11 (objective consistency) could not be validated: "
            f"{type(error).__name__}: {str(error)[:300]}",
            0.0, 1.0, 1.0,
        )

    # Build result
    feasible = len(violated_set) == 0
    violated_constraints = sorted(violated_set)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for SDORS")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE - no constraint violations found")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
