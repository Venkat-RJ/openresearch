"""
Feasibility checker for PP-DESUP (Hewitt & Pantuso, 2024/2025).
Checks all hard constraints from the original bilinear formulation (Eqs. 2-13).

Constraint numbering (top to bottom in formulation):
  1 = Eq.(2): Facility capacity
  2 = Eq.(3): Exactly one production level per (p,f)
  3 = Eq.(4): Production allocation bounds (two-sided)
  4 = Eq.(5): Exactly one distribution per product
  5 = Eq.(6): Endogeneity link
  6 = Eq.(7): Inventory definition z = sum Y*x
  7 = Eq.(8): Sales limited by demand w <= D
  8 = Eq.(9): Inventory balance w + o = z
  9 = Eq.(10): x >= 0
 10 = Eq.(11): y binary
 11 = Eq.(12): delta binary
 12 = Eq.(13): z, w, o >= 0
 13 = Objective consistency (Tier C): the reported objective_value must
      match the objective recomputed from the solution variables via
      Eq.(1):  obj = -sum_{p,f} C_pf x_pf
                     + sum_p sum_d delta_pd sum_s pi_sd (P_p w_pds + O_p o_pds)
      Defends against LLM score-gaming exploits that report a fabricated
      objective_value while keeping the decision variables constraint-feasible.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'solution')
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



def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []

    # If solution is empty, report infeasible but no specific violations
    sol = solution.get("solution", {})
    if not sol:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution is empty — no decision variables provided."],
            "violation_magnitudes": []
        }

    # --- Parse instance ---
    P = [p["id"] for p in instance["products"]]
    F = [fac["id"] for fac in instance["facilities"]]

    B = {fac["id"]: fac["capacity"] for fac in instance["facilities"]}

    # Per-product pricing (used by the objective recompute, constraint 13).
    price = {p["id"]: p["sale_price"] for p in instance["products"]}
    disc = {p["id"]: p["discounted_price"] for p in instance["products"]}

    C = {}
    L_bound = {}
    U_bound = {}
    levels = {}
    for pfd in instance["product_facility_data"]:
        p = pfd["product_id"]
        f = pfd["facility_id"]
        C[(p, f)] = pfd["manufacturing_cost"]
        lv_list = []
        for lv in pfd["production_levels"]:
            l = lv["level_id"]
            lv_list.append(l)
            L_bound[(p, f, l)] = lv["lower_bound"]
            U_bound[(p, f, l)] = lv["upper_bound"]
        levels[(p, f)] = lv_list

    distributions_data = {}
    for pd_entry in instance["products_distributions"]:
        distributions_data[pd_entry["product_id"]] = pd_entry["distributions"]

    D_p = {p: [d["distribution_id"] for d in distributions_data[p]] for p in P}
    S_pd = {}
    pi = {}
    Y = {}
    D_dem = {}
    level_combo = {}
    l_map = {}

    for p in P:
        for d_dict in distributions_data[p]:
            d = d_dict["distribution_id"]
            level_combo[(p, d)] = d_dict["level_combination"]
            S_pd[(p, d)] = [s["scenario_id"] for s in d_dict["scenarios"]]
            combo = d_dict["level_combination"]
            if combo is not None:
                for fi, f in enumerate(F):
                    l_map[(p, f, d)] = combo[fi]
            for s_dict in d_dict["scenarios"]:
                s = s_dict["scenario_id"]
                pi[(p, d, s)] = s_dict["probability"]
                D_dem[(p, d, s)] = s_dict["demand"]
                for f in F:
                    if "yields" in s_dict:
                        Y[(p, f, d, s)] = s_dict["yields"].get(str(f), 0.0)
                    else:
                        Y[(p, f, d, s)] = 0.0

    # --- Parse solution variables ---
    x_sol = sol.get("x", {})
    y_sol = sol.get("y", {})
    delta_sol = sol.get("delta", {})

    x = {}
    for p in P:
        for f in F:
            x[(p, f)] = x_sol.get(f"{p}_{f}", 0.0)

    y = {}
    for p in P:
        for f in F:
            for l in levels[(p, f)]:
                y[(p, f, l)] = y_sol.get(f"{p}_{f}_{l}", 0.0)

    delta = {}
    for p in P:
        for d in D_p[p]:
            delta[(p, d)] = delta_sol.get(f"{p}_{d}", 0.0)

    # z, w, o are auxiliary linearisation variables. They are not part of
    # the original solution structure — original primary vars are x, y, delta.
    # When the solution does not carry them (e.g. older solver outputs only
    # the primary decisions), derive them from x:
    #   z_pds = sum_f Y_pfds * x_pf            (Eq. 7, definition)
    #   w_pds = min(z_pds, D_pds)              (optimal recourse, Eq. 8)
    #   o_pds = z_pds - w_pds                  (Eq. 9)
    # so the constraint check does not fail just because the auxiliaries
    # are missing from the saved solution.
    z_sol = sol.get("z", {})
    w_sol = sol.get("w", {})
    o_sol = sol.get("o", {})

    z = {}
    w = {}
    o = {}
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                key = f"{p}_{d}_{s}"
                if key in z_sol:
                    z[(p, d, s)] = z_sol[key]
                else:
                    z[(p, d, s)] = sum(Y[(p, f, d, s)] * x[(p, f)] for f in F)
                if key in w_sol:
                    w[(p, d, s)] = w_sol[key]
                else:
                    w[(p, d, s)] = max(0.0, min(z[(p, d, s)], D_dem[(p, d, s)]))
                if key in o_sol:
                    o[(p, d, s)] = o_sol[key]
                else:
                    o[(p, d, s)] = max(0.0, z[(p, d, s)] - w[(p, d, s)])

    def add_violation(constraint_idx, msg, lhs_val, rhs_val, raw_excess):
        normalizer = max(abs(rhs_val), eps)
        ratio = raw_excess / normalizer
        violations.append((constraint_idx, msg))
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs_val,
            "rhs": rhs_val,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": ratio
        })

    # ============================================================
    # Constraint 1 = Eq.(2): sum_p x_pf <= B_f for all f
    # ============================================================
    for f in F:
        lhs = sum(x[(p, f)] for p in P)
        rhs = B[f]
        excess = lhs - rhs
        if excess > tol:
            add_violation(1, f"Facility {f} capacity exceeded: sum(x_pf)={lhs:.6f} > B_f={rhs:.6f}", lhs, rhs, excess)

    # ============================================================
    # Constraint 2 = Eq.(3): sum_l y_pfl = 1 for all p, f
    # ============================================================
    for p in P:
        for f in F:
            lhs = sum(y[(p, f, l)] for l in levels[(p, f)])
            rhs = 1.0
            excess = abs(lhs - rhs)
            if excess > tol:
                add_violation(2, f"Product {p}, facility {f}: sum(y_pfl)={lhs:.6f} != 1", lhs, rhs, excess)

    # ============================================================
    # Constraint 3 = Eq.(4): L*y <= x_pf <= U*y for all p, f
    # ============================================================
    for p in P:
        for f in F:
            lb = sum(L_bound[(p, f, l)] * y[(p, f, l)] for l in levels[(p, f)])
            ub = sum(U_bound[(p, f, l)] * y[(p, f, l)] for l in levels[(p, f)])
            xpf = x[(p, f)]
            # Lower bound: lb <= x_pf  →  violation if lb > x_pf
            excess_lb = lb - xpf
            if excess_lb > tol:
                add_violation(3, f"Product {p}, facility {f}: x_pf={xpf:.6f} < lower bound={lb:.6f}",
                              lb, xpf, excess_lb)
            # Upper bound: x_pf <= ub  →  violation if x_pf > ub
            excess_ub = xpf - ub
            if excess_ub > tol:
                add_violation(3, f"Product {p}, facility {f}: x_pf={xpf:.6f} > upper bound={ub:.6f}",
                              xpf, ub, excess_ub)

    # ============================================================
    # Constraint 4 = Eq.(5): sum_d delta_pd = 1 for all p
    # ============================================================
    for p in P:
        lhs = sum(delta[(p, d)] for d in D_p[p])
        rhs = 1.0
        excess = abs(lhs - rhs)
        if excess > tol:
            add_violation(4, f"Product {p}: sum(delta_pd)={lhs:.6f} != 1", lhs, rhs, excess)

    # ============================================================
    # Constraint 5 = Eq.(6): sum_f y_{p,f,l(p,f,d)} >= |F|*delta_pd
    #   for all p, d (only yield distributions with level_combination != null)
    # ============================================================
    num_F = len(F)
    for p in P:
        for d in D_p[p]:
            if level_combo[(p, d)] is None:
                continue
            lhs = sum(y[(p, f, l_map[(p, f, d)])] for f in F)
            rhs = num_F * delta[(p, d)]
            excess = rhs - lhs  # >= constraint: violation if RHS > LHS
            if excess > tol:
                add_violation(5, f"Product {p}, distribution {d}: endogeneity link violated, "
                              f"sum(y)={lhs:.6f} < |F|*delta={rhs:.6f}", lhs, rhs, excess)

    # ============================================================
    # Compute second-stage variables for constraints 6-8, 12
    # z_pds = sum_f Y_pfds * x_pf
    # w_pds = min(z_pds, D_pds)  (optimal choice)
    # o_pds = z_pds - w_pds
    # ============================================================
    z_val = {}
    w_val = {}
    o_val = {}
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                z_val[(p, d, s)] = sum(Y[(p, f, d, s)] * x[(p, f)] for f in F)
                w_val[(p, d, s)] = min(z_val[(p, d, s)], D_dem[(p, d, s)])
                if w_val[(p, d, s)] < 0:
                    w_val[(p, d, s)] = 0.0
                o_val[(p, d, s)] = z_val[(p, d, s)] - w_val[(p, d, s)]

    # ============================================================
    # Constraint 6 = Eq.(7): z_pds = sum_f Y_pfds * x_pf
    # ============================================================
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                lhs = z[(p, d, s)]
                rhs = sum(Y[(p, f, d, s)] * x[(p, f)] for f in F)
                excess = abs(lhs - rhs)
                if excess > tol:
                    add_violation(6, f"Product {p}, dist {d}, scenario {s}: z={lhs:.6f} != sum(Y*x)={rhs:.6f}",
                                  lhs, rhs, excess)

    # ============================================================
    # Constraint 7 = Eq.(8): w_pds <= D_pds
    # ============================================================
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                lhs = w[(p, d, s)]
                rhs = D_dem[(p, d, s)]
                excess = lhs - rhs
                if excess > tol:
                    add_violation(7, f"Product {p}, dist {d}, scenario {s}: w={lhs:.6f} > D={rhs:.6f}",
                                  lhs, rhs, excess)

    # ============================================================
    # Constraint 8 = Eq.(9): w_pds + o_pds = z_pds
    # ============================================================
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                lhs = w[(p, d, s)] + o[(p, d, s)]
                rhs = z[(p, d, s)]
                excess = abs(lhs - rhs)
                if excess > tol:
                    add_violation(8, f"Product {p}, dist {d}, scenario {s}: w+o={lhs:.6f} != z={rhs:.6f}",
                                  lhs, rhs, excess)

    # ============================================================
    # Constraint 9 = Eq.(10): x_pf >= 0
    # ============================================================
    for p in P:
        for f in F:
            xpf = x[(p, f)]
            if xpf < -tol:
                excess = -xpf
                add_violation(9, f"Product {p}, facility {f}: x_pf={xpf:.6f} < 0", xpf, 0.0, excess)

    # ============================================================
    # Constraint 10 = Eq.(11): y_pfl in {0,1}
    # ============================================================
    for p in P:
        for f in F:
            for l in levels[(p, f)]:
                val = y[(p, f, l)]
                dist = min(abs(val), abs(val - 1.0))
                if dist > tol:
                    add_violation(10, f"Product {p}, facility {f}, level {l}: y={val:.6f} not binary",
                                  val, round(val), dist)

    # ============================================================
    # Constraint 11 = Eq.(12): delta_pd in {0,1}
    # ============================================================
    for p in P:
        for d in D_p[p]:
            val = delta[(p, d)]
            dist = min(abs(val), abs(val - 1.0))
            if dist > tol:
                add_violation(11, f"Product {p}, distribution {d}: delta={val:.6f} not binary",
                              val, round(val), dist)

    # ============================================================
    # Constraint 12 = Eq.(13): z_pds, w_pds, o_pds >= 0
    # ============================================================
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                zv = z_val[(p, d, s)]
                if zv < -tol:
                    add_violation(12, f"Product {p}, dist {d}, scenario {s}: z={zv:.6f} < 0",
                                  zv, 0.0, -zv)
                wv = w_val[(p, d, s)]
                if wv < -tol:
                    add_violation(12, f"Product {p}, dist {d}, scenario {s}: w={wv:.6f} < 0",
                                  wv, 0.0, -wv)
                ov = o_val[(p, d, s)]
                if ov < -tol:
                    add_violation(12, f"Product {p}, dist {d}, scenario {s}: o={ov:.6f} < 0",
                                  ov, 0.0, -ov)

    # ============================================================
    # Constraint 13 (Tier C): Objective consistency.
    #   The eval pipeline trusts the candidate's self-reported
    #   objective_value; LLM-evolved candidates exploit that trust by
    #   reporting a fabricated value while keeping x/y/delta/z/w/o
    #   constraint-feasible. Defend by recomputing the objective directly
    #   from the solution variables via Eq.(1):
    #     obj = -sum_{p,f} C_pf x_pf
    #           + sum_p sum_d delta_pd sum_s pi_sd (P_p w_pds + O_p o_pds)
    #   All variables the objective depends on (x, delta, w, o) are present
    #   in the solution, so a FULL recompute is possible (no second-stage
    #   re-optimisation needed). w/o use the solution-provided values when
    #   present and the optimal-recourse derivation otherwise — exactly the
    #   `w`/`o` dicts already parsed above.
    #   Tolerance: max(1e-3 absolute, 1e-3 relative) — 0.1% of |recomputed|,
    #   well above observed solver round-off (~1e-9 on Gurobi references).
    # ============================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            cost_term = sum(C[(p, f)] * x[(p, f)] for p in P for f in F)
            revenue_term = 0.0
            for p in P:
                Pp = price[p]
                Op = disc[p]
                for d in D_p[p]:
                    inner = sum(
                        pi[(p, d, s)] * (Pp * w[(p, d, s)] + Op * o[(p, d, s)])
                        for s in S_pd[(p, d)]
                    )
                    revenue_term += delta[(p, d)] * inner
            true_obj = -cost_term + revenue_term
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                add_violation(
                    13,
                    f"Objective consistency violated: reported objective_value={reported} "
                    f"differs from recomputed objective={true_obj:.6f} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    reported, true_obj, abs_diff)
        elif reported is not None:
            # Non-finite reported objective (inf / nan) can never equal a
            # finite recomputed objective.
            add_violation(
                13,
                f"Objective consistency violated: reported objective_value={reported_obj} "
                f"is not a finite number",
                reported, 0.0, float("inf"))

    # --- Build output ---
    violated_set = sorted(set(v[0] for v in violations))
    violation_msgs = []
    for idx in violated_set:
        msgs = [v[1] for v in violations if v[0] == idx]
        violation_msgs.append("; ".join(msgs))

    feasible = len(violated_set) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_set,
        "violations": violation_msgs,
        "violation_magnitudes": violation_magnitudes if not feasible else []
    }


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for PP-DESUP.")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON file.")
    parser.add_argument("--result_path", required=True, help="Path for output feasibility result JSON.")
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
    print(f"Feasibility: {status}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
