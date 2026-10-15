#!/usr/bin/env python3
"""
Feasibility checker for the Two-Stage Stochastic Three-Level Lot Sizing and
Replenishment Problem with a Distribution Structure (2S-3LSPD).

Reference: Gruson, Cordeau, and Jans (2021), European Journal of Operational
Research.

Per project rule: this checker verifies the original problem in *aggregated*
form (per-period, per-scenario quantities at plant / each warehouse / each
retailer), independent of any per-retailer-per-period disaggregation that
the paper's MC reformulation uses internally. Any solver — disaggregated
MC, aggregated heuristic, two-stage Benders, etc. — that converts its
output back into the aggregated solution structure can be checked against
the same rules.

Constraints checked:
  1. Plant inventory balance   (aggregated of paper Eq. 12)
  2. Warehouse inventory balance per warehouse (aggregated of Eq. 13)
  3. Retailer inventory balance per retailer (aggregated of Eq. 14;
     non-negativity of retailer end-of-period inventory captures the
     no-stockout condition)
  4. Setup forcing at plant   (aggregated of Eq. 15)
  5. Setup forcing at each warehouse (aggregated of Eq. 16)
  6. Setup forcing at each retailer  (aggregated of Eq. 17)
  7. Non-negativity of all continuous quantities (Eq. 18)
  8. Binary domain of setup variables y (Eq. 19)
  9. Initial setups imposed (Section 5.1 / math_model assumption #5):
     y_{i,0} = 1 for every facility i.
 10. Initial conditions: end-of-period inventories at t = -1 are zero
     (math_model assumption #4 / Section 5.1).
 11. Objective consistency: reported `objective_value` matches the cost
     recomputed from the aggregated quantities.

Solution structure expected (all keys optional except setup_variables and
objective_value; missing aggregated quantities default to zero, in which
case constraints will simply read zero for those terms — they will fire
naturally if zero violates the balance):

  setup_variables       : dict "y_<i>_<t>" -> 0/1
  production_plant      : dict "<t>_<w>" -> float            (per period & scenario)
  delivery_warehouse    : dict "<w_idx>_<t>_<scenario>" -> float
  delivery_retailer     : dict "<r_idx>_<t>_<scenario>" -> float
  inventory_plant       : dict "<t>_<scenario>" -> float
  inventory_warehouse   : dict "<w_idx>_<t>_<scenario>" -> float
  inventory_retailer    : dict "<r_idx>_<t>_<scenario>" -> float
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('delivery_retailer', 'delivery_warehouse', 'inventory_plant', 'inventory_retailer', 'inventory_warehouse', 'objective_value', 'production_plant', 'setup_variables')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('inventory_plant',)
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



# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_y_key(key):
    """y_<facility_id>_<period> -> (facility_id, period)."""
    parts = key.split("_")
    return int(parts[1]), int(parts[2])


def _to_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _lookup(var_dict, key):
    """var_dict may use string or tuple keys; return float, default 0."""
    if not var_dict:
        return 0.0
    v = var_dict.get(key)
    if v is None:
        return 0.0
    return _to_float(v)


# ---------------------------------------------------------------------------
# Violation recording
# ---------------------------------------------------------------------------

def _record(violated_set, violations, magnitudes, cidx, message,
            lhs, rhs, raw, normalizer):
    violated_set.add(cidx)
    violations.append(message)
    magnitudes.append({
        "constraint": cidx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(raw),
        "normalizer": float(normalizer),
        "ratio": float(raw / normalizer) if normalizer > 0 else float(raw),
    })


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_feasibility(instance, solution):
    tol = 1e-4
    eps = 1e-6

    violations = []
    magnitudes = []
    violated_set = set()

    # ------------------------------------------------------------------
    # Instance
    # ------------------------------------------------------------------
    T = instance["num_periods"]
    nW = instance["num_warehouses"]
    nR = instance["num_retailers"]
    nS = instance["num_scenarios"]
    r2w = instance["retailer_to_warehouse"]
    demands = instance["demands"]              # demands[scenario][retailer][period]
    p_omega = instance["scenario_probabilities"]
    sc_plant = instance["setup_costs_plant"]
    sc_warehouses = instance["setup_costs_warehouses"]
    sc_retailers = instance["setup_costs_retailers"]
    hc_plant_raw = instance["holding_cost_plant"]
    hc_warehouse_raw = instance["holding_cost_warehouse"]
    hc_retailers = instance["holding_costs_retailers"]
    initial_setups = instance.get("initial_setups_imposed", False)

    # Normalize holding-cost shapes: scalar -> indexed.
    hc_plant = (hc_plant_raw if isinstance(hc_plant_raw, list)
                else [hc_plant_raw] * T)
    if isinstance(hc_warehouse_raw, list) and hc_warehouse_raw and isinstance(hc_warehouse_raw[0], list):
        hc_warehouse = hc_warehouse_raw
    elif isinstance(hc_warehouse_raw, list):
        hc_warehouse = [[v] * T for v in hc_warehouse_raw]
    else:
        hc_warehouse = [[hc_warehouse_raw] * T for _ in range(nW)]

    PLANT_ID = 0
    def warehouse_id(w):
        return 1 + w
    def retailer_id(r):
        return 1 + nW + r
    nF = 1 + nW + nR

    # ------------------------------------------------------------------
    # Solution
    # ------------------------------------------------------------------
    y_vars = solution.get("setup_variables", {}) or {}
    y = {}
    for key, val in y_vars.items():
        try:
            fid, t = _parse_y_key(key)
        except (ValueError, IndexError):
            continue
        y[(fid, t)] = _to_float(val)

    prod_plant_raw = solution.get("production_plant", {}) or {}
    delv_w_raw = solution.get("delivery_warehouse", {}) or {}
    delv_r_raw = solution.get("delivery_retailer", {}) or {}
    inv_p_raw = solution.get("inventory_plant", {}) or {}
    inv_w_raw = solution.get("inventory_warehouse", {}) or {}
    inv_r_raw = solution.get("inventory_retailer", {}) or {}

    def prod_plant(t, omega):
        return _lookup(prod_plant_raw, f"{t}_{omega}")
    def delv_warehouse(w, t, omega):
        return _lookup(delv_w_raw, f"{w}_{t}_{omega}")
    def delv_retailer(r, t, omega):
        return _lookup(delv_r_raw, f"{r}_{t}_{omega}")
    def inv_plant(t, omega):
        if t < 0:
            return 0.0  # Constraint 10: zero initial inventory
        return _lookup(inv_p_raw, f"{t}_{omega}")
    def inv_warehouse(w, t, omega):
        if t < 0:
            return 0.0
        return _lookup(inv_w_raw, f"{w}_{t}_{omega}")
    def inv_retailer(r, t, omega):
        if t < 0:
            return 0.0
        return _lookup(inv_r_raw, f"{r}_{t}_{omega}")

    # ------------------------------------------------------------------
    # Constraint 8 (Eq. 19): y in {0,1}
    # ------------------------------------------------------------------
    for fid in range(nF):
        for t in range(T):
            v = y.get((fid, t), 0.0)
            d = min(abs(v), abs(v - 1.0))
            if d > tol:
                _record(violated_set, violations, magnitudes, 8,
                        f"y[{fid},{t}]={v} not binary",
                        v, round(v), d, max(abs(round(v)), eps))

    # ------------------------------------------------------------------
    # Constraint 9: Initial setups imposed (paper assumption #5)
    # ------------------------------------------------------------------
    if initial_setups:
        for fid in range(nF):
            v = y.get((fid, 0), 0.0)
            if abs(v - 1.0) > tol:
                _record(violated_set, violations, magnitudes, 9,
                        f"Initial setup y[{fid},0]={v} should be 1",
                        v, 1.0, abs(v - 1.0), 1.0)

    # ------------------------------------------------------------------
    # Per-scenario per-period checks
    # ------------------------------------------------------------------
    for omega in range(nS):
        # Big-M values for setup forcing (per-scenario totals)
        M_plant = sum(demands[omega][r][t] for r in range(nR) for t in range(T))
        M_w = {w: sum(demands[omega][r][t] for r in range(nR) for t in range(T)
                       if r2w[r] == w)
               for w in range(nW)}
        M_r = {r: sum(demands[omega][r][t] for t in range(T))
               for r in range(nR)}

        for t in range(T):
            # ------------- Constraint 1: plant balance -------------
            # inv_plant[t,omega] = inv_plant[t-1,omega]
            #                      + production_plant[t,omega]
            #                      - sum_w delivery_warehouse[w,t,omega]
            lhs = inv_plant(t, omega)
            rhs = (inv_plant(t - 1, omega) + prod_plant(t, omega)
                   - sum(delv_warehouse(w, t, omega) for w in range(nW)))
            diff = abs(lhs - rhs)
            if diff > tol:
                _record(violated_set, violations, magnitudes, 1,
                        f"Plant balance t={t} ω={omega}: "
                        f"lhs={lhs:.6g} rhs={rhs:.6g}",
                        lhs, rhs, diff, max(abs(rhs), eps))

            # ------------- Constraint 2: warehouse balance ---------
            for w in range(nW):
                lhs = inv_warehouse(w, t, omega)
                rhs = (inv_warehouse(w, t - 1, omega)
                       + delv_warehouse(w, t, omega)
                       - sum(delv_retailer(r, t, omega)
                             for r in range(nR) if r2w[r] == w))
                d = abs(lhs - rhs)
                if d > tol:
                    _record(violated_set, violations, magnitudes, 2,
                            f"Warehouse {w} balance t={t} ω={omega}: "
                            f"lhs={lhs:.6g} rhs={rhs:.6g}",
                            lhs, rhs, d, max(abs(rhs), eps))

            # ------------- Constraint 3: retailer balance ----------
            for r in range(nR):
                lhs = inv_retailer(r, t, omega)
                rhs = (inv_retailer(r, t - 1, omega)
                       + delv_retailer(r, t, omega)
                       - demands[omega][r][t])
                d = abs(lhs - rhs)
                if d > tol:
                    _record(violated_set, violations, magnitudes, 3,
                            f"Retailer {r} balance t={t} ω={omega}: "
                            f"lhs={lhs:.6g} rhs={rhs:.6g}",
                            lhs, rhs, d, max(abs(rhs), eps))

            # ------------- Constraint 4-6: setup forcing -----------
            v = prod_plant(t, omega)
            cap = M_plant * y.get((PLANT_ID, t), 0.0)
            if v > cap + tol:
                _record(violated_set, violations, magnitudes, 4,
                        f"Plant forcing t={t} ω={omega}: "
                        f"prod={v:.6g} > cap={cap:.6g}",
                        v, cap, v - cap, max(abs(cap), eps))
            for w in range(nW):
                v = delv_warehouse(w, t, omega)
                cap = M_w[w] * y.get((warehouse_id(w), t), 0.0)
                if v > cap + tol:
                    _record(violated_set, violations, magnitudes, 5,
                            f"Warehouse {w} forcing t={t} ω={omega}: "
                            f"delv={v:.6g} > cap={cap:.6g}",
                            v, cap, v - cap, max(abs(cap), eps))
            for r in range(nR):
                v = delv_retailer(r, t, omega)
                cap = M_r[r] * y.get((retailer_id(r), t), 0.0)
                if v > cap + tol:
                    _record(violated_set, violations, magnitudes, 6,
                            f"Retailer {r} forcing t={t} ω={omega}: "
                            f"delv={v:.6g} > cap={cap:.6g}",
                            v, cap, v - cap, max(abs(cap), eps))

            # ------------- Constraint 7: non-negativity ------------
            for label, v in (
                ("production_plant", prod_plant(t, omega)),
                ("inventory_plant", inv_plant(t, omega)),
            ):
                if v < -tol:
                    _record(violated_set, violations, magnitudes, 7,
                            f"{label} t={t} ω={omega} = {v:.6g} < 0",
                            v, 0.0, -v, eps)
            for w in range(nW):
                for label, v in (
                    ("delivery_warehouse", delv_warehouse(w, t, omega)),
                    ("inventory_warehouse", inv_warehouse(w, t, omega)),
                ):
                    if v < -tol:
                        _record(violated_set, violations, magnitudes, 7,
                                f"{label}[{w}] t={t} ω={omega} = {v:.6g} < 0",
                                v, 0.0, -v, eps)
            for r in range(nR):
                for label, v in (
                    ("delivery_retailer", delv_retailer(r, t, omega)),
                    ("inventory_retailer", inv_retailer(r, t, omega)),
                ):
                    if v < -tol:
                        _record(violated_set, violations, magnitudes, 7,
                                f"{label}[{r}] t={t} ω={omega} = {v:.6g} < 0",
                                v, 0.0, -v, eps)

    # ------------------------------------------------------------------
    # Constraint 11: objective consistency
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        # setup cost
        setup_cost = 0.0
        for t in range(T):
            setup_cost += sc_plant[t] * y.get((PLANT_ID, t), 0.0)
            for w in range(nW):
                setup_cost += sc_warehouses[w][t] * y.get((warehouse_id(w), t), 0.0)
            for r in range(nR):
                setup_cost += sc_retailers[r][t] * y.get((retailer_id(r), t), 0.0)
        # expected holding cost
        hold_cost = 0.0
        for omega in range(nS):
            pw = p_omega[omega]
            for t in range(T):
                hold_cost += pw * hc_plant[t] * inv_plant(t, omega)
                for w in range(nW):
                    hold_cost += pw * hc_warehouse[w][t] * inv_warehouse(w, t, omega)
                for r in range(nR):
                    hold_cost += pw * hc_retailers[r] * inv_retailer(r, t, omega)
        recomputed = setup_cost + hold_cost
        rel_tol = 1e-3 * max(abs(recomputed), 1.0)
        if abs(reported_obj - recomputed) > rel_tol:
            _record(violated_set, violations, magnitudes, 11,
                    f"Objective mismatch: reported={reported_obj:.6g}, "
                    f"recomputed={recomputed:.6g}",
                    reported_obj, recomputed,
                    abs(reported_obj - recomputed),
                    max(abs(recomputed), 1.0))

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregated-form feasibility checker for 2S-3LSPD "
                    "(Gruson et al. 2021)."
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
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
        print("Solution is FEASIBLE.")
    else:
        print("Solution is INFEASIBLE.")
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"][:10]:
            print(f"  - {v}")
        if len(result["violations"]) > 10:
            print(f"  ... and {len(result['violations']) - 10} more violations")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
