#!/usr/bin/env python3
"""
Feasibility checker (Tier C variant) for the Energy Systems Planning problem.

From: Jacobson et al. (2023) - "A computationally efficient Benders
decomposition for energy systems planning problems with detailed
operations and time-coupling constraints"

Constraints 1-12 validate the submitted investment witness. Constraint 49
substitutes the complete investment-and-operation vector into trusted offline
model coefficients without invoking an optimizer.

Constraint 49 (objective consistency): trusted fixed-investment recourse solve.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solverless_witness import require_linked_variables, validate_complete_witness

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('investments', 'model_variables', 'objective_value', 'storage_energy_investments', 'transmission_investments')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('investments', 'model_variables')
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


def check_feasibility(instance, solution, instance_path=None):
    """Run all checks and return the same dict shape as the original main()
    writes to result_path. Exposed as a function so the diff-test harness
    can call it directly (parallel to other paper's _new.py files)."""
    inst = instance
    sol = solution

    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violation_details = {}    # constraint_idx -> list of message strings
    violation_magnitudes = []  # list of dicts

    def record_violation(constraint_idx, lhs, rhs, raw_excess, msg):
        violated_constraints.add(constraint_idx)
        if constraint_idx not in violation_details:
            violation_details[constraint_idx] = []
        violation_details[constraint_idx].append(msg)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": raw_excess / normalizer
        })

    def check_leq(constraint_idx, lhs, rhs, msg):
        excess = lhs - rhs
        if excess > tol:
            record_violation(constraint_idx, lhs, rhs, excess, msg)

    def check_geq(constraint_idx, lhs, rhs, msg):
        excess = rhs - lhs
        if excess > tol:
            record_violation(constraint_idx, lhs, rhs, excess, msg)

    def check_eq(constraint_idx, lhs, rhs, msg):
        excess = abs(lhs - rhs)
        if excess > tol:
            record_violation(constraint_idx, lhs, rhs, excess, msg)

    # =========================================================================
    # Check if solution has investment data
    # =========================================================================
    has_investments = "investments" in sol

    if not has_investments:
        return {
            "feasible": False,
            "violated_constraints": ["solution_schema"],
            "violations": ["No investment solution provided (null or missing)"],
            "violation_magnitudes": []
        }

    # =========================================================================
    # Extract instance data
    # =========================================================================
    generators = inst["generators"]
    gen_params = inst["generator_parameters"]
    existing_cap = inst["existing_capacity"]
    trans_lines = inst["transmission_lines"]
    release_convention = inst["released_investment_convention"]

    num_gen = len(generators)
    G = list(range(num_gen))
    G_UC = [g for g in G if generators[g]["is_UC"]]
    G_STOR = [g for g in G if generators[g]["is_storage"]]
    G_HYDRO = [g for g in G if generators[g]["is_hydro"]]
    G_NONRET = set(g for g in G if not generators[g]["can_retire"])
    G_UC_set = set(G_UC)
    G_STOR_set = set(G_STOR)
    L = list(range(len(trans_lines)))

    # Released convention: physical UC blocks and integer 1-MW non-UC blocks.
    gamma_P = {}
    for g in G:
        if g in G_UC_set:
            gamma_P[g] = generators[g]["capacity_size_MW"]
        else:
            gamma_P[g] = float(release_convention["non_uc_power_increment_MW"])

    # gamma_E: energy_capacity_size_MWh for UC storage, 1.0 otherwise
    gamma_E = {}
    for g in G_STOR:
        if g in G_UC_set:
            rtype = generators[g]["resource_type"]
            gamma_E[g] = gen_params[rtype]["energy_capacity_size_MWh"]
        else:
            gamma_E[g] = float(
                release_convention["non_uc_storage_energy_increment_MWh"]
            )

    # Existing power capacity
    y_P_0 = {}
    for g in G:
        y_P_0[g] = existing_cap.get(str(g), 0.0)

    # Existing energy capacity for storage
    y_E_0 = {}
    for g in G_STOR:
        rtype = generators[g]["resource_type"]
        bp = gen_params[rtype]
        phys_cap_size = bp.get("capacity_size_MW", 1.0)
        e_size = bp.get("energy_capacity_size_MWh", 1.0)
        if phys_cap_size > 0:
            y_E_0[g] = (y_P_0[g] / phys_cap_size) * e_size
        else:
            y_E_0[g] = 0.0

    # Max capacities
    y_P_max = {}
    for g in G:
        rtype = generators[g]["resource_type"]
        y_P_max[g] = gen_params[rtype]["max_capacity_MW"]

    y_E_max = {}
    for g in G_STOR:
        rtype = generators[g]["resource_type"]
        y_E_max[g] = gen_params[rtype]["max_energy_capacity_MWh"]

    # Storage duration limits
    rho_min_stor = {}
    rho_max_stor = {}
    for g in G_STOR:
        rtype = generators[g]["resource_type"]
        rho_min_stor[g] = gen_params[rtype]["min_duration_MWh_per_MW"]
        rho_max_stor[g] = gen_params[rtype]["max_duration_MWh_per_MW"]

    # =========================================================================
    # Extract solution investment variables
    # =========================================================================
    inv = sol.get("investments", {})
    stor_inv = sol.get("storage_energy_investments", {})
    trans_inv = sol.get("transmission_investments", {})

    # Power capacity variables
    y_P_new = {}
    y_P_ret = {}
    y_P = {}
    for g in G:
        gs = str(g)
        if gs in inv:
            y_P_new[g] = inv[gs]["new_units"]
            y_P_ret[g] = abs(inv[gs]["retired_units"])
            y_P[g] = inv[gs]["total_capacity_MW"]
        else:
            y_P_new[g] = 0.0
            y_P_ret[g] = 0.0
            y_P[g] = y_P_0[g]

    # Storage energy capacity variables
    y_E_new = {}
    y_E = {}
    for g in G_STOR:
        gs = str(g)
        if gs in stor_inv:
            y_E_new[g] = stor_inv[gs]["new_units"]
            y_E[g] = stor_inv[gs]["total_energy_MWh"]
        else:
            y_E_new[g] = 0.0
            y_E[g] = y_E_0[g]

    # Derive y_E_ret from constraint (13d)
    y_E_ret = {}
    for g in G_STOR:
        if gamma_E[g] > 0:
            y_E_ret[g] = y_E_new[g] - (y_E[g] - y_E_0[g]) / gamma_E[g]
        else:
            y_E_ret[g] = 0.0

    # Transmission capacity variables
    y_F_new = {}
    y_F = {}
    for l_idx in L:
        ls = str(l_idx)
        if ls in trans_inv:
            y_F_new[l_idx] = trans_inv[ls]["new_MW"]
            y_F[l_idx] = trans_inv[ls]["total_capacity_MW"]
        else:
            y_F_new[l_idx] = 0.0
            y_F[l_idx] = trans_lines[l_idx]["existing_capacity_MW"]

    # MILP investment variables, including transmission expansion, are integer
    # counts of the increments declared above.
    if sol.get("formulation", "MILP") == "MILP":
        integer_values = [
            (f"y_P_new[{g}]", y_P_new[g]) for g in G
        ] + [
            (f"y_P_ret[{g}]", y_P_ret[g]) for g in G
        ] + [
            (f"y_E_new[{g}]", y_E_new[g]) for g in G_STOR
        ] + [
            (f"y_F_new[{line}]", y_F_new[line]) for line in L
        ]
        for name, value in integer_values:
            difference = abs(float(value) - round(float(value)))
            if difference > 1e-4:
                record_violation(
                    50, float(value), float(round(float(value))), difference,
                    f"MILP investment variable {name}={value} is not integral",
                )

    # =========================================================================
    # Check Investment Constraints (Constraints 1-12) -- unchanged from original
    # =========================================================================

    # Constraint 1 (12a)
    for g in G:
        lhs_val = gamma_P[g] * y_P_new[g]
        rhs_val = y_P_max[g]
        check_leq(1, lhs_val, rhs_val,
                  f"Gen {g} ({generators[g]['resource_type']}): "
                  f"gamma_P*y_P_new={lhs_val:.2f} exceeds max_cap={rhs_val:.2f}")

    # Constraint 2 (12b)
    for g in G:
        lhs_val = gamma_P[g] * y_P_ret[g]
        rhs_val = y_P_0[g]
        check_leq(2, lhs_val, rhs_val,
                  f"Gen {g} ({generators[g]['resource_type']}): "
                  f"gamma_P*y_P_ret={lhs_val:.2f} exceeds existing_cap={rhs_val:.2f}")

    # Constraint 3 (12c)
    for g in G:
        if g in G_NONRET:
            lhs_val = y_P_ret[g]
            rhs_val = 0.0
            check_eq(3, lhs_val, rhs_val,
                     f"Gen {g} ({generators[g]['resource_type']}): "
                     f"non-retirable but y_P_ret={lhs_val:.6f}")

    # Constraint 4 (12d)
    for g in G:
        lhs_val = y_P[g]
        rhs_val = y_P_0[g] + gamma_P[g] * (y_P_new[g] - y_P_ret[g])
        check_eq(4, lhs_val, rhs_val,
                 f"Gen {g} ({generators[g]['resource_type']}): "
                 f"y_P={lhs_val:.4f} != y_P_0+gamma_P*(new-ret)={rhs_val:.4f}")

    # Constraint 5 (13a)
    for g in G_STOR:
        lhs_val = gamma_E[g] * y_E_new[g]
        rhs_val = y_E_max[g]
        check_leq(5, lhs_val, rhs_val,
                  f"Storage {g}: gamma_E*y_E_new={lhs_val:.2f} exceeds "
                  f"max_E_cap={rhs_val:.2f}")

    # Constraint 6 (13b)
    for g in G_STOR:
        lhs_val = gamma_E[g] * y_E_ret[g]
        rhs_val = y_E_0[g]
        check_leq(6, lhs_val, rhs_val,
                  f"Storage {g}: gamma_E*y_E_ret={lhs_val:.2f} exceeds "
                  f"existing_E_cap={rhs_val:.2f}")

    # Constraint 7 (13c)
    for g in G_STOR:
        if g in G_NONRET:
            lhs_val = y_E_ret[g]
            rhs_val = 0.0
            check_eq(7, lhs_val, rhs_val,
                     f"Storage {g}: non-retirable but y_E_ret={lhs_val:.6f}")

    # Constraint 8 (13d)
    for g in G_STOR:
        lhs_val = y_E[g]
        rhs_val = y_E_0[g] + gamma_E[g] * (y_E_new[g] - y_E_ret[g])
        check_eq(8, lhs_val, rhs_val,
                 f"Storage {g}: y_E={lhs_val:.4f} != "
                 f"y_E_0+gamma_E*(new-ret)={rhs_val:.4f}")

    # Constraint 9 (13e)
    for g in G_STOR:
        lhs_val = rho_min_stor[g] * y_P[g]
        rhs_val = y_E[g]
        check_leq(9, lhs_val, rhs_val,
                  f"Storage {g}: rho_min*y_P={lhs_val:.2f} exceeds "
                  f"y_E={rhs_val:.2f}")

    # Constraint 10 (13f)
    for g in G_STOR:
        lhs_val = rho_max_stor[g] * y_P[g]
        rhs_val = y_E[g]
        check_geq(10, lhs_val, rhs_val,
                  f"Storage {g}: rho_max*y_P={lhs_val:.2f} less than "
                  f"y_E={rhs_val:.2f}")

    # Constraint 11 (14a)
    for l_idx in L:
        lhs_val = y_F_new[l_idx]
        rhs_val = max(
            0.0,
            trans_lines[l_idx]["max_capacity_MW"]
            - trans_lines[l_idx]["existing_capacity_MW"],
        )
        check_leq(11, lhs_val, rhs_val,
                  f"Line {l_idx}: y_F_new={lhs_val:.2f} exceeds "
                  f"max_F_cap={rhs_val:.2f}")

    # Constraint 12 (14b)
    for l_idx in L:
        lhs_val = y_F[l_idx]
        rhs_val = trans_lines[l_idx]["existing_capacity_MW"] + y_F_new[l_idx]
        check_eq(12, lhs_val, rhs_val,
                 f"Line {l_idx}: y_F={lhs_val:.4f} != "
                 f"y_F_0+y_F_new={rhs_val:.4f}")

    # =========================================================================
    # Constraints 13-48: skipped (operational variables not in solution).
    # =========================================================================

    # =========================================================================
    # Constraint 49 (NEW): Objective-value envelope (Tier C anti-exploit)
    #
    # obj_lower = c^FIXED   (Eq. 29, exact from investment variables)
    # obj_upper = obj_lower + crude upper bounds on c^VAR, c^NSE, c^START,
    #             c^RPS, c^CO2 (loose but data-driven).
    # =========================================================================
    reported_obj = sol.get("objective_value")
    try:
        reported = float(reported_obj) if reported_obj is not None else None
    except (TypeError, ValueError):
        reported = None

    if reported is not None:
        # ------ Cost / parameter lookups (same conventions as gurobi_code.py) ------
        def gp_val(g, key, default=0.0):
            rtype = generators[g]["resource_type"]
            return gen_params[rtype].get(key, default)

        c_P_INV = {g: gp_val(g, "investment_cost_per_MW_yr") for g in G}
        c_P_FOM = {g: gp_val(g, "fixed_OM_cost_per_MW_yr") for g in G}
        c_VAR = {g: gp_val(g, "variable_cost_per_MWh") for g in G}
        c_START = {g: gp_val(g, "start_up_cost", 0.0) for g in G_UC}
        co2_rate = {g: gp_val(g, "CO2_emission_tons_per_MWh", 0.0) for g in G}

        # Energy-side costs are explicit data, including released zeros.
        if G_STOR and "battery" in gen_params:
            c_E_INV_stor = {g: gen_params["battery"].get(
                "energy_investment_cost_per_MWh_yr", 0.0) for g in G_STOR}
        else:
            c_E_INV_stor = {g: 0.0 for g in G_STOR}
        c_E_FOM_stor = {
            g: gen_params[generators[g]["resource_type"]].get(
                "energy_fixed_OM_cost_per_MWh_yr", 0.0
            )
            for g in G_STOR
        }
        c_E_INV_hydro = {
            g: gen_params[generators[g]["resource_type"]].get(
                "energy_investment_cost_per_MWh_yr", 0.0
            )
            for g in G_HYDRO
        }
        c_E_FOM_hydro = {
            g: gen_params[generators[g]["resource_type"]].get(
                "energy_fixed_OM_cost_per_MWh_yr", 0.0
            )
            for g in G_HYDRO
        }

        # Transmission investment cost.
        c_F_INV = {l_idx: trans_lines[l_idx].get("investment_cost_per_MW_yr", 0.0)
                   for l_idx in L}

        # ------ Subperiod weights and total weighted hours / demand ------
        params = inst.get("problem_parameters", {})
        num_weeks = params.get("num_weeks", 0)
        delta_w = params.get("hours_per_subperiod", 168)
        total_hours = params.get("total_hours", num_weeks * delta_w)
        subperiod_weights = inst.get("subperiod_weights", [1.0] * num_weeks)

        # alpha_t = beta_w / delta_w; sum over t = sum_w beta_w
        if delta_w > 0:
            total_weighted_hours = float(sum(subperiod_weights))
        else:
            total_weighted_hours = float(total_hours)

        # Total weighted demand = sum_{w,t in H_w, z} alpha_t * d_{z,t}
        # alpha is constant beta_w/delta_w within each subperiod; we collapse by week.
        demand = inst.get("demand", {})
        total_weighted_demand = 0.0
        if delta_w > 0:
            for z_key, dem_list in demand.items():
                # weight every hour by its subperiod's alpha
                for w in range(num_weeks):
                    beta_w = subperiod_weights[w] if w < len(subperiod_weights) else 1.0
                    alpha = beta_w / delta_w
                    t0 = w * delta_w
                    t1 = min((w + 1) * delta_w, len(dem_list))
                    for t in range(t0, t1):
                        total_weighted_demand += alpha * float(dem_list[t])

        # ------ obj_lower = c^FIXED (Eq. 29, exact) ------
        c_fixed = 0.0
        # Investment costs: sum_g c_P_INV * gamma_P * y_P_new = c_P_INV * new_MW
        for g in G:
            new_mw = gamma_P[g] * float(y_P_new[g])
            c_fixed += float(c_P_INV[g]) * new_mw
        # Storage energy investment: sum c_E_INV * gamma_E * y_E_new
        for g in G_STOR:
            c_fixed += float(c_E_INV_stor[g]) * gamma_E[g] * float(y_E_new[g])
            c_fixed += float(c_E_FOM_stor[g]) * float(y_E[g])
        for g in G_HYDRO:
            duration = float(gen_params[generators[g]["resource_type"]]["duration_MWh_per_MW"])
            c_fixed += float(c_E_INV_hydro[g]) * duration * float(y_P_new[g])
            c_fixed += float(c_E_FOM_hydro[g]) * duration * float(y_P[g])
        # Fixed O&M: sum_g c_P_FOM * y_P
        for g in G:
            c_fixed += float(c_P_FOM[g]) * float(y_P[g])
        # Transmission investment
        for l_idx in L:
            c_fixed += float(c_F_INV[l_idx]) * float(y_F_new[l_idx])

        obj_lower = c_fixed

        # ------ obj_upper = c^FIXED + crude upper bound on operational costs ------
        # Each generator can produce at most y_P[g] MWh per hour for all hours;
        # similarly storage withdrawals are bounded by y_P[g].
        c_var_max = 0.0
        for g in G:
            c_var_max += float(c_VAR[g]) * float(y_P[g]) * total_weighted_hours
        for g in G_STOR:
            c_var_max += float(c_VAR[g]) * float(y_P[g]) * total_weighted_hours

        # NSE upper bound: every segment's max fraction of demand, at its cost
        c_nse_max = 0.0
        for seg in inst.get("consumer_segments", []):
            seg_zone = seg.get("zone")
            seg_cost = float(seg.get("cost_per_MWh", 0.0))
            seg_frac = float(seg.get("max_nse_fraction", 0.0))
            if seg_cost <= 0.0 or seg_frac <= 0.0:
                continue
            zone_key = str(seg_zone)
            if zone_key not in demand:
                continue
            dem_list = demand[zone_key]
            zone_weighted_demand = 0.0
            if delta_w > 0:
                for w in range(num_weeks):
                    beta_w = subperiod_weights[w] if w < len(subperiod_weights) else 1.0
                    alpha = beta_w / delta_w
                    t0 = w * delta_w
                    t1 = min((w + 1) * delta_w, len(dem_list))
                    for t in range(t0, t1):
                        zone_weighted_demand += alpha * float(dem_list[t])
            c_nse_max += seg_cost * seg_frac * zone_weighted_demand

        # START upper bound: each UC unit could start every hour
        c_start_max = 0.0
        for g in G_UC:
            if gamma_P[g] <= 0:
                continue
            num_units_bound = float(y_P[g]) / gamma_P[g]
            c_start_max += float(c_START.get(g, 0.0)) * num_units_bound * total_weighted_hours

        # RPS / CO2 noncompliance penalties (loose upper bounds)
        policy = inst.get("policy", {})
        c_RPS = float(policy.get("RPS_noncompliance_cost_per_MWh", 0.0) or 0.0)
        c_CO2 = float(policy.get("CO2_noncompliance_cost_per_ton", 0.0) or 0.0)
        # RPS noncompliance bounded by total weighted demand (full RPS shortfall)
        c_rps_max = c_RPS * total_weighted_demand
        # CO2 noncompliance bounded by max possible emissions (all gens at full)
        co2_max_emissions = 0.0
        for g in G:
            co2_max_emissions += float(co2_rate.get(g, 0.0)) * float(y_P[g]) * total_weighted_hours
        for g in G_STOR:
            co2_max_emissions += float(co2_rate.get(g, 0.0)) * float(y_P[g]) * total_weighted_hours
        c_co2_max = c_CO2 * co2_max_emissions

        obj_upper = c_fixed + c_var_max + c_nse_max + c_start_max + c_rps_max + c_co2_max

        # Safety floor so even an all-zero investment doesn't make obj_upper
        # too small to catch MAX_FLOAT; tied to total demand at $1/MWh.
        obj_upper = max(obj_upper, 1.0 + total_weighted_demand)

        # ------ Compare ------
        tol_obj_lo = max(1e-3, 1e-3 * abs(obj_lower))
        tol_obj_hi = max(1e-3, 1e-3 * abs(obj_upper))

        # NaN / Inf are always rejected.
        if math.isnan(reported) or math.isinf(reported):
            record_violation(
                49, float("nan" if math.isnan(reported) else reported),
                float(obj_upper),
                float("inf"),
                f"Constraint 49 (Objective consistency): reported objective_value="
                f"{reported_obj!r} is not a finite number; "
                f"expected value in [{obj_lower:.6g}, {obj_upper:.6g}]."
            )
        else:
            shortfall = obj_lower - reported
            overshoot = reported - obj_upper
            if shortfall > tol_obj_lo:
                record_violation(
                    49, float(obj_lower), float(reported), float(shortfall),
                    f"Constraint 49 (Objective consistency): reported objective_value="
                    f"{reported:.6f} is below the lower bound obj_lower=c^FIXED="
                    f"{obj_lower:.6f}; shortfall={shortfall:.6g}, tol={tol_obj_lo:.6g}"
                )
            if overshoot > tol_obj_hi:
                record_violation(
                    49, float(reported), float(obj_upper), float(overshoot),
                    f"Constraint 49 (Objective consistency): reported objective_value="
                    f"{reported:.6f} exceeds worst-case upper bound obj_upper="
                    f"{obj_upper:.6f} (c^FIXED={c_fixed:.6f}, c^VAR_max={c_var_max:.6g}, "
                    f"c^NSE_max={c_nse_max:.6g}, c^START_max={c_start_max:.6g}, "
                    f"c^RPS_max={c_rps_max:.6g}, c^CO2_max={c_co2_max:.6g}); "
                    f"overshoot={overshoot:.6g}, tol={tol_obj_hi:.6g}"
                )

    # The envelope above remains a cheap early screen. For an otherwise valid
    # plan, validate the complete hourly witness by direct substitution.
    if not violated_constraints and reported is not None and math.isfinite(reported):
        try:
            if instance_path is None:
                raise ValueError("instance_path is required for solverless validation")
            expected, witness_errors, _ = validate_complete_witness(
                instance_path, sol, Path(__file__).resolve().parent
            )
            links = {}
            investments = sol.get("investments", {})
            for generator in range(len(inst["generators"])):
                entry = investments.get(str(generator), {})
                links[f"y_P_new[{generator}]"] = float(entry.get("new_units", 0.0))
                links[f"y_P_ret[{generator}]"] = abs(float(entry.get("retired_units", 0.0)))
                if "total_capacity_MW" in entry:
                    links[f"y_P[{generator}]"] = float(entry["total_capacity_MW"])
            storage = sol.get("storage_energy_investments", {})
            for key, entry in storage.items():
                generator = int(key)
                links[f"y_E_new[{generator}]"] = float(entry.get("new_units", 0.0))
                links[f"y_E[{generator}]"] = float(entry["total_energy_MWh"])
            transmission = sol.get("transmission_investments", {})
            for line in range(len(inst["transmission_lines"])):
                entry = transmission.get(str(line), {})
                links[f"y_F_new[{line}]"] = float(entry.get("new_MW", 0.0))
                if "total_capacity_MW" in entry:
                    links[f"y_F[{line}]"] = float(entry["total_capacity_MW"])
            witness_errors.extend(require_linked_variables(sol, links))
            if witness_errors or expected is None:
                raise ValueError("; ".join(witness_errors[:20]) or "missing witness objective")
        except Exception as error:
            record_violation(
                49,
                1.0,
                0.0,
                1.0,
                "Constraint 49 (Objective consistency): trusted fixed-investment "
                f"recourse failed: {type(error).__name__}: {str(error)[:300]}",
            )
        else:
            difference = abs(reported - expected)
            objective_tol = max(1e-3, 1e-4 * max(1.0, abs(expected)))
            if difference > objective_tol:
                record_violation(
                    49,
                    reported,
                    expected,
                    difference,
                    "Constraint 49 (Objective consistency): reported "
                    f"objective_value={reported:.6f} does not equal trusted "
                    f"fixed-investment cost={expected:.6f} "
                    f"(difference={difference:.6g}, tol={objective_tol:.6g})",
                )

    # =========================================================================
    # Build output (matches original feasibility_check.py shape)
    # =========================================================================
    sorted_violated = sorted(violated_constraints)
    aggregated_violations = []
    for c in sorted_violated:
        msgs = violation_details[c]
        if len(msgs) == 1:
            aggregated_violations.append(msgs[0])
        else:
            aggregated_violations.append(
                f"Constraint {c}: {len(msgs)} violations. "
                f"Example: {msgs[0]}")

    return {
        "feasible": len(sorted_violated) == 0,
        "violated_constraints": sorted_violated,
        "violations": aggregated_violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for energy systems planning solutions")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)
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

    result = check_feasibility(inst, sol, args.instance_path)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE (investment constraints + exact recourse objective).")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
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
