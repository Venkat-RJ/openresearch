"""
Nurse Scheduling Problem (NSP) — Compact ILP formulation solved with Gurobi.

This implements a compact reformulation equivalent to the extended formulation
(1a)-(1f) from Legrain & Omer (2024). The paper uses an exponential number of
roster variables in a set-partitioning style formulation, solved via column
generation. Since enumerating all rosters is impractical for direct ILP
submission, we instead use a compact formulation with direct shift-assignment
variables that models the same constraints and objective.

Variables:
    w[i,d,s]       binary  — 1 if nurse i works shift s on day d
                              (s includes "None" for day off)
    y[i,d,s,sigma] binary  — 1 if nurse i performs skill sigma on day d, shift s
                              (only for work shifts s != "None"; only created for
                               skills the nurse possesses)
    z[d,s,sigma]   >= 0    — unsatisfied demand for skill sigma on day d, shift s

ASSUMPTION (boundary handling for min-consecutive constraints): Streaks of
working days (or days off) that start at day 1 or end at the last day of the
horizon are *not* penalized for being shorter than the minimum. This is because
those streaks may be part of a longer streak that extends beyond the planning
horizon.
"""

import argparse
import json
import sys

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
import time
# Walk up from this file's directory to find repo root (containing scripts/).
_GUROBI_CODE_START_TIME = time.time()
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass
def load_instance(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_model(inst: dict) -> tuple:
    """Build the Gurobi model and return (model, metadata needed for solution extraction)."""

    meta = inst["instance_metadata"]
    num_nurses = meta["num_nurses"]
    num_days = meta["num_days"]
    shifts = inst["shifts"]  # work shifts only (e.g. ["Early", "Night"])
    skills = inst["skills"]
    nurses = inst["nurses"]
    weekends = inst["weekends"]
    demand = inst["demand"]
    demand_penalty = inst["unsatisfied_demand_penalty"]
    hard = inst["constraints"]["hard_constraints"]
    soft = inst["constraints"]["soft_constraints"]

    all_shifts = shifts + ["None"]  # include "None" as day-off shift
    days = list(range(1, num_days + 1))

    # Build nurse skill sets
    nurse_skills = {}
    for nurse in nurses:
        nurse_skills[nurse["id"]] = set(nurse["skills"])

    nurse_ids = [n["id"] for n in nurses]

    model = gp.Model("NSP_Legrain2024")
    model.setParam("Threads", 1)

    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------

    # w[i,d,s]: nurse i works shift s on day d
    w = {}
    for i in nurse_ids:
        for d in days:
            for s in all_shifts:
                w[i, d, s] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{d}_{s}")

    # y[i,d,s,sigma]: nurse i uses skill sigma on day d, shift s.  When skill
    # possession is hard, unavailable skills are omitted.  The INRC-I release
    # variant makes possession soft, so all skills must remain selectable and
    # a mismatch is charged in the objective.
    y = {}
    hard_skill_requirement = bool(hard.get("skill_requirement", False))
    for i in nurse_ids:
        for d in days:
            for s in shifts:
                for sigma in skills:
                    if not hard_skill_requirement or sigma in nurse_skills[i]:
                        y[i, d, s, sigma] = model.addVar(
                            vtype=GRB.BINARY, name=f"y_{i}_{d}_{s}_{sigma}"
                        )

    # z[d,s,sigma]: unsatisfied demand
    z = {}
    for d in days:
        for s in shifts:
            for sigma in skills:
                z[d, s, sigma] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"z_{d}_{s}_{sigma}"
                )

    model.update()

    # -----------------------------------------------------------------------
    # Hard constraints
    # -----------------------------------------------------------------------

    # (HC1) Single assignment per day: each nurse works exactly one shift
    # (including "None") per day.
    for i in nurse_ids:
        for d in days:
            model.addConstr(
                gp.quicksum(w[i, d, s] for s in all_shifts) == 1,
                name=f"single_assign_{i}_{d}",
            )

    # (HC2) Forbidden patterns
    for fp in hard.get("forbidden_patterns", []):
        pattern = fp["pattern"]
        # Pattern is a list of [day_type, shift] pairs for consecutive days.
        # e.g. [["AnyDay","Night"],["AnyDay","Early"]] means Night on day d
        # followed by Early on day d+1 is forbidden.
        pat_len = len(pattern)
        if pat_len < 2:
            continue  # single-day patterns don't constrain consecutive days

        for i in nurse_ids:
            for d in days:
                # Check if the window d..d+pat_len-1 fits in the horizon
                if d + pat_len - 1 > num_days:
                    continue
                # Build the constraint: sum of matching w variables <= pat_len - 1
                terms = []
                valid = True
                for offset, (day_type, shift_name) in enumerate(pattern):
                    dd = d + offset
                    if shift_name not in all_shifts:
                        valid = False
                        break
                    terms.append(w[i, dd, shift_name])
                if valid and terms:
                    model.addConstr(
                        gp.quicksum(terms) <= pat_len - 1,
                        name=f"forbidden_{fp['description'][:20]}_{i}_{d}",
                    )

    # -----------------------------------------------------------------------
    # Linking constraints
    # -----------------------------------------------------------------------

    # (LC1) For work shifts: w[i,d,s] = sum_sigma y[i,d,s,sigma]
    for i in nurse_ids:
        for d in days:
            for s in shifts:
                skill_vars = [
                    y[i, d, s, sigma]
                    for sigma in skills
                    if (i, d, s, sigma) in y
                ]
                if skill_vars:
                    model.addConstr(
                        w[i, d, s] == gp.quicksum(skill_vars),
                        name=f"link_wy_{i}_{d}_{s}",
                    )
                else:
                    # Nurse has no skills for this shift — cannot work it
                    model.addConstr(
                        w[i, d, s] == 0,
                        name=f"no_skill_{i}_{d}_{s}",
                    )

    # (LC2) Demand coverage.  Most variants allow penalized shortfall z.  The
    # INRC-I-style instance explicitly requests exact hard demand.
    hard_demand_equality = bool(hard.get("demand_equality", False))
    for d in days:
        day_key = f"day_{d}"
        if day_key not in demand:
            continue
        for s in shifts:
            if s not in demand[day_key]:
                continue
            for sigma in skills:
                if sigma not in demand[day_key][s]:
                    continue
                b = demand[day_key][s][sigma]
                assigned = [
                    y[i, d, s, sigma]
                    for i in nurse_ids
                    if (i, d, s, sigma) in y
                ]
                if hard_demand_equality:
                    model.addConstr(
                        gp.quicksum(assigned) == b,
                        name=f"hard_demand_equal_{d}_{s}_{sigma}",
                    )
                    model.addConstr(
                        z[d, s, sigma] == 0,
                        name=f"hard_demand_no_slack_{d}_{s}_{sigma}",
                    )
                else:
                    model.addConstr(
                        gp.quicksum(assigned) + z[d, s, sigma] >= b,
                        name=f"demand_{d}_{s}_{sigma}",
                    )

    # -----------------------------------------------------------------------
    # Helper: work[i,d] = 1 - w[i,d,"None"] (continuous, derived)
    # We'll use expressions inline rather than extra variables.
    # -----------------------------------------------------------------------

    def work_expr(i, d):
        """Expression equal to 1 if nurse i works on day d, 0 otherwise."""
        return 1 - w[i, d, "None"]

    # -----------------------------------------------------------------------
    # Soft constraints — penalty variables and objective terms
    # -----------------------------------------------------------------------
    obj_terms = []

    # Soft skill possession: every mismatch is visible in y and charged once.
    if not hard_skill_requirement:
        mismatch_penalty = float(soft.get("skill_requirement_penalty", 0) or 0)
        for (i, d, s, sigma), variable in y.items():
            if sigma not in nurse_skills[i]:
                obj_terms.append(mismatch_penalty * variable)

    # Soft forbidden patterns (the hard versions were added above).
    for fp_index, fp in enumerate(soft.get("forbidden_patterns", []) or []):
        pattern = fp.get("pattern", [])
        pat_len = len(pattern)
        if not pattern:
            continue
        penalty = float(fp.get("penalty", 0) or 0)
        for i in nurse_ids:
            for d in days:
                if d + pat_len - 1 > num_days:
                    continue
                terms = []
                valid = True
                for offset, (_day_type, shift_name) in enumerate(pattern):
                    if shift_name not in all_shifts:
                        valid = False
                        break
                    terms.append(w[i, d + offset, shift_name])
                if valid:
                    violation = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"soft_forbidden_{fp_index}_{i}_{d}",
                    )
                    model.addConstr(
                        violation >= gp.quicksum(terms) - (pat_len - 1),
                        name=f"soft_forbidden_link_{fp_index}_{i}_{d}",
                    )
                    obj_terms.append(penalty * violation)

    # --- (SC1) Total working days [min, max] per nurse ---
    tw = soft.get("total_working_days")
    if tw:
        tw_min = tw["min"]
        tw_max = tw["max"]
        pen_under = tw["penalty_under"]
        pen_over = tw["penalty_over"]
        for i in nurse_ids:
            over = model.addVar(lb=0, name=f"over_total_{i}")
            under = model.addVar(lb=0, name=f"under_total_{i}")
            total_work = gp.quicksum(work_expr(i, d) for d in days)
            model.addConstr(over >= total_work - tw_max, name=f"total_over_{i}")
            model.addConstr(under >= tw_min - total_work, name=f"total_under_{i}")
            obj_terms.append(pen_over * over)
            obj_terms.append(pen_under * under)

    # --- (SC2) Total worked weekends [max] per nurse ---
    # Always build wk_worked indicators so downstream HARD constraints
    # (consecutive_worked_weekends) can reference them even when total_worked
    # _weekends is in hard_constraints rather than soft_constraints.
    tww = soft.get("total_worked_weekends")
    wk_worked = {}  # wk_worked[i,wk_idx] binary
    h_tww_present = isinstance(hard.get("consecutive_worked_weekends"), dict) or \
                    isinstance(hard.get("total_worked_weekends"), dict)
    if tww or h_tww_present:
        for i in nurse_ids:
            for wk in weekends:
                wk_idx = wk["week"]
                weekend_days = [
                    wk[key] for key in ("friday", "saturday", "sunday")
                    if key in wk
                ]
                wk_var = model.addVar(vtype=GRB.BINARY, name=f"wk_worked_{i}_{wk_idx}")
                wk_worked[i, wk_idx] = wk_var
                for weekend_day in weekend_days:
                    for s in shifts:
                        model.addConstr(
                            wk_var >= w[i, weekend_day, s],
                            name=f"wk_link_{weekend_day}_{i}_{wk_idx}_{s}",
                        )
                model.addConstr(
                    wk_var
                    <= gp.quicksum(
                        w[i, weekend_day, s]
                        for weekend_day in weekend_days for s in shifts
                    ),
                    name=f"wk_upper_{i}_{wk_idx}",
                )

    if tww:
        max_wk = tww["max"]
        pen_wk = tww["penalty"]
        for i in nurse_ids:
            over_wk = model.addVar(lb=0, name=f"over_weekends_{i}")
            model.addConstr(
                over_wk
                >= gp.quicksum(
                    wk_worked[i, wk["week"]] for wk in weekends
                )
                - max_wk,
                name=f"total_wk_over_{i}",
            )
            obj_terms.append(pen_wk * over_wk)

    # --- (SC3) Consecutive working days [min, max] ---
    cwd = soft.get("consecutive_working_days")
    if cwd:
        cwd_min = cwd["min"]
        cwd_max = cwd["max"]
        pen_cwd_under = cwd["penalty_under"]
        pen_cwd_over = cwd["penalty_over"]

        for i in nurse_ids:
            # Max consecutive working days: in any window of (cwd_max+1) days,
            # at most cwd_max are working days.
            window = cwd_max + 1
            for d in days:
                if d + window - 1 > num_days:
                    break
                slack = model.addVar(lb=0, name=f"consec_work_over_{i}_{d}")
                model.addConstr(
                    gp.quicksum(work_expr(i, d + k) for k in range(window))
                    <= cwd_max + slack,
                    name=f"max_consec_work_{i}_{d}",
                )
                obj_terms.append(pen_cwd_over * slack)

            # Min consecutive working days: penalize short streaks.
            # A streak of length g (1 <= g < cwd_min) starting at day d is
            # detected when day d-1 is off (or d==1), days d..d+g-1 are working,
            # and day d+g is off (or d+g > num_days).
            #
            # ASSUMPTION: We do NOT penalize boundary streaks (starting at day 1
            # or ending at the last day) since they may extend beyond the horizon.
            for g in range(1, cwd_min):
                for d in days:
                    end_day = d + g - 1
                    if end_day > num_days:
                        break

                    # We need day d-1 to be off and day d+g to be off
                    # (boundary: d==1 or d+g > num_days means boundary streak,
                    #  skip per assumption)
                    if d == 1 or end_day + 1 > num_days:
                        continue

                    # Violation if: off on d-1 AND work on d..end_day AND off on d+g
                    # That is: w[i,d-1,None]=1, work(i,d')=1 for d'=d..end_day, w[i,end_day+1,None]=1
                    #
                    # Linearize: viol >= w[i,d-1,None] + sum_{d'=d}^{end_day} work(i,d') + w[i,end_day+1,None] - (g+1)
                    # If all conditions hold, RHS = 1 + g + 1 - (g+1) = 1, so viol >= 1
                    # Otherwise RHS <= 0
                    viol = model.addVar(lb=0, name=f"short_work_{i}_{d}_{g}")
                    model.addConstr(
                        viol
                        >= w[i, d - 1, "None"]
                        + gp.quicksum(work_expr(i, d + k) for k in range(g))
                        + w[i, end_day + 1, "None"]
                        - (g + 1),
                        name=f"min_consec_work_{i}_{d}_{g}",
                    )
                    obj_terms.append(pen_cwd_under * viol)

    # --- (SC4) Consecutive days off [min, max] ---
    cdo = soft.get("consecutive_days_off")
    if cdo:
        cdo_min = cdo["min"]
        cdo_max = cdo["max"]
        pen_cdo_under = cdo["penalty_under"]
        pen_cdo_over = cdo["penalty_over"]

        for i in nurse_ids:
            # Max consecutive days off: in any window of (cdo_max+1) days,
            # at most cdo_max are off.
            window = cdo_max + 1
            for d in days:
                if d + window - 1 > num_days:
                    break
                slack = model.addVar(lb=0, name=f"consec_off_over_{i}_{d}")
                model.addConstr(
                    gp.quicksum(w[i, d + k, "None"] for k in range(window))
                    <= cdo_max + slack,
                    name=f"max_consec_off_{i}_{d}",
                )
                obj_terms.append(pen_cdo_over * slack)

            # Min consecutive days off: penalize short rest streaks.
            # ASSUMPTION: boundary streaks not penalized (same as working days).
            for g in range(1, cdo_min):
                for d in days:
                    end_day = d + g - 1
                    if end_day > num_days:
                        break
                    if d == 1 or end_day + 1 > num_days:
                        continue

                    # Violation if: work on d-1, off on d..end_day, work on d+g
                    viol = model.addVar(lb=0, name=f"short_off_{i}_{d}_{g}")
                    model.addConstr(
                        viol
                        >= work_expr(i, d - 1)
                        + gp.quicksum(w[i, d + k, "None"] for k in range(g))
                        + work_expr(i, end_day + 1)
                        - (g + 1),
                        name=f"min_consec_off_{i}_{d}_{g}",
                    )
                    obj_terms.append(pen_cdo_under * viol)

    # --- (SC5) Consecutive same shift [max] ---
    css = soft.get("consecutive_same_shift")
    if css:
        css_max = css["max"]
        pen_css = css["penalty"]
        window = css_max + 1
        for i in nurse_ids:
            for s in shifts:
                for d in days:
                    if d + window - 1 > num_days:
                        break
                    slack = model.addVar(lb=0, name=f"consec_shift_{i}_{s}_{d}")
                    model.addConstr(
                        gp.quicksum(w[i, d + k, s] for k in range(window))
                        <= css_max + slack,
                        name=f"max_consec_shift_{i}_{s}_{d}",
                    )
                    obj_terms.append(pen_css * slack)

    # --- (SC6) Consecutive worked weekends [max] ---
    cww = soft.get("consecutive_worked_weekends")
    if cww and wk_worked:
        cww_max = cww["max"]
        pen_cww = cww["penalty"]
        num_weekends = len(weekends)
        wk_indices = [wk["week"] for wk in weekends]

        window = cww_max + 1
        for i in nurse_ids:
            for start in range(len(wk_indices)):
                if start + window > num_weekends:
                    break
                slack = model.addVar(
                    lb=0,
                    name=f"consec_wk_{i}_{wk_indices[start]}",
                )
                model.addConstr(
                    gp.quicksum(
                        wk_worked[i, wk_indices[start + k]] for k in range(window)
                    )
                    <= cww_max + slack,
                    name=f"max_consec_wk_{i}_{wk_indices[start]}",
                )
                obj_terms.append(pen_cww * slack)

    # --- (SC7) Identical weekend ---
    iw = soft.get("identical_weekend")
    if iw and iw.get("enabled", False):
        pen_iw = iw["penalty"]
        for i in nurse_ids:
            for wk in weekends:
                wk_idx = wk["week"]
                weekend_days = [
                    wk[key] for key in ("friday", "saturday", "sunday")
                    if key in wk
                ]
                if len(weekend_days) < 2:
                    continue
                base_day = weekend_days[0]
                viol = model.addVar(
                    vtype=GRB.BINARY, name=f"ident_wk_viol_{i}_{wk_idx}"
                )
                # If any constituent day differs from the first weekend day,
                # charge the weekend once.
                for other_day in weekend_days[1:]:
                    for s in all_shifts:
                        model.addConstr(
                            viol >= w[i, base_day, s] - w[i, other_day, s],
                            name=f"ident_wk_a_{i}_{wk_idx}_{other_day}_{s}",
                        )
                        model.addConstr(
                            viol >= w[i, other_day, s] - w[i, base_day, s],
                            name=f"ident_wk_b_{i}_{wk_idx}_{other_day}_{s}",
                        )
                obj_terms.append(pen_iw * viol)

    # --- (SC8) Nurse preferences ---
    for nurse in nurses:
        i = nurse["id"]
        for pref in nurse.get("preferences", []):
            day = pref["day"]
            ptype = pref["type"]
            shift_pref = pref["shift"]
            penalty = pref["penalty"]

            if day < 1 or day > num_days:
                continue

            if ptype == "day_off":
                # Nurse prefers day off: penalize if not off
                viol = model.addVar(lb=0, name=f"pref_off_{i}_{day}")
                model.addConstr(
                    viol >= 1 - w[i, day, "None"],
                    name=f"pref_off_c_{i}_{day}",
                )
                obj_terms.append(penalty * viol)
            elif ptype == "day_on":
                # Nurse prefers to work: penalize if off
                viol = model.addVar(lb=0, name=f"pref_on_{i}_{day}")
                model.addConstr(
                    viol >= w[i, day, "None"],
                    name=f"pref_on_c_{i}_{day}",
                )
                obj_terms.append(penalty * viol)
            elif ptype == "shift_on":
                # Nurse prefers a specific shift: penalize if not assigned that shift
                if shift_pref in all_shifts:
                    viol = model.addVar(lb=0, name=f"pref_shift_{i}_{day}_{shift_pref}")
                    model.addConstr(
                        viol >= 1 - w[i, day, shift_pref],
                        name=f"pref_shift_c_{i}_{day}_{shift_pref}",
                    )
                    obj_terms.append(penalty * viol)
            elif ptype == "shift_off":
                # Nurse prefers not to work a specific shift
                if shift_pref in all_shifts:
                    viol = model.addVar(lb=0, name=f"pref_noshift_{i}_{day}_{shift_pref}")
                    model.addConstr(
                        viol >= w[i, day, shift_pref],
                        name=f"pref_noshift_c_{i}_{day}_{shift_pref}",
                    )
                    obj_terms.append(penalty * viol)

    # --- Unsatisfied demand penalty (c_z, math_model §PARAMETERS) ---
    for key in z:
        obj_terms.append(demand_penalty * z[key])

    # -----------------------------------------------------------------------
    # Hard versions of the same constraints (when instance puts a constraint
    # in `hard_constraints` instead of `soft_constraints`, e.g. large_31).
    # These have no slack — strict feasibility.
    # -----------------------------------------------------------------------

    # Hard: total_working_days
    h_tw = hard.get("total_working_days") if isinstance(hard.get("total_working_days"), dict) else None
    if h_tw:
        for i in nurse_ids:
            total_work = gp.quicksum(work_expr(i, d) for d in days)
            if "max" in h_tw:
                model.addConstr(total_work <= h_tw["max"], name=f"hard_total_max_{i}")
            if "min" in h_tw:
                model.addConstr(total_work >= h_tw["min"], name=f"hard_total_min_{i}")

    # Hard: consecutive_working_days (max + min)
    h_cwd = hard.get("consecutive_working_days") if isinstance(hard.get("consecutive_working_days"), dict) else None
    if h_cwd:
        cmax = h_cwd.get("max")
        cmin = h_cwd.get("min")
        for i in nurse_ids:
            if cmax is not None:
                window = cmax + 1
                for d in days:
                    if d + window - 1 > num_days:
                        break
                    model.addConstr(
                        gp.quicksum(work_expr(i, d + k) for k in range(window)) <= cmax,
                        name=f"hard_max_consec_work_{i}_{d}",
                    )
            if cmin is not None:
                # Hard min: a working streak shorter than cmin is forbidden.
                # Forbid pattern off-(work for g)-off where 1 <= g < cmin.
                for g in range(1, cmin):
                    for d in days:
                        end_day = d + g - 1
                        if end_day > num_days or d == 1 or end_day + 1 > num_days:
                            continue
                        # Disallow: w[i,d-1,None] + work(d..end_day) + w[i,end_day+1,None] >= g+2
                        # i.e. forbid the case where this exactly equals g+2.
                        # Linear: sum <= g+1 ensures NOT all three conditions fully meet.
                        model.addConstr(
                            w[i, d - 1, "None"]
                            + gp.quicksum(work_expr(i, d + k) for k in range(g))
                            + w[i, end_day + 1, "None"]
                            <= g + 1,
                            name=f"hard_min_consec_work_{i}_{d}_{g}",
                        )

    # Hard: consecutive_days_off (max + min)
    h_cdo = hard.get("consecutive_days_off") if isinstance(hard.get("consecutive_days_off"), dict) else None
    if h_cdo:
        cmax = h_cdo.get("max")
        cmin = h_cdo.get("min")
        for i in nurse_ids:
            if cmax is not None:
                window = cmax + 1
                for d in days:
                    if d + window - 1 > num_days:
                        break
                    model.addConstr(
                        gp.quicksum(w[i, d + k, "None"] for k in range(window)) <= cmax,
                        name=f"hard_max_consec_off_{i}_{d}",
                    )
            if cmin is not None:
                for g in range(1, cmin):
                    for d in days:
                        end_day = d + g - 1
                        if end_day > num_days or d == 1 or end_day + 1 > num_days:
                            continue
                        model.addConstr(
                            work_expr(i, d - 1)
                            + gp.quicksum(w[i, d + k, "None"] for k in range(g))
                            + work_expr(i, end_day + 1)
                            <= g + 1,
                            name=f"hard_min_consec_off_{i}_{d}_{g}",
                        )

    # Hard: consecutive_same_shift (max only)
    h_css = hard.get("consecutive_same_shift") if isinstance(hard.get("consecutive_same_shift"), dict) else None
    if h_css and "max" in h_css:
        cmax = h_css["max"]
        window = cmax + 1
        for i in nurse_ids:
            for s in shifts:
                for d in days:
                    if d + window - 1 > num_days:
                        break
                    model.addConstr(
                        gp.quicksum(w[i, d + k, s] for k in range(window)) <= cmax,
                        name=f"hard_max_consec_shift_{i}_{s}_{d}",
                    )

    # Hard: consecutive_worked_weekends (max only)
    h_cww = hard.get("consecutive_worked_weekends") if isinstance(hard.get("consecutive_worked_weekends"), dict) else None
    if h_cww and "max" in h_cww and wk_worked:
        cmax = h_cww["max"]
        window = cmax + 1
        wks_sorted = sorted({k[1] for k in wk_worked})
        for i in nurse_ids:
            for start_idx in range(len(wks_sorted) - window + 1):
                wk_window = wks_sorted[start_idx : start_idx + window]
                model.addConstr(
                    gp.quicksum(wk_worked[i, wk] for wk in wk_window) <= cmax,
                    name=f"hard_max_consec_wkends_{i}_{start_idx}",
                )

    # -----------------------------------------------------------------------
    # Objective: minimize total penalty
    # -----------------------------------------------------------------------
    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

    return model, w, y, nurse_ids, days, all_shifts, shifts, num_days


def extract_solution(model, w, y, nurse_ids, days, all_shifts, shifts, num_days):
    """Extract the solution from a solved model and return a dict."""
    if model.SolCount == 0:
        return {"objective_value": None, "schedule": {}}

    schedule = {}
    skill_assignments = {}
    reported_skills = sorted({key[3] for key in y})
    for i in nurse_ids:
        nurse_schedule = {}
        nurse_skill_schedule = {}
        for d in days:
            assigned_shift = "None"
            for s in all_shifts:
                if w[i, d, s].X > 0.5:
                    assigned_shift = s
                    break
            nurse_schedule[f"day_{d}"] = assigned_shift
            assigned_skill = "None"
            if assigned_shift != "None":
                for sigma in reported_skills:
                    key = (i, d, assigned_shift, sigma)
                    if key in y and y[key].X > 0.5:
                        assigned_skill = sigma
                        break
            nurse_skill_schedule[f"day_{d}"] = assigned_skill
        schedule[i] = nurse_schedule
        skill_assignments[i] = nurse_skill_schedule

    return {
        "objective_value": model.ObjVal,
        "schedule": schedule,
        "skill_assignments": skill_assignments,
        "solver_status": model.Status,
        "best_bound": model.ObjBound,
        "mip_gap": model.MIPGap,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Solve Nurse Scheduling Problem (Legrain & Omer 2024) with Gurobi"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to instance JSON file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path for output solution JSON (default: gurobi_solution_1.json)",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=300,
        help="Time limit in seconds (default: 300)",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    inst = load_instance(args.instance_path)

    # Build model
    model, w, y, nurse_ids, days, all_shifts, shifts, num_days = build_model(inst)

    # Set parameters
    model.setParam("TimeLimit", args.time_limit)

    # Solve
    model.optimize()

    # Extract and save solution
    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        solution = extract_solution(model, w, y, nurse_ids, days, all_shifts, shifts, num_days)
    else:
        print(f"Model status: {model.Status} — no feasible solution found.")
        solution = {"objective_value": None, "schedule": {}}

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
