"""
Gurobi implementation of a compact MIP for the Nurse Scheduling Problem from:
Legrain, Omer, Rosat (2020),
"A rotation-based branch-and-price approach for the nurse scheduling problem",
Computers & Operations Research, 122, 104944.

The paper uses an INRC-II formulation with rotation-based branch-and-price.
Since rotation enumeration is specific to the B&P approach, this file implements
an equivalent compact MIP formulation that captures the same constraints (H1-H4)
and soft-constraint penalties (S1-S7) using standard decision variables.
"""

import argparse
import json
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


def load_instance(path):
    """Load an INRC-II nurse scheduling instance from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    num_nurses = data["num_nurses"]
    num_days = data["num_days"]
    num_weeks = data["num_weeks"]
    skills = data["skills"]
    shifts_data = data["shifts"]
    shift_names = [s["name"] for s in shifts_data]
    num_shifts = len(shift_names)
    num_skills = len(skills)

    # Index mappings
    shift_idx = {name: idx for idx, name in enumerate(shift_names)}
    skill_idx = {name: idx for idx, name in enumerate(skills)}

    # Shift parameters
    CS_minus = {s: shifts_data[shift_idx[s]]["CS_minus"] for s in shift_names}
    CS_plus = {s: shifts_data[shift_idx[s]]["CS_plus"] for s in shift_names}

    # Forbidden successions as index pairs
    forbidden = []
    for pair in data["forbidden_shift_successions"]:
        forbidden.append((shift_idx[pair[0]], shift_idx[pair[1]]))

    # Penalty weights
    pw = data["penalty_weights"]
    c_S1 = pw["c_S1"]
    c_S2a = pw["c_S2a"]
    c_S2b = pw["c_S2b"]
    c_S3 = pw["c_S3"]
    c_S4 = pw["c_S4"]
    c_S5 = pw["c_S5"]
    c_S6 = pw["c_S6"]
    c_S7 = pw["c_S7"]

    # Nurse data
    nurses = data["nurses"]
    nurse_skills = {}  # nurse_id -> set of skill indices
    for nurse in nurses:
        i = nurse["id"]
        nurse_skills[i] = set(skill_idx[sk] for sk in nurse["skills"])

    # Nurses that can perform each skill
    nurses_with_skill = {sigma: [] for sigma in range(num_skills)}
    for i in range(num_nurses):
        for sigma in nurse_skills[i]:
            nurses_with_skill[sigma].append(i)

    # Nurse contracts
    L_minus = {}
    L_plus = {}
    CD_minus = {}
    CD_plus = {}
    CR_minus = {}
    CR_plus = {}
    WE_plus = {}
    for nurse in nurses:
        i = nurse["id"]
        c = nurse["contract"]
        L_minus[i] = c["L_minus"]
        L_plus[i] = c["L_plus"]
        CD_minus[i] = c["CD_minus"]
        CD_plus[i] = c["CD_plus"]
        CR_minus[i] = c["CR_minus"]
        CR_plus[i] = c["CR_plus"]
        WE_plus[i] = c["WE_plus"]

    # Nurse preferences (undesired assignments)
    prefs = {}  # (nurse_id, day, shift_idx) -> True
    for nurse in nurses:
        i = nurse["id"]
        for p in nurse["preferences"]:
            prefs[(i, p["day"], shift_idx[p["shift"]])] = True

    # Demand data: demand[day][shift_name][skill_name] -> {minimum, optimal}
    demand_data = data["demand"]
    D_min = {}  # (day, shift_idx, skill_idx) -> minimum demand
    D_opt = {}  # (day, shift_idx, skill_idx) -> optimal demand
    for k in range(num_days):
        for s_name in shift_names:
            s = shift_idx[s_name]
            for sk_name in skills:
                sigma = skill_idx[sk_name]
                d_entry = demand_data[k][s_name][sk_name]
                D_min[(k, s, sigma)] = d_entry["minimum"]
                D_opt[(k, s, sigma)] = d_entry["optimal"]

    # Weekend days: assuming week starts Monday (day 0 = Monday of week 1)
    # Saturday = day 5, 12, 19, 26; Sunday = day 6, 13, 20, 27
    weekends = []
    for m in range(num_weeks):
        sat = 7 * m + 5
        sun = 7 * m + 6
        if sat < num_days and sun < num_days:
            weekends.append((m, sat, sun))

    # -------------------------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("NSP_Legrain2020")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    model.setParam("MIPFocus", 1)  # focus on finding feasible solutions quickly

    # ----- Decision variables -----

    # x[i,k,s]: nurse i works shift s on day k
    x = {}
    for i in range(num_nurses):
        for k in range(num_days):
            for s in range(num_shifts):
                x[i, k, s] = model.addVar(vtype=GRB.BINARY,
                                          name=f"x_{i}_{k}_{s}")

    # work[i,k]: nurse i works on day k (any shift)
    work = {}
    for i in range(num_nurses):
        for k in range(num_days):
            work[i, k] = model.addVar(vtype=GRB.BINARY,
                                      name=f"work_{i}_{k}")

    # y[i,k,s,sigma]: nurse i performs skill sigma on shift s, day k
    y = {}
    for i in range(num_nurses):
        for k in range(num_days):
            for s in range(num_shifts):
                for sigma in nurse_skills[i]:
                    y[i, k, s, sigma] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"y_{i}_{k}_{s}_{sigma}")

    # z[k,s,sigma]: optimal demand violation
    z = {}
    for k in range(num_days):
        for s in range(num_shifts):
            for sigma in range(num_skills):
                z[k, s, sigma] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                              name=f"z_{k}_{s}_{sigma}")

    # S6: total assignment violations
    w_under = {}
    w_over = {}
    for i in range(num_nurses):
        w_under[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                  name=f"w_under_{i}")
        w_over[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                 name=f"w_over_{i}")

    # S7: weekend violation
    w_we = {}
    for i in range(num_nurses):
        w_we[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                name=f"w_we_{i}")

    # Weekend worked indicator: we_worked[i,m] = 1 if nurse works at least
    # one day of weekend m
    we_worked = {}
    for i in range(num_nurses):
        for m, sat, sun in weekends:
            we_worked[i, m] = model.addVar(vtype=GRB.BINARY,
                                           name=f"we_worked_{i}_{m}")

    # S5: complete weekend violation
    we_viol = {}
    for i in range(num_nurses):
        for m, sat, sun in weekends:
            we_viol[i, m] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                         name=f"we_viol_{i}_{m}")

    # S2a: consecutive worked days upper-bound violation
    cd_over = {}
    for i in range(num_nurses):
        cdp = CD_plus[i]
        for k in range(num_days - cdp):
            cd_over[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                         name=f"cd_over_{i}_{k}")

    # S2a: consecutive worked days lower-bound violation
    # start_work[i,k] indicates start of a work streak at day k
    start_work = {}
    for i in range(num_nurses):
        for k in range(num_days):
            start_work[i, k] = model.addVar(vtype=GRB.BINARY,
                                            name=f"start_work_{i}_{k}")

    cd_under = {}
    for i in range(num_nurses):
        cdm = CD_minus[i]
        for k in range(num_days):
            for j in range(1, cdm):
                if k + j < num_days:
                    cd_under[i, k, j] = model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0,
                        name=f"cd_under_{i}_{k}_{j}")

    # S2b: consecutive same-shift upper-bound violation
    cs_over = {}
    for i in range(num_nurses):
        for s in range(num_shifts):
            csp = CS_plus[shift_names[s]]
            for k in range(num_days - csp):
                cs_over[i, k, s] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0,
                    name=f"cs_over_{i}_{k}_{s}")

    # S2b: consecutive same-shift lower-bound violation
    # start_shift[i,k,s] indicates start of a consecutive-same-shift streak
    start_shift = {}
    for i in range(num_nurses):
        for k in range(num_days):
            for s in range(num_shifts):
                start_shift[i, k, s] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"start_shift_{i}_{k}_{s}")

    cs_under = {}
    for i in range(num_nurses):
        for s in range(num_shifts):
            csm = CS_minus[shift_names[s]]
            for k in range(num_days):
                for j in range(1, csm):
                    if k + j < num_days:
                        cs_under[i, k, s, j] = model.addVar(
                            vtype=GRB.CONTINUOUS, lb=0,
                            name=f"cs_under_{i}_{k}_{s}_{j}")

    # S3: consecutive days off violations
    # rest[i,k] = 1 - work[i,k]
    # Upper bound: no rest streak longer than CR_plus
    cr_over = {}
    for i in range(num_nurses):
        crp = CR_plus[i]
        for k in range(num_days - crp):
            cr_over[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                         name=f"cr_over_{i}_{k}")

    # Lower bound on consecutive rest
    start_rest = {}
    for i in range(num_nurses):
        for k in range(num_days):
            start_rest[i, k] = model.addVar(vtype=GRB.BINARY,
                                            name=f"start_rest_{i}_{k}")

    cr_under = {}
    for i in range(num_nurses):
        crm = CR_minus[i]
        for k in range(num_days):
            for j in range(1, crm):
                if k + j < num_days:
                    cr_under[i, k, j] = model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0,
                        name=f"cr_under_{i}_{k}_{j}")

    model.update()

    # ----- Constraints -----

    # Link work[i,k] = sum_s x[i,k,s]
    for i in range(num_nurses):
        for k in range(num_days):
            model.addConstr(
                work[i, k] == gp.quicksum(x[i, k, s] for s in range(num_shifts)),
                name=f"link_work_{i}_{k}")

    # H1: At most one shift per day (work is binary, so sum <= 1)
    for i in range(num_nurses):
        for k in range(num_days):
            model.addConstr(
                gp.quicksum(x[i, k, s] for s in range(num_shifts)) <= 1,
                name=f"H1_{i}_{k}")

    # H3: Forbidden shift successions
    for i in range(num_nurses):
        for k in range(num_days - 1):
            for s1, s2 in forbidden:
                model.addConstr(
                    x[i, k, s1] + x[i, k + 1, s2] <= 1,
                    name=f"H3_{i}_{k}_{s1}_{s2}")

    # H4: Skill linking - nurse can only use skills they have
    # sum_{sigma in Sigma_i} y[i,k,s,sigma] = x[i,k,s]
    for i in range(num_nurses):
        for k in range(num_days):
            for s in range(num_shifts):
                model.addConstr(
                    gp.quicksum(y[i, k, s, sigma]
                                for sigma in nurse_skills[i]) == x[i, k, s],
                    name=f"H4_{i}_{k}_{s}")

    # H2: Minimum demand
    for k in range(num_days):
        for s in range(num_shifts):
            for sigma in range(num_skills):
                if D_min[(k, s, sigma)] > 0:
                    model.addConstr(
                        gp.quicksum(y[i, k, s, sigma]
                                    for i in nurses_with_skill[sigma]) >=
                        D_min[(k, s, sigma)],
                        name=f"H2_{k}_{s}_{sigma}")

    # S1: Optimal coverage
    for k in range(num_days):
        for s in range(num_shifts):
            for sigma in range(num_skills):
                model.addConstr(
                    gp.quicksum(y[i, k, s, sigma]
                                for i in nurses_with_skill[sigma]) +
                    z[k, s, sigma] >= D_opt[(k, s, sigma)],
                    name=f"S1_{k}_{s}_{sigma}")

    # S6: Total assignments bounds
    for i in range(num_nurses):
        total_work = gp.quicksum(work[i, k] for k in range(num_days))
        model.addConstr(w_under[i] >= L_minus[i] - total_work,
                        name=f"S6_under_{i}")
        model.addConstr(w_over[i] >= total_work - L_plus[i],
                        name=f"S6_over_{i}")

    # S7: Total working weekends
    for i in range(num_nurses):
        for m, sat, sun in weekends:
            model.addConstr(we_worked[i, m] >= work[i, sat],
                            name=f"we_worked_sat_{i}_{m}")
            model.addConstr(we_worked[i, m] >= work[i, sun],
                            name=f"we_worked_sun_{i}_{m}")
            model.addConstr(we_worked[i, m] <= work[i, sat] + work[i, sun],
                            name=f"we_worked_ub_{i}_{m}")
        model.addConstr(
            w_we[i] >= gp.quicksum(we_worked[i, m]
                                   for m, _, _ in weekends) - WE_plus[i],
            name=f"S7_{i}")

    # S5: Complete weekend
    for i in range(num_nurses):
        for m, sat, sun in weekends:
            model.addConstr(we_viol[i, m] >= work[i, sat] - work[i, sun],
                            name=f"S5a_{i}_{m}")
            model.addConstr(we_viol[i, m] >= work[i, sun] - work[i, sat],
                            name=f"S5b_{i}_{m}")

    # S2a: Consecutive worked days - upper bound
    for i in range(num_nurses):
        cdp = CD_plus[i]
        for k in range(num_days - cdp):
            model.addConstr(
                cd_over[i, k] >= gp.quicksum(
                    work[i, k + j] for j in range(cdp + 1)) - cdp,
                name=f"S2a_over_{i}_{k}")

    # S2a: Consecutive worked days - lower bound
    # start_work[i,k] >= work[i,k] - work[i,k-1]
    for i in range(num_nurses):
        cdm = CD_minus[i]
        model.addConstr(start_work[i, 0] >= work[i, 0],
                        name=f"start_work_{i}_0")
        for k in range(1, num_days):
            model.addConstr(
                start_work[i, k] >= work[i, k] - work[i, k - 1],
                name=f"start_work_{i}_{k}")
        # If a work streak starts at k, then days k+1..k+cdm-1 should be worked
        for k in range(num_days):
            for j in range(1, cdm):
                if k + j < num_days:
                    model.addConstr(
                        cd_under[i, k, j] >=
                        start_work[i, k] - work[i, k + j],
                        name=f"S2a_under_{i}_{k}_{j}")

    # S2b: Consecutive same-shift - upper bound
    for i in range(num_nurses):
        for s in range(num_shifts):
            csp = CS_plus[shift_names[s]]
            for k in range(num_days - csp):
                model.addConstr(
                    cs_over[i, k, s] >= gp.quicksum(
                        x[i, k + j, s] for j in range(csp + 1)) - csp,
                    name=f"S2b_over_{i}_{k}_{s}")

    # S2b: Consecutive same-shift - lower bound
    # start_shift[i,k,s] >= x[i,k,s] - x[i,k-1,s]
    for i in range(num_nurses):
        for s in range(num_shifts):
            csm = CS_minus[shift_names[s]]
            model.addConstr(start_shift[i, 0, s] >= x[i, 0, s],
                            name=f"start_shift_{i}_0_{s}")
            for k in range(1, num_days):
                model.addConstr(
                    start_shift[i, k, s] >= x[i, k, s] - x[i, k - 1, s],
                    name=f"start_shift_{i}_{k}_{s}")
            for k in range(num_days):
                for j in range(1, csm):
                    if k + j < num_days:
                        model.addConstr(
                            cs_under[i, k, s, j] >=
                            start_shift[i, k, s] - x[i, k + j, s],
                            name=f"S2b_under_{i}_{k}_{s}_{j}")

    # S3: Consecutive days off - upper bound
    for i in range(num_nurses):
        crp = CR_plus[i]
        for k in range(num_days - crp):
            # sum of rest days in window of crp+1: (crp+1) - sum(work)
            # violation if rest > crp, i.e., sum(work) < 1
            model.addConstr(
                cr_over[i, k] >= 1 - gp.quicksum(
                    work[i, k + j] for j in range(crp + 1)),
                name=f"S3_over_{i}_{k}")

    # S3: Consecutive days off - lower bound
    # start_rest[i,k] >= (1-work[i,k]) - (1-work[i,k-1]) = work[i,k-1] - work[i,k]
    for i in range(num_nurses):
        crm = CR_minus[i]
        # Day 0: rest starts if not working
        model.addConstr(start_rest[i, 0] >= 1 - work[i, 0],
                        name=f"start_rest_{i}_0")
        for k in range(1, num_days):
            model.addConstr(
                start_rest[i, k] >= work[i, k - 1] - work[i, k],
                name=f"start_rest_{i}_{k}")
        # If rest streak starts at k, days k+1..k+crm-1 should also be rest
        for k in range(num_days):
            for j in range(1, crm):
                if k + j < num_days:
                    model.addConstr(
                        cr_under[i, k, j] >=
                        start_rest[i, k] - (1 - work[i, k + j]),
                        name=f"S3_under_{i}_{k}_{j}")

    # ----- Objective -----
    obj = gp.LinExpr()

    # S1: Optimal coverage penalty
    obj += c_S1 * gp.quicksum(z[k, s, sigma]
                              for k in range(num_days)
                              for s in range(num_shifts)
                              for sigma in range(num_skills))

    # S2a: Consecutive worked days violations
    obj += c_S2a * gp.quicksum(cd_over[i, k]
                               for i in range(num_nurses)
                               for k in range(num_days - CD_plus[i]))
    obj += c_S2a * gp.quicksum(cd_under[i, k, j]
                               for i in range(num_nurses)
                               for k in range(num_days)
                               for j in range(1, CD_minus[i])
                               if k + j < num_days)

    # S2b: Consecutive same-shift violations
    obj += c_S2b * gp.quicksum(cs_over[i, k, s]
                               for i in range(num_nurses)
                               for s in range(num_shifts)
                               for k in range(num_days -
                                              CS_plus[shift_names[s]]))
    obj += c_S2b * gp.quicksum(cs_under[i, k, s, j]
                               for i in range(num_nurses)
                               for s in range(num_shifts)
                               for k in range(num_days)
                               for j in range(1, CS_minus[shift_names[s]])
                               if k + j < num_days)

    # S3: Consecutive days off violations
    obj += c_S3 * gp.quicksum(cr_over[i, k]
                              for i in range(num_nurses)
                              for k in range(num_days - CR_plus[i]))
    obj += c_S3 * gp.quicksum(cr_under[i, k, j]
                              for i in range(num_nurses)
                              for k in range(num_days)
                              for j in range(1, CR_minus[i])
                              if k + j < num_days)

    # S4: Preferences penalty
    obj += c_S4 * gp.quicksum(x[i, k, s]
                              for (i, k, s) in prefs)

    # S5: Complete weekend penalty
    obj += c_S5 * gp.quicksum(we_viol[i, m]
                              for i in range(num_nurses)
                              for m, _, _ in weekends)

    # S6: Total assignments penalty
    obj += c_S6 * gp.quicksum(w_under[i] + w_over[i]
                              for i in range(num_nurses))

    # S7: Working weekends penalty
    obj += c_S7 * gp.quicksum(w_we[i] for i in range(num_nurses))

    model.setObjective(obj, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    result = {}
    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["status"] = model.Status
        result["status_description"] = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, f"STATUS_{model.Status}")

        # Extract schedule: for each nurse, list of (day, shift, skill)
        schedule = []
        for i in range(num_nurses):
            nurse_schedule = {
                "nurse_id": i,
                "nurse_name": nurses[i]["name"],
                "assignments": []
            }
            for k in range(num_days):
                for s in range(num_shifts):
                    if x[i, k, s].X > 0.5:
                        assigned_skill = None
                        for sigma in nurse_skills[i]:
                            if (i, k, s, sigma) in y and y[i, k, s, sigma].X > 0.5:
                                assigned_skill = skills[sigma]
                                break
                        nurse_schedule["assignments"].append({
                            "day": k,
                            "shift": shift_names[s],
                            "skill": assigned_skill
                        })
            schedule.append(nurse_schedule)

        result["schedule"] = schedule

        # Compute penalty breakdown
        penalty_breakdown = {}
        val_S1 = c_S1 * sum(z[k, s, sigma].X
                            for k in range(num_days)
                            for s in range(num_shifts)
                            for sigma in range(num_skills))
        penalty_breakdown["S1_optimal_coverage"] = round(val_S1, 2)

        val_S2a_over = c_S2a * sum(
            cd_over[i, k].X
            for i in range(num_nurses)
            for k in range(num_days - CD_plus[i]))
        val_S2a_under = c_S2a * sum(
            cd_under[i, k, j].X
            for i in range(num_nurses)
            for k in range(num_days)
            for j in range(1, CD_minus[i])
            if k + j < num_days)
        penalty_breakdown["S2a_consecutive_days"] = round(
            val_S2a_over + val_S2a_under, 2)

        val_S2b_over = c_S2b * sum(
            cs_over[i, k, s].X
            for i in range(num_nurses)
            for s in range(num_shifts)
            for k in range(num_days - CS_plus[shift_names[s]]))
        val_S2b_under = c_S2b * sum(
            cs_under[i, k, s, j].X
            for i in range(num_nurses)
            for s in range(num_shifts)
            for k in range(num_days)
            for j in range(1, CS_minus[shift_names[s]])
            if k + j < num_days)
        penalty_breakdown["S2b_consecutive_shifts"] = round(
            val_S2b_over + val_S2b_under, 2)

        val_S3_over = c_S3 * sum(
            cr_over[i, k].X
            for i in range(num_nurses)
            for k in range(num_days - CR_plus[i]))
        val_S3_under = c_S3 * sum(
            cr_under[i, k, j].X
            for i in range(num_nurses)
            for k in range(num_days)
            for j in range(1, CR_minus[i])
            if k + j < num_days)
        penalty_breakdown["S3_consecutive_rest"] = round(
            val_S3_over + val_S3_under, 2)

        val_S4 = c_S4 * sum(x[i, k, s].X for (i, k, s) in prefs)
        penalty_breakdown["S4_preferences"] = round(val_S4, 2)

        val_S5 = c_S5 * sum(we_viol[i, m].X
                            for i in range(num_nurses)
                            for m, _, _ in weekends)
        penalty_breakdown["S5_complete_weekend"] = round(val_S5, 2)

        val_S6 = c_S6 * sum(w_under[i].X + w_over[i].X
                            for i in range(num_nurses))
        penalty_breakdown["S6_total_assignments"] = round(val_S6, 2)

        val_S7 = c_S7 * sum(w_we[i].X for i in range(num_nurses))
        penalty_breakdown["S7_working_weekends"] = round(val_S7, 2)

        result["penalty_breakdown"] = penalty_breakdown

    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["status_description"] = "NO_SOLUTION_FOUND"
        result["schedule"] = []

    # Add solver info
    result["solver"] = "Gurobi"
    result["time_limit"] = time_limit
    if hasattr(model, "Runtime"):
        result["runtime_seconds"] = round(model.Runtime, 2)
    if hasattr(model, "MIPGap") and model.SolCount > 0:
        result["mip_gap"] = round(model.MIPGap, 6)
    if model.SolCount > 0 and hasattr(model, "ObjBound"):
        result["best_bound"] = round(model.ObjBound, 2)

    # Write solution
    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    print(f"Objective value: {result['objective_value']}")
    if result.get("penalty_breakdown"):
        print("Penalty breakdown:")
        for key, val in result["penalty_breakdown"].items():
            print(f"  {key}: {val}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="NSP Gurobi solver (Legrain et al. 2020 INRC-II formulation)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    build_and_solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
