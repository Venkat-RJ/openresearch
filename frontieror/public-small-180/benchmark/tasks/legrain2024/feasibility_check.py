#!/usr/bin/env python3
"""
Feasibility checker for Nurse Scheduling Problem (Legrain & Omer 2024).

Checks all hard constraints from the mathematical formulation (1b)-(1f) and
subproblem constraints (3-9) that are marked as hard in the instance.

Constraints numbered top-to-bottom from the formulation:
  1  (1b)  Assignment: each nurse exactly one shift per day
  2  (1c)  Skill-assignment linking
  3  (1d)  Demand coverage (with z slack, always satisfiable)
  4  (1e)  Integrality (trivially satisfied for discrete schedules)
  5  (1f)  Non-negativity of z (trivially satisfied)
  6        Total shift types / total working days
  7        Total worked weekends
  8        Consecutive shift types (working days, days off, same shift)
  9        Consecutive worked weekends
  10       Identical weekend
  11       Forbidden pattern
  12       Preferences
  13       Objective consistency (exactly recomputed from the schedule after
           optimally completing the omitted nurse-to-skill assignments)
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'schedule', 'skill_assignments')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('schedule', 'skill_assignments')
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


def _minimum_unmet_demand(workers, skill_sets, skill_demand):
    """Return the exact minimum uncovered demand for one day/shift.

    The model assigns every scheduled nurse to exactly one skill they possess.
    Because every uncovered unit has the same penalty, the recourse problem is
    a maximum-cardinality bipartite matching between nurses and unit-demand
    skill slots.  Nurses left unmatched to a demand slot may be assigned to any
    other possessed skill and do not change the objective.
    """
    demand_slots = []
    for skill, raw_demand in skill_demand.items():
        value = float(raw_demand)
        rounded = round(value)
        if not math.isfinite(value) or value < 0 or abs(value - rounded) > 1e-8:
            raise ValueError(
                f"demand for skill {skill!r} must be a non-negative integer"
            )
        demand_slots.extend([skill] * int(rounded))

    owner = [None] * len(demand_slots)

    def augment(nurse_id, seen):
        for slot, skill in enumerate(demand_slots):
            if seen[slot] or skill not in skill_sets[nurse_id]:
                continue
            seen[slot] = True
            if owner[slot] is None or augment(owner[slot], seen):
                owner[slot] = nurse_id
                return True
        return False

    covered = 0
    for nurse_id in workers:
        if augment(nurse_id, [False] * len(demand_slots)):
            covered += 1
    return float(len(demand_slots) - covered)



def check_feasibility(inst, sol):
    tol = 1e-5
    eps = 1e-5

    meta = inst["instance_metadata"]
    num_days = meta["num_days"]
    shifts = inst["shifts"]
    skills = inst.get("skills", [])
    nurses = inst["nurses"]
    weekends = inst["weekends"]
    hard_con = inst["constraints"]["hard_constraints"]
    soft_con = inst["constraints"].get("soft_constraints", {}) or {}
    demand = inst.get("demand", {}) or {}
    demand_penalty = inst.get("unsatisfied_demand_penalty", 0)

    schedule = sol["schedule"]
    skill_assignments = sol["skill_assignments"]
    all_shifts = shifts + ["None"]

    nurse_ids = [n["id"] for n in nurses]
    nurse_skills = {n["id"]: set(n["skills"]) for n in nurses}

    def weekend_days(weekend):
        return [
            weekend[key]
            for key in ("friday", "saturday", "sunday")
            if key in weekend
        ]

    violations = []       # list of (constraint_idx, message)
    magnitudes = []       # list of magnitude dicts

    def record(idx, msg, lhs, rhs, ctype="eq"):
        """Record a violation if it exceeds tolerance."""
        if ctype == "eq":
            va = abs(lhs - rhs)
        elif ctype == "leq":
            va = max(0.0, lhs - rhs)
        elif ctype == "geq":
            va = max(0.0, rhs - lhs)
        else:
            va = 0.0
        if va > tol:
            normalizer = max(abs(rhs), eps)
            violations.append((idx, msg))
            magnitudes.append({
                "constraint": idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(va),
                "normalizer": float(normalizer),
                "ratio": float(va / normalizer),
            })

    def is_work(s):
        return s != "None"

    def get_shifts(nurse_id):
        """Return list of shifts for days 1..num_days (0-indexed list)."""
        return [schedule.get(nurse_id, {}).get(f"day_{d}", "None")
                for d in range(1, num_days + 1)]

    # ==================================================================
    # Constraint 1 (1b): Each nurse exactly one shift per day
    # ==================================================================
    for nid in nurse_ids:
        if nid not in schedule:
            record(1, f"Nurse {nid} missing from schedule", 0, 1, "eq")
            continue
        for d in range(1, num_days + 1):
            dk = f"day_{d}"
            if dk not in schedule[nid]:
                record(1, f"Nurse {nid} missing assignment on day {d}", 0, 1, "eq")
            else:
                s = schedule[nid][dk]
                if s not in all_shifts:
                    record(1, f"Nurse {nid} day {d}: invalid shift '{s}'", 0, 1, "eq")

    # ==================================================================
    # Constraint 2 (1c): exact skill-assignment witness and hard/soft
    # possession rule.  Every worked assignment has exactly one reported skill;
    # a day off reports "None".
    # ==================================================================
    hard_skill = bool(hard_con.get("skill_requirement", False))
    for nid in nurse_ids:
        nurse_skill_days = skill_assignments.get(nid, {})
        if not isinstance(nurse_skill_days, dict):
            record(2, f"Nurse {nid} has no skill-assignment mapping", 0, 1, "eq")
            continue
        for d in range(1, num_days + 1):
            day_key = f"day_{d}"
            shift = schedule.get(nid, {}).get(day_key, "None")
            if day_key not in nurse_skill_days:
                record(2, f"Nurse {nid} missing skill assignment on day {d}", 0, 1, "eq")
                continue
            assigned_skill = nurse_skill_days[day_key]
            if not is_work(shift):
                if assigned_skill != "None":
                    record(2,
                           f"Nurse {nid} is off on day {d} but reports skill {assigned_skill}",
                           1, 0, "eq")
            elif assigned_skill not in skills:
                record(2,
                       f"Nurse {nid} works {shift} on day {d} but skill {assigned_skill!r} is invalid",
                       0, 1, "eq")
            elif hard_skill and assigned_skill not in nurse_skills[nid]:
                record(2,
                       f"Nurse {nid} lacks assigned skill {assigned_skill} on day {d}",
                       0, 1, "geq")

    # ==================================================================
    # Constraint 3 (1d): exact hard demand equality where requested.  In soft
    # variants, nonnegative z absorbs the shortfall and is recomputed below.
    # ==================================================================
    if hard_con.get("demand_equality", False):
        for d in range(1, num_days + 1):
            day_key = f"day_{d}"
            for shift in shifts:
                for skill in skills:
                    required = float(
                        demand.get(day_key, {}).get(shift, {}).get(skill, 0)
                    )
                    assigned = sum(
                        1
                        for nid in nurse_ids
                        if schedule.get(nid, {}).get(day_key) == shift
                        and skill_assignments.get(nid, {}).get(day_key) == skill
                    )
                    if abs(assigned - required) > tol:
                        record(3,
                               f"Hard demand equality day={d}, shift={shift}, skill={skill}: assigned={assigned}, required={required}",
                               assigned, required, "eq")

    # ==================================================================
    # Constraint 4 (1e): Integrality — trivially satisfied for discrete
    # ==================================================================

    # ==================================================================
    # Constraint 5 (1f): Non-negativity of z — trivially satisfied
    # ==================================================================

    # ==================================================================
    # Constraint 6: Total shift types / total working days
    # Hard only if present in hard_constraints.
    # ==================================================================
    tw = hard_con.get("total_working_days")
    if tw:
        tw_min = tw.get("min", 0)
        tw_max = tw.get("max", num_days)
        for nid in nurse_ids:
            total = sum(1 for s in get_shifts(nid) if is_work(s))
            if total > tw_max:
                record(6,
                       f"Nurse {nid} works {total} days, max allowed {tw_max}",
                       total, tw_max, "leq")
            if total < tw_min:
                record(6,
                       f"Nurse {nid} works {total} days, min required {tw_min}",
                       total, tw_min, "geq")

    # ==================================================================
    # Constraint 7: Total worked weekends
    # ==================================================================
    tww = hard_con.get("total_worked_weekends")
    if tww:
        tww_max = tww.get("max", len(weekends))
        for nid in nurse_ids:
            wk_count = 0
            for wk in weekends:
                if any(
                    is_work(schedule.get(nid, {}).get(f"day_{day}", "None"))
                    for day in weekend_days(wk)
                ):
                    wk_count += 1
            if wk_count > tww_max:
                record(7,
                       f"Nurse {nid} works {wk_count} weekends, max {tww_max}",
                       wk_count, tww_max, "leq")

    # ==================================================================
    # Constraint 8: Consecutive shift types
    # Covers consecutive_working_days, consecutive_days_off,
    # and consecutive_same_shift (all sub-cases of paper constraint 5).
    # ==================================================================
    cw = hard_con.get("consecutive_working_days")
    if cw:
        cw_min = cw.get("min", 0)
        cw_max = cw.get("max", num_days)
        for nid in nurse_ids:
            sched = get_shifts(nid)
            streak = 0
            for d in range(num_days):
                if is_work(sched[d]):
                    streak += 1
                    if streak > cw_max:
                        record(8,
                               f"Nurse {nid}: {streak} consecutive working days at day {d+1}, max {cw_max}",
                               streak, cw_max, "leq")
                else:
                    if streak > 0 and streak < cw_min:
                        # Don't penalise boundary streaks (starting at day 1)
                        start_day = d - streak  # 0-indexed start
                        if start_day > 0:
                            record(8,
                                   f"Nurse {nid}: {streak} consecutive working days ending day {d}, min {cw_min}",
                                   streak, cw_min, "geq")
                    streak = 0
            # Final streak: don't penalise if ends at last day (boundary)

    co = hard_con.get("consecutive_days_off")
    if co:
        co_min = co.get("min", 0)
        co_max = co.get("max", num_days)
        for nid in nurse_ids:
            sched = get_shifts(nid)
            streak = 0
            for d in range(num_days):
                if not is_work(sched[d]):
                    streak += 1
                    if streak > co_max:
                        record(8,
                               f"Nurse {nid}: {streak} consecutive days off at day {d+1}, max {co_max}",
                               streak, co_max, "leq")
                else:
                    if streak > 0 and streak < co_min:
                        start_day = d - streak
                        if start_day > 0:
                            record(8,
                                   f"Nurse {nid}: {streak} consecutive days off ending day {d}, min {co_min}",
                                   streak, co_min, "geq")
                    streak = 0

    cs = hard_con.get("consecutive_same_shift")
    if cs:
        cs_max = cs.get("max", num_days)
        for nid in nurse_ids:
            sched = get_shifts(nid)
            for s in shifts:
                streak = 0
                for d in range(num_days):
                    if sched[d] == s:
                        streak += 1
                        if streak > cs_max:
                            record(8,
                                   f"Nurse {nid}: {streak} consecutive '{s}' shifts at day {d+1}, max {cs_max}",
                                   streak, cs_max, "leq")
                    else:
                        streak = 0

    # ==================================================================
    # Constraint 9: Consecutive worked weekends
    # ==================================================================
    cwk = hard_con.get("consecutive_worked_weekends")
    if cwk:
        cwk_max = cwk.get("max", len(weekends))
        for nid in nurse_ids:
            streak = 0
            for wk in weekends:
                if any(
                    is_work(schedule.get(nid, {}).get(f"day_{day}", "None"))
                    for day in weekend_days(wk)
                ):
                    streak += 1
                    if streak > cwk_max:
                        record(9,
                               f"Nurse {nid}: {streak} consecutive worked weekends at week {wk['week']}, max {cwk_max}",
                               streak, cwk_max, "leq")
                else:
                    streak = 0

    # ==================================================================
    # Constraint 10: Identical weekend
    # ==================================================================
    iw = hard_con.get("identical_weekend")
    if iw and iw.get("enabled", False):
        for nid in nurse_ids:
            for wk in weekends:
                constituent_shifts = [
                    schedule.get(nid, {}).get(f"day_{day}", "None")
                    for day in weekend_days(wk)
                ]
                if len(set(constituent_shifts)) > 1:
                    record(10,
                           f"Nurse {nid} week {wk['week']}: weekend shifts {constituent_shifts} must be identical",
                           1, 0, "leq")

    # ==================================================================
    # Constraint 11: Forbidden patterns
    # ==================================================================
    fps = hard_con.get("forbidden_patterns", [])
    for fp in fps:
        pattern = fp["pattern"]
        pat_len = len(pattern)
        desc = fp.get("description", "forbidden pattern")
        for nid in nurse_ids:
            for d in range(1, num_days - pat_len + 2):
                match = True
                for offset, (day_type, shift_name) in enumerate(pattern):
                    dd = d + offset
                    actual = schedule.get(nid, {}).get(f"day_{dd}", "None")
                    if actual != shift_name:
                        match = False
                        break
                if match:
                    day_range = f"days {d}-{d + pat_len - 1}"
                    record(11,
                           f"Nurse {nid}: {desc} at {day_range}",
                           pat_len, pat_len - 1, "leq")

    # ==================================================================
    # Constraint 12: Preferences (only if marked as hard)
    # ==================================================================
    if "preferences" in hard_con or hard_con.get("hard_preferences", False):
        for nurse in nurses:
            nid = nurse["id"]
            for pref in nurse.get("preferences", []):
                day = pref["day"]
                ptype = pref["type"]
                shift_pref = pref.get("shift", "None")
                dk = f"day_{day}"
                actual = schedule.get(nid, {}).get(dk, "None")

                violated = False
                if ptype == "day_off" and actual != "None":
                    violated = True
                    msg = f"Nurse {nid} day {day}: prefers day off, assigned {actual}"
                elif ptype == "day_on" and actual == "None":
                    violated = True
                    msg = f"Nurse {nid} day {day}: prefers working, assigned off"
                elif ptype == "shift_on" and actual != shift_pref:
                    violated = True
                    msg = f"Nurse {nid} day {day}: prefers {shift_pref}, assigned {actual}"
                elif ptype == "shift_off" and actual == shift_pref:
                    violated = True
                    msg = f"Nurse {nid} day {day}: prefers not {shift_pref}, assigned {actual}"

                if violated:
                    record(12, msg, 1, 0, "leq")

    # ==================================================================
    # Constraint 13: Objective consistency from the concrete schedule and
    # nurse-to-skill witness.
    # ==================================================================
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None

        if reported is not None:
            schedule_pen = 0.0

            # --- SC1: total_working_days (soft) ---
            s_tw = soft_con.get("total_working_days")
            if isinstance(s_tw, dict):
                tw_min_s = s_tw.get("min", 0)
                tw_max_s = s_tw.get("max", num_days)
                pen_under = s_tw.get("penalty_under", 0)
                pen_over = s_tw.get("penalty_over", 0)
                for nid in nurse_ids:
                    total = sum(1 for s in get_shifts(nid) if is_work(s))
                    schedule_pen += pen_over * max(0, total - tw_max_s)
                    schedule_pen += pen_under * max(0, tw_min_s - total)

            # --- SC2: total_worked_weekends (soft) ---
            s_tww = soft_con.get("total_worked_weekends")
            if isinstance(s_tww, dict):
                tww_max_s = s_tww.get("max", len(weekends))
                pen_wk = s_tww.get("penalty", 0)
                for nid in nurse_ids:
                    wk_count = 0
                    for wk in weekends:
                        if any(
                            is_work(schedule.get(nid, {}).get(f"day_{day}", "None"))
                            for day in weekend_days(wk)
                        ):
                            wk_count += 1
                    schedule_pen += pen_wk * max(0, wk_count - tww_max_s)

            # --- SC3: consecutive_working_days (soft) ---
            s_cwd = soft_con.get("consecutive_working_days")
            if isinstance(s_cwd, dict):
                cwd_max_s = s_cwd.get("max")
                cwd_min_s = s_cwd.get("min")
                pen_cwd_over = s_cwd.get("penalty_over", 0)
                pen_cwd_under = s_cwd.get("penalty_under", 0)
                for nid in nurse_ids:
                    sched = get_shifts(nid)
                    if cwd_max_s is not None:
                        window = cwd_max_s + 1
                        for d_start in range(1, num_days + 1):
                            if d_start + window - 1 > num_days:
                                break
                            sum_work = sum(1 for k in range(window)
                                           if is_work(sched[d_start - 1 + k]))
                            schedule_pen += pen_cwd_over * max(0, sum_work - cwd_max_s)
                    if cwd_min_s is not None and cwd_min_s > 1:
                        for g in range(1, cwd_min_s):
                            for d_start in range(1, num_days + 1):
                                end_day = d_start + g - 1
                                if end_day > num_days:
                                    break
                                if d_start == 1 or end_day + 1 > num_days:
                                    continue
                                if (not is_work(sched[d_start - 2]) and
                                    all(is_work(sched[d_start - 1 + k]) for k in range(g)) and
                                    not is_work(sched[end_day])):
                                    schedule_pen += pen_cwd_under

            # --- SC4: consecutive_days_off (soft) ---
            s_cdo = soft_con.get("consecutive_days_off")
            if isinstance(s_cdo, dict):
                cdo_max_s = s_cdo.get("max")
                cdo_min_s = s_cdo.get("min")
                pen_cdo_over = s_cdo.get("penalty_over", 0)
                pen_cdo_under = s_cdo.get("penalty_under", 0)
                for nid in nurse_ids:
                    sched = get_shifts(nid)
                    if cdo_max_s is not None:
                        window = cdo_max_s + 1
                        for d_start in range(1, num_days + 1):
                            if d_start + window - 1 > num_days:
                                break
                            sum_off = sum(1 for k in range(window)
                                          if not is_work(sched[d_start - 1 + k]))
                            schedule_pen += pen_cdo_over * max(0, sum_off - cdo_max_s)
                    if cdo_min_s is not None and cdo_min_s > 1:
                        for g in range(1, cdo_min_s):
                            for d_start in range(1, num_days + 1):
                                end_day = d_start + g - 1
                                if end_day > num_days:
                                    break
                                if d_start == 1 or end_day + 1 > num_days:
                                    continue
                                if (is_work(sched[d_start - 2]) and
                                    all(not is_work(sched[d_start - 1 + k]) for k in range(g)) and
                                    is_work(sched[end_day])):
                                    schedule_pen += pen_cdo_under

            # --- SC5: consecutive_same_shift (soft) ---
            s_css = soft_con.get("consecutive_same_shift")
            if isinstance(s_css, dict):
                css_max_s = s_css.get("max")
                pen_css = s_css.get("penalty", 0)
                if css_max_s is not None:
                    window = css_max_s + 1
                    for nid in nurse_ids:
                        sched = get_shifts(nid)
                        for s in shifts:
                            for d_start in range(1, num_days + 1):
                                if d_start + window - 1 > num_days:
                                    break
                                sum_s = sum(1 for k in range(window)
                                            if sched[d_start - 1 + k] == s)
                                schedule_pen += pen_css * max(0, sum_s - css_max_s)

            # --- SC6: consecutive_worked_weekends (soft) ---
            s_cww = soft_con.get("consecutive_worked_weekends")
            if isinstance(s_cww, dict):
                cww_max_s = s_cww.get("max")
                pen_cww = s_cww.get("penalty", 0)
                if cww_max_s is not None:
                    num_wks = len(weekends)
                    window = cww_max_s + 1
                    for nid in nurse_ids:
                        wk_worked_flag = []
                        for wk in weekends:
                            wk_worked_flag.append(
                                1 if any(
                                    is_work(schedule.get(nid, {}).get(f"day_{day}", "None"))
                                    for day in weekend_days(wk)
                                ) else 0)
                        for start in range(num_wks):
                            if start + window > num_wks:
                                break
                            sum_w = sum(wk_worked_flag[start + k] for k in range(window))
                            schedule_pen += pen_cww * max(0, sum_w - cww_max_s)

            # --- SC7: identical_weekend (soft) ---
            s_iw = soft_con.get("identical_weekend")
            if isinstance(s_iw, dict) and s_iw.get("enabled", False):
                pen_iw = s_iw.get("penalty", 0)
                for nid in nurse_ids:
                    for wk in weekends:
                        constituent_shifts = [
                            schedule.get(nid, {}).get(f"day_{day}", "None")
                            for day in weekend_days(wk)
                        ]
                        if len(set(constituent_shifts)) > 1:
                            schedule_pen += pen_iw

            # --- Soft forbidden patterns ---
            for fp in soft_con.get("forbidden_patterns", []) or []:
                pattern = fp.get("pattern", [])
                pat_len = len(pattern)
                penalty = float(fp.get("penalty", 0) or 0)
                if not pattern:
                    continue
                for nid in nurse_ids:
                    for d_start in range(1, num_days - pat_len + 2):
                        if all(
                            schedule.get(nid, {}).get(f"day_{d_start + offset}", "None")
                            == shift_name
                            for offset, (_day_type, shift_name) in enumerate(pattern)
                        ):
                            schedule_pen += penalty

            # --- SC8: preferences (gurobi always applies these, regardless of
            # whether the instance lists them in hard/soft; each pref carries
            # its own penalty field).
            for nurse in nurses:
                nid = nurse["id"]
                for pref in nurse.get("preferences", []):
                    day = pref["day"]
                    if day < 1 or day > num_days:
                        continue
                    ptype = pref["type"]
                    shift_pref = pref.get("shift", "None")
                    penalty = pref.get("penalty", 0)
                    actual = schedule.get(nid, {}).get(f"day_{day}", "None")
                    if ptype == "day_off":
                        if actual != "None":
                            schedule_pen += penalty
                    elif ptype == "day_on":
                        if actual == "None":
                            schedule_pen += penalty
                    elif ptype == "shift_on":
                        if shift_pref in all_shifts and actual != shift_pref:
                            schedule_pen += penalty
                    elif ptype == "shift_off":
                        if shift_pref in all_shifts and actual == shift_pref:
                            schedule_pen += penalty

            # --- Soft skill-possession mismatches ---
            if not hard_skill:
                mismatch_penalty = float(
                    soft_con.get("skill_requirement_penalty", 0) or 0
                )
                for nid in nurse_ids:
                    for d in range(1, num_days + 1):
                        day_key = f"day_{d}"
                        if is_work(schedule.get(nid, {}).get(day_key, "None")):
                            assigned_skill = skill_assignments.get(nid, {}).get(day_key)
                            if assigned_skill in skills and assigned_skill not in nurse_skills[nid]:
                                schedule_pen += mismatch_penalty

            # --- Exact unsatisfied demand from the submitted skill witness ---
            unmet_demand = 0.0
            pen_z = float(demand_penalty or 0)
            for d in range(1, num_days + 1):
                day_key = f"day_{d}"
                day_dem = demand.get(day_key, {})
                if not day_dem:
                    continue
                for s in shifts:
                    s_dem = day_dem.get(s, {})
                    if not s_dem:
                        continue
                    for skill, required in s_dem.items():
                        assigned = sum(
                            1
                            for nid in nurse_ids
                            if schedule.get(nid, {}).get(day_key, "None") == s
                            and skill_assignments.get(nid, {}).get(day_key) == skill
                        )
                        unmet_demand += max(0.0, float(required) - assigned)

            demand_cost = pen_z * unmet_demand
            expected_obj = schedule_pen + demand_cost
            objective_tol = max(1e-5, 1e-7 * max(1.0, abs(expected_obj)))

            if abs(reported - expected_obj) > objective_tol:
                record(13,
                       (f"Objective consistency: reported objective_value="
                        f"{reported} does not equal the exactly recomputed value "
                        f"(schedule_penalty={schedule_pen:.6g} + "
                        f"demand_penalty={demand_cost:.6g} = {expected_obj:.6g}); "
                        f"difference={abs(reported - expected_obj):.3g}, "
                        f"tol={objective_tol:.3g}"),
                       reported, expected_obj, "eq")

    # ==================================================================
    # Build result
    # ==================================================================
    violated_indices = sorted(set(c for c, _ in violations))
    # Aggregate messages per constraint, keeping order
    seen_msgs = []
    for c in violated_indices:
        for ci, m in violations:
            if ci == c and m not in seen_msgs:
                seen_msgs.append(m)

    feasible = len(violated_indices) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": seen_msgs,
        "violation_magnitudes": magnitudes if not feasible else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Nurse Scheduling (Legrain & Omer 2024)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write feasibility result JSON")
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

    result = check_feasibility(inst, sol)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {result['feasible']}")
    if not result["feasible"]:
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
