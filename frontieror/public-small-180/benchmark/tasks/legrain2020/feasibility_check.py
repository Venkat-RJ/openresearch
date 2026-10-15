#!/usr/bin/env python3
"""
Feasibility checker for the Nurse Scheduling Problem (INRC-II static)
Based on Legrain, Omer, Rosat (2020) - Formulation (1)

Hard constraints checked (numbered top-to-bottom in formulation):
  Constraint 1 (1b): Flow at source - each nurse has exactly one roster
  Constraint 2 (1c): Flow conservation - single assignment per day (H1) +
                      forbidden shift successions (H3)
  Constraint 4 (1e): Skill linking - nurse only performs skills they have (H4)
  Constraint 5 (1f): Minimum demand - staffing >= D_{p,sigma} (H2)
  Constraint 7 (1h): Integrality of x, y
  Constraint 8 (1i): Non-negativity of w, z
  Constraint 9 (Tier C): Objective consistency - reported objective_value
                          must lie within [LB, UB] derived from the schedule
                          and instance parameters.

Soft constraints skipped:
  Constraint 3 (1d): S6/S7 total assignments / weekends
  Constraint 6 (1g): S1 optimal coverage
"""

import argparse
import json
import math
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'schedule')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('objective_value',)
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



def _compute_obj_bounds(instance, solution):
    """Return (lower_bound, upper_bound) on the total objective implied by the
    schedule + instance. The LB is the *exact* minimum obj attainable given the
    schedule (every continuous slack at its lower bound, every binary indicator
    at its forced value); the UB is a generous instance-derived ceiling that
    safely covers any feasible incumbent including LP-suboptimal MIP heuristic
    solutions, while remaining finite enough to reject sys.float_info.max-style
    exploits."""
    num_nurses = instance["num_nurses"]
    num_days = instance["num_days"]
    num_weeks = instance["num_weeks"]
    nurses = instance["nurses"]
    skills_list = instance["skills"]
    shifts_data = instance["shifts"]
    shift_names = [s["name"] for s in shifts_data]
    CS_minus = {s["name"]: s["CS_minus"] for s in shifts_data}
    CS_plus = {s["name"]: s["CS_plus"] for s in shifts_data}
    demand = instance["demand"]
    pw = instance["penalty_weights"]
    c_S1 = pw["c_S1"]; c_S2a = pw["c_S2a"]; c_S2b = pw["c_S2b"]
    c_S3 = pw["c_S3"]; c_S4 = pw["c_S4"]; c_S5 = pw["c_S5"]
    c_S6 = pw["c_S6"]; c_S7 = pw["c_S7"]

    valid_shifts = set(shift_names)
    valid_skills = set(skills_list)

    work = [[0] * num_days for _ in range(num_nurses)]
    x = {}
    skill_used = {}
    schedule_by_nurse = {
        entry["nurse_id"]: entry["assignments"]
        for entry in solution.get("schedule", []) or []
    }
    for nid, assignments in schedule_by_nurse.items():
        if not isinstance(nid, int) or nid < 0 or nid >= num_nurses:
            continue
        seen_days = set()
        for a in assignments:
            d = a.get("day")
            sn = a.get("shift")
            sk = a.get("skill")
            if d is None or sn is None:
                continue
            if not (isinstance(d, int) and 0 <= d < num_days):
                continue
            if sn not in valid_shifts:
                continue
            if d in seen_days:
                continue
            seen_days.add(d)
            work[nid][d] = 1
            x[(nid, d, sn)] = 1
            if sk in valid_skills:
                skill_used[(nid, d, sn)] = sk

    weekends = []
    for m in range(num_weeks):
        sat = 7 * m + 5
        sun = 7 * m + 6
        if sat < num_days and sun < num_days:
            weekends.append((m, sat, sun))

    # ----- LB components (tight minimum given the schedule) -----
    actual_y = defaultdict(int)
    for (nid, d, sn), sk in skill_used.items():
        actual_y[(d, sn, sk)] += 1

    s1_total = 0
    for d in range(num_days):
        for sn in shift_names:
            for sk in skills_list:
                D_opt = demand[d][sn][sk]["optimal"]
                actual = actual_y.get((d, sn, sk), 0)
                s1_total += max(0, D_opt - actual)

    s2a_over = 0
    s2a_under = 0
    for nurse in nurses:
        i = nurse["id"]
        if not isinstance(i, int) or i < 0 or i >= num_nurses:
            continue
        cdp = nurse["contract"]["CD_plus"]
        cdm = nurse["contract"]["CD_minus"]
        for k in range(num_days - cdp):
            window_sum = sum(work[i][k + j] for j in range(cdp + 1))
            s2a_over += max(0, window_sum - cdp)
        for k in range(num_days):
            sw = work[i][0] if k == 0 else max(0, work[i][k] - work[i][k - 1])
            if sw == 0:
                continue
            for j in range(1, cdm):
                if k + j < num_days:
                    s2a_under += max(0, sw - work[i][k + j])

    s2b_over = 0
    s2b_under = 0
    for nurse in nurses:
        i = nurse["id"]
        if not isinstance(i, int) or i < 0 or i >= num_nurses:
            continue
        for sn in shift_names:
            csp = CS_plus[sn]
            csm = CS_minus[sn]
            x_arr = [x.get((i, k, sn), 0) for k in range(num_days)]
            for k in range(num_days - csp):
                window_sum = sum(x_arr[k + j] for j in range(csp + 1))
                s2b_over += max(0, window_sum - csp)
            for k in range(num_days):
                ss = x_arr[0] if k == 0 else max(0, x_arr[k] - x_arr[k - 1])
                if ss == 0:
                    continue
                for j in range(1, csm):
                    if k + j < num_days:
                        s2b_under += max(0, ss - x_arr[k + j])

    s3_over = 0
    s3_under = 0
    for nurse in nurses:
        i = nurse["id"]
        if not isinstance(i, int) or i < 0 or i >= num_nurses:
            continue
        crp = nurse["contract"]["CR_plus"]
        crm = nurse["contract"]["CR_minus"]
        for k in range(num_days - crp):
            window_work = sum(work[i][k + j] for j in range(crp + 1))
            s3_over += max(0, 1 - window_work)
        for k in range(num_days):
            sr = (1 - work[i][0]) if k == 0 else max(0, work[i][k - 1] - work[i][k])
            if sr == 0:
                continue
            for j in range(1, crm):
                if k + j < num_days:
                    s3_under += max(0, sr - (1 - work[i][k + j]))

    s4_count = 0
    for nurse in nurses:
        i = nurse["id"]
        if not isinstance(i, int) or i < 0 or i >= num_nurses:
            continue
        for p in nurse.get("preferences", []) or []:
            if x.get((i, p.get("day"), p.get("shift")), 0) == 1:
                s4_count += 1

    s5_total = 0
    for nurse in nurses:
        i = nurse["id"]
        if not isinstance(i, int) or i < 0 or i >= num_nurses:
            continue
        for _, sat, sun in weekends:
            s5_total += abs(work[i][sat] - work[i][sun])

    s6_total = 0
    for nurse in nurses:
        i = nurse["id"]
        if not isinstance(i, int) or i < 0 or i >= num_nurses:
            continue
        Lm = nurse["contract"]["L_minus"]
        Lp = nurse["contract"]["L_plus"]
        total_work = sum(work[i])
        s6_total += max(0, Lm - total_work) + max(0, total_work - Lp)

    s7_total = 0
    for nurse in nurses:
        i = nurse["id"]
        if not isinstance(i, int) or i < 0 or i >= num_nurses:
            continue
        WEp = nurse["contract"]["WE_plus"]
        we_worked_count = 0
        for _, sat, sun in weekends:
            if work[i][sat] == 1 or work[i][sun] == 1:
                we_worked_count += 1
        s7_total += max(0, we_worked_count - WEp)

    lb = (
        c_S1 * s1_total
        + c_S2a * (s2a_over + s2a_under)
        + c_S2b * (s2b_over + s2b_under)
        + c_S3 * (s3_over + s3_under)
        + c_S4 * s4_count
        + c_S5 * s5_total
        + c_S6 * s6_total
        + c_S7 * s7_total
    )

    # ----- UB: instance-derived ceiling that any feasible obj must respect.
    # Each violation variable in the model is non-negative with positive obj
    # coefficient; we bound each by the largest value it could conceivably
    # take in any feasible solution (LP-suboptimal incumbents included).
    s1_max_total = 0
    for d in range(num_days):
        for sn in shift_names:
            for sk in skills_list:
                s1_max_total += demand[d][sn][sk]["optimal"]
    ub_S1 = c_S1 * s1_max_total

    ub_S2a_over = c_S2a * num_nurses * num_days
    ub_S2a_under = c_S2a * sum(
        num_days * max(0, nurse["contract"]["CD_minus"] - 1)
        for nurse in nurses
        if isinstance(nurse.get("id"), int)
    )
    ub_S2b_over = c_S2b * num_nurses * num_days * len(shift_names)
    ub_S2b_under = c_S2b * num_nurses * num_days * sum(
        max(0, CS_minus[sn] - 1) for sn in shift_names
    )
    ub_S3_over = c_S3 * num_nurses * num_days
    ub_S3_under = c_S3 * sum(
        num_days * max(0, nurse["contract"]["CR_minus"] - 1)
        for nurse in nurses
        if isinstance(nurse.get("id"), int)
    )
    ub_S4 = c_S4 * sum(
        len(nurse.get("preferences", []) or []) for nurse in nurses
    )
    ub_S5 = c_S5 * num_nurses * len(weekends)
    ub_S6 = c_S6 * num_nurses * num_days
    ub_S7 = c_S7 * num_nurses * len(weekends)

    ub = (
        ub_S1
        + ub_S2a_over + ub_S2a_under
        + ub_S2b_over + ub_S2b_under
        + ub_S3_over + ub_S3_under
        + ub_S4 + ub_S5 + ub_S6 + ub_S7
    )
    # Safety margin (10x) -- generous enough that real Gurobi heuristic
    # incumbents never exceed it while still rejecting sys.float_info.max.
    ub = max(ub * 10, lb * 10 + 1e6)
    return float(lb), float(ub)


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for NSP (INRC-II)")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
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

    # Handle degenerate solutions (solver found no solution / infeasible)
    sol_status = solution.get("status")
    sol_schedule = solution.get("schedule", [])
    if sol_status in (3, "3", "INFEASIBLE", "NO_SOLUTION_FOUND") or \
       (solution.get("objective_value") is None and len(sol_schedule) == 0):
        status_desc = solution.get("status_description", solution.get("status", "unknown"))
        result = {
            "feasible": False,
            "violated_constraints": [1],
            "violations": [
                f"No solution provided (solver status={status_desc}). "
                f"Empty schedule cannot satisfy constraint 1b (every nurse must have a roster)."
            ],
            "violation_magnitudes": [{
                "constraint": 1,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            }],
        }
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Solution is INFEASIBLE (no solution from solver, status={status_desc}).")
        return

    tol = 1e-5
    eps = 1e-5

    num_nurses = instance["num_nurses"]
    num_days = instance["num_days"]
    nurses = instance["nurses"]
    demand = instance["demand"]  # list of dicts indexed by day
    forbidden = instance["forbidden_shift_successions"]
    skills_list = instance["skills"]

    # Build nurse lookup
    nurse_by_id = {n["id"]: n for n in nurses}
    nurse_skills = {n["id"]: set(n["skills"]) for n in nurses}

    schedule = solution["schedule"]
    schedule_by_nurse = {entry["nurse_id"]: entry["assignments"] for entry in schedule}

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs, rhs, operator="eq"):
        violated_constraints.add(constraint_idx)
        violations.append(msg)
        if operator == "eq":
            violation_amount = abs(lhs - rhs)
        elif operator == "geq":
            # LHS >= RHS required, violation = max(0, RHS - LHS)
            violation_amount = max(0.0, rhs - lhs)
        elif operator == "leq":
            # LHS <= RHS required, violation = max(0, LHS - RHS)
            violation_amount = max(0.0, lhs - rhs)
        else:
            violation_amount = abs(lhs - rhs)

        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # =========================================================================
    # Constraint 1 (1b): Flow at source - each nurse has exactly one roster
    # sum_{j in A+_{i,o_i}} x_{ij} = 1, for all i
    # In practice: each nurse in the instance must appear in the solution.
    # =========================================================================
    for nurse in nurses:
        nid = nurse["id"]
        if nid not in schedule_by_nurse:
            # LHS = 0 (no roster), RHS = 1
            record_violation(1,
                f"Nurse {nid} ({nurse['name']}) has no roster in the solution",
                0.0, 1.0, "eq")

    # =========================================================================
    # Constraint 2 (1c): Flow conservation at intermediate nodes
    # Enforces H1 (single assignment per day) and H3 (forbidden successions)
    # =========================================================================
    for nurse in nurses:
        nid = nurse["id"]
        assignments = schedule_by_nurse.get(nid, [])

        # H1: At most one shift per day
        day_counts = defaultdict(int)
        for a in assignments:
            day_counts[a["day"]] += 1
        for day, count in day_counts.items():
            if count > 1:
                # LHS = count (shifts on that day), RHS = 1
                record_violation(2,
                    f"Nurse {nid} has {count} assignments on day {day} (H1: single assignment per day)",
                    float(count), 1.0, "leq")

        # H3: Forbidden shift successions
        sorted_assignments = sorted(assignments, key=lambda a: a["day"])
        for i in range(len(sorted_assignments) - 1):
            a1 = sorted_assignments[i]
            a2 = sorted_assignments[i + 1]
            if a2["day"] == a1["day"] + 1:
                for fs in forbidden:
                    if a1["shift"] == fs[0] and a2["shift"] == fs[1]:
                        # Forbidden succession: LHS = 1 (occurs), RHS = 0 (must not occur)
                        record_violation(2,
                            f"Nurse {nid} has forbidden succession {fs[0]}->{fs[1]} on days {a1['day']}->{a2['day']} (H3)",
                            1.0, 0.0, "leq")

    # =========================================================================
    # Constraint 4 (1e): Skill assignment linking (H4)
    # Nurse can only perform skills they have.
    # =========================================================================
    for nurse in nurses:
        nid = nurse["id"]
        assignments = schedule_by_nurse.get(nid, [])
        nurse_sk = nurse_skills[nid]
        for a in assignments:
            skill = a["skill"]
            if skill not in nurse_sk:
                # LHS = 1 (assignment with invalid skill), RHS = 0
                record_violation(4,
                    f"Nurse {nid} assigned skill '{skill}' on day {a['day']} shift {a['shift']}, "
                    f"but only has skills {sorted(nurse_sk)} (H4)",
                    1.0, 0.0, "leq")

    # =========================================================================
    # Constraint 5 (1f): Minimum demand (H2)
    # sum_{i in N_sigma} y_{ip,sigma} >= D_{p,sigma}, for all p, sigma
    # =========================================================================
    # Count assignments per (day, shift, skill)
    staffing = defaultdict(int)
    for entry in schedule:
        for a in entry["assignments"]:
            staffing[(a["day"], a["shift"], a["skill"])] += 1

    for day_idx in range(num_days):
        day_demand = demand[day_idx]
        for shift_name, skill_demands in day_demand.items():
            for skill_name, dem in skill_demands.items():
                min_dem = dem["minimum"]
                actual = staffing.get((day_idx, shift_name, skill_name), 0)
                if actual < min_dem - tol:
                    # LHS = actual, RHS = min_dem, constraint is LHS >= RHS
                    record_violation(5,
                        f"Day {day_idx} shift {shift_name} skill {skill_name}: "
                        f"staffing {actual} < minimum demand {min_dem} (H2)",
                        float(actual), float(min_dem), "geq")

    # =========================================================================
    # Constraint 7 (1h): Integrality of x, y
    # x_{ij}, y_{ip,sigma} in {0,1}
    # Trivially satisfied by the solution format (discrete assignments).
    # We verify that assignment values are valid (known shifts and skills).
    # =========================================================================
    valid_shifts = {s["name"] for s in instance["shifts"]}
    valid_skills = set(instance["skills"])
    for entry in schedule:
        nid = entry["nurse_id"]
        for a in entry["assignments"]:
            if a["shift"] not in valid_shifts:
                record_violation(7,
                    f"Nurse {nid} day {a['day']}: unknown shift '{a['shift']}'",
                    1.0, 0.0, "leq")
            if a["skill"] not in valid_skills:
                record_violation(7,
                    f"Nurse {nid} day {a['day']}: unknown skill '{a['skill']}'",
                    1.0, 0.0, "leq")
            if not (0 <= a["day"] < num_days):
                record_violation(7,
                    f"Nurse {nid}: day {a['day']} out of range [0, {num_days - 1}]",
                    float(a["day"]), float(num_days - 1), "leq")

    # =========================================================================
    # Constraint 8 (1i): Non-negativity of w, z
    # w, z >= 0. These are slack/surplus variables implicitly >= 0 by definition
    # when computed from a valid schedule. Trivially satisfied.
    # =========================================================================
    # No explicit check needed; w and z are derived from the schedule and
    # are always non-negative by construction.

    # =========================================================================
    # Constraint 9 (Tier C): Objective consistency
    # Recompute the canonical minimum penalty implied by the schedule.  Legacy
    # Gurobi references also serialize a penalty_breakdown because a
    # time-limited incumbent can retain nonminimal continuous slack values;
    # in that case bind objective_value to the serialized component sum instead
    # of accepting the former, extremely broad instance-level upper bound.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            lb, _ = _compute_obj_bounds(instance, solution)
            obj_tol = max(1e-3, 1e-3 * abs(lb))
            if reported < lb - obj_tol:
                record_violation(
                    9,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} is below the schedule-derived lower bound "
                    f"LB={lb} (shortfall={lb - reported:.6g}, tol={obj_tol:.3g}).",
                    reported, lb, "geq",
                )
            elif abs(reported - lb) > obj_tol:
                breakdown = solution.get("penalty_breakdown")
                component_names = (
                    "S1_optimal_coverage",
                    "S2a_consecutive_days",
                    "S2b_consecutive_shifts",
                    "S3_consecutive_rest",
                    "S4_preferences",
                    "S5_complete_weekend",
                    "S6_total_assignments",
                    "S7_working_weekends",
                )
                breakdown_total = None
                if isinstance(breakdown, dict) and all(
                        name in breakdown for name in component_names):
                    try:
                        values = [float(breakdown[name])
                                  for name in component_names]
                        if all(math.isfinite(value) and value >= 0.0
                               for value in values):
                            breakdown_total = sum(values)
                    except (TypeError, ValueError):
                        breakdown_total = None

                legacy_tol = max(1e-3, 1e-8 * max(abs(reported), 1.0))
                if (breakdown_total is None
                        or abs(reported - breakdown_total) > legacy_tol):
                    record_violation(
                        9,
                        f"Objective consistency violated: reported "
                        f"objective_value={reported} is above the canonical "
                        f"schedule cost LB={lb}, but does not equal a complete "
                        f"legacy penalty_breakdown (sum={breakdown_total}, "
                        f"tol={legacy_tol:.3g}).",
                        reported,
                        breakdown_total if breakdown_total is not None else lb,
                        "eq",
                    )

    # Build result
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if feasible:
        print("Solution is FEASIBLE (all hard constraints satisfied).")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {sorted(violated_constraints)}")
        for v in violations:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
