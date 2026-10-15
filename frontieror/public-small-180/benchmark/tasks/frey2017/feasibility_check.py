#!/usr/bin/env python3
"""
Feasibility Checker for the Outbound Baggage Handling Problem (OBHP)
Based on Frey, Kolisch, and Artigues (2017)

Checks hard constraints (2)-(9) from the TIF formulation:
  Constraint (2): Flight assignment - each flight assigned exactly once
  Constraint (3): Working station capacity per carousel per period
  Constraint (4): Parking position capacity per carousel per period
  Constraint (5): Central storage system capacity per period
  Constraint (6): Utilization threshold linking constraint
  Constraint (7): At most one threshold level per carousel per peak interval
  Constraint (8): Binary domain for flight assignment variables x
  Constraint (9): Binary domain for threshold violation variables y
  Constraint (10): Objective consistency (Tier C anti-gaming check) -
                   reported objective_value must match the objective (1)
                   recomputed from the solution's y variables

Objective (1) is not a hard constraint of the original model, but constraint
(10) below rejects solutions whose self-reported objective_value disagrees
with the value recomputed from the solution variables, defeating LLM
score-gaming exploits that return fabricated objective values.
"""

import argparse
import json
import math
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignments', 'objective_value', 'utilization_violations')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('assignments',)
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



# ---------------------------------------------------------------------------
# Tolerance parameters
# ---------------------------------------------------------------------------
TOL = 1e-5
EPS = 1e-5


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_solution(path):
    with open(path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Gamma / Phi computation (Online Appendix B simulation)
# Mirrors compute_gamma_phi in efficient_algorithm.py
# ---------------------------------------------------------------------------

def compute_gamma_phi(flight, w, s_h, s_d, working_rate, depletion_rate):
    """
    Compute Gamma^{i,w}_{t,tau} (belt workload) and Phi^{i}_{t,tau}
    (central storage occupancy) for a flight.

    Returns:
        gamma: dict {t: bags_on_belt}
        phi:   dict {t: bags_in_storage}
    """
    es = flight['earliest_start_period']
    se = flight['end_period']
    profile = flight['arrival_profile']

    gamma = {}
    phi = {}

    stored = 0
    belt_bags = 0

    for t in range(es, se):
        idx = t - es
        arrivals = profile[idx] if 0 <= idx < len(profile) else 0

        if t < s_h:
            stored += arrivals
            phi[t] = stored
            gamma[t] = 0
        else:
            belt_bags += arrivals
            if t >= s_d and stored > 0:
                depleted = min(stored, depletion_rate)
                belt_bags += depleted
                stored -= depleted
            loaded = min(belt_bags, w * working_rate)
            belt_bags -= loaded
            gamma[t] = belt_bags
            phi[t] = stored

    return gamma, phi


# ---------------------------------------------------------------------------
# Build lookup structures from instance
# ---------------------------------------------------------------------------

def build_carousel_map(instance):
    """Return dict {carousel_id: carousel_dict}."""
    return {c['carousel_id']: c for c in instance['carousels']}


def build_flight_map(instance):
    """Return dict {flight_id: flight_dict}."""
    return {f['flight_id']: f for f in instance['flights']}


# ---------------------------------------------------------------------------
# Normalise solution assignment keys across efficient / gurobi formats
# ---------------------------------------------------------------------------

def normalise_assignment(a):
    """Return (flight_id, carousel_id, w, s_h, s_d) regardless of format."""
    fid = a['flight_id']
    cid = a['carousel_id']
    w = a['working_stations']
    s_h = a.get('handling_start', a.get('handling_start_period'))
    s_d = a.get('depletion_start', a.get('depletion_start_period'))
    return fid, cid, w, s_h, s_d


# ---------------------------------------------------------------------------
# Violation recording helpers
# ---------------------------------------------------------------------------

def make_magnitude(constraint_idx, lhs, rhs, raw_excess):
    normalizer = max(abs(rhs), EPS)
    return {
        'constraint': constraint_idx,
        'lhs': float(lhs),
        'rhs': float(rhs),
        'raw_excess': float(raw_excess),
        'normalizer': float(normalizer),
        'ratio': float(raw_excess / normalizer),
    }


# ---------------------------------------------------------------------------
# Precompute time-dependent storage capacity (Section 3.3 preprocessing)
# ---------------------------------------------------------------------------

def precompute_storage_capacity(instance):
    num_periods = instance['planning_horizon']['num_periods']
    Ks_base = instance['central_storage']['capacity_bags']
    flights = instance['flights']

    Ks = [Ks_base] * num_periods
    for t in range(num_periods):
        reserved = 0
        for f in flights:
            es = f['earliest_start_period']
            if t < es:
                cum = 0
                for idx, bags in enumerate(f['arrival_profile']):
                    arr_period = es + idx
                    if arr_period > t:
                        break
                    cum += bags
                reserved += cum
        Ks[t] = max(0, Ks_base - reserved)
    return Ks


# ---------------------------------------------------------------------------
# Main feasibility checks
# ---------------------------------------------------------------------------

def check_feasibility(instance, solution):
    violations = []           # textual messages
    violated_set = set()      # constraint indices (unique)
    magnitudes = []           # violation magnitude dicts

    # Handle null / infeasible solutions – when the solver found no solution,
    # feasibility is indeterminate (null), not false.
    status = solution.get("status", solution.get("status_name", ""))
    status_lower = str(status).lower() if status else ""
    has_no_solution = (
        solution.get("objective_value") is None
        or "infeasible" in status_lower
        or "no_solution" in status_lower
        or (not solution.get("assignments"))
    )
    if has_no_solution:
        return {
            "feasible": None,
            "violated_constraints": ["no_solution"],
            "violations": ["No feasible solution provided"],
            "violation_magnitudes": [],
        }

    flights_map = build_flight_map(instance)
    carousels_map = build_carousel_map(instance)

    num_periods = instance['planning_horizon']['num_periods']
    working_rate = instance['working_rate_bags_per_period']
    depletion_rate = instance['central_storage']['depletion_rate_bags_per_period']
    depletion_offset = instance['central_storage']['depletion_offset_periods']

    assignments = solution.get('assignments', [])

    # Normalise assignments
    norm_assignments = []
    for a in assignments:
        fid, cid, w, s_h, s_d = normalise_assignment(a)
        norm_assignments.append((fid, cid, w, s_h, s_d))

    # ==================================================================
    # Constraint (2): Each flight assigned to exactly one configuration
    # sum_{c,w,tau} x_{i,c,w,tau} = 1  for all i in F
    # ==================================================================
    all_flight_ids = set(f['flight_id'] for f in instance['flights'])
    assigned_counts = defaultdict(int)
    for fid, cid, w, s_h, s_d in norm_assignments:
        assigned_counts[fid] += 1

    # Check: every flight must appear exactly once
    for fid in sorted(all_flight_ids):
        cnt = assigned_counts.get(fid, 0)
        if cnt == 0:
            lhs = 0.0
            rhs = 1.0
            excess = abs(lhs - rhs)
            if excess > TOL:
                violated_set.add(2)
                violations.append(f"Constraint (2): Flight {fid} is not assigned (count=0, required=1)")
                magnitudes.append(make_magnitude(2, lhs, rhs, excess))
        elif cnt > 1:
            lhs = float(cnt)
            rhs = 1.0
            excess = abs(lhs - rhs)
            if excess > TOL:
                violated_set.add(2)
                violations.append(f"Constraint (2): Flight {fid} is assigned {cnt} times (required=1)")
                magnitudes.append(make_magnitude(2, lhs, rhs, excess))

    # Also check that each assignment uses a feasible start time tuple and
    # valid working station count for the carousel type. These are part of
    # the domain of constraint (2) (the sums range only over feasible tuples).
    for fid, cid, w, s_h, s_d in norm_assignments:
        flight = flights_map.get(fid)
        car = carousels_map.get(cid)
        if flight is None or car is None:
            continue

        tr = str(car['type_r'])
        ws_bounds = flight.get('working_station_bounds_by_carousel_type', {}).get(tr)

        # Check working station bounds
        if ws_bounds is not None:
            wmin = ws_bounds['Wmin']
            wmax = ws_bounds['Wmax']
            if w < wmin or w > wmax:
                # working station count out of range – this means the
                # assignment uses an (i,c,w) combination for which no
                # x variable exists → violates constraint (2)
                lhs_val = float(w)
                rhs_val = float(wmax)  # use upper bound for normalisation
                excess = 0.0
                if w < wmin:
                    excess = float(wmin - w)
                    rhs_val = float(wmin)
                else:
                    excess = float(w - wmax)
                    rhs_val = float(wmax)
                if excess > TOL:
                    violated_set.add(2)
                    violations.append(
                        f"Constraint (2): Flight {fid} on carousel {cid}: "
                        f"working stations w={w} outside [{wmin},{wmax}] for type {tr}"
                    )
                    magnitudes.append(make_magnitude(2, lhs_val, rhs_val, excess))

        # Check handling start within time window
        es = flight['earliest_start_period']
        ls = flight['latest_start_period']
        if s_h < es or s_h > ls:
            excess = 0.0
            if s_h < es:
                excess = float(es - s_h)
                rhs_val = float(es)
            else:
                excess = float(s_h - ls)
                rhs_val = float(ls)
            if excess > TOL:
                violated_set.add(2)
                violations.append(
                    f"Constraint (2): Flight {fid}: handling_start={s_h} "
                    f"outside window [{es},{ls}]"
                )
                magnitudes.append(make_magnitude(2, float(s_h), rhs_val, excess))

        # Check depletion start feasibility: s_h <= s_d < S^e_i - depletion_offset
        se = flight['end_period']
        s_d_max = se - depletion_offset - 1
        if s_d < s_h:
            excess = float(s_h - s_d)
            if excess > TOL:
                violated_set.add(2)
                violations.append(
                    f"Constraint (2): Flight {fid}: depletion_start={s_d} < handling_start={s_h}"
                )
                magnitudes.append(make_magnitude(2, float(s_d), float(s_h), excess))
        if s_d > s_d_max:
            excess = float(s_d - s_d_max)
            if excess > TOL:
                violated_set.add(2)
                violations.append(
                    f"Constraint (2): Flight {fid}: depletion_start={s_d} > max allowed {s_d_max} "
                    f"(end_period={se}, depletion_offset={depletion_offset})"
                )
                magnitudes.append(make_magnitude(2, float(s_d), float(s_d_max), excess))

    # ==================================================================
    # Constraint (3): Working station capacity per carousel per period
    # sum_{eligible i} sum_w sum_tau w * x_{i,c,w,tau} <= K^ws_r
    # for all r, c in C_r, t in T (only when s_h <= t and S^es_i <= t < S^e_i)
    # ==================================================================

    # Group assignments by carousel
    carousel_assignments = defaultdict(list)
    for fid, cid, w, s_h, s_d in norm_assignments:
        carousel_assignments[cid].append((fid, w, s_h, s_d))

    for car in instance['carousels']:
        cid = car['carousel_id']
        type_data = instance['carousel_types'][str(car['type_r'])]
        Kws = type_data['Kws']
        assigns = carousel_assignments.get(cid, [])
        if not assigns:
            continue

        for t in range(num_periods):
            ws_sum = 0
            for fid, w, s_h, s_d in assigns:
                flight = flights_map.get(fid)
                if flight is None:
                    continue
                es = flight['earliest_start_period']
                se = flight['end_period']
                if es <= t < se and s_h <= t:
                    ws_sum += w
            lhs = float(ws_sum)
            rhs = float(Kws)
            excess = lhs - rhs
            if excess > TOL:
                violated_set.add(3)
                violations.append(
                    f"Constraint (3): Carousel {cid}, period {t}: "
                    f"working station usage {ws_sum} exceeds capacity {Kws}"
                )
                magnitudes.append(make_magnitude(3, lhs, rhs, excess))

    # ==================================================================
    # Constraint (4): Parking position capacity per carousel per period
    # sum_{eligible i} sum_w sum_tau P_i * x_{i,c,w,tau} <= K^pp_r
    # for all r, c in C_r, t in T (only when s_h <= t and S^es_i <= t < S^e_i)
    # ==================================================================

    for car in instance['carousels']:
        cid = car['carousel_id']
        type_data = instance['carousel_types'][str(car['type_r'])]
        Kpp = type_data['Kpp']
        assigns = carousel_assignments.get(cid, [])
        if not assigns:
            continue

        for t in range(num_periods):
            pp_sum = 0
            for fid, w, s_h, s_d in assigns:
                flight = flights_map.get(fid)
                if flight is None:
                    continue
                es = flight['earliest_start_period']
                se = flight['end_period']
                if es <= t < se and s_h <= t:
                    pp_sum += flight['num_containers']
            lhs = float(pp_sum)
            rhs = float(Kpp)
            excess = lhs - rhs
            if excess > TOL:
                violated_set.add(4)
                violations.append(
                    f"Constraint (4): Carousel {cid}, period {t}: "
                    f"parking positions {pp_sum} exceeds capacity {Kpp}"
                )
                magnitudes.append(make_magnitude(4, lhs, rhs, excess))

    # ==================================================================
    # Constraint (5): Central storage system capacity per period
    # sum_{eligible i} sum_c sum_w sum_tau Phi^i_{t,tau} * x_{i,c,w,tau} <= K^s_t
    # for all t in T
    # ==================================================================

    storage_cap = precompute_storage_capacity(instance)

    for t in range(num_periods):
        total_phi = 0.0
        for fid, cid, w, s_h, s_d in norm_assignments:
            flight = flights_map.get(fid)
            if flight is None:
                continue
            es = flight['earliest_start_period']
            se = flight['end_period']
            if not (es <= t < se):
                continue
            _, phi = compute_gamma_phi(flight, w, s_h, s_d,
                                       working_rate, depletion_rate)
            total_phi += phi.get(t, 0)

        lhs = float(total_phi)
        rhs = float(storage_cap[t])
        excess = lhs - rhs
        if excess > TOL:
            violated_set.add(5)
            violations.append(
                f"Constraint (5): Period {t}: central storage usage "
                f"{total_phi:.1f} exceeds capacity {storage_cap[t]:.1f}"
            )
            magnitudes.append(make_magnitude(5, lhs, rhs, excess))

    # ==================================================================
    # Constraint (6): Utilization threshold linking constraint
    # (1/K^cb_r) * sum Gamma * x - sum_k u_k * y_{c,k,m} <= u^ta
    # for all r, c in C_r, 1 <= m <= M, t in T_m
    # ==================================================================
    peak_intervals = instance.get('peak_intervals', [])
    obj_data = instance.get('objective', {})
    threshold_values = obj_data.get('threshold_values', [])
    u_ta = float(obj_data.get('target_utilization', 0.0))

    y_set = set()
    for viol in solution.get('utilization_violations') or []:
        y_set.add((viol['carousel_id'],
                   viol['threshold_level'] - 1,
                   viol['peak_interval'] - 1))

    for m_idx, peak in enumerate(peak_intervals):
        t_start = peak['start_period']
        t_end = peak['end_period']
        for car in instance['carousels']:
            cid = car['carousel_id']
            type_data = instance['carousel_types'][str(car['type_r'])]
            Kcb = type_data.get('Kcb')
            if not Kcb:
                continue
            assigns = carousel_assignments.get(cid, [])
            for t in range(t_start, t_end + 1):
                total_gamma = 0.0
                for fid, w, s_h, s_d in assigns:
                    flight = flights_map.get(fid)
                    if flight is None:
                        continue
                    es = flight['earliest_start_period']
                    se = flight['end_period']
                    if not (es <= t < se) or s_h > t:
                        continue
                    gamma, _ = compute_gamma_phi(
                        flight, w, s_h, s_d,
                        working_rate, depletion_rate)
                    total_gamma += gamma.get(t, 0)
                y_sum = sum(threshold_values[k]
                            for k in range(len(threshold_values))
                            if (cid, k, m_idx) in y_set)
                lhs = (total_gamma / Kcb) - y_sum
                excess = lhs - u_ta
                if excess > TOL:
                    violated_set.add(6)
                    violations.append(
                        f"Constraint (6): Carousel {cid}, peak "
                        f"{m_idx + 1}, period {t}: util "
                        f"{total_gamma / Kcb:.4f} - slack "
                        f"{y_sum:.4f} > u_ta={u_ta} by {excess:.4f}")
                    magnitudes.append(
                        make_magnitude(6, lhs, u_ta, excess))

    # ==================================================================
    # Constraint (7): At most one threshold level per carousel per peak
    # sum_k y_{c,k,m} <= 1  for all c in C, 1 <= m <= M
    # ==================================================================
    for car in instance['carousels']:
        cid = car['carousel_id']
        for m_idx in range(len(peak_intervals)):
            y_count = sum(1 for k in range(len(threshold_values))
                          if (cid, k, m_idx) in y_set)
            if y_count > 1:
                excess = float(y_count - 1)
                violated_set.add(7)
                violations.append(
                    f"Constraint (7): Carousel {cid}, peak "
                    f"{m_idx + 1}: {y_count} threshold levels "
                    f"activated (max 1)")
                magnitudes.append(
                    make_magnitude(7, float(y_count), 1.0, excess))

    # ==================================================================
    # Constraint (8): Binary domain for x_{i,c,w,tau}
    # Each assignment implicitly sets x=1; verify integer values and
    # uniqueness of (flight, carousel, w, tau) combinations.
    # ==================================================================
    for fid, cid, w, s_h, s_d in norm_assignments:
        for name, val in (('working_stations', w),
                          ('handling_start', s_h),
                          ('depletion_start', s_d)):
            if val is None:
                continue
            fval = float(val)
            excess = abs(fval - round(fval))
            if excess > TOL:
                violated_set.add(8)
                violations.append(
                    f"Constraint (8): Flight {fid}: non-integer "
                    f"{name}={val} (x must be binary)")
                magnitudes.append(
                    make_magnitude(8, fval, float(round(fval)), excess))

    x_key_counts = defaultdict(int)
    for fid, cid, w, s_h, s_d in norm_assignments:
        x_key_counts[(fid, cid, w, s_h, s_d)] += 1
    for key, cnt in x_key_counts.items():
        if cnt > 1:
            excess = float(cnt - 1)
            violated_set.add(8)
            violations.append(
                f"Constraint (8): x for {key} selected {cnt} times "
                f"(must be binary)")
            magnitudes.append(
                make_magnitude(8, float(cnt), 1.0, excess))

    # ==================================================================
    # Constraint (9): Binary domain for y_{c,k,m}
    # Each utilization_violation entry implicitly sets y=1; verify no
    # (carousel, k, m) duplicates.
    # ==================================================================
    y_key_counts = defaultdict(int)
    for viol in solution.get('utilization_violations') or []:
        y_key_counts[(viol['carousel_id'],
                      viol['threshold_level'] - 1,
                      viol['peak_interval'] - 1)] += 1
    for key, cnt in y_key_counts.items():
        if cnt > 1:
            excess = float(cnt - 1)
            violated_set.add(9)
            violations.append(
                f"Constraint (9): y for {key} selected {cnt} times "
                f"(must be binary)")
            magnitudes.append(
                make_magnitude(9, float(cnt), 1.0, excess))

    # ==================================================================
    # Constraint (10): Objective consistency (Tier C anti-gaming check)
    # Recompute objective (1):  sum_{c} sum_{k} sum_{m} p_k * y_{c,k,m}
    # Every utilization_violations entry is exactly one activated
    # y_{c,k,m}=1, and the penalty p_k equals penalties[k-1] (threshold_level
    # is 1-indexed -- the same 1->0 offset the constraint (6)/(9) blocks use).
    # The objective depends ONLY on the y variables, ALL of which are carried
    # in the solution -> a full exact recompute applies (no second-stage
    # variables, no LP subproblem). Reject when the self-reported
    # objective_value disagrees with the recomputed value.
    # ==================================================================
    penalties = obj_data.get('penalties')
    reported_obj = solution.get('objective_value')
    if penalties is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            recompute_ok = True
            for viol in solution.get('utilization_violations') or []:
                try:
                    k = int(viol['threshold_level'])
                except (TypeError, ValueError, KeyError):
                    # malformed y entry -- domain checks (9) cover this;
                    # skip the obj recompute rather than crash.
                    recompute_ok = False
                    break
                if 1 <= k <= len(penalties):
                    true_obj += float(penalties[k - 1])
                else:
                    # threshold level outside the penalty table -> cannot
                    # recompute reliably; defer to constraint (6)/(9).
                    recompute_ok = False
                    break
            if recompute_ok:
                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with a 1e-3 absolute floor.
                tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > tol:
                    violated_set.add(10)
                    violations.append(
                        f"Constraint (10): Objective consistency violated: "
                        f"reported objective_value={reported} differs from "
                        f"recomputed sum_(c,k,m) p_k*y_(c,k,m)={true_obj} "
                        f"(|diff|={abs_diff:.3g}, tol={tol:.3g})")
                    magnitudes.append(
                        make_magnitude(10, reported, true_obj, abs_diff))

    # ==================================================================
    # Build result
    # ==================================================================
    feasible = len(violated_set) == 0
    result = {
        'feasible': feasible,
        'violated_constraints': sorted(violated_set),
        'violations': violations,
        'violation_magnitudes': magnitudes,
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Feasibility checker for OBHP (Frey et al. 2017)'
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to solution JSON file')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write feasibility result JSON file')
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
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

    if result['feasible'] is None:
        tag = "INDETERMINATE (no solution)"
    elif result['feasible']:
        tag = "FEASIBLE"
    else:
        tag = "INFEASIBLE"
    n_violated = len(result['violated_constraints'])
    print(f"[{tag}] {n_violated} constraint(s) violated, "
          f"{len(result['violation_magnitudes'])} violation instance(s).")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
