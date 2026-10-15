#!/usr/bin/env python3
"""
Feasibility checker for the Integrated Berth Allocation and Quay Crane
Assignment Problem (BACAP) from Xie, Wu, and Zhang (2019).

Checks:
1. Each vessel is assigned exactly one berth.
2. Each vessel starts within its time window [a_i, b_i].
3. Each vessel starts within its assigned berth's time window [a_k, b_k].
4. Vessel end time = start time + profile duration - 1, and end <= H_size.
5. No two vessels overlap on the same berth at the same time step.
6. QC capacity constraint: total QCs used at each time step <= total_qc.
7. Objective value matches recomputation.
"""

import argparse
import json
import random
from itertools import product

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'vessels')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('vessels',)
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


def generate_qc_profiles(instance, rng):
    """Generate QC profiles (same logic as gurobi_code.py)."""
    S_size = instance["time_steps_per_shift"]
    big_cfg = instance["qc_profiles"]["big"]
    small_cfg = instance["qc_profiles"]["small"]
    num_big = instance["num_big_profiles"]
    num_small = instance["num_small_profiles"]

    def make_profiles(cfg, count):
        """Stratified sampling -- must match gurobi_code.py exactly so profile
        indices line up. One profile per duration bucket first, then random
        fill from the rest using the same RNG sequence."""
        min_qc, max_qc = cfg["min_qc"], cfg["max_qc"]
        min_s, max_s = cfg["min_handling_shifts"], cfg["max_handling_shifts"]
        qc_range = list(range(min_qc, max_qc + 1))
        by_dur = {}
        for dur in range(min_s, max_s + 1):
            by_dur[dur] = [
                {"duration_shifts": dur, "qc_per_shift": list(combo)}
                for combo in product(qc_range, repeat=dur)
            ]
        all_possible = [p for ps in by_dur.values() for p in ps]
        if len(all_possible) <= count:
            return all_possible
        n_buckets = len(by_dur)
        selected = []
        for dur in by_dur:
            selected.append(rng.choice(by_dur[dur]))
        remaining_count = count - n_buckets
        if remaining_count > 0:
            taken_ids = {id(p) for p in selected}
            rest = [p for p in all_possible if id(p) not in taken_ids]
            extra = rng.sample(rest, min(remaining_count, len(rest)))
            selected.extend(extra)
        return selected

    all_raw = make_profiles(big_cfg, num_big) + make_profiles(small_cfg, num_small)

    profiles_per_vessel = {}
    for v in instance["vessels"]:
        vid, a_i, b_i = v["id"], v["a"], v["b"]
        feasible = []
        for praw in all_raw:
            dur_steps = praw["duration_shifts"] * S_size
            if dur_steps <= (b_i - a_i + 1):
                qc_per_step = []
                for sq in praw["qc_per_shift"]:
                    qc_per_step.extend([sq] * S_size)
                feasible.append({
                    "pid": len(feasible),
                    "duration_steps": dur_steps,
                    "duration_shifts": praw["duration_shifts"],
                    "qc_per_shift": praw["qc_per_shift"],
                    "qc_per_step": qc_per_step,
                })
        profiles_per_vessel[vid] = feasible
    return profiles_per_vessel


def get_shift_index(h, S_size):
    return ((h - 1) % S_size) + 1


def compute_d_sp(profiles_per_vessel, S_size):
    d = {}
    for vid, profs in profiles_per_vessel.items():
        for prof in profs:
            pid = prof["pid"]
            D = prof["duration_shifts"]
            for s in range(1, S_size + 1):
                d[(vid, s, pid)] = (S_size - s + 1) + (D - 1) * S_size
    return d


def compute_q_spu(profiles_per_vessel, S_size):
    q = {}
    for vid, profs in profiles_per_vessel.items():
        for prof in profs:
            pid = prof["pid"]
            D = prof["duration_shifts"]
            qc_shifts = prof["qc_per_shift"]
            for s in range(1, S_size + 1):
                u = 1
                for _ in range(S_size - s + 1):
                    q[(vid, s, pid, u)] = qc_shifts[0]
                    u += 1
                for si in range(1, D):
                    for _ in range(S_size):
                        q[(vid, s, pid, u)] = qc_shifts[si]
                        u += 1
    return q


def check_feasibility(instance, solution):
    violations = []
    violated_constraints = set()

    # If no solution
    if solution.get("objective_value") is None or solution.get("status") in ("NO_SOLUTION", "INFEASIBLE"):
        return {
            "feasible": False,
            "violated_constraints": ["no_solution"],
            "violations": ["No solution found."],
            "objective_value": None,
            "recomputed_objective": None
        }

    rng = random.Random(instance.get("random_seed", 42))
    S_size = instance["time_steps_per_shift"]
    F_size = instance["num_shifts_per_step"]
    H_size = S_size * F_size
    total_qc = instance["total_qc"]
    c1, c2 = instance["c1"], instance["c2"]
    vessels_data = {v["id"]: v for v in instance["vessels"]}
    berths_data = {b["id"]: b for b in instance["berths"]}
    n = instance["num_vessels"]

    profiles_per_vessel = generate_qc_profiles(instance, rng)
    d_sp = compute_d_sp(profiles_per_vessel, S_size)
    q_spu = compute_q_spu(profiles_per_vessel, S_size)

    sol_vessels = solution.get("vessels", [])
    if len(sol_vessels) != n:
        violations.append(f"Expected {n} vessel assignments, got {len(sol_vessels)}.")
        violated_constraints.add("assignment_count")

    expected_vessel_ids = set(vessels_data)
    submitted_vessel_ids = []

    # Track berth occupancy and QC usage
    berth_usage = {}  # (k, h) -> list of vessel ids
    qc_usage = {}     # h -> total QC count
    recomputed_obj = 0.0

    for vs in sol_vessels:
        if not isinstance(vs, dict):
            violations.append("Each vessel assignment must be an object.")
            violated_constraints.add("assignment_domain")
            continue

        missing_fields = [
            key for key in ("id", "berth", "start_time", "end_time", "profile")
            if key not in vs
        ]
        if missing_fields:
            violations.append(
                "Vessel assignment is missing field(s): " + ", ".join(missing_fields)
            )
            violated_constraints.add("assignment_domain")
            continue

        vid = vs["id"]
        berth = vs["berth"]
        start = vs["start_time"]
        end = vs["end_time"]
        pid = vs["profile"]

        integer_fields = {
            "id": vid,
            "berth": berth,
            "start_time": start,
            "end_time": end,
            "profile": pid,
        }
        invalid_integer_fields = [
            key for key, value in integer_fields.items()
            if not isinstance(value, int) or isinstance(value, bool)
        ]
        if invalid_integer_fields:
            violations.append(
                "Vessel assignment requires integer field(s): "
                + ", ".join(invalid_integer_fields)
            )
            violated_constraints.add("assignment_domain")
            continue

        submitted_vessel_ids.append(vid)
        if vid not in vessels_data:
            violations.append(f"Unknown vessel id {vid}.")
            violated_constraints.add("assignment_ids")
            continue
        v = vessels_data[vid]

        # Check berth assignment
        if berth not in berths_data:
            violations.append(f"Vessel {vid}: invalid berth {berth}.")
            violated_constraints.add("berth_assignment")
            continue

        bk = berths_data[berth]

        # Check vessel time window
        if start < v["a"] or start > v["b"]:
            violations.append(f"Vessel {vid}: start={start} outside time window [{v['a']}, {v['b']}].")
            violated_constraints.add("vessel_time_window")

        # The complete service interval must lie in the berth window.
        if start < bk["a_k"] or end > bk["b_k"]:
            violations.append(
                f"Vessel {vid}: service interval [{start}, {end}] outside "
                f"berth {berth} window [{bk['a_k']}, {bk['b_k']}]."
            )
            violated_constraints.add("berth_time_window")

        # A profile ID is an ordinary zero-based array index.  Negative IDs
        # previously invoked Python's negative indexing and disabled the
        # duration and crane-use checks.
        profiles = profiles_per_vessel.get(vid, [])
        valid_profile = 0 <= pid < len(profiles)
        if not valid_profile:
            violations.append(
                f"Vessel {vid}: profile={pid} outside valid range "
                f"[0, {len(profiles) - 1}]."
            )
            violated_constraints.add("profile_domain")
        else:
            s = get_shift_index(start, S_size)
            expected_end = start + d_sp[(vid, s, pid)] - 1
            if end != expected_end:
                violations.append(f"Vessel {vid}: end={end} expected {expected_end}.")
                violated_constraints.add("end_time")

        # A reversed interval used to execute both resource-accounting loops
        # zero times.  Validate the entire interval before accounting for it.
        valid_interval = 1 <= start <= end <= H_size
        if end < start:
            violations.append(
                f"Vessel {vid}: end={end} is earlier than start={start}."
            )
            violated_constraints.add("end_time")
        if not valid_interval:
            violations.append(
                f"Vessel {vid}: service interval [{start}, {end}] outside "
                f"horizon [1, {H_size}]."
            )
            violated_constraints.add("horizon")

        # Track berth occupancy
        if valid_interval:
            for h in range(start, end + 1):
                key = (berth, h)
                if key not in berth_usage:
                    berth_usage[key] = []
                berth_usage[key].append(vid)

        # Track QC usage
        if valid_profile and valid_interval:
            s = get_shift_index(start, S_size)
            dur = d_sp[(vid, s, pid)]
            for u in range(1, dur + 1):
                h = start + u - 1
                qc_val = q_spu[(vid, s, pid, u)]
                qc_usage[h] = qc_usage.get(h, 0) + qc_val

        # Compute cost
        delta_k = abs(berth - v["k_bar"])
        delta_t = abs(start - v["t_bar"])
        recomputed_obj += c1 * delta_k + c2 * delta_t

    if (
        set(submitted_vessel_ids) != expected_vessel_ids
        or len(submitted_vessel_ids) != len(set(submitted_vessel_ids))
    ):
        submitted_id_set = set(submitted_vessel_ids)
        duplicate_ids = sorted({
            vid for vid in submitted_vessel_ids
            if submitted_vessel_ids.count(vid) > 1
        })
        violations.append(
            "Vessel IDs must cover the instance exactly once; "
            f"missing={sorted(expected_vessel_ids - submitted_id_set)}, "
            f"duplicate={duplicate_ids}, "
            f"unexpected={sorted(submitted_id_set - expected_vessel_ids)}."
        )
        violated_constraints.add("assignment_ids")

    # Check no-overlap constraint
    for (k, h), vids in berth_usage.items():
        if len(vids) > 1:
            violations.append(f"Berth {k} at time {h}: overlapping vessels {vids}.")
            violated_constraints.add("no_overlap")

    # Check QC capacity
    for h, qc_total in qc_usage.items():
        if qc_total > total_qc:
            violations.append(f"Time step {h}: QC usage {qc_total} > capacity {total_qc}.")
            violated_constraints.add("qc_capacity")

    # Check objective
    reported_obj = solution.get("objective_value")
    obj_match = True
    if reported_obj is not None:
        # c1, c2 are integers and delta_k, delta_t are integer differences,
        # so the recomputed objective is always exactly integer. Use 1e-6 to
        # tolerate float-representation noise without permitting any real
        # integer-level objective drift.
        if abs(reported_obj - recomputed_obj) > 1e-6:
            violations.append(f"Objective mismatch: reported={reported_obj}, recomputed={recomputed_obj}.")
            violated_constraints.add("objective")
            obj_match = False

    feasible = len(violations) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "objective_value": reported_obj,
        "recomputed_objective": recomputed_obj
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility check for BACAP (Xie et al. 2019)"
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
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

    print(f"Feasibility: {result['feasible']}")
    if result['violations']:
        for v in result['violations'][:10]:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
