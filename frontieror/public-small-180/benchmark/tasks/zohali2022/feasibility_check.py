#!/usr/bin/env python3
"""
Feasibility checker for SUALBP-2 (Zohali et al. 2022).

Checks the original SUALBP-2 problem against the original solution
structure (task-to-station assignment + per-station task sequence +
cycle time). It does NOT enforce the SSBF-1 / SSBF-2 MIP reformulation
variables or the D_{ij} pairwise lower-bound valid inequality (Eq. 2g):
those are reformulation artifacts and a heuristic / non-MIP solver that
returns a valid assignment + sequence must not be falsely flagged
infeasible.

Constraints checked (against the original solution structure):
  1. Each task is assigned to exactly one station, with the station
     index in the valid range [0, m-1].
  2. Each station receives at least one task.
  3. station_sequences[k] is a permutation of the tasks assigned to
     station k (covers them all without repetition).
  4. Precedence: for every (i, j) in E, station(i) <= station(j); when
     station(i) == station(j), i appears before j in the sequence.
  5. Per-station cycle time matches: T_k = sum(processing) + sum(forward
     setups along the sequence) + backward setup from last to first.
  6. Reported `cycle_time` equals max_k T_k.
  7. cycle_time lies within [c_lower, c_upper] (when given).
  8. Reported `objective_value` matches `cycle_time`.
  9. Objective consistency: reported `objective_value` matches the
     objective recomputed directly from assignment + station_sequences
     (max_k T_k). This blocks score-gaming exploits that fabricate a
     tiny / huge objective_value while leaving the decision variables
     truthful (which would otherwise slip past constraint-only checks).
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'cycle_time', 'objective_value', 'station_sequences')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('assignment', 'station_sequences')
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
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-4
    eps = 1e-6

    violations = []
    magnitudes = []
    violated_set = set()

    def record(cidx, msg, lhs, rhs, raw):
        violated_set.add(cidx)
        violations.append(msg)
        normalizer = max(abs(rhs), eps)
        magnitudes.append({
            "constraint": cidx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(raw),
            "normalizer": float(normalizer),
            "ratio": float(raw / normalizer) if normalizer > 0 else float(raw),
        })

    # ----- Instance -----
    n = instance["num_tasks"]
    m = instance["num_stations"]
    t = instance["processing_times"]                  # 0-indexed list of length n
    tau = instance["forward_setup_times"]              # 0-indexed n x n matrix
    mu = instance["backward_setup_times"]              # 0-indexed n x n matrix
    precedence = instance.get("precedence_relations", [])  # list of [i, j] pairs (1-indexed)
    c_lower = instance.get("cycle_time_lower_bound")
    c_upper = instance.get("cycle_time_upper_bound")
    # Tasks and stations in the gurobi solution are 1-indexed; convert
    # processing times and setup matrices to 1-indexed lookups for clarity.
    def proc(i):
        return t[i - 1]
    def fwd(i, j):
        return tau[i - 1][j - 1]
    def back(i, j):
        return mu[i - 1][j - 1]

    # ----- Solution (graceful degradation) -----
    obj = solution.get("objective_value")
    cycle_time = solution.get("cycle_time", obj)

    if obj is None:
        # No solver incumbent — feasibility is indeterminate.
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No solver solution to check."],
            "violation_magnitudes": [],
        }

    raw_assign = solution.get("assignment")
    raw_seq = solution.get("station_sequences")

    # If neither original-structure field is present, only verify the
    # cycle-time bound (necessary but not sufficient).
    if not raw_assign or not raw_seq:
        if cycle_time is not None:
            if c_lower is not None and cycle_time + tol < c_lower:
                record(7, f"cycle_time {cycle_time} < c_lower {c_lower}",
                       cycle_time, c_lower, c_lower - cycle_time)
            if c_upper is not None and cycle_time - tol > c_upper:
                record(7, f"cycle_time {cycle_time} > c_upper {c_upper}",
                       cycle_time, c_upper, cycle_time - c_upper)
        feasible = None if not violated_set else False
        return {
            "feasible": feasible,
            "violated_constraints": sorted(violated_set),
            "violations": violations + (
                ["assignment / station_sequences missing — only cycle-time "
                 "bounds were checked; full feasibility cannot be verified."]
                if feasible is None else []
            ),
            "violation_magnitudes": magnitudes,
        }

    # Normalize keys to int.
    assignment = {int(k): int(v) for k, v in raw_assign.items()}
    station_sequences = {int(k): [int(x) for x in v]
                         for k, v in raw_seq.items()}

    # ----- C1: every task in V = {1..n} is assigned to a station in {1..m} -----
    for i in range(1, n + 1):
        if i not in assignment:
            record(1, f"Task {i} has no station assignment", 0, 1, 1)
        else:
            k = assignment[i]
            if not (1 <= k <= m):
                record(1,
                       f"Task {i} assigned to invalid station {k} "
                       f"(must be in [1, {m}])",
                       float(k), float(m), 1)

    # ----- C2: every station has >= 1 task -----
    tasks_per_station = {k: [] for k in range(1, m + 1)}
    for i, k in assignment.items():
        if 1 <= k <= m:
            tasks_per_station[k].append(i)
    for k in range(1, m + 1):
        if not tasks_per_station[k]:
            record(2, f"Station {k} has no tasks assigned", 0, 1, 1)

    # ----- C3: station_sequences[k] is a permutation of tasks_per_station[k] -----
    for k in range(1, m + 1):
        seq = station_sequences.get(k, [])
        expected = set(tasks_per_station[k])
        got = set(seq)
        if got != expected or len(seq) != len(expected):
            record(3,
                   f"Station {k} sequence {seq} is not a permutation of "
                   f"its assigned tasks {sorted(expected)}",
                   float(len(seq)), float(len(expected)), 1)

    # ----- C4: precedence -----
    seq_index = {}  # (k, i) -> position in station k's sequence
    for k, seq in station_sequences.items():
        for pos, i in enumerate(seq):
            seq_index[(k, i)] = pos
    for pair in precedence:
        i, j = int(pair[0]), int(pair[1])
        ki = assignment.get(i)
        kj = assignment.get(j)
        if ki is None or kj is None:
            continue  # already flagged in C1
        if ki > kj:
            record(4,
                   f"Precedence ({i}->{j}) violated: station(i)={ki} > "
                   f"station(j)={kj}",
                   float(ki), float(kj), float(ki - kj))
        elif ki == kj:
            pi = seq_index.get((ki, i))
            pj = seq_index.get((kj, j))
            if pi is None or pj is None:
                continue
            if pi >= pj:
                record(4,
                       f"Precedence ({i}->{j}) violated within station {ki}: "
                       f"position(i)={pi} >= position(j)={pj}",
                       float(pi), float(pj), 1)

    # ----- C5: per-station cycle time -----
    station_times = {}
    station_times_ok = True
    for k in range(1, m + 1):
        seq = station_sequences.get(k, [])
        if not seq:
            station_times[k] = 0.0
            continue
        try:
            T_k = sum(proc(i) for i in seq)
            for q in range(len(seq) - 1):
                T_k += fwd(seq[q], seq[q + 1])
            T_k += back(seq[-1], seq[0])
            station_times[k] = T_k
        except (IndexError, TypeError):
            station_times_ok = False
            station_times[k] = 0.0

    # C5 itself is computational; any inconsistency surfaces via C6.

    # ----- C6: reported cycle_time = max station_time -----
    if station_times:
        recomputed_cycle = max(station_times.values())
        if cycle_time is None:
            record(6, "cycle_time field missing in solution",
                   0.0, recomputed_cycle, recomputed_cycle)
        elif abs(cycle_time - recomputed_cycle) > tol * max(abs(recomputed_cycle), 1.0):
            record(6,
                   f"Reported cycle_time {cycle_time} != max station time "
                   f"{recomputed_cycle:.6g}",
                   float(cycle_time), float(recomputed_cycle),
                   abs(cycle_time - recomputed_cycle))

    # ----- C7: cycle_time within [c_lower, c_upper] -----
    if cycle_time is not None:
        if c_lower is not None and cycle_time + tol < c_lower:
            record(7, f"cycle_time {cycle_time} < c_lower {c_lower}",
                   float(cycle_time), float(c_lower),
                   float(c_lower - cycle_time))
        if c_upper is not None and cycle_time - tol > c_upper:
            record(7, f"cycle_time {cycle_time} > c_upper {c_upper}",
                   float(cycle_time), float(c_upper),
                   float(cycle_time - c_upper))

    # ----- C8: objective_value == cycle_time -----
    if obj is not None and cycle_time is not None:
        if abs(obj - cycle_time) > tol * max(abs(cycle_time), 1.0):
            record(8,
                   f"objective_value {obj} != cycle_time {cycle_time}",
                   float(obj), float(cycle_time),
                   abs(obj - cycle_time))

    # ----- C9: objective_value consistent with recomputed objective -----
    # Tier C defense against score-gaming exploits that fake objective_value
    # (and possibly cycle_time too) while leaving the decision variables
    # truthful. Recompute the true objective directly from assignment +
    # station_sequences (max_k T_k) and compare to reported obj.
    if station_times and station_times_ok:
        try:
            reported_obj = float(obj)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None:
            true_obj = float(max(station_times.values()))
            abs_diff = abs(reported_obj - true_obj)
            obj_tol = max(tol, tol * abs(true_obj))
            if abs_diff > obj_tol:
                record(9,
                       f"Objective consistency violated: reported "
                       f"objective_value={reported_obj} differs from "
                       f"recomputed max_k T_k={true_obj:.6g} "
                       f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                       reported_obj, true_obj, abs_diff)

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SUALBP-2 (Zohali et al. 2022)"
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

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"] is None:
        print("Solution is INDETERMINATE.")
    elif result["feasible"]:
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
