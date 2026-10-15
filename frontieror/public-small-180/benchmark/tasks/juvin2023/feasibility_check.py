#!/usr/bin/env python3
"""
Feasibility checker for the preemptive Flexible Job-Shop Scheduling Problem (pFJSSP).

Based on: Juvin, Houssin, Lopez (2023) - "Logic-based Benders decomposition
for the preemptive Flexible Job-Shop Scheduling Problem"

Checks constraints (2)-(12) from the mathematical model (Section 2.1).
Constraint numbering (top to bottom in formulation):
  Constraint 1  = paper (2):  Assignment — each op to exactly one machine
  Constraint 2  = paper (3):  Duration — enough processing time allocated
  Constraint 3  = paper (4):  Precedence — consecutive ops ordering
  Constraint 4  = paper (5):  Resource (nonlinear form) — at most 1 op per machine per time
  Constraint 5  = paper (6):  Makespan definition
  Constraint 6  = paper (7):  Binary domain for x
  Constraint 7  = paper (8):  Binary domain for y
  Constraint 8  = paper (9):  Linearisation z >= x + y - 1
  Constraint 9  = paper (10): Linearisation z <= x
  Constraint 10 = paper (11): Linearisation z <= y
  Constraint 11 = paper (12): Linearised resource constraint
  Constraint 12 = objective consistency — reported objective_value must equal
                  the makespan implied by the schedule (Tier C defense against
                  LLM score-gaming exploits that fabricate objective_value).

NOTE: This is the obj-recompute ("_new") variant. It is identical to
feasibility_check.py except for the additional Constraint 12. The original
checker is left untouched.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'objective_value', 'schedule')
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


TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_instance(instance):
    """Parse instance JSON into convenient structures."""
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    jobs_data = instance["jobs"]

    eligible = []  # eligible[i][j] = list of machine ids
    proc = []      # proc[i][j][m] = processing time
    n_ops = []     # number of operations per job

    for job in jobs_data:
        ops = job["operations"]
        n_ops.append(len(ops))
        elig_i = []
        proc_i = []
        for op in ops:
            machines = op["eligible_machines"]
            pt = {int(k): v for k, v in op["processing_times"].items()}
            elig_i.append(machines)
            proc_i.append(pt)
        eligible.append(elig_i)
        proc.append(proc_i)

    # I_m: operations that can be processed on machine m
    I_m = {m: [] for m in range(num_machines)}
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            for m in eligible[i][j]:
                I_m[m].append((i, j))

    # Time horizon h (sum of max processing times)
    h = 0
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            h += max(proc[i][j].values())
    h = max(h, 1)

    return {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "n_ops": n_ops,
        "eligible": eligible,
        "proc": proc,
        "I_m": I_m,
        "h": h,
    }


def parse_solution(solution, data):
    """
    Parse solution JSON and reconstruct x, y, z variables.

    Returns:
        x: dict (i, j, m) -> {0, 1}
        y: dict (i, j, t) -> {0, 1}
        z: dict (i, j, m, t) -> {0, 1}  (derived as x * y)
        C_max: float
    """
    num_jobs = data["num_jobs"]
    n_ops = data["n_ops"]
    eligible = data["eligible"]
    h = data["h"]

    assignment = solution.get("assignment", {})
    schedule = solution.get("schedule", {})
    C_max = solution.get("objective_value", None)

    # Build x_{i,j,m}
    x = {}
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            key = f"({i},{j})"
            assigned_m = assignment.get(key, None)
            for m in eligible[i][j]:
                x[(i, j, m)] = 1 if (assigned_m is not None and m == assigned_m) else 0

    # Build y_{i,j,t}
    y = {}
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            key = f"({i},{j})"
            active_times = set(schedule.get(key, []))
            for t in range(1, h + 1):
                y[(i, j, t)] = 1 if t in active_times else 0

    # Build z_{i,j,m,t} = x_{i,j,m} * y_{i,j,t}
    z = {}
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            for m in eligible[i][j]:
                for t in range(1, h + 1):
                    z[(i, j, m, t)] = x.get((i, j, m), 0) * y.get((i, j, t), 0)

    return x, y, z, C_max


def record_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs, violation_amount):
    """Record a single constraint violation."""
    normalizer = max(abs(rhs), EPS)
    ratio = violation_amount / normalizer
    violations_list.append((constraint_idx, message))
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def check_feasibility(instance, solution):
    data = parse_instance(instance)
    x, y, z, C_max = parse_solution(solution, data)

    num_jobs = data["num_jobs"]
    num_machines = data["num_machines"]
    n_ops = data["n_ops"]
    eligible = data["eligible"]
    proc = data["proc"]
    I_m = data["I_m"]
    h = data["h"]

    violations_list = []   # list of (constraint_idx, message)
    magnitudes_list = []   # list of violation magnitude dicts

    # =========================================================================
    # Constraint 1 (paper 2): Assignment — each op assigned to exactly one machine
    # sum_{m in M_{i,j}} x_{i,j,m} = 1  for all i, O_{i,j}
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            lhs = sum(x.get((i, j, m), 0) for m in eligible[i][j])
            rhs = 1
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                record_violation(
                    violations_list, magnitudes_list, 1,
                    f"Operation O({i},{j}): assigned to {int(lhs)} machines instead of 1",
                    lhs, rhs, violation_amount,
                )

    # =========================================================================
    # Constraint 2 (paper 3): Duration
    # sum_{t=1}^{h} y_{i,j,t} >= sum_{m} x_{i,j,m} * p_{i,j,m}
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            lhs = sum(y.get((i, j, t), 0) for t in range(1, h + 1))
            rhs = sum(x.get((i, j, m), 0) * proc[i][j][m] for m in eligible[i][j])
            # >= constraint: violation if rhs > lhs
            violation_amount = max(0, rhs - lhs)
            if violation_amount > TOL:
                record_violation(
                    violations_list, magnitudes_list, 2,
                    f"Operation O({i},{j}): scheduled {int(lhs)} time units but needs {rhs}",
                    lhs, rhs, violation_amount,
                )

    # =========================================================================
    # Constraint 3 (paper 4): Precedence
    # sum_{t'=t}^{h} y_{i,j,t'} <= max_{m} p_{i,j,m} * (1 - y_{i,j+1,t})
    # for all i, O_{i,j} not last, t in H
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i] - 1):
            max_p = max(proc[i][j].values())
            for t in range(1, h + 1):
                lhs = sum(y.get((i, j, tp), 0) for tp in range(t, h + 1))
                y_next = y.get((i, j + 1, t), 0)
                rhs = max_p * (1 - y_next)
                # <= constraint: violation if lhs > rhs
                violation_amount = max(0, lhs - rhs)
                if violation_amount > TOL:
                    record_violation(
                        violations_list, magnitudes_list, 3,
                        f"Precedence violated: O({i},{j}) at t={t}, "
                        f"remaining y sum={int(lhs)} > {rhs} (max_p={max_p}, y_next={y_next})",
                        lhs, rhs, violation_amount,
                    )

    # =========================================================================
    # Constraint 4 (paper 5): Resource (nonlinear form)
    # sum_i sum_j x_{i,j,m} * y_{i,j,t} <= 1  for all m, t
    # =========================================================================
    for m in range(num_machines):
        for t in range(1, h + 1):
            lhs = sum(
                x.get((i, j, m), 0) * y.get((i, j, t), 0)
                for (i, j) in I_m.get(m, [])
            )
            rhs = 1
            violation_amount = max(0, lhs - rhs)
            if violation_amount > TOL:
                record_violation(
                    violations_list, magnitudes_list, 4,
                    f"Machine {m} at t={t}: {int(lhs)} operations active (max 1)",
                    lhs, rhs, violation_amount,
                )

    # =========================================================================
    # Constraint 5 (paper 6): Makespan definition
    # C_max >= (t+1) * sum_i y_{i,n_i,t}  for all t in H
    # (n_i is 1-indexed in paper; last op index = n_ops[i]-1 in 0-indexed)
    # =========================================================================
    if C_max is not None:
        for t in range(1, h + 1):
            sum_last = sum(y.get((i, n_ops[i] - 1, t), 0) for i in range(num_jobs))
            rhs_val = (t + 1) * sum_last
            lhs_val = C_max
            # >= constraint: violation if rhs > lhs
            violation_amount = max(0, rhs_val - lhs_val)
            if violation_amount > TOL:
                record_violation(
                    violations_list, magnitudes_list, 5,
                    f"Makespan C_max={C_max} < (t+1)*sum_last = {rhs_val} at t={t}",
                    lhs_val, rhs_val, violation_amount,
                )

    # =========================================================================
    # Constraint 6 (paper 7): Binary domain for x
    # x_{i,j,m} in {0, 1}
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            for m in eligible[i][j]:
                val = x.get((i, j, m), 0)
                if val not in (0, 1):
                    violation_amount = min(abs(val - 0), abs(val - 1))
                    rhs = round(val)
                    record_violation(
                        violations_list, magnitudes_list, 6,
                        f"x({i},{j},{m})={val} is not binary",
                        val, rhs, violation_amount,
                    )

    # =========================================================================
    # Constraint 7 (paper 8): Binary domain for y
    # y_{i,j,t} in {0, 1}
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            for t in range(1, h + 1):
                val = y.get((i, j, t), 0)
                if val not in (0, 1):
                    violation_amount = min(abs(val - 0), abs(val - 1))
                    rhs = round(val)
                    record_violation(
                        violations_list, magnitudes_list, 7,
                        f"y({i},{j},{t})={val} is not binary",
                        val, rhs, violation_amount,
                    )

    # =========================================================================
    # Constraint 8 (paper 9): z >= x + y - 1
    # z_{i,j,m,t} >= x_{i,j,m} + y_{i,j,t} - 1
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            for m in eligible[i][j]:
                for t in range(1, h + 1):
                    z_val = z.get((i, j, m, t), 0)
                    x_val = x.get((i, j, m), 0)
                    y_val = y.get((i, j, t), 0)
                    rhs = x_val + y_val - 1
                    lhs = z_val
                    # >= constraint: violation if rhs > lhs
                    violation_amount = max(0, rhs - lhs)
                    if violation_amount > TOL:
                        record_violation(
                            violations_list, magnitudes_list, 8,
                            f"z({i},{j},{m},{t})={z_val} < x+y-1={rhs}",
                            lhs, rhs, violation_amount,
                        )

    # =========================================================================
    # Constraint 9 (paper 10): z <= x
    # z_{i,j,m,t} <= x_{i,j,m}
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            for m in eligible[i][j]:
                for t in range(1, h + 1):
                    z_val = z.get((i, j, m, t), 0)
                    x_val = x.get((i, j, m), 0)
                    lhs = z_val
                    rhs = x_val
                    violation_amount = max(0, lhs - rhs)
                    if violation_amount > TOL:
                        record_violation(
                            violations_list, magnitudes_list, 9,
                            f"z({i},{j},{m},{t})={z_val} > x({i},{j},{m})={x_val}",
                            lhs, rhs, violation_amount,
                        )

    # =========================================================================
    # Constraint 10 (paper 11): z <= y
    # z_{i,j,m,t} <= y_{i,j,t}
    # =========================================================================
    for i in range(num_jobs):
        for j in range(n_ops[i]):
            for m in eligible[i][j]:
                for t in range(1, h + 1):
                    z_val = z.get((i, j, m, t), 0)
                    y_val = y.get((i, j, t), 0)
                    lhs = z_val
                    rhs = y_val
                    violation_amount = max(0, lhs - rhs)
                    if violation_amount > TOL:
                        record_violation(
                            violations_list, magnitudes_list, 10,
                            f"z({i},{j},{m},{t})={z_val} > y({i},{j},{t})={y_val}",
                            lhs, rhs, violation_amount,
                        )

    # =========================================================================
    # Constraint 11 (paper 12): Linearised resource
    # sum_i sum_j z_{i,j,m,t} <= 1  for all m, t
    # =========================================================================
    for m in range(num_machines):
        for t in range(1, h + 1):
            lhs = sum(
                z.get((i, j, m, t), 0)
                for (i, j) in I_m.get(m, [])
            )
            rhs = 1
            violation_amount = max(0, lhs - rhs)
            if violation_amount > TOL:
                record_violation(
                    violations_list, magnitudes_list, 11,
                    f"Machine {m} at t={t}: {int(lhs)} z-values active (max 1)",
                    lhs, rhs, violation_amount,
                )

    # =========================================================================
    # Constraint 12 (objective consistency): obj-recompute / Tier C defense
    # The objective is  min C_max  (paper eq. (1)).  C_max is a continuous
    # variable bounded only by the makespan-definition constraint (paper (6) /
    # Constraint 5):  C_max >= (t+1) * sum_i y_{i,n_i,t}  for all t in H.
    # Hence the *minimum feasible* makespan implied by the schedule (the y
    # variables) is exactly:
    #     true_obj = max_{t in 1..h} (t+1) * sum_i y_{i, n_i, t}
    # All variables this depends on (the y_{i,j,t} schedule) are fully present
    # in the solution, so this is an exact FULL RECOMPUTE.  We reject when the
    # reported objective_value disagrees with true_obj in either direction.
    #
    # (Constraint 5 above already rejects reported < true_obj; this check adds
    # the reciprocal direction — reported > true_obj — and pins the value to an
    # exact equality, closing the score-gaming / sys.float_info.max exploit.)
    # =========================================================================
    if C_max is not None:
        try:
            reported = float(C_max)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for t in range(1, h + 1):
                sum_last = sum(y.get((i, n_ops[i] - 1, t), 0) for i in range(num_jobs))
                cand = (t + 1) * sum_last
                if cand > true_obj:
                    true_obj = float(cand)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor; the makespan is
            # integer-valued, so any genuine mismatch (>= 1) fires while float
            # round-off (Gurobi ObjVal vs integer) stays under tolerance.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                record_violation(
                    violations_list, magnitudes_list, 12,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed makespan "
                    f"max_t (t+1)*sum_i y(i,last,t) = {true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})",
                    reported, true_obj, abs_diff,
                )

    # =========================================================================
    # Build output
    # =========================================================================
    violated_indices = sorted(set(idx for idx, _ in violations_list))
    # Aggregate messages per constraint index
    messages_by_idx = {}
    for idx, msg in violations_list:
        if idx not in messages_by_idx:
            messages_by_idx[idx] = []
        messages_by_idx[idx].append(msg)

    # Deduplicate and summarize violation messages per constraint
    violation_messages = []
    for idx in violated_indices:
        msgs = messages_by_idx[idx]
        if len(msgs) <= 3:
            for m in msgs:
                violation_messages.append(m)
        else:
            for m in msgs[:2]:
                violation_messages.append(m)
            violation_messages.append(f"... and {len(msgs) - 2} more violations of constraint {idx}")

    feasible = len(violated_indices) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": magnitudes_list if not feasible else [],
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for pFJSSP solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
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
