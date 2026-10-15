#!/usr/bin/env python3
"""
Feasibility checker for the Pm||sum w_j C_j set-covering formulation
from Kowalczyk & Leus (2018).

Hard constraints (from the mathematical model, in order):
  Constraint 1 (6b): sum_{s in F: j in s} lambda_s >= 1  for all j in J
      Every job must be assigned to at least one machine.
  Constraint 2 (6c): sum_{s in F} lambda_s <= m
      At most m machines (schedules) are used.
  Constraint 3 (6d): lambda_s in {0,1}  for all s in F
      Binary integrality of schedule selection variables.
  Constraint 4 (SWPT order): jobs in each selected schedule s must be
      sequenced in nonincreasing w_j/p_j order (Smith's rule).
  Constraint 5 (release/deadline): r_j + p_j <= C_j(s) <= d_bar_j for
      every job j in each selected schedule s (with r_j, d_bar_j
      tightened per Properties 5-6 of the paper).
  Constraint 6 (makespan bound): H_min <= makespan(s) <= H_max for
      every selected schedule s (Properties 3-4).
  Constraint 7 (objective consistency): the self-reported
      objective_value must equal the recomputed total weighted
      completion time  min sum_s c_s  with  c_s = sum_{j in s} w_j C_j(s)
      (Eqs. 5, 6a).  C_j(s) is the prefix sum of processing times in the
      listed order on the machine; w_j and p_j are taken from the
      INSTANCE (not the self-reported solution fields), so the check
      cannot be gamed by fabricating a small objective_value alongside
      fake per-job p/w/C entries.  This is a Tier-C defense against
      LLM score-gaming exploits.

NOTE: this is the obj-recompute ("_new") variant.  It is identical to
feasibility_check.py except for the added Constraint 7 block; the
original file is kept untouched until this variant is validated by the
differential test (feasibility_check_new_diff_test.json).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('machines', 'objective_value')
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



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_solution_machines(sol_data):
    """
    Parse solution into a unified list of machines.
    Each machine is a list of (job_id, p, w, C) tuples.
    Handles both efficient_solution and gurobi_solution formats.
    """
    machines = []
    for machine in sol_data["machines"]:
        if isinstance(machine, list):
            # efficient_algorithm format: list of dicts with original_job_id, p, w, C
            jobs = [(j["original_job_id"], j["p"], j["w"], j["C"])
                    for j in machine]
        elif isinstance(machine, dict):
            # gurobi_code format: dict with "jobs" key
            jobs = [(j["j_orig"], j["p"], j["w"], j["C_j"])
                    for j in machine["jobs"]]
        else:
            jobs = []
        machines.append(jobs)
    return machines


def compute_schedule_bounds(instance):
    """
    Replicate the per-schedule feasibility envelope used by the column
    generator (gurobi_code.preprocess): SWPT positions, tightened
    release dates r_j and deadlines d_bar_j (Properties 5-6), and the
    global makespan window [H_min, H_max] (Properties 3-4 of
    Belouadah & Potts 1994; Azizoglu & Kirca 1999).

    Returns:
        pos     : dict j_orig -> SWPT position index in [0, n)
        p, w    : processing times and weights in SWPT order
        r       : tightened release dates in SWPT order
        d_bar   : tightened deadlines in SWPT order
        H_min, H_max : global makespan bounds
    """
    m = instance["m"]
    jobs = instance["jobs"]
    n = len(jobs)

    sorted_jobs = sorted(jobs, key=lambda jj: (-jj["w"] / jj["p"], jj["p"]))
    pos = {jj["j"]: i for i, jj in enumerate(sorted_jobs)}
    p = [jj["p"] for jj in sorted_jobs]
    w = [jj["w"] for jj in sorted_jobs]

    total_p = sum(p)
    p_max = max(p)
    p_sorted_asc = sorted(p)

    H_max_real = total_p / m + (m - 1) / m * p_max
    H_max = math.ceil(H_max_real)

    sum_m1_largest = sum(p_sorted_asc[-(m - 1):]) if m > 1 else 0
    H_min_real = (total_p - sum_m1_largest) / m
    H_min = math.ceil(H_min_real)

    r = [0] * n
    d_bar = [H_max] * n

    prefix_p = [0] * (n + 1)
    for i in range(n):
        prefix_p[i + 1] = prefix_p[i] + p[i]
    suffix_p = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_p[i] = suffix_p[i + 1] + p[i]

    threshold = (total_p + sum_m1_largest) / m

    for jj in range(n):
        P_j = set()
        for k in range(n):
            if k == jj:
                continue
            if (w[k] > w[jj] and p[k] <= p[jj]) or (w[k] >= w[jj] and p[k] < p[jj]):
                P_j.add(k)
        rhs_p2 = threshold - suffix_p[jj]
        for k in range(jj):
            if prefix_p[k] <= rhs_p2:
                P_j.add(k)
        if len(P_j) >= m - 1:
            p_in_P = sorted([p[k] for k in P_j])
            count = len(P_j) - m + 1
            rho_j = sum(p_in_P[:count])
            r[jj] = math.ceil(rho_j / m)

    for jj in range(n):
        E_j = set()
        for k in range(n):
            if k == jj:
                continue
            if (w[k] < w[jj] and p[k] >= p[jj]) or (w[k] <= w[jj] and p[k] > p[jj]):
                E_j.add(k)
        for k in range(jj + 1, n):
            if prefix_p[k] <= threshold - suffix_p[k]:
                E_j.add(k)
        sum_E_p = sum(p[k] for k in E_j)
        delta_j = H_max - math.ceil((sum_E_p + p[jj]) / m)
        if delta_j + p[jj] < d_bar[jj]:
            d_bar[jj] = delta_j + p[jj]

    return pos, p, w, r, d_bar, H_min, H_max


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n"]
    m = instance["m"]
    instance_jobs = {j["j"]: j for j in instance["jobs"]}

    machines = parse_solution_machines(solution)

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    # ------------------------------------------------------------------
    # Constraint 1 (6b): sum_{s: j in s} lambda_s >= 1  for all j in J
    #   Every job must be covered by at least one selected schedule.
    # ------------------------------------------------------------------
    job_coverage = {jid: 0 for jid in instance_jobs}
    for machine_jobs in machines:
        for (job_id, p, w, C) in machine_jobs:
            if job_id in job_coverage:
                job_coverage[job_id] += 1

    uncovered = []
    for jid in sorted(job_coverage):
        lhs = float(job_coverage[jid])
        rhs = 1.0
        # >= constraint: violation_amount = max(0, rhs - lhs)
        violation_amount = max(0.0, rhs - lhs)
        if violation_amount > tol:
            uncovered.append(jid)
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    if uncovered:
        violated_constraints_set.add(1)
        violations.append(
            f"Constraint 1 (covering): job(s) {uncovered} not assigned to any machine"
        )

    # ------------------------------------------------------------------
    # Constraint 2 (6c): sum_{s in F} lambda_s <= m
    #   At most m schedules (machines) are used.
    # ------------------------------------------------------------------
    num_used = float(len(machines))
    rhs2 = float(m)
    # <= constraint: violation_amount = max(0, lhs - rhs)
    violation_amount_2 = max(0.0, num_used - rhs2)
    if violation_amount_2 > tol:
        violated_constraints_set.add(2)
        violations.append(
            f"Constraint 2 (capacity): {int(num_used)} machines used, "
            f"limit is {m}"
        )
        normalizer2 = max(abs(rhs2), eps)
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": num_used,
            "rhs": rhs2,
            "raw_excess": violation_amount_2,
            "normalizer": normalizer2,
            "ratio": violation_amount_2 / normalizer2
        })

    # ------------------------------------------------------------------
    # Constraint 3 (6d): lambda_s in {0,1}  for all s in F
    #   Binary integrality. The solution format encodes selected schedules
    #   explicitly (one per machine entry), so each lambda is implicitly 1.
    #   We verify that no fractional or non-binary values sneak in by
    #   checking that each schedule selection is integral (always satisfied
    #   by the discrete machine list, but we confirm no anomalies).
    # ------------------------------------------------------------------
    # In the solution representation each listed machine corresponds to
    # lambda_s = 1 and all others are 0 -- inherently binary.
    # We still confirm by treating each listed machine as lambda=1.0 and
    # checking |lambda - round(lambda)| <= tol.
    lambda_values = [1.0] * len(machines)
    for idx, lv in enumerate(lambda_values):
        nearest_int = round(lv)
        violation_amount_3 = abs(lv - nearest_int)
        if violation_amount_3 > tol:
            violated_constraints_set.add(3)
            violations.append(
                f"Constraint 3 (integrality): schedule {idx} has "
                f"lambda={lv}, not binary"
            )
            rhs3 = nearest_int
            normalizer3 = max(abs(rhs3), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lv,
                "rhs": float(rhs3),
                "raw_excess": violation_amount_3,
                "normalizer": normalizer3,
                "ratio": violation_amount_3 / normalizer3
            })

    # ------------------------------------------------------------------
    # Constraints 4, 5, 6 SKIPPED (Rule 4 cleanup):
    #   - Constraint 4 (SWPT order) is a proven-optimal ordering for
    #     single-machine weighted completion time; enforcing it as a hard
    #     constraint rejects valid but sub-optimal orderings that an LLM
    #     algorithm (e.g., LPT, random) might produce. The global objective
    #     check below catches sub-optimality.
    #   - Constraint 5 (tightened r_j, d_bar_j from Properties 5-6) is a
    #     dominance-based pruning bound, not an original-problem constraint.
    #   - Constraint 6 (H_min <= makespan <= H_max from Properties 3-4) is
    #     also a dominance bound, not a business rule.
    # Replaced by Constraint 4 below: completion-time consistency within a
    # schedule (C_j accumulates processing times in listed order).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 4 (completion-time consistency on listed order):
    # For each machine, if jobs are listed in processing order, then:
    #   C of first job = p of first job
    #   C of k-th job  = C of (k-1)-th job + p of k-th job
    # The listed order defines the schedule; permuting does NOT affect
    # feasibility, only the objective value.
    # ------------------------------------------------------------------
    for s_idx, machine_jobs in enumerate(machines):
        if not machine_jobs:
            continue
        expected_C = 0
        for idx, (job_id, p_j, _, C_j) in enumerate(machine_jobs):
            expected_C += p_j
            if abs(C_j - expected_C) > tol:
                violated_constraints_set.add(4)
                violations.append(
                    f"Constraint 4 (completion-time consistency): schedule "
                    f"{s_idx} position {idx}, job {job_id} has C_j={C_j} "
                    f"but expected {expected_C} = sum of p's up to position"
                )
                excess = float(abs(C_j - expected_C))
                normalizer = max(abs(float(expected_C)), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(C_j),
                    "rhs": float(expected_C),
                    "raw_excess": excess,
                    "normalizer": normalizer,
                    "ratio": excess / normalizer
                })

    # ------------------------------------------------------------------
    # Constraint 7 (objective consistency -- Tier-C anti-gaming check):
    # The reported objective_value must equal the recomputed total
    # weighted completion time
    #     obj = sum_{s} c_s ,   c_s = sum_{j in s} w_j * C_j(s)   (Eqs. 5, 6a)
    # where C_j(s) is the prefix sum of processing times along the listed
    # order on the machine that carries job j.
    #
    # Both p_j and w_j are taken from the INSTANCE (indexed by the job's
    # one-based original id) -- never from the self-reported solution
    # fields -- so an LLM candidate cannot pass this check by fabricating
    # a small objective_value together with matching fake per-job p/w/C
    # entries.  If a listed job id is absent from the instance the
    # solution-reported p/w are used as a last-resort fallback (such a
    # solution is already flagged elsewhere via the covering constraint).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    try:
        reported = float(reported_obj)
    except (TypeError, ValueError):
        reported = None
    if reported is not None:
        true_obj = 0.0
        for machine_jobs in machines:
            elapsed = 0
            for (job_id, sol_p, sol_w, sol_C) in machine_jobs:
                inst_job = instance_jobs.get(job_id)
                p_j = inst_job["p"] if inst_job is not None else sol_p
                w_j = inst_job["w"] if inst_job is not None else sol_w
                elapsed += p_j
                true_obj += w_j * elapsed
        abs_diff = abs(reported - true_obj)
        # The objective is an exact integer sum of w_j*C_j; a 0.5 absolute
        # floor makes any integer-level fabrication fire, while the 1e-6
        # relative term only absorbs floating-point noise at large scale.
        obj_tol = max(0.5, 1e-6 * abs(true_obj))
        if abs_diff > obj_tol:
            violated_constraints_set.add(7)
            violations.append(
                f"Constraint 7 (objective consistency): reported "
                f"objective_value={reported} differs from recomputed total "
                f"weighted completion time sum_s sum_j w_j*C_j(s)={true_obj} "
                f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
            )
            normalizer7 = max(abs(float(true_obj)), eps)
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": float(reported),
                "rhs": float(true_obj),
                "raw_excess": float(abs_diff),
                "normalizer": normalizer7,
                "ratio": float(abs_diff) / normalizer7
            })

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    feasible = len(violated_constraints_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else []
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Pm||sum w_j C_j "
                    "(Kowalczyk & Leus 2018)"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", required=True,
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

    if result["feasible"]:
        print(f"FEASIBLE — no constraint violations.")
    else:
        print(f"INFEASIBLE — violated constraints: {result['violated_constraints']}")
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
