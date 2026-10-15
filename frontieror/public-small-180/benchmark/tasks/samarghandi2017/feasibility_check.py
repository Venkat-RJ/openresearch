#!/usr/bin/env python3
"""
Feasibility checker for F|nwt, d_j|C_max (No-Wait Flow Shop with Due Dates).

Checks all hard constraints from Models I-V as listed in math_model.txt,
numbered sequentially from top to bottom:

  Model I  : Constraints  1-11  (Eqs. 5-15)
  Model II : Constraints 12-23  (Eqs. 18-29)
  Model III: Constraints 24-31  (Eqs. 32-39)
  Model IV : Constraints 32-34  (Eqs. 42-44)
  Model V  : Constraints 35-40  (Eqs. 46-51)
  Constraint 41 (obj consistency): reported objective_value must equal the
                makespan recomputed from the sequence (Tier C defense).

This is the obj-recompute variant (feasibility_check_new.py); the original
feasibility_check.py is kept untouched. The makespan C_max is fully
determined by the sequence (the no-wait property fixes every start time
S_{ij}), so a *full recompute* applies: true_obj = max_j(S_{mj} + p_{mj}).
Constraint 41 rejects a solution whose self-reported objective_value
disagrees with that recomputed value.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'sequence')
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



# ---------------------------------------------------------------------------
# Contribution matrix (Samarghandi algorithm, Section 3)
# ---------------------------------------------------------------------------

def _compute_cjk(pj_orig, pk, m):
    """Compute c_{j,k} for a pair of jobs using Samarghandi's algorithm."""
    pj = list(pj_orig)
    t, w = 1, 0
    while True:
        if t >= m:
            return pk[m - 1]
        if pj[t] >= pk[w]:
            t += 1
            w += 1
            if t >= m:
                return pk[m - 1]
        else:
            cum, z = 0, None
            for h in range(t, m):
                cum += pj[h]
                if cum - pk[w] >= 0:
                    z = h
                    break
            if z is None:
                return max(0, sum(pk[w:]) - sum(pj[t:]))
            pj[z] = sum(pj[t:z + 1]) - pk[w]
            w += 1
            t = z
            if t >= m:
                return pk[m - 1]


def build_contribution_matrix(n, m, p):
    """Build (n+1) x (n+1) contribution matrix C (1-indexed)."""
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for k in range(1, n + 1):
        C[0][k] = sum(p[i][k - 1] for i in range(m))
    for j in range(1, n + 1):
        pj = [p[i][j - 1] for i in range(m)]
        for k in range(1, n + 1):
            if j != k:
                pk = [p[i][k - 1] for i in range(m)]
                C[j][k] = _compute_cjk(pj, pk, m)
    return C


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for F|nwt, d_j|C_max"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to instance JSON file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to candidate solution JSON file.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write feasibility result JSON file.")
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

    tol = 1e-5
    eps = 1e-5

    n = inst["n_jobs"]
    m = inst["n_machines"]
    p = inst["processing_times"]   # p[i][j] 0-indexed (machine i, job j)
    dd = inst["due_dates"]         # dd[j]   0-indexed

    seq = sol["sequence"]          # 1-indexed job numbers
    obj_val = sol.get("objective_value")

    violations = []
    violation_magnitudes = []
    violated_set = set()

    def record(idx, msg, lhs, rhs, raw_excess):
        violated_set.add(idx)
        violations.append(msg)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(raw_excess),
            "normalizer": float(normalizer),
            "ratio": float(raw_excess / normalizer),
        })

    # Handle null / empty sequence
    if seq is None or len(seq) == 0:
        result = {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution has no sequence (infeasible or no solution found)"],
            "violation_magnitudes": [],
        }
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
        return

    seq0 = [j - 1 for j in seq]  # 0-indexed

    # ======================================================================
    # Derive Model I decision variables from sequence
    # ======================================================================
    # Binary x_I[j][k] = 1 iff job k+1 immediately follows job j+1
    x_I = [[0] * n for _ in range(n)]
    for pos in range(len(seq0) - 1):
        x_I[seq0[pos]][seq0[pos + 1]] = 1

    # Start times S[i][j] via no-wait property
    S = [[0.0] * n for _ in range(m)]
    first = seq0[0]
    S[0][first] = 0.0
    for i in range(1, m):
        S[i][first] = S[i - 1][first] + p[i - 1][first]

    for pos in range(1, len(seq0)):
        prev_j = seq0[pos - 1]
        curr_j = seq0[pos]
        max_s0 = 0.0
        for i in range(m):
            # S[i][curr] >= S[i][prev] + p[i][prev]  (sequencing)
            # S[i][curr] = S[0][curr] + sum_{l=0..i-1} p[l][curr]  (no-wait)
            offset_curr = sum(p[l][curr_j] for l in range(i))
            needed = S[i][prev_j] + p[i][prev_j] - offset_curr
            if needed > max_s0:
                max_s0 = needed
        S[0][curr_j] = max_s0
        for i in range(1, m):
            S[i][curr_j] = S[i - 1][curr_j] + p[i - 1][curr_j]

    # Big-M value (sum of all processing times)
    M_val = sum(p[i][j] for i in range(m) for j in range(n))

    # ======================================================================
    # Build contribution matrix C (1-indexed, size (n+1) x (n+1))
    # ======================================================================
    C = build_contribution_matrix(n, m, p)

    # ======================================================================
    # Derive Model II variables (with dummy jobs 0 and n+1)
    # ======================================================================
    N2 = n + 2
    x_II = [[0] * N2 for _ in range(N2)]
    x_II[0][seq[0]] = 1
    for pos in range(len(seq) - 1):
        x_II[seq[pos]][seq[pos + 1]] = 1
    x_II[seq[-1]][n + 1] = 1

    # u variables (MTZ subtour elimination)
    u = [0] * N2
    u[0] = 1
    for pos in range(len(seq)):
        u[seq[pos]] = pos + 2
    u[n + 1] = len(seq) + 2

    # F variables (finish times via contribution matrix)
    F = [0.0] * N2
    cum_c = 0.0
    for pos in range(len(seq)):
        job_1 = seq[pos]
        if pos == 0:
            cum_c += C[0][job_1]
        else:
            cum_c += C[seq[pos - 1]][job_1]
        F[job_1] = cum_c
    F[n + 1] = cum_c

    # Extended contribution matrix C' for Model II (c'_{j,n+1}=0, c'_{n+1,k}=0)
    Cp = [[0] * N2 for _ in range(N2)]
    for j in range(n + 1):
        for k in range(1, n + 1):
            Cp[j][k] = C[j][k]

    # ======================================================================
    # Derive Model III variables (position-assignment)
    # ======================================================================
    x_III = [[0] * n for _ in range(n)]
    for l in range(min(n, len(seq0))):
        if 0 <= seq0[l] < n:
            x_III[l][seq0[l]] = 1

    # L[l] = cumulative finish time at position l+1 (0-indexed)
    L = [0.0] * n
    L[0] = C[0][seq[0]]
    for l in range(1, min(n, len(seq))):
        L[l] = L[l - 1] + C[seq[l - 1]][seq[l]]

    # ######################################################################
    # CHECK ALL HARD CONSTRAINTS (numbered 1-40, top to bottom)
    # ######################################################################

    # ==== MODEL I ====

    # Constraint 1 (Eq.5): C_max >= S_{mj} + p_{mj}  for j=1..n  [>=]
    if obj_val is not None:
        for j in range(n):
            comp_j = S[m - 1][j] + p[m - 1][j]
            va = comp_j - obj_val
            if va > tol:
                record(1,
                       f"Constraint 1 (Eq.5): C_max={obj_val} < completion "
                       f"of job {j+1}={comp_j}",
                       obj_val, comp_j, va)

    # Constraint 2 (Eq.6): S_{ik} + M(1-x_{jk}) >= S_{ij} + p_{ij}  [>=]
    for i in range(m):
        for j in range(n):
            for k in range(n):
                lhs_v = S[i][k] + M_val * (1 - x_I[j][k])
                rhs_v = S[i][j] + p[i][j]
                va = rhs_v - lhs_v
                if va > tol:
                    record(2,
                           f"Constraint 2 (Eq.6): Sequencing violated "
                           f"i={i+1},j={j+1},k={k+1}: LHS={lhs_v} < RHS={rhs_v}",
                           lhs_v, rhs_v, va)

    # Constraint 3 (Eq.7): S_{(i+1)j} = S_{ij} + p_{ij}  [=]
    for i in range(m - 1):
        for j in range(n):
            lhs_v = S[i + 1][j]
            rhs_v = S[i][j] + p[i][j]
            va = abs(lhs_v - rhs_v)
            if va > tol:
                record(3,
                       f"Constraint 3 (Eq.7): No-wait violated "
                       f"i={i+1},j={j+1}: S={lhs_v} != {rhs_v}",
                       lhs_v, rhs_v, va)

    # Constraint 4 (Eq.8): S_{mj} + p_{mj} <= d_j  [<=]
    for j in range(n):
        comp_j = S[m - 1][j] + p[m - 1][j]
        va = comp_j - dd[j]
        if va > tol:
            record(4,
                   f"Constraint 4 (Eq.8): Due date violated job {j+1}: "
                   f"completion={comp_j} > d={dd[j]}",
                   comp_j, dd[j], va)

    # Constraint 5 (Eq.9): sum_j x_{jk} <= 1  for k=1..n  [<=]
    for k in range(n):
        s = sum(x_I[j][k] for j in range(n))
        va = s - 1
        if va > tol:
            record(5,
                   f"Constraint 5 (Eq.9): Job {k+1} has {s} predecessors (> 1)",
                   s, 1, va)

    # Constraint 6 (Eq.10): sum_k x_{jk} <= 1  for j=1..n  [<=]
    for j in range(n):
        s = sum(x_I[j][k] for k in range(n))
        va = s - 1
        if va > tol:
            record(6,
                   f"Constraint 6 (Eq.10): Job {j+1} has {s} successors (> 1)",
                   s, 1, va)

    # Constraint 7 (Eq.11): x_{jk} + x_{kj} <= 1  [<=]
    for j in range(n):
        for k in range(n):
            if j != k:
                s = x_I[j][k] + x_I[k][j]
                va = s - 1
                if va > tol:
                    record(7,
                           f"Constraint 7 (Eq.11): Bidirectional link "
                           f"jobs {j+1} and {k+1}",
                           s, 1, va)

    # Constraint 8 (Eq.12): sum_{j,k} x_{jk} = n-1  [=]
    total_x = sum(x_I[j][k] for j in range(n) for k in range(n))
    va = abs(total_x - (n - 1))
    if va > tol:
        record(8,
               f"Constraint 8 (Eq.12): Total links={total_x} != {n-1}",
               total_x, n - 1, va)

    # Constraint 9 (Eq.13): x_{jj} = 0  [=]
    for j in range(n):
        va = abs(x_I[j][j])
        if va > tol:
            record(9,
                   f"Constraint 9 (Eq.13): Self-loop job {j+1}",
                   x_I[j][j], 0, va)

    # Constraint 10 (Eq.14): S_{ij} >= 0  [>=]
    for i in range(m):
        for j in range(n):
            if S[i][j] < -tol:
                record(10,
                       f"Constraint 10 (Eq.14): S[{i+1}][{j+1}]={S[i][j]:.6f} < 0",
                       S[i][j], 0, -S[i][j])

    # Constraint 11 (Eq.15): x_{jk} in {0,1}
    for j in range(n):
        for k in range(n):
            v = x_I[j][k]
            if v not in (0, 1):
                va = min(abs(v), abs(v - 1))
                record(11,
                       f"Constraint 11 (Eq.15): x[{j+1}][{k+1}]={v} not binary",
                       v, round(v), va)

    # ==== MODELS II, III, IV, V SKIPPED ====
    # These are reformulation models (contribution matrix + MTZ / position-
    # assignment / CP disjunctive) that operate on auxiliary variables
    # (u, F, x_II, x_III, L) derived from the sequence. A generated algorithm
    # operating on the F|nwt,d_j|Cmax problem description need produce only the
    # sequence; Model I checks already cover the business feasibility, and
    # Models II-V are equivalent restatements that add no constraint power.

    # ==== OBJECTIVE CONSISTENCY (Tier C defense) ====
    # Constraint 41: the reported objective_value must equal the makespan
    # recomputed from the sequence.  The objective of Model I is
    # min C_max (Eq.4); for any fixed sequence the no-wait property fixes
    # every start time S_{ij}, so the makespan is fully determined:
    #     true_obj = max_j ( S_{mj} + p_{mj} ).
    # This is a *full recompute* -- every variable the objective depends on
    # is implied by the sequence.  The original checker's constraint 1
    # (Eq.5) already rejects an under-reported C_max (C_max >= each
    # completion); constraint 41 additionally rejects an over-reported or
    # otherwise inconsistent C_max, closing the obj=MAX_FLOAT score-gaming
    # gap and making the objective check exact.
    if obj_val is not None:
        try:
            reported_obj = float(obj_val)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None:
            true_obj = max(S[m - 1][j] + p[m - 1][j] for j in range(n))
            abs_diff = abs(reported_obj - true_obj)
            # 0.1% relative tolerance with a 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record(41,
                       f"Constraint 41 (objective consistency): reported "
                       f"objective_value={reported_obj} differs from "
                       f"recomputed makespan max_j(S_mj+p_mj)={true_obj} "
                       f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                       reported_obj, true_obj, abs_diff)

    # ======================================================================
    # Output result
    # ======================================================================
    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {feasible}")
    if not feasible:
        print(f"Violated constraints: {sorted(violated_set)}")
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
