#!/usr/bin/env python3
"""
Feasibility checker for the three paper-supported CSCP variants.

Checks all hard constraints from the mathematical formulation (equations 16-23)
in Pérez Martínez, Adulyasak, and Jans (2022).

Constraints are numbered 1-8 counting top to bottom from the formulation:
  Constraint 1 (eq 16): sum_{i in T} l_i * a_{ij} <= L * s_j,  forall j in P
  Constraint 2 (eq 17): z_j <= M * s_j,  forall j in P
  Constraint 3 (eq 18): q_{ij} = a_{ij} * z_j,  forall i in T, j in P
  Constraint 4 (eq 19): sum_j q_{ij} + x_i >= d_i,  forall i in T
  Constraint 5 (eq 20): sum_j z_j >= sum_i x_i
  Constraint 6 (eq 21): z_{j-1} >= z_j,  forall j in P, j > 1
  Constraint 7 (eq 22): s_j in {0,1}, a_{ij} in Z_+,  forall i,j
  Constraint 8 (eq 23): x_i >= 0, z_j in Z_+, q_{ij} >= 0,  forall i,j
  Constraint 9 (eq 15): objective consistency -- the reported objective_value
                        must equal the recomputed total time
                        C_s * sum_j s_j + C_r * sum_j z_j, plus the
                        knife-dependent setup term when applicable.

Constraint 9 is a Tier-C anti-gaming check: the eval pipeline otherwise trusts
the program's self-reported objective_value. Every variable the objective
(eq 15) depends on -- the pattern setup flags s_j and the repetition counts
z_j -- is present in the solution, so a full recompute is possible and exact.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('extra_pieces', 'objective_value', 'patterns', 'production_quantities')
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



def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CSCP variants"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
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

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # --- Parse instance data ---
    products = instance["products"]
    num_products = instance["num_products"]
    num_patterns = instance["num_patterns"]
    L = instance["machine_length"]
    M_val = instance["M"]
    problem_type = instance.get("problem_type", "open_ended_cscp")
    supported_types = {
        "open_ended_cscp",
        "closed_ended_cscp",
        "cscp_with_knife_dependent_setups",
    }
    if problem_type not in supported_types:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    is_closed_ended = problem_type == "closed_ended_cscp"
    is_knife_dependent = problem_type == "cscp_with_knife_dependent_setups"

    T = list(range(num_products))  # 0-indexed
    P = list(range(num_patterns))  # 0-indexed

    d = {i: products[i]["demand"] for i in T}
    l = {i: products[i]["length"] for i in T}
    product_ids = {i: products[i]["id"] for i in T}
    # Reverse map: product_id (int) -> 0-indexed
    id_to_idx = {products[i]["id"]: i for i in T}

    # --- Parse solution ---
    patterns = solution["patterns"]
    extra_pieces = solution["extra_pieces"]

    # s_j: whether pattern j is used
    s = {}
    # z_j: repetitions of pattern j
    z = {}
    # a[i][j]: pieces of product i in pattern j
    a = {}
    knife_counts = {}
    for j in P:
        pdata = patterns[str(j)]
        s[j] = 1 if pdata["used"] else 0
        z[j] = pdata["repetitions"]
        for i in T:
            pid_str = str(product_ids[i])
            a[(i, j)] = pdata["pieces"].get(pid_str, 0)
        if is_knife_dependent:
            if "knife_count" not in pdata:
                raise ValueError(
                    f"pattern {j} missing knife_count for knife-dependent instance"
                )
            knife_counts[j] = pdata["knife_count"]

    # x_i: extra pieces
    x = {}
    for i in T:
        pid_str = str(product_ids[i])
        x[i] = extra_pieces.get(pid_str, 0.0)

    # q_{ij} = a_{ij} * z_j (computed from solution, this IS the production quantity)
    q = {}
    for i in T:
        for j in P:
            q[(i, j)] = a[(i, j)] * z[j]

    # If the solution provides explicit production_quantities, use those for
    # checking constraint 3 (eq 18) mismatch
    q_reported = None
    if "production_quantities" in solution:
        q_reported = {}
        for i in T:
            for j in P:
                key = f"product_{product_ids[i]}_pattern_{j}"
                q_reported[(i, j)] = solution["production_quantities"].get(key, 0.0)

    # --- Check constraints ---
    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs, rhs, violation_amount):
        violated_constraints.add(constraint_idx)
        violations.append(msg)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer)
        })

    # ---- Constraint 1 (eq 16): sum_{i in T} l_i * a_{ij} <= L * s_j ----
    for j in P:
        lhs = sum(l[i] * a[(i, j)] for i in T)
        rhs = L * s[j]
        violation_amount = lhs - rhs
        if violation_amount > tol:
            record_violation(
                1,
                f"Constraint 1 (eq 16) violated for pattern {j}: "
                f"sum of piece lengths = {lhs}, but L*s_j = {rhs}",
                lhs, rhs, violation_amount
            )

    # ---- Constraint 2 (eq 17): z_j <= M * s_j ----
    for j in P:
        lhs = z[j]
        rhs = M_val * s[j]
        violation_amount = lhs - rhs
        if violation_amount > tol:
            record_violation(
                2,
                f"Constraint 2 (eq 17) violated for pattern {j}: "
                f"z_j = {lhs}, but M*s_j = {rhs}",
                lhs, rhs, violation_amount
            )

    # ---- Constraint 3 (eq 18): q_{ij} = a_{ij} * z_j ----
    # If solution reports explicit q values, check they match a_{ij}*z_j
    if q_reported is not None:
        for i in T:
            for j in P:
                lhs = q_reported[(i, j)]
                rhs = a[(i, j)] * z[j]
                violation_amount = abs(lhs - rhs)
                if violation_amount > tol:
                    record_violation(
                        3,
                        f"Constraint 3 (eq 18) violated for product {product_ids[i]}, "
                        f"pattern {j}: q_reported = {lhs}, a*z = {rhs}",
                        lhs, rhs, violation_amount
                    )

    # ---- Constraint 4 (eq 19): sum_j q_{ij} + x_i >= d_i ----
    for i in T:
        total_q = sum(q[(i, j)] for j in P)
        lhs = total_q + x[i]
        rhs = d[i]
        violation_amount = rhs - lhs  # >= constraint: RHS exceeds LHS
        if violation_amount > tol:
            record_violation(
                4,
                f"Constraint 4 (eq 19) violated for product {product_ids[i]}: "
                f"total production + extra = {lhs}, demand = {rhs}",
                lhs, rhs, violation_amount
            )

    # ---- Constraint 5 (eq 20): sum_j z_j >= sum_i x_i ----
    lhs_5 = sum(z[j] for j in P)
    rhs_5 = sum(x[i] for i in T)
    violation_amount_5 = rhs_5 - lhs_5  # >= constraint
    if violation_amount_5 > tol:
        record_violation(
            5,
            f"Constraint 5 (eq 20) violated: "
            f"sum(z_j) = {lhs_5}, sum(x_i) = {rhs_5}",
            lhs_5, rhs_5, violation_amount_5
        )

    # ---- Constraint 6 (eq 21): z_{j-1} >= z_j, for j > 0 ----
    for j in P:
        if j > 0:
            lhs = z[j - 1]
            rhs = z[j]
            violation_amount = rhs - lhs  # >= constraint: z_{j-1} >= z_j
            if violation_amount > tol:
                record_violation(
                    6,
                    f"Constraint 6 (eq 21) violated: "
                    f"z_{j-1} = {lhs} < z_{j} = {rhs}",
                    lhs, rhs, violation_amount
                )

    # ---- Constraint 7 (eq 22): s_j in {0,1}, a_{ij} in Z_+ ----
    for j in P:
        val = s[j]
        if val not in (0, 1):
            record_violation(
                7,
                f"Constraint 7 (eq 22) violated: s_{j} = {val} not in {{0,1}}",
                val, 0, 1.0
            )
        for i in T:
            val_a = a[(i, j)]
            if not isinstance(val_a, int) or val_a < 0:
                if isinstance(val_a, float) and abs(val_a - round(val_a)) < tol and val_a >= -tol:
                    pass  # close enough to non-negative integer
                else:
                    violation_amount = abs(val_a - round(val_a)) if val_a >= 0 else abs(val_a)
                    record_violation(
                        7,
                        f"Constraint 7 (eq 22) violated: a_{product_ids[i]},{j} = {val_a} "
                        f"not a non-negative integer",
                        val_a, round(max(val_a, 0)), violation_amount
                    )

    # ---- Constraint 8 (eq 23): x_i >= 0, z_j in Z_+, q_{ij} >= 0 ----
    for i in T:
        if x[i] < -tol:
            record_violation(
                8,
                f"Constraint 8 (eq 23) violated: x_{product_ids[i]} = {x[i]} < 0",
                x[i], 0.0, abs(x[i])
            )

        if is_closed_ended and abs(x[i]) > tol:
            record_violation(
                10,
                f"Closed-ended constraint violated: x_{product_ids[i]} = "
                f"{x[i]}, expected 0",
                x[i], 0.0, abs(x[i])
            )

    for j in P:
        val_z = z[j]
        if isinstance(val_z, float):
            if abs(val_z - round(val_z)) > tol or val_z < -tol:
                violation_amount = abs(val_z - round(val_z)) if val_z >= 0 else abs(val_z)
                record_violation(
                    8,
                    f"Constraint 8 (eq 23) violated: z_{j} = {val_z} "
                    f"not a non-negative integer",
                    val_z, round(max(val_z, 0)), violation_amount
                )
        elif isinstance(val_z, int):
            if val_z < 0:
                record_violation(
                    8,
                    f"Constraint 8 (eq 23) violated: z_{j} = {val_z} < 0",
                    val_z, 0, abs(val_z)
                )

    for i in T:
        for j in P:
            if q[(i, j)] < -tol:
                record_violation(
                    8,
                    f"Constraint 8 (eq 23) violated: q_{product_ids[i]},{j} = {q[(i,j)]} < 0",
                    q[(i, j)], 0.0, abs(q[(i, j)])
                )

    # ---- Constraint 11: knife_count_j equals pieces cut per pass ----
    if is_knife_dependent:
        for j in P:
            reported_knives = knife_counts[j]
            expected_knives = sum(a[(i, j)] for i in T)
            valid_numeric = (
                not isinstance(reported_knives, bool)
                and isinstance(reported_knives, (int, float))
                and math.isfinite(float(reported_knives))
            )
            if not valid_numeric:
                record_violation(
                    11,
                    f"Knife-count constraint violated for pattern {j}: "
                    f"reported {reported_knives!r}, expected a finite integer",
                    0.0, expected_knives, abs(expected_knives)
                )
            elif (
                isinstance(reported_knives, bool)
                or reported_knives < -tol
                or abs(reported_knives - round(reported_knives)) > tol
                or abs(reported_knives - expected_knives) > tol
            ):
                record_violation(
                    11,
                    f"Knife-count constraint violated for pattern {j}: "
                    f"reported {reported_knives}, expected {expected_knives}",
                    reported_knives, expected_knives,
                    abs(reported_knives - expected_knives)
                )

    # ---- Constraint 9 (eq 15): objective consistency ----
    # Tier-C anti-gaming check. The objective (eq 15) is
    #   C_s * sum_j s_j + C_r * sum_j z_j
    # and every term -- the setup flags s_j and repetition counts z_j -- is
    # present in the solution, so this is an exact full recompute. Reject the
    # solution when the reported objective_value disagrees.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            C_s = instance["cost_setup_pattern"]
            C_r = instance["cost_repetition"]
            true_obj = float(C_s * sum(s[j] for j in P)
                             + C_r * sum(z[j] for j in P))
            if is_knife_dependent:
                C_k = instance["cost_knife_setup"]
                true_obj += float(C_k * sum(knife_counts[j] for j in P))
            abs_diff = abs(reported - true_obj)
            # The objective is an integer count (C_s, C_r, s_j, z_j all
            # integral); a 0.5 absolute floor fires on any integer mismatch,
            # with a tiny relative term for large-magnitude safety.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(
                    9,
                    f"Constraint 9 (eq 15) violated: reported objective_value "
                    f"= {reported} differs from recomputed "
                    f"variant-specific objective = {true_obj} "
                    f"(|diff| = {abs_diff:.6g}, tol = {obj_tol:.6g})",
                    reported, true_obj, abs_diff
                )

    # --- Build output ---
    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
