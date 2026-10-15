"""
Feasibility checker for the Generalized Assignment Problem (GAP).

Reference: Savelsbergh (1997), "A Branch-and-Price Algorithm for the
Generalized Assignment Problem", Operations Research 45(6):831-841.

Checks the Standard Integer Programming Formulation (Formulation 1):

  max  sum_{i,j} p_{ij} x_{ij}

  subject to
    (1) sum_{i=1..m} x_{ij} = 1,          for j = 1,...,n
    (2) sum_{j=1..n} w_{ij} x_{ij} <= c_i, for i = 1,...,m
    (3) x_{ij} in {0, 1},                  for all i, j
    (4) reported objective_value must equal recomputed
        sum_{i,j} p_{ij} x_{ij} within tolerance (anti-gaming check).
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'objective_value')
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


def check_feasibility(instance, solution):
    """
    Check all hard constraints of the GAP standard IP formulation.

    Returns a dict with feasibility results.
    """
    m = instance["num_agents"]
    n = instance["num_jobs"]
    profits = instance["profits"]        # profits[i][j]
    weights = instance["weights"]      # weights[i][j]
    capacities = instance["capacities"]  # capacities[i]

    assignment = solution["assignment"]  # dict: str(j) -> agent i

    violations = []
    violation_magnitudes = []

    # ---------------------------------------------------------------
    # Build the x_{ij} matrix from the assignment dict
    # ---------------------------------------------------------------
    x = [[0] * n for _ in range(m)]
    for j_str, i_val in assignment.items():
        j = int(j_str)
        i = int(i_val)
        if 0 <= i < m and 0 <= j < n:
            x[i][j] += 1  # use += to detect duplicates

    # ---------------------------------------------------------------
    # Constraint (1): sum_{i=1..m} x_{ij} = 1, for each j = 1,...,n
    #   Equality constraint: violation_amount = |LHS - RHS|
    # ---------------------------------------------------------------
    constraint_1_violated = False
    for j in range(n):
        lhs = sum(x[i][j] for i in range(m))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            constraint_1_violated = True
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            if lhs == 0:
                violations.append(
                    f"Constraint (1): Job {j} is not assigned to any agent "
                    f"(sum_i x_{{i,{j}}} = {lhs}, expected 1)"
                )
            elif lhs > 1:
                violations.append(
                    f"Constraint (1): Job {j} is assigned to {lhs} agents "
                    f"(sum_i x_{{i,{j}}} = {lhs}, expected 1)"
                )
            else:
                violations.append(
                    f"Constraint (1): Job {j} assignment sum is {lhs} "
                    f"(expected 1)"
                )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ---------------------------------------------------------------
    # Constraint (2): sum_{j=1..n} w_{ij} x_{ij} <= c_i, for each i
    #   <= constraint: violation_amount = max(LHS - RHS, 0)
    # ---------------------------------------------------------------
    constraint_2_violated = False
    for i in range(m):
        lhs = sum(weights[i][j] * x[i][j] for j in range(n))
        rhs = float(capacities[i])
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > TOL:
            constraint_2_violated = True
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint (2): Agent {i} capacity exceeded "
                f"(load = {lhs}, capacity = {rhs}, excess = {violation_amount})"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ---------------------------------------------------------------
    # Constraint (3): x_{ij} in {0, 1} for all i, j
    #   Each x_{ij} must be exactly 0 or 1.
    #   Treat as equality to nearest integer: violation = |x_{ij} - round(x_{ij})|
    # ---------------------------------------------------------------
    constraint_3_violated = False
    for i in range(m):
        for j in range(n):
            val = x[i][j]
            if val not in (0, 1):
                nearest_bin = round(val)
                violation_amount = abs(val - nearest_bin)
                if violation_amount > TOL:
                    constraint_3_violated = True
                    rhs = float(nearest_bin)
                    normalizer = max(abs(rhs), EPS)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint (3): x_{{{i},{j}}} = {val} is not binary "
                        f"(expected 0 or 1)"
                    )
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": float(val),
                        "rhs": rhs,
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(ratio),
                    })
                elif val > 1:
                    # x_{ij} > 1 means job assigned multiple times to same agent
                    constraint_3_violated = True
                    rhs = 1.0
                    violation_amount = abs(val - rhs)
                    normalizer = max(abs(rhs), EPS)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint (3): x_{{{i},{j}}} = {val} is not binary "
                        f"(expected 0 or 1)"
                    )
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": float(val),
                        "rhs": rhs,
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(ratio),
                    })

    # Also check that assignment keys reference valid jobs in range [0, n)
    for j_str, i_val in assignment.items():
        j = int(j_str)
        i = int(i_val)
        if j < 0 or j >= n:
            violations.append(
                f"Constraint (1): Job index {j} is out of range [0, {n-1}]"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })
        if i < 0 or i >= m:
            violations.append(
                f"Constraint (3): Agent index {i} is out of range [0, {m-1}] "
                f"for job {j}"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(i),
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": EPS,
                "ratio": 1.0 / EPS,
            })

    # ---------------------------------------------------------------
    # Constraint (4): objective consistency (anti-gaming).
    #   Reported objective_value must equal recomputed
    #     sum_{i,j} p_{ij} x_{ij}
    #   within tolerance. Catches LLM-evolved exploits that fabricate
    #   objective_value while keeping assignment feasible.
    # ---------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for i in range(m):
                for j in range(n):
                    if x[i][j]:
                        true_obj += float(profits[i][j]) * x[i][j]
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor. Profits are
            # integers so any honest reporter will match exactly; the slack
            # only forgives float-roundtrip noise.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                normalizer = max(abs(true_obj), EPS)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint (4): Objective consistency violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed sum_{{i,j}} p_{{ij}} x_{{ij}}={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.3g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })

    # ---------------------------------------------------------------
    # Build result
    # ---------------------------------------------------------------
    violated_constraints = sorted(set(
        vm["constraint"] for vm in violation_magnitudes
    ))
    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the GAP (Savelsbergh 1997)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON.")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    with open(args.solution_path, 'r') as f:
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE — no constraint violations detected.")
    else:
        print(f"INFEASIBLE — {len(result['violations'])} violation(s) found:")
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
