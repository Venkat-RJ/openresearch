"""
Feasibility checker for the Maximum Clique Problem (ILP Formulation 3)
from: "A Branch and Bound Algorithm for the Maximum Clique Problem"
      Pardalos & Rodgers, Computers Ops Res. Vol. 19, No. 5, pp. 363-375, 1992

Constraints (numbered top-to-bottom from the formulation):
  Constraint (1): x_i + x_j <= 1  for all (v_i, v_j) in complement edge set bar{E}
  Constraint (2): x_i in {0, 1}   for all i = 1, ..., n
  Constraint (3): Binary domain check (auto-generated; no vars registered)
  Constraint (4): Integer domain check (auto-generated; no vars registered)
  Constraint (5): Objective consistency -- the reported objective_value must
                  equal the recomputed clique size sum_i(x_i). This is a
                  Tier C defense: the maximum clique objective is fully
                  determined by the clique_vertices in the solution, so a
                  full recompute is possible. Rejects fabricated objective
                  values (e.g. obj=0 or obj=sys.float_info.max) that pass
                  the constraint checks but lie about solution quality.
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('clique_vertices', 'objective_value')
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



def check_feasibility(instance_path, solution_path, result_path):
    tol = 1e-5
    eps = 1e-5

    with open(instance_path, 'r') as f:
        instance = json.load(f)
    with open(solution_path, 'r') as f:
        solution = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
        if _frontieror_contract_result is not None:
            with open(result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return _frontieror_contract_result

    n = instance["num_vertices"]
    edges = instance["edges"]

    # Build edge set of original graph G
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    # Extract solution: build x vector from clique_vertices
    clique_vertices = solution["clique_vertices"]
    x = [0] * n
    for v in clique_vertices:
        if 0 <= v < n:
            x[v] = 1

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ---- Constraint (1): x_i + x_j <= 1 for all (v_i, v_j) in bar{E} ----
    # bar{E} = pairs (i, j) with i < j that are NOT edges in G
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                # This is a complement edge
                lhs = x[i] + x[j]
                rhs = 1.0
                violation_amount = lhs - rhs
                if violation_amount > tol:
                    violated_constraints.add(1)
                    violations.append(
                        f"Constraint (1) violated: x_{i} + x_{j} = {lhs} > {rhs} "
                        f"(vertices {i} and {j} are both in clique but not adjacent in G)"
                    )
                    normalizer = max(abs(rhs), eps)
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": float(lhs),
                        "rhs": float(rhs),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(violation_amount / normalizer),
                    })

    # ---- Constraint (2): x_i in {0, 1} for all i ----
    # Check that all clique_vertices are valid vertex indices
    for v in clique_vertices:
        if not isinstance(v, int) or v < 0 or v >= n:
            lhs = float(v) if isinstance(v, (int, float)) else 0.0
            rhs = 0.0  # nearest valid bound
            violation_amount = 1.0  # binary domain violation
            violated_constraints.add(2)
            violations.append(
                f"Constraint (2) violated: vertex index {v} is not a valid vertex in [0, {n-1}]"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # Check for duplicate vertices in the clique (would mean x_i > 1)
    seen = set()
    for v in clique_vertices:
        if v in seen:
            lhs = float(clique_vertices.count(v))
            rhs = 1.0
            violation_amount = lhs - rhs
            if violation_amount > tol:
                violated_constraints.add(2)
                violations.append(
                    f"Constraint (2) violated: vertex {v} appears {int(lhs)} times "
                    f"in clique (x_{v} must be 0 or 1)"
                )
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })
        seen.add(v)

    _domain_check_vars_binary = []
    _domain_check_vars_integer = []

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 3: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > tol or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > tol:
                        violated_constraints.add(3)
                        violations.append(
                            f"Constraint 3 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        violation_magnitudes.append({
                            "constraint": 3,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    # Constraint 4: Integer domain — variables must be integral
    for var_name, var_dict in _domain_check_vars_integer:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                frac = abs(v - round(v))
                if frac > tol:
                    violated_constraints.add(4)
                    violations.append(
                        f"Constraint 4 (integer domain): {var_name}[{key}] = {v} is not integer")
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": v,
                        "rhs": round(v),
                        "raw_excess": float(frac),
                        "normalizer": max(abs(round(v)), eps),
                        "ratio": float(frac / max(abs(round(v)), eps)),
                    })

    # =====================================================================
    # Constraint 5: Objective consistency (Tier C obj-recompute defense)
    # =====================================================================
    # The maximum clique objective is f(x) = sum_i(x_i) = clique size (the
    # solution_schema documents objective_value as "Number of vertices in the
    # selected clique"). Every variable that determines the objective is in
    # the solution (clique_vertices -> x), so a FULL recompute is possible:
    # true_obj = sum(x) where x is the de-duplicated, in-range 0/1 vector
    # built above -- identical to the math model's f(x). Since the objective
    # is an integer count, the tolerance is 0.5 so any integer mismatch fires.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(x))
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5  # integer-valued objective: any mismatch >= 1 fires
            if abs_diff > obj_tol:
                violated_constraints.add(5)
                violations.append(
                    f"Constraint (5) violated: reported objective_value={reported} "
                    f"differs from recomputed clique size sum_i(x_i)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer),
                })

    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {feasible}")
    if not feasible:
        for v in violations:
            print(f"  - {v}")
    print(f"Result written to: {result_path}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Feasibility checker for Maximum Clique Problem (Formulation 3)")
        parser.add_argument("--instance_path", type=str, required=True,
                            help="Path to JSON instance file")
        parser.add_argument("--solution_path", type=str, required=True,
                            help="Path to JSON solution file")
        parser.add_argument("--result_path", type=str, required=True,
                            help="Path to write JSON feasibility result")
        args = parser.parse_args()

        check_feasibility(args.instance_path, args.solution_path, args.result_path)
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
