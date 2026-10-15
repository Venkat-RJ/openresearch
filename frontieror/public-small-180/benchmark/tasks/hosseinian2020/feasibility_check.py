"""
Feasibility checker for the Maximum Edge Weight Clique (MEWC) problem.

Constraints (numbered top-to-bottom from the formulation):
  (1) y_{ij} <= x_i          for all {i,j} in E
  (2) y_{ij} <= x_j          for all {i,j} in E
  (3) x_i + x_j <= 1         for all {i,j} not in E
  (4) x_i in {0,1}           for all i in V
  (5) y_{ij} in {0,1}        for all {i,j} in E
  (6) Objective consistency  reported objective_value must equal the
      recomputed clique edge weight W(C) = sum_{{i,j} in E(C)} w_{ij}
      (Tier C defense against fabricated objective values).
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('clique', 'objective_value')
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



def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    vertices = instance["vertices"]
    edges = instance["edges"]
    vertex_set = set(vertices)

    # Build edge set (undirected)
    edge_set = set()
    edge_weights = {}
    for e in edges:
        i, j = e["i"], e["j"]
        edge_set.add((min(i, j), max(i, j)))
        edge_weights[(min(i, j), max(i, j))] = e["weight"]

    clique = solution["clique"]
    clique_set = set(clique)

    # Derive x and y from the solution
    # x_i = 1 if i in clique, else 0
    x = {v: (1 if v in clique_set else 0) for v in vertices}

    # y_{ij} = 1 if both i,j in clique AND {i,j} in E, else 0
    y = {}
    for key in edge_set:
        i, j = key
        y[key] = 1 if (i in clique_set and j in clique_set) else 0

    def record_violation(constraint_idx, message, lhs, rhs, violation_amount):
        violated_constraints_set.add(constraint_idx)
        violations.append(message)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    # Constraint (1): y_{ij} <= x_i for all {i,j} in E
    for (i, j) in edge_set:
        lhs = y[(i, j)]
        rhs = x[i]
        violation_amount = lhs - rhs  # for <= constraint
        if violation_amount > tol:
            record_violation(
                1,
                f"Constraint 1 violated: y_{{{i},{j}}} = {lhs} > x_{i} = {rhs}",
                lhs, rhs, violation_amount,
            )

    # Constraint (2): y_{ij} <= x_j for all {i,j} in E
    for (i, j) in edge_set:
        lhs = y[(i, j)]
        rhs = x[j]
        violation_amount = lhs - rhs
        if violation_amount > tol:
            record_violation(
                2,
                f"Constraint 2 violated: y_{{{i},{j}}} = {lhs} > x_{j} = {rhs}",
                lhs, rhs, violation_amount,
            )

    # Constraint (3): x_i + x_j <= 1 for all {i,j} NOT in E
    for idx_a in range(len(vertices)):
        for idx_b in range(idx_a + 1, len(vertices)):
            i = vertices[idx_a]
            j = vertices[idx_b]
            key = (min(i, j), max(i, j))
            if key not in edge_set:
                lhs = x[i] + x[j]
                rhs_val = 1
                violation_amount = lhs - rhs_val
                if violation_amount > tol:
                    record_violation(
                        3,
                        f"Constraint 3 violated: x_{i} + x_{j} = {lhs} > 1 "
                        f"(vertices {i} and {j} are both in clique but not adjacent)",
                        lhs, rhs_val, violation_amount,
                    )

    # Constraint (4): x_i in {0,1} for all i in V
    for v in vertices:
        val = x[v]
        if val not in (0, 1):
            violation_amount = min(abs(val - 0), abs(val - 1))
            if violation_amount > tol:
                record_violation(
                    4,
                    f"Constraint 4 violated: x_{v} = {val} is not binary",
                    val, round(val), violation_amount,
                )

    # Also check that clique vertices are valid vertices in V
    for v in clique:
        if v not in vertex_set:
            record_violation(
                4,
                f"Constraint 4 violated: vertex {v} in clique is not in V",
                1, 0, 1.0,
            )

    # Constraint (5): y_{ij} in {0,1} for all {i,j} in E
    for key in edge_set:
        val = y[key]
        if val not in (0, 1):
            violation_amount = min(abs(val - 0), abs(val - 1))
            if violation_amount > tol:
                i, j = key
                record_violation(
                    5,
                    f"Constraint 5 violated: y_{{{i},{j}}} = {val} is not binary",
                    val, round(val), violation_amount,
                )

    # Constraint (6): objective consistency (Tier C defense).
    # The MEWC objective W(C) = sum_{{i,j} in E(C)} w_{ij} is FULLY determined
    # by the clique in the solution: y_{ij} is derived above as 1 iff both
    # endpoints are in the clique and {i,j} in E. Hence we can recompute the
    # true objective exactly and reject solutions whose reported
    # objective_value disagrees with it.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(edge_weights[key] for key in edge_set
                                 if y[key] == 1))
            abs_diff = abs(reported - true_obj)
            # Edge weights are positive integers, so W(C) is integral; a
            # tolerance of 0.5 fires on any integer mismatch >= 1, with a
            # relative term to stay robust for very large objectives.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(
                    6,
                    f"Constraint 6 violated: reported objective_value="
                    f"{reported} differs from recomputed clique edge weight "
                    f"W(C)=sum_{{{{i,j}} in E(C)}} w_ij={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    reported, true_obj, abs_diff,
                )

    feasible = len(violated_constraints_set) == 0
    violated_constraints = sorted(violated_constraints_set)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MEWC problem"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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
        print("FEASIBLE")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
