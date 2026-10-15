#!/usr/bin/env python3
"""
Feasibility checker for the Maximum Clique Problem.

Mathematical formulation (Standard IP for Maximum Clique, Section 4 of math_model.txt):

  Decision Variables:
    x_v in {0, 1}, v in V   (1 if vertex v is in the clique, 0 otherwise)

  Objective:
    max  sum_{v in V} x_v

  Constraints (numbered top to bottom as they appear in the formulation):
    Constraint 1:  x_i + x_j <= 1,  for all {i, j} in E_bar  (non-edges of G)
    Constraint 2:  objective_value == sum_v x_v
                   (Tier C obj-consistency: recompute the objective from the
                   solution variables and reject if the reported value lies.)

This script checks whether a candidate solution (a set of vertices claimed to
form a clique) satisfies all hard constraints of the IP formulation.
"""

import argparse
import json

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



def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n"]
    edges = instance["edges"]

    # Build adjacency set
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Extract clique vertices from solution (handle both solution formats)
    if "clique" in solution:
        clique = solution["clique"]
    elif "clique_vertices" in solution:
        clique = solution["clique_vertices"]
    else:
        clique = []

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # Detect duplicate vertices: each x_v ∈ {0,1}, so a repeated vertex implies
    # x_v >= 2 which violates the binary domain.
    seen = set()
    for v in clique:
        if v in seen:
            violations.append(
                f"Vertex {v} listed more than once in clique_vertices "
                f"(implies x_{v} >= 2; binary domain violated).")
            violation_magnitudes.append({
                "constraint": "binary_domain",
                "lhs": float(clique.count(v)),
                "rhs": 1.0,
                "raw_excess": float(clique.count(v) - 1),
                "normalizer": 1.0,
                "ratio": float(clique.count(v) - 1),
            })
            violated_constraint_set.add("binary_domain")
        seen.add(v)

    # Build binary vector x from the clique
    x = [0] * n
    for v in clique:
        if 0 <= v < n:
            x[v] = 1

    # ------------------------------------------------------------------
    # Constraint 1: x_i + x_j <= 1  for all {i, j} in E_bar (non-edges)
    # ------------------------------------------------------------------
    # For every pair of vertices in the clique, they must share an edge.
    # Equivalently, for every non-edge {i,j}, x_i + x_j <= 1.
    # We only need to check pairs where both x_i=1 and x_j=1.
    clique_sorted = sorted(clique)
    for idx_a in range(len(clique_sorted)):
        for idx_b in range(idx_a + 1, len(clique_sorted)):
            i = clique_sorted[idx_a]
            j = clique_sorted[idx_b]
            if j not in adj[i]:
                # {i, j} is a non-edge and both are selected => violation
                lhs = x[i] + x[j]  # = 2
                rhs = 1.0
                violation_amount = lhs - rhs  # = 1.0
                if violation_amount > tol:
                    violated_constraint_set.add(1)
                    violations.append(
                        f"Constraint 1 violated: vertices {i} and {j} are both "
                        f"in the clique but are not adjacent (non-edge), "
                        f"x_{i} + x_{j} = {lhs} > {rhs}"
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

    # ------------------------------------------------------------------
    # Additional validity checks (not from the IP but ensuring solution
    # is well-formed w.r.t. the instance)
    # ------------------------------------------------------------------
    # Check that all clique vertex indices are valid (0 <= v < n)
    for v in clique:
        if v < 0 or v >= n:
            violated_constraint_set.add(1)
            violations.append(
                f"Vertex {v} in clique is out of range [0, {n-1}]"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(v),
                "rhs": float(n - 1),
                "raw_excess": float(abs(v - (n - 1))),
                "normalizer": float(max(abs(n - 1), eps)),
                "ratio": float(abs(v - (n - 1)) / max(abs(n - 1), eps)),
            })

    # ------------------------------------------------------------------
    # Constraint 2 (Tier C obj-consistency): reported objective_value must
    # equal sum_v x_v -- i.e. the count of distinct in-range clique vertices.
    # The obj is integer-valued, so a tolerance of 0.5 fires on any
    # integer mismatch by >= 1 while permitting harmless float noise.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(x))  # number of distinct in-range vertices
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5
            if abs_diff > obj_tol:
                violated_constraint_set.add(2)
                violations.append(
                    f"Constraint 2 violated: reported objective_value="
                    f"{reported} differs from recomputed sum_v(x_v)="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer),
                })

    feasible = len(violated_constraint_set) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraint_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Maximum Clique solutions."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
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

    if result["feasible"]:
        print("FEASIBLE: All constraints satisfied.")
    else:
        print(f"INFEASIBLE: {len(result['violations'])} violation(s) found.")
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
