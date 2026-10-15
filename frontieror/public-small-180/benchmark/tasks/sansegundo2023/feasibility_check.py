#!/usr/bin/env python3
"""
Feasibility checker for the Maximum Clique Problem (MCP).

Paper: San Segundo et al. (2023), "CliSAT: A new exact algorithm for hard
       maximum clique problems", EJOR 307, 1008-1025.

Constraints (numbered top-to-bottom from the formulation):
  Constraint 1 (9b): x_u + x_v <= 1,  for all {u,v} in bar{E}(G)
  Constraint 2 (9c): x_u in {0, 1},   for all u in V(G)
  Constraint 3 (Tier-C obj recompute): reported objective_value must equal
    the recomputed value of sum_{u in V} x_u (size of the reported clique
    restricted to valid, distinct vertices in V).
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


TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    violations = []
    violation_magnitudes = []
    violated_set = set()

    # --- Extract instance data ---
    n = instance["num_vertices"]
    vertices = instance["vertices"]
    vertex_set = set(vertices)

    # Build adjacency sets
    if "adjacency_list" in instance:
        adj = {int(k): set(v) for k, v in instance["adjacency_list"].items()}
    else:
        adj = {v: set() for v in vertices}
        for u, w in instance["edges"]:
            adj[u].add(w)
            adj[w].add(u)
    for v in vertices:
        if v not in adj:
            adj[v] = set()

    # --- Extract solution clique ---
    if "clique" in solution:
        clique = solution["clique"]
    elif "clique_vertices" in solution:
        clique = solution["clique_vertices"]
    else:
        clique = []

    # Build x values: x_u = 1 if u in clique, else 0
    x = {v: 0 for v in vertices}
    for v in clique:
        x[v] = 1

    # --- Constraint 2 (9c): x_u in {0, 1}, for all u in V ---
    # Check that all clique vertices are valid vertices in the graph
    for v in clique:
        if v not in vertex_set:
            lhs = 1.0  # x_v = 1 but v not in V
            rhs = 0.0  # should not be selected
            violation_amount = 1.0
            violated_set.add(2)
            violations.append(
                f"Vertex {v} in clique is not a valid vertex in V (domain violation)"
            )
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # Check for duplicates in clique
    if len(clique) != len(set(clique)):
        from collections import Counter
        counts = Counter(clique)
        for v, cnt in counts.items():
            if cnt > 1:
                lhs = float(cnt)
                rhs = 1.0
                violation_amount = lhs - rhs
                violated_set.add(2)
                violations.append(
                    f"Vertex {v} appears {cnt} times in clique (must be binary)"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # --- Constraint 1 (9b): x_u + x_v <= 1, for all {u,v} in bar{E}(G) ---
    # For every pair of vertices in the clique, they must be adjacent in G.
    # If they are non-adjacent, then {u,v} in bar{E}(G) and x_u + x_v = 2 > 1.
    clique_valid = [v for v in clique if v in vertex_set]
    clique_sorted = sorted(set(clique_valid))
    for i in range(len(clique_sorted)):
        u = clique_sorted[i]
        for j in range(i + 1, len(clique_sorted)):
            v = clique_sorted[j]
            if v not in adj.get(u, set()):
                # Non-adjacent pair both selected: x_u + x_v = 2, limit is 1
                lhs = 2.0
                rhs = 1.0
                violation_amount = lhs - rhs  # 1.0
                violated_set.add(1)
                violations.append(
                    f"Non-adjacent vertices {u} and {v} both in clique "
                    f"(x_{u} + x_{v} = 2 > 1)"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # --- Constraint 3 (Tier-C obj recompute): reported objective_value must
    # match the recomputed sum_{u in V} x_u. The MCP objective (9a) is fully
    # determined by the clique indicator x, which is itself derived from the
    # solution's clique_vertices list. Full recompute applies. Tolerance is
    # 0.5 (the objective is integer-valued, so any mismatch of >=1 fires).
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(x[v] for v in vertices))
            abs_diff = abs(reported - true_obj)
            tol = 0.5
            if abs_diff > tol:
                lhs = reported
                rhs = true_obj
                violation_amount = abs_diff
                violated_set.add(3)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_v(x_v)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Maximum Clique Problem (MCP)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file")
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
        print("FEASIBLE: All constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
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
