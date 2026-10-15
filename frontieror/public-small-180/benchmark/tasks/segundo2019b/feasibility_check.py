"""
Feasibility Checker for Maximum Weighted Clique Problem (MWCP)
==============================================================
Checks whether a candidate solution satisfies all hard constraints from
Segundo, Furini, Artieda (2019).

Hard constraints (numbered top-to-bottom from the formulation section):
  Constraint 1 (Eq. 3 / Eq. 20): Independent-set / edge constraints.
      For every pair (u, v) in the complement edge set (i.e., u and v are
      NOT adjacent in G), at most one can be selected:  x_u + x_v <= 1.
      Equivalently, the selected vertices must form a clique in G.
  Constraint 2 (Eq. 4): Binary domain.
      x_v in {0, 1} for all v in V.  Each selected vertex must be a valid
      vertex of the graph, and no vertex may appear more than once.
  Constraint 3 (Eq. 2, obj consistency): Objective recomputation.
      The reported objective_value must equal sum_{v in clique} w(v)
      within a small tolerance.  Guards against LLM score-gaming where a
      candidate returns a fabricated objective_value while the clique
      itself satisfies constraints 1-2.
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



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    graph = instance["graph"]
    num_vertices = graph["num_vertices"]
    edges = graph["edges"]
    weights = instance["vertex_weights"]

    # Build adjacency set
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    # Extract clique from solution (handle both formats)
    if "clique" in solution:
        clique = list(solution["clique"])
    elif "clique_vertices" in solution:
        clique = list(solution["clique_vertices"])
    else:
        raise ValueError("Solution has neither 'clique' nor 'clique_vertices' field.")

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 2 (Eq. 4): Binary domain — x_v in {0, 1}, v in V
    # Each selected vertex must be a valid vertex index and appear at most once.
    # ------------------------------------------------------------------

    # Check for invalid vertex indices
    for v in clique:
        if not isinstance(v, int) or v < 0 or v >= num_vertices:
            violated_constraints.add(2)
            # LHS = x_v (which is 1 for a selected vertex), RHS = 0 (vertex doesn't exist)
            # violation_amount = 1 (selecting a non-existent vertex)
            lhs = 1.0
            rhs = 0.0
            raw_excess = abs(lhs - rhs)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Vertex {v} is not a valid vertex index (must be in 0..{num_vertices - 1})."
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": raw_excess / normalizer,
            })

    # Check for duplicate vertices
    seen = {}
    for v in clique:
        seen[v] = seen.get(v, 0) + 1
    for v, count in seen.items():
        if count > 1:
            violated_constraints.add(2)
            # LHS = count (number of times selected), RHS = 1 (binary)
            lhs = float(count)
            rhs = 1.0
            raw_excess = abs(lhs - rhs)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Vertex {v} appears {count} times in the clique (must appear at most once)."
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": raw_excess / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint 1 (Eq. 3 / Eq. 20): Clique / independent-set constraints
    # For every pair (u, v) NOT in E:  x_u + x_v <= 1
    # If both u and v are selected, LHS = 2 > RHS = 1  =>  violated.
    # ------------------------------------------------------------------
    unique_clique = list(set(clique))
    for i in range(len(unique_clique)):
        for j in range(i + 1, len(unique_clique)):
            u, v = unique_clique[i], unique_clique[j]
            pair = (min(u, v), max(u, v))
            if pair not in edge_set:
                # This pair is in the complement edge set — constraint violated
                violated_constraints.add(1)
                lhs = 2.0  # x_u + x_v = 1 + 1
                rhs = 1.0  # <= 1
                raw_excess = lhs - rhs  # 1.0
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Vertices {u} and {v} are both selected but not adjacent in G "
                    f"(complement edge constraint x_{u} + x_{v} <= 1 violated)."
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint 3 (Eq. 2): Objective consistency (full recompute)
    # The MWCP objective is sum_{v in clique} w(v).  Every variable that
    # determines the objective (the clique itself) is in the solution, so
    # we can recompute exactly and compare to the reported value.
    # Only valid, distinct vertex indices contribute (mirrors what the
    # Gurobi-style ILP would compute for x_v=1).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for v in set(clique):
                if isinstance(v, int) and 0 <= v < num_vertices:
                    true_obj += float(weights[v])
            abs_diff = abs(reported - true_obj)
            # Vertex weights are positive integers (w(v_i) = (i mod 200) + 1);
            # objective is an integer sum.  Use a strict tolerance: 0.5
            # catches any integer mismatch by >=1, plus a 0.1% relative
            # cushion for huge cliques where float drift could matter.
            obj_tol = max(0.5, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(3)
                lhs = reported
                rhs = true_obj
                raw_excess = abs_diff
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_v(w[v]*x_v)="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.3g})."
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Maximum Weighted Clique Problem (MWCP)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the candidate solution JSON file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file.")
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
        print("FEASIBLE")
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
