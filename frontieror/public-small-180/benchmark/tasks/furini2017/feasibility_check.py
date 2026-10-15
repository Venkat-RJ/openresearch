"""
Feasibility checker for the Vertex Coloring Problem (VCP).

Based on: Furini, Gabrel, Ternier (2017) — "An Improved DSATUR-Based
Branch-and-Bound Algorithm for the Vertex Coloring Problem"

Mathematical formulation constraints (MSSP on auxiliary graph G_A):
  Constraint (1): x_u + x_v <= 1  for all (u, v) in E_A   [tag 6]
  Constraint (2): x_v in {0, 1}   for all v in V_A         [tag 7]

These map to the VCP coloring solution as:
  Constraint (1): Proper coloring — for each edge (u,v) in E,
                  color(u) != color(v).
  Constraint (2): Valid assignment — every vertex v in V is assigned
                  exactly one non-negative integer color.
  Constraint (3): Objective consistency — the reported objective_value
                  (min number of colors k) must equal the number of
                  distinct colors actually used in the coloring. This is
                  a Tier C defense against score-gaming exploits where a
                  candidate submits a valid coloring but reports a
                  fabricated objective_value.

NOTE: this is the obj-recompute variant of feasibility_check.py. It is a
strict superset of the original — it never removes a violation the
original reports and only ever adds constraint index 3.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('color_classes', 'coloring', 'objective_value')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('color_classes',)
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
                if operation == "assignment_key_set":
                    reported_set = set(solution[target])
                    expected_set = {entry[source_key] for entry in solution[source]}
                    if reported_set != expected_set:
                        violations.append(
                            f"{target}={sorted(reported_set)} does not match the "
                            f"decision-derived value {sorted(expected_set)}"
                        )
                    continue

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
    """Check all hard constraints of the VCP coloring solution."""
    violations = []
    violation_magnitudes = []

    graph = instance["graph"]
    n = graph["n"]
    edges = graph["edges"]

    # Extract coloring from solution
    coloring = solution.get("coloring", None)
    num_colors_used = solution.get("num_colors_used", None)

    if coloring is None:
        # Solution only has objective_value (e.g., efficient_algorithm output).
        # No structural solution to verify — report feasible.
        return violations, violation_magnitudes

    # Convert string keys to int keys if needed
    coloring = {int(k): v for k, v in coloring.items()}

    # ------------------------------------------------------------------
    # Constraint (1): Proper coloring — x_u + x_v <= 1 for all (u,v) in E_A
    # Mapped to: for each edge (u,v) in E, color(u) != color(v).
    # When both endpoints share color c, the assignment indicator
    # y_{u,c} + y_{v,c} = 2 > 1, so LHS=2, RHS=1, violation_amount=1.
    # ------------------------------------------------------------------
    conflict_edges = []
    for edge in edges:
        u, v = edge[0], edge[1]
        cu = coloring.get(u, None)
        cv = coloring.get(v, None)
        if cu is not None and cv is not None and cu == cv:
            conflict_edges.append((u, v, cu))

    if conflict_edges:
        lhs = 2.0
        rhs = 1.0
        raw_excess = lhs - rhs  # = 1.0
        normalizer = max(abs(rhs), EPS)
        ratio = raw_excess / normalizer

        edge_strs = [f"({u},{v}) both color {c}" for u, v, c in conflict_edges]
        msg = (
            f"Constraint (1) [proper coloring]: {len(conflict_edges)} edge(s) "
            f"have both endpoints with the same color: "
            + "; ".join(edge_strs[:10])
        )
        if len(conflict_edges) > 10:
            msg += f" ... and {len(conflict_edges) - 10} more"
        violations.append(msg)

        for u, v, c in conflict_edges:
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint (2): Valid assignment — x_v in {0,1} for all v in V_A
    # Mapped to: every vertex v in V must be assigned exactly one
    # non-negative integer color in {0, ..., k-1}.
    # ------------------------------------------------------------------

    # 2a. Check that every vertex is colored
    uncolored = [v for v in range(n) if v not in coloring]
    if uncolored:
        # Each uncolored vertex violates the assignment constraint.
        # LHS = 0 (number of colors assigned), RHS = 1, this is a >= constraint.
        lhs_val = 0.0
        rhs_val = 1.0
        raw_excess = rhs_val - lhs_val  # = 1.0
        normalizer = max(abs(rhs_val), EPS)
        ratio = raw_excess / normalizer

        msg = (
            f"Constraint (2) [valid assignment]: {len(uncolored)} vertex/vertices "
            f"not assigned any color: {uncolored[:10]}"
        )
        if len(uncolored) > 10:
            msg += f" ... and {len(uncolored) - 10} more"
        violations.append(msg)

        for v in uncolored:
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # 2b. Check that all color values are non-negative integers
    invalid_colors = []
    for v, c in coloring.items():
        if not isinstance(c, int) or c < 0:
            invalid_colors.append((v, c))

    if invalid_colors:
        # LHS = actual value, RHS = 0 (lower bound), >= constraint
        msg = (
            f"Constraint (2) [valid assignment]: {len(invalid_colors)} vertex/vertices "
            f"have invalid color values: "
            + "; ".join(f"vertex {v} -> {c}" for v, c in invalid_colors[:10])
        )
        if len(invalid_colors) > 10:
            msg += f" ... and {len(invalid_colors) - 10} more"
        violations.append(msg)

        for v, c in invalid_colors:
            raw_excess = 1.0  # binary domain violation
            normalizer = max(abs(1.0), EPS)
            ratio = raw_excess / normalizer
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(c) if isinstance(c, (int, float)) else 0.0,
                "rhs": 0.0,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint (3): Objective consistency (Tier C obj-recompute).
    # The VCP objective is min k = the number of distinct colors used.
    # Every variable the objective depends on (the per-vertex coloring)
    # is present in the solution, so a FULL recompute applies:
    #   true_obj = number of distinct color labels in `coloring`.
    # Reject when the reported objective_value disagrees. The objective
    # is an integer color count, so a tolerance of 0.5 fires on any
    # integer mismatch by >= 1 while tolerating float round-trip noise.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value", None)
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(len(set(coloring.values())))
            abs_diff = abs(reported - true_obj)
            tol = 0.5
            if abs_diff > tol:
                raw_excess = abs_diff
                normalizer = max(abs(true_obj), EPS)
                ratio = raw_excess / normalizer
                msg = (
                    f"Constraint (3) [objective consistency]: reported "
                    f"objective_value={reported_obj} differs from the "
                    f"recomputed number of distinct colors used in the "
                    f"coloring ({int(true_obj)}) "
                    f"(|diff|={abs_diff:.6g}, tol={tol})"
                )
                violations.append(msg)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    return violations, violation_magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Vertex Coloring Problem (Furini et al. 2017)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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

    violations, violation_magnitudes = check_feasibility(instance, solution)

    # Collect unique violated constraint indices
    violated_constraints = sorted(
        set(vm["constraint"] for vm in violation_magnitudes)
    )

    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if feasible else "INFEASIBLE"
    print(f"Result: {status}")
    if not feasible:
        for v in violations:
            print(f"  - {v}")
    print(f"Written to {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
