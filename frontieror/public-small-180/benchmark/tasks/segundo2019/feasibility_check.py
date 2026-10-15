"""
Feasibility checker for the Maximum Edge-Weighted Clique Problem (MEWCP).

Checks the F1 (Nonedge) formulation from Segundo et al. (2019):

  Constraint 1 (tag 27): x_u + x_v <= 1,  for all {u,v} in E_bar (non-edges)
  Constraint 2 (tag 6):  y_e <= x_u,       for all e = {u,v} in E
  Constraint 3 (tag 7):  y_e <= x_v,       for all e = {u,v} in E
  Constraint 4 (tag 8):  x_u in {0,1},     for all u in V
  Constraint 5 (tag 9):  y_e in {0,1},     for all e in E
  Constraint 6 (Obj):    reported objective_value == sum_{e in E} c_e * y_e
                         (Tier C anti-exploit: reject solutions whose
                         reported obj disagrees with the value recomputed
                         from clique_edges and instance edge weights.)
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('clique_edges', 'clique_vertices', 'objective_value')
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


def edge_key(u, v):
    return (min(u, v), max(u, v))


def check_feasibility(instance_path, solution_path, result_path):
    with open(instance_path, "r") as f:
        instance = json.load(f)
    with open(solution_path, "r") as f:
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

    graph = instance["graph"]
    vertices = set(graph["vertices"])
    edges_raw = graph["edges"]  # [[u, v, w], ...]

    # Build edge set and weight map
    edge_set = set()
    edge_weight = {}
    for u, v, w in edges_raw:
        k = edge_key(u, v)
        edge_set.add(k)
        edge_weight[k] = w

    # Extract solution
    clique_vertices = solution["clique_vertices"]
    clique_edges = solution["clique_edges"]  # [[u, v], ...] per solution_schema.json

    # Build x and y from solution
    # x_u = 1 if u in clique_vertices, else 0
    x = {u: 1.0 for u in clique_vertices}
    # y_e = 1 if edge in clique_edges, else 0
    y = {}
    for entry in clique_edges:
        u, v = entry[0], entry[1]
        k = edge_key(u, v)
        y[k] = 1.0

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 (tag 27): x_u + x_v <= 1 for all non-edges {u,v} in E_bar
    # =========================================================================
    # If two clique vertices are NOT connected by an edge, this is violated.
    clique_list = list(clique_vertices)
    for i in range(len(clique_list)):
        for j in range(i + 1, len(clique_list)):
            u, v = clique_list[i], clique_list[j]
            k = edge_key(u, v)
            if k not in edge_set:
                # This is a non-edge, and both x_u=1, x_v=1 => LHS=2, RHS=1
                lhs = x.get(u, 0.0) + x.get(v, 0.0)
                rhs = 1.0
                violation_amount = lhs - rhs
                if violation_amount > TOL:
                    violated_constraints.add(1)
                    violations.append(
                        f"Constraint 1 (non-edge): vertices {u} and {v} are both "
                        f"in the clique but are not connected by an edge "
                        f"(x_{u} + x_{v} = {lhs} > {rhs})"
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

    # =========================================================================
    # Constraint 2 (tag 6): y_e <= x_u for all e = {u,v} in E
    # =========================================================================
    # For each edge in clique_edges, the first endpoint must be in clique_vertices.
    for entry in clique_edges:
        u, v = entry[0], entry[1]
        k = edge_key(u, v)
        y_val = y.get(k, 0.0)
        # Check y_e <= x_u (u is the smaller index in k)
        x_u_val = x.get(k[0], 0.0)
        lhs = y_val
        rhs = x_u_val
        violation_amount = lhs - rhs
        if violation_amount > TOL:
            violated_constraints.add(2)
            violations.append(
                f"Constraint 2 (edge-vertex linking): edge ({k[0]},{k[1]}) has "
                f"y=1 but vertex {k[0]} not in clique (y_e={lhs} > x_{k[0]}={rhs})"
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

    # =========================================================================
    # Constraint 3 (tag 7): y_e <= x_v for all e = {u,v} in E
    # =========================================================================
    for entry in clique_edges:
        u, v = entry[0], entry[1]
        k = edge_key(u, v)
        y_val = y.get(k, 0.0)
        # Check y_e <= x_v (v is the larger index in k)
        x_v_val = x.get(k[1], 0.0)
        lhs = y_val
        rhs = x_v_val
        violation_amount = lhs - rhs
        if violation_amount > TOL:
            violated_constraints.add(3)
            violations.append(
                f"Constraint 3 (edge-vertex linking): edge ({k[0]},{k[1]}) has "
                f"y=1 but vertex {k[1]} not in clique (y_e={lhs} > x_{k[1]}={rhs})"
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

    # =========================================================================
    # Constraint 4 (tag 8): x_u in {0, 1} for all u in V
    # =========================================================================
    # Check that all clique vertices are valid graph vertices (x=1 only for
    # valid vertices; all others are implicitly 0).
    for u in clique_vertices:
        x_val = x.get(u, 0.0)
        if u not in vertices:
            # Vertex not in graph, so x_u = 1 is assigned to a non-existent vertex
            lhs = x_val
            rhs = 0.0  # should be 0 since vertex doesn't exist
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(4)
                violations.append(
                    f"Constraint 4 (binary x): vertex {u} is not in the graph "
                    f"but assigned x_{u} = {x_val}"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })
        # Check integrality
        if abs(x_val - round(x_val)) > TOL:
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4 (binary x): x_{u} = {x_val} is not binary"
            )
            nearest_int = round(x_val)
            violation_amount = abs(x_val - nearest_int)
            normalizer = max(abs(nearest_int), EPS)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": x_val,
                "rhs": nearest_int,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint 5 (tag 9): y_e in {0, 1} for all e in E
    # =========================================================================
    # Check that all clique edges are valid graph edges.
    for entry in clique_edges:
        u, v = entry[0], entry[1]
        k = edge_key(u, v)
        y_val = y.get(k, 0.0)
        if k not in edge_set:
            # Edge not in graph, y_e = 1 for a non-existent edge
            lhs = y_val
            rhs = 0.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(5)
                violations.append(
                    f"Constraint 5 (binary y): edge ({k[0]},{k[1]}) is not in "
                    f"the graph but assigned y_e = {y_val}"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })
        # Check integrality
        if abs(y_val - round(y_val)) > TOL:
            violated_constraints.add(5)
            violations.append(
                f"Constraint 5 (binary y): y_({k[0]},{k[1]}) = {y_val} is not binary"
            )
            nearest_int = round(y_val)
            violation_amount = abs(y_val - nearest_int)
            normalizer = max(abs(nearest_int), EPS)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": y_val,
                "rhs": nearest_int,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint 6 (Obj-consistency): reported objective_value must equal
    # sum_{e in E} c_e * y_e  (Tier C: catch LLMs that lie about their obj
    # while keeping the clique structure feasible).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Full recompute: y is binary-valued (1 if edge in clique_edges,
            # 0 otherwise); obj = sum of weights of clique_edges that are in E.
            # Edges in clique_edges but not in E contribute 0 (and are already
            # flagged by Constraint 5).
            true_obj = 0.0
            for k_ce in y:
                if k_ce in edge_weight:
                    true_obj += float(edge_weight[k_ce])
            abs_diff = abs(reported - true_obj)
            # Edge weights c_e are positive integers (c_e = (u+v) mod 200 + 1
            # for DIMACS instances); any honest obj is an integer sum. Use
            # tol = max(0.5, 1e-3 * |true_obj|) so a unit-off integer
            # mismatch still fires but pure float noise doesn't.
            tol = max(0.5, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(6)
                violations.append(
                    f"Constraint 6 (objective consistency): reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_e(c_e * y_e)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # Build result
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check feasibility of a candidate MEWCP solution."
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
    result = check_feasibility(args.instance_path, args.solution_path, args.result_path)
    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
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
