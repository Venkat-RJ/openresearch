#!/usr/bin/env python3
"""
Feasibility checker for Min-Power Symmetric Connectivity (MinPSC).

Checks candidate solutions against the hard constraints from the mathematical
formulation in Bentert et al. (2020):

  Constraint 1 (C1): Spanning subgraph — F ⊆ E, and all edges in F must use
                      valid vertices from V.
  Constraint 2 (C2): Connectivity — T = (V, F) must be a connected subgraph.
  Constraint 3 (C3): Cost definition — For each v ∈ V,
                      p_v = max_{{u,v} ∈ F} w({u,v}).
"""

import argparse
import json
from collections import defaultdict, deque

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'selected_edges', 'vertex_costs')
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



def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    vertices = data["vertices"]
    V = []
    coords = {}
    for v_info in vertices:
        vid = v_info["id"]
        V.append(vid)
        coords[vid] = (v_info["x"], v_info["y"])

    # Build edge weight lookup (squared Euclidean distance, complete graph)
    edge_weights = {}
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            u, v = V[i], V[j]
            dx = coords[u][0] - coords[v][0]
            dy = coords[u][1] - coords[v][1]
            w = dx * dx + dy * dy
            edge_weights[(u, v)] = w
            edge_weights[(v, u)] = w

    return V, coords, edge_weights


def check_connectivity(V, selected_edges):
    """Check if (V, selected_edges) is connected. Returns number of components."""
    adj = defaultdict(set)
    for u, v in selected_edges:
        adj[u].add(v)
        adj[v].add(u)

    if not V:
        return 0

    visited = set()
    queue = deque([V[0]])
    visited.add(V[0])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    V_set = set(V)
    num_visited = len(visited & V_set)
    if num_visited == len(V_set):
        return 1

    # Count actual components
    components = 0
    visited = set()
    for start in V:
        if start in visited:
            continue
        components += 1
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb in V_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return components


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MinPSC solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    tol = 1e-5
    eps = 1e-5

    V, coords, edge_weights = load_instance(args.instance_path)
    V_set = set(V)

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

    selected_edges = [tuple(e) for e in solution.get("selected_edges", [])]
    objective_value = solution.get("objective_value", None)
    vertex_costs = solution.get("vertex_costs", {})
    # Keys in vertex_costs may be strings
    vertex_costs = {int(k): float(v) for k, v in vertex_costs.items()}

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 (C1): Spanning subgraph — F ⊆ E
    # All edges in F must connect valid vertices from V, and each edge must
    # exist in E (the complete graph over V, so just check vertex validity).
    # =========================================================================
    invalid_edges = []
    for u, v in selected_edges:
        if u not in V_set or v not in V_set:
            invalid_edges.append((u, v))
        elif u == v:
            invalid_edges.append((u, v))

    if invalid_edges:
        violated_constraints.add(1)
        violations.append(
            f"C1: {len(invalid_edges)} edge(s) in F use invalid or duplicate "
            f"vertex IDs not in V: {invalid_edges[:5]}"
        )
        # LHS = number of invalid edges, RHS = 0 (equality: should be 0)
        lhs = float(len(invalid_edges))
        rhs = 0.0
        violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # =========================================================================
    # Constraint 2 (C2): Connectivity — T = (V, F) must be connected
    # =========================================================================
    num_components = check_connectivity(V, selected_edges)
    if num_components != 1:
        violated_constraints.add(2)
        violations.append(
            f"C2: Subgraph (V, F) has {num_components} connected components "
            f"(must be 1)"
        )
        # LHS = num_components, RHS = 1 (equality: should be exactly 1)
        lhs = float(num_components)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # =========================================================================
    # Constraint 3 (C3): Cost definition — p_v = max_{{u,v} ∈ F} w({u,v})
    # For each vertex v, the reported cost must equal the maximum weight of
    # any selected edge incident to v. Vertices with no incident selected
    # edges have an implied cost of 0.
    # =========================================================================
    # Compute actual max incident edge weight for each vertex
    actual_costs = {v: 0.0 for v in V}
    for u, v in selected_edges:
        if u in V_set and v in V_set and u != v:
            w = edge_weights[(u, v)]
            if w > actual_costs[u]:
                actual_costs[u] = w
            if w > actual_costs[v]:
                actual_costs[v] = w

    c3_violations = []
    for v in V:
        reported = vertex_costs.get(v, 0.0)
        actual = actual_costs[v]
        diff = abs(reported - actual)
        if diff > tol:
            c3_violations.append((v, reported, actual, diff))

    if c3_violations:
        violated_constraints.add(3)
        # Summarize
        example_strs = []
        for v, reported, actual, diff in c3_violations[:5]:
            example_strs.append(
                f"v={v}: reported={reported:.6f}, actual={actual:.6f}, "
                f"diff={diff:.6f}"
            )
        violations.append(
            f"C3: {len(c3_violations)} vertex cost(s) do not match "
            f"max incident edge weight. Examples: {'; '.join(example_strs)}"
        )
        # Record one violation magnitude entry per violated vertex
        for v, reported, actual, diff in c3_violations:
            lhs = reported
            rhs = actual
            violation_amount = diff
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint 4 (C4): Objective value — objective_value == sum of p_v
    # The reported objective_value must equal the sum of all vertex costs.
    # From math_model: min sum_{v in V} p_v
    # =========================================================================
    if objective_value is not None:
        computed_obj = sum(vertex_costs.get(v, 0.0) for v in V)
        obj_diff = abs(objective_value - computed_obj)
        if obj_diff > tol:
            violated_constraints.add(4)
            violations.append(
                f"C4: Reported objective_value={objective_value:.6f} does not "
                f"match sum of vertex_costs={computed_obj:.6f}, "
                f"diff={obj_diff:.6f}"
            )
            lhs = float(objective_value)
            rhs = computed_obj
            violation_amount = obj_diff
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Build output
    # =========================================================================
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if feasible:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: "
              f"{sorted(violated_constraints)}")
        for msg in violations:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
