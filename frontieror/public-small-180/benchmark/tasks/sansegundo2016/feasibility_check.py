#!/usr/bin/env python3
"""
Feasibility checker for the Maximum Clique Problem (MCP).

Mathematical formulation:
  max  sum_{i in V} x_i
  s.t.
    (C1) x_i + x_j <= 1,  for all (i,j) not in E, i < j
    (C2) x_i in {0, 1},   for all i in V

Constraints enforced by this checker:
    1  (C1) Clique constraint: every pair of selected vertices must be adjacent.
    2  (C2) Binary domain: clique vertices must be distinct valid indices in [0, n).
    3  (Tier C obj-consistency) Reported objective_value must equal the recomputed
       clique size sum_i x_i (= number of distinct valid vertices in clique_vertices).
       This blocks LLM exploits that pass constraints but lie about the objective
       (e.g. reporting sys.float_info.max or 0 on a valid clique).
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


def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    graph = data['graph']
    n = graph['num_vertices']
    adj = [set() for _ in range(n)]
    for u_str, neighbors in graph['adjacency_list'].items():
        u = int(u_str)
        for v in neighbors:
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
    return n, adj


def load_solution(path):
    with open(path, 'r') as f:
        sol = json.load(f)
    return sol['clique_vertices']


def check_feasibility(instance_path, solution_path, result_path):
    n, adj = load_instance(instance_path)
    with open(solution_path, 'r') as f:
        sol = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
        if _frontieror_contract_result is not None:
            with open(result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return _frontieror_contract_result
    clique_vertices = sol.get('clique_vertices', []) or []

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # --- Constraint (C1): x_i + x_j <= 1 for all (i,j) not in E, i < j ---
    # If both i and j are in the clique, then x_i + x_j = 2.
    # For this to be feasible, (i,j) must be in E. If not, C1 is violated.
    clique_set = set(clique_vertices)
    sorted_clique = sorted(clique_vertices)
    c1_violations = []
    for idx_a in range(len(sorted_clique)):
        i = sorted_clique[idx_a]
        for idx_b in range(idx_a + 1, len(sorted_clique)):
            j = sorted_clique[idx_b]
            if not isinstance(i, int) or not isinstance(j, int):
                continue
            if i < 0 or i >= n or j < 0 or j >= n:
                continue
            if j not in adj[i]:
                # LHS = x_i + x_j = 2, RHS = 1
                lhs = 2.0
                rhs = 1.0
                violation_amount = lhs - rhs  # 1.0
                normalizer = max(abs(rhs), EPS)
                ratio = violation_amount / normalizer
                c1_violations.append((i, j, lhs, rhs, violation_amount, normalizer, ratio))

    if c1_violations:
        violated_constraints.add(1)
        if len(c1_violations) <= 10:
            for i, j, *_ in c1_violations:
                violations.append(
                    f"C1 violated: vertices {i} and {j} are both in the clique but are not adjacent"
                )
        else:
            violations.append(
                f"C1 violated: {len(c1_violations)} pairs of clique vertices are not adjacent"
            )
        for i, j, lhs, rhs, raw_excess, normalizer, ratio in c1_violations:
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # --- Constraint (C2): x_i in {0, 1} for all i in V ---
    # Check that all clique vertices are valid vertex indices in [0, n).
    c2_violations = []
    for v in clique_vertices:
        if not isinstance(v, int) or v < 0 or v >= n:
            lhs = float(v) if isinstance(v, (int, float)) else 0.0
            # For domain violation, treat as |x - round(x)| or out-of-range
            rhs = 0.0  # nearest valid boundary
            violation_amount = 1.0  # binary domain violation
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            c2_violations.append((v, lhs, rhs, violation_amount, normalizer, ratio))

    if c2_violations:
        violated_constraints.add(2)
        for v, *_ in c2_violations:
            violations.append(f"C2 violated: vertex {v} is not a valid vertex in V = {{0, ..., {n-1}}}")
        for v, lhs, rhs, raw_excess, normalizer, ratio in c2_violations:
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # Also check for duplicates (a vertex selected more than once violates binary domain)
    seen = set()
    for v in clique_vertices:
        if v in seen:
            if 2 not in violated_constraints:
                violated_constraints.add(2)
            violations.append(f"C2 violated: vertex {v} appears more than once in the clique")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": 2.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })
        seen.add(v)

    # --- Constraint (C3): obj-consistency (Tier C) ---
    # True objective = sum_i x_i = number of distinct valid vertices selected.
    # Reject when the reported objective_value disagrees by more than 0.5
    # (objective is integer-valued, so a half-unit tolerance fires on any
    # integer-magnitude mismatch while remaining robust to harmless float noise).
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            valid_distinct = set(
                v for v in clique_vertices
                if isinstance(v, int) and 0 <= v < n
            )
            true_obj = float(len(valid_distinct))
            abs_diff = abs(reported - true_obj)
            tol = 0.5  # objective is integer-valued
            if abs_diff > tol:
                violated_constraints.add(3)
                violations.append(
                    f"C3 violated: reported objective_value={reported} differs "
                    f"from recomputed sum_i x_i={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
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

    status = "FEASIBLE" if feasible else "INFEASIBLE"
    print(f"Result: {status}")
    if not feasible:
        print(f"Violated constraints: {sorted(violated_constraints)}")
        for v in violations[:10]:
            print(f"  - {v}")
    print(f"Result written to {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for Maximum Clique Problem")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    check_feasibility(args.instance_path, args.solution_path, args.result_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
