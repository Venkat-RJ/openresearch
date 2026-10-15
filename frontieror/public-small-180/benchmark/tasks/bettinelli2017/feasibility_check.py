#!/usr/bin/env python3
"""
Feasibility checker for the Knapsack Problem with Conflict Graph (KPCG).
Based on Bettinelli, Cacchiani, and Malaguti (2017).

Constraints checked (original edge-based formulation, eqs. 1b-1d):
  1: (1b) Capacity: sum w_i x_i <= c
  2: (1c) Edge conflict: x_i + x_j <= 1 for (i,j) in E
  3: (1d) Binary: x_i in {0,1}
  4: (1a) Objective consistency: reported objective_value must equal
         sum_{i in selected} p_i (Tier C anti-exploit check).
"""

import argparse
import json
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'selected_items')
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



def build_adjacency(num_vertices, edges):
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def generate_clique_cover(num_vertices, edges, adj):
    """
    Heuristically generate a clique cover as described in the paper:
    iteratively pick an uncovered edge, grow a maximal clique from it.
    """
    uncovered = set()
    for e in edges:
        uncovered.add((min(e), max(e)))

    cliques = []
    while uncovered:
        # Pick an uncovered edge
        i, j = next(iter(uncovered))
        clique = {i, j}

        # Try to extend: add vertices connected to all current clique members
        candidates = adj[i] & adj[j]
        for v in sorted(candidates):
            if all(v in adj[u] for u in clique):
                clique.add(v)

        # Mark edges within this clique as covered
        clique_list = sorted(clique)
        for a_idx in range(len(clique_list)):
            for b_idx in range(a_idx + 1, len(clique_list)):
                edge = (clique_list[a_idx], clique_list[b_idx])
                uncovered.discard(edge)

        cliques.append(clique_list)

    return cliques


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["num_items"]
    capacity = instance["capacity"]
    weights = instance["items"]["weights"]
    profits = instance["items"]["profits"]
    edges = instance["conflict_graph"]["edges"]
    num_vertices = instance["conflict_graph"]["num_vertices"]

    selected = solution["selected_items"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Build selection set and vector
    selected_set = set(selected)
    x = [1 if i in selected_set else 0 for i in range(n)]

    # --- Constraint 1 (1b): Capacity constraint: sum w_i x_i <= c ---
    total_weight = sum(weights[i] for i in selected)
    lhs_1 = float(total_weight)
    rhs_1 = float(capacity)
    violation_amount = max(0.0, lhs_1 - rhs_1)
    if violation_amount > tol:
        violated_constraints.add(1)
        violations.append(
            f"Capacity exceeded: total weight {total_weight} > capacity {capacity}"
        )
        normalizer = max(abs(rhs_1), eps)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs_1,
            "rhs": rhs_1,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # --- Constraint 2 (1c): Edge conflict: x_i + x_j <= 1 for (i,j) in E ---
    conflict_violations = []
    for i, j in edges:
        lhs_val = float(x[i] + x[j])
        rhs_val = 1.0
        va = max(0.0, lhs_val - rhs_val)
        if va > tol:
            conflict_violations.append((i, j, lhs_val, rhs_val, va))

    if conflict_violations:
        violated_constraints.add(2)
        pairs = [(i, j) for i, j, _, _, _ in conflict_violations]
        violations.append(
            f"Conflict edge constraint violated for {len(conflict_violations)} "
            f"edge(s): {pairs}"
        )
        for i, j, lhs_val, rhs_val, va in conflict_violations:
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": va,
                "normalizer": normalizer,
                "ratio": va / normalizer,
            })

    # --- Constraint 3 (1d): Binary: x_i in {0,1} ---
    non_binary = []
    for i in selected:
        if i < 0 or i >= n:
            non_binary.append(i)
    # Also check that selected_items values are valid indices and binary-compatible
    for i in selected:
        if not isinstance(i, int) and not (isinstance(i, float) and i == int(i)):
            non_binary.append(i)

    if non_binary:
        violated_constraints.add(3)
        violations.append(
            f"Non-binary or out-of-range item indices: {non_binary}"
        )
        for idx in non_binary:
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(idx),
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })

    # Check for duplicate selections (violates binary constraint)
    if len(selected) != len(selected_set):
        violated_constraints.add(3)
        from collections import Counter
        counts = Counter(selected)
        dups = {k: v for k, v in counts.items() if v > 1}
        violations.append(f"Duplicate item selections (violates binary): {dups}")
        for item, count in dups.items():
            # x_i effectively > 1
            lhs_val = float(count)
            rhs_val = 1.0
            va = lhs_val - rhs_val
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": va,
                "normalizer": normalizer,
                "ratio": va / normalizer,
            })

    # --- Constraint 4 (1a): Objective consistency (Tier C anti-exploit) ---
    # Profits are positive integers and obj = sum_{i in selected} p_i is a
    # full recompute (every variable that determines the obj is in the
    # solution). Reject when reported objective_value disagrees.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Only sum profits of valid in-range indices; out-of-range items
            # are already caught by constraint 3 and may IndexError here.
            true_obj = float(sum(
                profits[i] for i in selected
                if isinstance(i, int) and 0 <= i < n
            ))
            abs_diff = abs(reported - true_obj)
            # profits are integers, so an integer mismatch (>=1) must fire;
            # keep a 0.1% relative floor for safety on large objs.
            obj_tol = max(0.5, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(4)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_{{i in selected}} p_i="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for KPCG (Bettinelli et al., 2017)"
    )
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
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
        print("FEASIBLE")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}")
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
