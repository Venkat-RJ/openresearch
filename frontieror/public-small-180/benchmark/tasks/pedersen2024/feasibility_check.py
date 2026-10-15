#!/usr/bin/env python3
"""
Feasibility checker for the Quota Steiner Tree Problem (QSTP).

Checks a candidate solution against all hard constraints from the mathematical
formulation in Pedersen et al. (2024). Constraints are numbered 1-14 counting
strictly from top to bottom in the formulation sections of math_model.txt,
plus a new constraint 15 added by this _new variant for obj-consistency
(Tier C defense against fabricated objective values):

  Constraint  1 (Eq  2): Quota in original QSTP:  sum q_i >= Q
  Constraint  2 (Eq  5): Steiner cut for fixed terminals (connectivity)
  Constraint  3 (Eq  6): Steiner cut for selected potential terminals
  Constraint  4 (Eq  7): Quota in directed cut formulation
  Constraint  5 (Eq  8): Binary x, y variables + valid edges/turbines
  Constraint  6 (Eq 11): Steiner cut in transformed graph
  Constraint  7 (Eq 12): Transformed quota constraint
  Constraint  8 (Eq 13): Binary x in transformed graph
  Constraint  9 (Eq 25): Quota in flow-based MIP
  Constraint 10 (Eq 26): Flow balance (connectivity via flow)
  Constraint 11 (Eq 27): Arc-terminal linking: x_a <= y_v
  Constraint 12 (Eq 28): Flow capacity: f_a <= M * x_a
  Constraint 13 (Eq 29): Binary y in flow-based MIP
  Constraint 14 (Eq 30): Binary x, f >= 0 in flow-based MIP
  Constraint 15 (Eq 21/23 obj): Reported objective_value must match
                                alpha * (edge cost + turbine cost)
                              + (1-alpha) * (edge scenic + turbine scenic).
"""

import argparse
import json
from collections import defaultdict, deque

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'selected_edges', 'selected_turbines')
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

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # --- Extract instance data ---
    nodes = instance["nodes"]
    edges = instance["edges"]
    params = instance["parameters"]
    Q = params["quota"]
    alpha = params["alpha"]

    node_by_id = {n["id"]: n for n in nodes}
    T_f = [n["id"] for n in nodes if n["type"] == "substation"]
    T_p = [n["id"] for n in nodes if n["type"] == "potential_turbine"]
    T_p_set = set(T_p)
    T_f_set = set(T_f)

    # Valid undirected edges from instance
    valid_edges = set()
    edge_by_pair = {}
    for e in edges:
        valid_edges.add((e["from"], e["to"]))
        valid_edges.add((e["to"], e["from"]))
        edge_by_pair[(e["from"], e["to"])] = e
        edge_by_pair[(e["to"], e["from"])] = e

    # --- Extract solution data ---
    selected_turbines = set(solution.get("selected_turbines", []))
    selected_edges = solution.get("selected_edges", [])
    selected_arcs = [(e["from"], e["to"]) for e in selected_edges]

    # --- Vertex profits ---
    vertex_profit = {n["id"]: n["profit"] for n in nodes}
    total_profit = sum(vertex_profit[t] for t in selected_turbines)
    total_available_profit = sum(vertex_profit[t] for t in T_p)

    # --- Connectivity (undirected) from root ---
    r = T_f[0]
    adj_undirected = defaultdict(set)
    for u, v in selected_arcs:
        adj_undirected[u].add(v)
        adj_undirected[v].add(u)

    reachable = set()
    queue = deque([r])
    reachable.add(r)
    while queue:
        u = queue.popleft()
        for v in adj_undirected[u]:
            if v not in reachable:
                reachable.add(v)
                queue.append(v)

    # Big-M value for flow formulation
    M = len(T_f) + len(T_p)

    # --- Helper to record a violation ---
    def record(constraint_idx, msg, lhs, rhs, violation_amount):
        if violation_amount > tol:
            violated_constraints.add(constraint_idx)
            violations.append(msg)
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # ================================================================
    # Constraint 1 (Eq 2): Q(S) = sum_{i in T_p cap V'} q_i >= Q
    # ================================================================
    lhs_val = total_profit
    rhs_val = Q
    va = max(rhs_val - lhs_val, 0.0)
    record(1, f"Quota not met (original QSTP): total profit {lhs_val:.4f} < Q = {rhs_val:.4f}", lhs_val, rhs_val, va)

    # ================================================================
    # Constraint 2 (Eq 5): Steiner cut for fixed terminals
    #   All fixed terminals must be reachable from root via selected edges.
    # ================================================================
    for t in T_f:
        if t != r and t not in reachable:
            record(2, f"Fixed terminal {t} not connected to root {r}", 0.0, 1.0, 1.0)

    # ================================================================
    # Constraint 3 (Eq 6): Steiner cut for selected potential terminals
    #   Each selected potential terminal must be reachable from root.
    # ================================================================
    for t in sorted(selected_turbines):
        if t not in reachable:
            record(3, f"Selected turbine {t} not connected to root {r}", 0.0, 1.0, 1.0)

    # ================================================================
    # Constraint 4 (Eq 7): sum_{v_i in T_p} q_i * y_i >= Q
    # ================================================================
    lhs_val = total_profit
    rhs_val = Q
    va = max(rhs_val - lhs_val, 0.0)
    record(4, f"Quota not met (directed cut): sum q_i*y_i = {lhs_val:.4f} < Q = {rhs_val:.4f}", lhs_val, rhs_val, va)

    # ================================================================
    # Constraint 5 (Eq 8): x_{ij}, y_k in {0,1}
    #   Check that selected turbines are valid potential turbines
    #   and selected edges correspond to valid edges in the instance.
    # ================================================================
    for t in sorted(selected_turbines):
        if t not in T_p_set:
            record(5, f"Selected turbine {t} is not a valid potential turbine", 1.0, 0.0, 1.0)
    for e in selected_edges:
        u, v = e["from"], e["to"]
        if (u, v) not in valid_edges:
            record(5, f"Selected arc ({u},{v}) does not correspond to a valid edge", 1.0, 0.0, 1.0)

    # ================================================================
    # Constraint 6 (Eq 11): Steiner cut in transformed graph
    #   In the transformed QSTP, all terminals (original fixed + new fixed)
    #   must be connected. In original space: all fixed terminals and all
    #   selected potential terminals must be reachable from root.
    # ================================================================
    disconnected = []
    for t in T_f:
        if t != r and t not in reachable:
            disconnected.append(t)
    for t in sorted(selected_turbines):
        if t not in reachable:
            disconnected.append(t)
    if disconnected:
        record(6, f"Steiner cut violated (transformed): nodes {disconnected} disconnected from root", 0.0, 1.0, 1.0)

    # ================================================================
    # Constraint 7 (Eq 12): Transformed quota
    #   sum_{v_{i'} in T'_f} q_{i'} * x_{r,i'} <= sum_{v_{i'} in T'_f} q_{i'} - Q
    #   In original space: profit of UNselected turbines <= total_available - Q
    #   Equivalently: selected profit >= Q (same as constraint 1).
    # ================================================================
    unselected_profit = total_available_profit - total_profit
    rhs_val = total_available_profit - Q
    lhs_val = unselected_profit
    # This is a <= constraint: lhs <= rhs
    va = max(lhs_val - rhs_val, 0.0)
    record(7, f"Transformed quota violated: unselected profit {lhs_val:.4f} > allowed {rhs_val:.4f}", lhs_val, rhs_val, va)

    # ================================================================
    # Constraint 8 (Eq 13): x_{ij} in {0,1} (transformed)
    #   Auto-satisfied by the solution representation (list-based).
    #   Check that no duplicate arcs exist.
    # ================================================================
    arc_counts = defaultdict(int)
    for u, v in selected_arcs:
        arc_counts[(u, v)] += 1
    for arc, count in arc_counts.items():
        if count > 1:
            record(8, f"Duplicate arc {arc} appears {count} times (should be binary)", float(count), 1.0, float(count - 1))

    # ================================================================
    # Constraint 9 (Eq 25): sum_{v in T_p} q_v * y_v >= Q (flow MIP)
    # ================================================================
    lhs_val = total_profit
    rhs_val = Q
    va = max(rhs_val - lhs_val, 0.0)
    record(9, f"Quota not met (flow MIP): sum q_v*y_v = {lhs_val:.4f} < Q = {rhs_val:.4f}", lhs_val, rhs_val, va)

    # ================================================================
    # Constraint 10 (Eq 26): Flow balance
    #   Fixed terminals (not root): net inflow = 1
    #   Potential terminals: net inflow = y_v
    #   Steiner nodes: net inflow = 0
    #   Check indirectly: each fixed terminal (not root) and each selected
    #   potential terminal must be reachable from root.
    #   Also: non-selected potential terminals should not have net inflow.
    # ================================================================
    for t in T_f:
        if t != r and t not in reachable:
            record(10, f"Flow balance violated: fixed terminal {t} unreachable from root", 0.0, 1.0, 1.0)
    for t in sorted(selected_turbines):
        if t not in reachable:
            record(10, f"Flow balance violated: selected turbine {t} unreachable from root", 0.0, 1.0, 1.0)

    # ================================================================
    # Constraint 11 (Eq 27): x_a <= y_v for all a in delta^-(v), v in T_p
    #   If a selected arc enters a potential turbine, that turbine must be
    #   selected.
    # ================================================================
    for e in selected_edges:
        v = e["to"]
        if v in T_p_set and v not in selected_turbines:
            record(11, f"Arc ({e['from']},{v}) enters non-selected turbine {v} (x_a=1 but y_{v}=0)", 1.0, 0.0, 1.0)

    # ================================================================
    # Constraint 12 (Eq 28): f_a <= M * x_a
    #   For selected arcs (x_a = 1), flow is bounded by M (auto-satisfied).
    #   For non-selected arcs (x_a = 0), no flow passes (not checkable
    #   without flow variables, but structurally satisfied).
    #   No violation possible from the solution representation.
    # ================================================================
    # Auto-satisfied: flow variables are internal to the solver.

    # ================================================================
    # Constraint 13 (Eq 29): y_v in {0,1} for all v in T_p
    #   Auto-satisfied: solution represents y as a list of selected turbines.
    # ================================================================
    # Already checked in constraint 5 that selected turbines are valid T_p.

    # ================================================================
    # Constraint 14 (Eq 30): x_a in {0,1}, f_a >= 0
    #   Auto-satisfied by the solution representation.
    # ================================================================
    # Already checked in constraints 5 and 8.

    # ================================================================
    # Constraint 15 (Eq 21/23 obj): Reported objective_value must match
    #   the recomputed value
    #     obj = sum_{a in selected_edges} (alpha * c_e + (1-alpha) * s(e))
    #         + sum_{v in selected_turbines} (alpha * w(v) + (1-alpha) * s^v(v))
    #   where for each directed arc (u,v) in selected_edges, c_e and s(e)
    #   come from the corresponding undirected edge in the instance (Gurobi
    #   selects directed arcs from the flow MIP, so each undirected edge may
    #   appear in either or both orientations). Invalid arcs (already flagged
    #   by constraint 5) are skipped here so they don't double-fire.
    #   Tolerance: max(1e-3, 1e-3 * |recomputed|) — 0.1% relative.
    # ================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            edge_obj = 0.0
            for a in selected_edges:
                u, v = a["from"], a["to"]
                e = edge_by_pair.get((u, v))
                if e is None:
                    continue  # invalid arc, already flagged by constraint 5
                edge_obj += alpha * float(e["cost"]) + (1.0 - alpha) * float(e["scenic_impact"])
            turbine_obj = 0.0
            for t in selected_turbines:
                n = node_by_id.get(t)
                if n is None or n["type"] != "potential_turbine":
                    continue  # invalid turbine, already flagged by constraint 5
                turbine_obj += alpha * float(n["cost"]) + (1.0 - alpha) * float(n["scenic_impact"])
            true_obj = edge_obj + turbine_obj
            abs_diff = abs(reported - true_obj)
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record(
                    15,
                    f"Objective consistency violated: reported objective_value={reported} "
                    f"differs from recomputed alpha*cost+(1-alpha)*scenic_impact "
                    f"(edges={edge_obj:.6f}, turbines={turbine_obj:.6f}, "
                    f"total={true_obj:.6f}); |diff|={abs_diff:.6g}, tol={obj_tol:.3g}",
                    reported, true_obj, abs_diff,
                )

    # --- Build result ---
    result = {
        "feasible": len(violated_constraints) == 0,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if violation_magnitudes else [],
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check feasibility of a QSTP candidate solution."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
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
        print(f"FEASIBLE - No constraint violations found.")
    else:
        print(f"INFEASIBLE - Violated constraints: {result['violated_constraints']}")
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
