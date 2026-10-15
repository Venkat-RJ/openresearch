#!/usr/bin/env python3
"""
Feasibility checker for the Orienteering Problem (OP).

Based on Kobeaga et al. (2024) formulation:
  Constraint 1 (2b): sum_e d_e * x_e <= d_0                     (distance limit)
  Constraint 2 (2c): x(delta(v)) - 2*y_v = 0, for all v        (degree)
  Constraint 3 (2d): SEC - no disconnected subtours              (subtour elimination)
  Constraint 4 (2e): y_v - x_e >= 0, for v in V, e in delta(v)  (logical)
  Constraint 5 (2f): 0 <= y_v <= 1, for all v                   (vertex bounds)
  Constraint 6 (2g): 0 <= x_e <= 1, for all e                   (edge bounds)
  Constraint 7 (2h): y_o = 1                                    (depot visited)
  Constraint 8 (2i): x_e in Z, for all e                        (integrality)

  Constraint 9 (objective): the program's reported objective_value must
                            match the true objective sum_v(scores[v]*y_v)
                            recomputed from the solution variables; this
                            blocks score-exploits where the program returns
                            an inflated obj that the scorer would trust.
"""

import argparse
import json
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'tour_edges', 'visited_vertices')
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


def edge_key(u, v):
    return (min(u, v), max(u, v))


def record_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs, violation_amount):
    """Record a single violation instance."""
    violations_list.append(message)
    normalizer = max(abs(rhs), EPS)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(violation_amount / normalizer),
    })


def extract_solution(instance, solution):
    """
    Extract y (vertex) and x (edge) variables from either efficient or gurobi
    solution format.
    Returns: y dict {v: value}, x dict {(u,v): value} with u < v
    """
    n = instance["num_nodes"]
    y = {v: 0 for v in range(n)}
    x = {}

    if "tour" in solution:
        # Efficient algorithm format: tour is ordered vertex list forming a cycle
        tour = solution["tour"]
        for v in tour:
            y[v] = 1
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            ek = edge_key(u, v)
            x[ek] = x.get(ek, 0) + 1
    elif "visited_vertices" in solution and "tour_edges" in solution:
        # Gurobi format
        for v in solution["visited_vertices"]:
            y[v] = 1
        for e in solution["tour_edges"]:
            ek = edge_key(e[0], e[1])
            x[ek] = x.get(ek, 0) + 1
    else:
        raise ValueError("Unrecognized solution format")

    return y, x


def check_feasibility(instance, solution):
    n = instance["num_nodes"]
    depot = instance["depot_vertex"]
    d0 = instance["distance_limitation_d0"]
    dist = instance["distance_matrix"]

    y, x = extract_solution(instance, solution)

    violated_constraints = set()
    violations = []
    magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 1 (2b): sum_e d_e * x_e <= d_0
    # ------------------------------------------------------------------
    total_dist = 0.0
    for (u, v), x_val in x.items():
        total_dist += dist[u][v] * x_val
    violation_amount = total_dist - d0
    if violation_amount > TOL:
        violated_constraints.add(1)
        record_violation(violations, magnitudes, 1,
                         f"Distance limit exceeded: total distance {total_dist} > limit {d0}",
                         total_dist, d0, violation_amount)

    # ------------------------------------------------------------------
    # Constraint 2 (2c): x(delta(v)) - 2*y_v = 0 for all v
    # ------------------------------------------------------------------
    for v in range(n):
        degree_sum = 0.0
        for (u, w), x_val in x.items():
            if u == v or w == v:
                degree_sum += x_val
        lhs = degree_sum - 2 * y[v]
        rhs = 0.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            violated_constraints.add(2)
            record_violation(violations, magnitudes, 2,
                             f"Degree constraint violated at vertex {v}: "
                             f"x(delta({v}))={degree_sum}, 2*y_{v}={2*y[v]}",
                             lhs, rhs, violation_amount)

    # ------------------------------------------------------------------
    # Constraint 3 (2d): Subtour elimination - solution edges must form
    # a single connected cycle containing the depot.
    # SEC: x(delta(H)) - 2*y_l - 2*y_r >= -2 for l in H, r in V\H
    # Violated when there exist disconnected subtours.
    # ------------------------------------------------------------------
    # Build adjacency from edges with x_e > 0
    adj = defaultdict(set)
    active_vertices = set()
    for (u, v), x_val in x.items():
        if x_val > TOL:
            adj[u].add(v)
            adj[v].add(u)
            active_vertices.add(u)
            active_vertices.add(v)

    # Find connected components among active vertices
    visited_set = set()
    components = []
    for start in active_vertices:
        if start in visited_set:
            continue
        component = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited_set:
                continue
            visited_set.add(node)
            component.add(node)
            for nb in adj[node]:
                if nb not in visited_set:
                    stack.append(nb)
        components.append(component)

    # Check for subtours not containing depot
    for comp in components:
        if depot in comp:
            continue
        if len(comp) < 2:
            continue
        # This component H forms a subtour disconnected from depot.
        # Pick any l in H and any r not in H.
        # x(delta(H)) = 0 (disconnected), so LHS = 0 - 2*y_l - 2*y_r
        # For l in H with y_l = 1, and r = depot with y_r = 1:
        # LHS = 0 - 2 - 2 = -4, RHS = -2, violation = -2 - (-4) = 2
        l_node = next(iter(comp))
        r_node = depot
        lhs_val = 0 - 2 * y[l_node] - 2 * y[r_node]
        rhs_val = -2.0
        violation_amount = rhs_val - lhs_val  # >= constraint: RHS - LHS when violated
        if violation_amount > TOL:
            violated_constraints.add(3)
            record_violation(violations, magnitudes, 3,
                             f"Subtour elimination violated: disconnected component "
                             f"{sorted(comp)} not connected to depot {depot}",
                             lhs_val, rhs_val, violation_amount)

    # ------------------------------------------------------------------
    # Constraint 4 (2e): y_v - x_e >= 0 for all v in V, e in delta(v)
    # ------------------------------------------------------------------
    for (u, v), x_val in x.items():
        # Check for endpoint u
        lhs_u = y[u] - x_val
        if -lhs_u > TOL:  # y_u - x_e < 0
            violation_amount = -lhs_u
            violated_constraints.add(4)
            record_violation(violations, magnitudes, 4,
                             f"Logical constraint violated: y_{u}={y[u]} < x_({u},{v})={x_val}",
                             y[u] - x_val, 0.0, violation_amount)
        # Check for endpoint v
        lhs_v = y[v] - x_val
        if -lhs_v > TOL:
            violation_amount = -lhs_v
            violated_constraints.add(4)
            record_violation(violations, magnitudes, 4,
                             f"Logical constraint violated: y_{v}={y[v]} < x_({u},{v})={x_val}",
                             y[v] - x_val, 0.0, violation_amount)

    # ------------------------------------------------------------------
    # Constraint 5 (2f): 0 <= y_v <= 1 for all v
    # ------------------------------------------------------------------
    for v in range(n):
        if y[v] < -TOL:
            violation_amount = -y[v]
            violated_constraints.add(5)
            record_violation(violations, magnitudes, 5,
                             f"Vertex bound violated: y_{v}={y[v]} < 0",
                             y[v], 0.0, violation_amount)
        elif y[v] > 1 + TOL:
            violation_amount = y[v] - 1
            violated_constraints.add(5)
            record_violation(violations, magnitudes, 5,
                             f"Vertex bound violated: y_{v}={y[v]} > 1",
                             y[v], 1.0, violation_amount)

    # ------------------------------------------------------------------
    # Constraint 6 (2g): 0 <= x_e <= 1 for all e
    # ------------------------------------------------------------------
    for (u, v), x_val in x.items():
        if x_val < -TOL:
            violation_amount = -x_val
            violated_constraints.add(6)
            record_violation(violations, magnitudes, 6,
                             f"Edge bound violated: x_({u},{v})={x_val} < 0",
                             x_val, 0.0, violation_amount)
        elif x_val > 1 + TOL:
            violation_amount = x_val - 1
            violated_constraints.add(6)
            record_violation(violations, magnitudes, 6,
                             f"Edge bound violated: x_({u},{v})={x_val} > 1",
                             x_val, 1.0, violation_amount)

    # ------------------------------------------------------------------
    # Constraint 7 (2h): y_o = 1
    # ------------------------------------------------------------------
    y_depot = y[depot]
    violation_amount = abs(y_depot - 1)
    if violation_amount > TOL:
        violated_constraints.add(7)
        record_violation(violations, magnitudes, 7,
                         f"Depot constraint violated: y_{depot}={y_depot} != 1",
                         y_depot, 1.0, violation_amount)

    # ------------------------------------------------------------------
    # Constraint 8 (2i): x_e in Z (integer) for all e
    # ------------------------------------------------------------------
    for (u, v), x_val in x.items():
        frac = abs(x_val - round(x_val))
        if frac > TOL:
            violated_constraints.add(8)
            rhs_val = round(x_val)
            record_violation(violations, magnitudes, 8,
                             f"Integrality violated: x_({u},{v})={x_val} is not integer",
                             x_val, rhs_val, frac)

    # ------------------------------------------------------------------
    # Constraint 9 (objective consistency):
    #   The program's reported ``objective_value`` must match the true OP
    #   objective recomputed from the solution variables:
    #     true_obj = sum_v(scores[v] * y_v)
    #   A mismatch means the program is *lying* about how good its solution
    #   is -- a score-gaming exploit (e.g. returning obj=1.8e308 or obj=128141
    #   while the actual route collects only ~30 points). Reject as infeasible
    #   so the scorer doesn't credit the fake objective.
    # ------------------------------------------------------------------
    scores = instance.get("scores")
    reported_obj = solution.get("objective_value")
    if scores is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(int(scores[v]) * y[v] for v in range(n)))
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance or 1e-3 absolute, whichever is larger
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(9)
                record_violation(
                    violations, magnitudes, 9,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_v(scores[v]*y_v)="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={tol:.3g})",
                    reported, true_obj, abs_diff,
                )

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Orienteering Problem (Kobeaga et al. 2024)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file")
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
