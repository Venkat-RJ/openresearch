"""
Feasibility checker for the Steiner Tree-Star (STS) Problem.

Checks a candidate solution against all hard constraints from the mathematical
formulation in Lee, Chiu, Ryan (1996).

Constraints (numbered by formulation order):
  (1) sum_j x_{ij} = 1,  for each i in M              [assignment]
  (2) x_{ij} <= u_j,     for all i in M, j in N        [link to active]
  (3) y_{jk} <= u_j,     for j < k, j,k in N           [edge requires active j]
  (4) y_{jk} <= u_k,     for j < k, j,k in N           [edge requires active k]
  (5) sum y_{jk} = sum u_j - 1                          [spanning tree count]
  (6) subtour elimination (anti-cycle)                   [no cycles in Steiner tree]
  (7) x_{ij} >= 0,       for all i,j                    [non-negativity of x]
  (8) y_{jk} >= 0,       for j < k, j,k in N            [non-negativity of y]
  (9) u_j in {0,1},      for j in N                     [binary domain]
 (10) objective consistency: reported objective_value must equal the
      objective recomputed from the solution variables
      sum_{i,j} c_{ij} x_{ij} + sum_{j<k} d_{jk} y_{jk} + sum_j b_j u_j
      [Tier C defense against fabricated objective values]
"""

import argparse
import json
from collections import defaultdict, deque

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('active_steiner_nodes', 'assignments', 'objective_value', 'steiner_tree_edges')
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
    violated_constraints = set()
    violation_magnitudes = []

    m = instance["target_nodes"]["count"]
    n = instance["steiner_nodes"]["count"]
    c = instance["cost_matrices"]["assignment_costs"]["matrix"]
    d = instance["cost_matrices"]["interconnection_costs"]["matrix"]
    b = instance["steiner_nodes"]["node_weights"]

    # ----- Reconstruct decision variables from the solution -----
    # u_j: 1 if Steiner node j is active, 0 otherwise
    active_set = set(solution["active_steiner_nodes"])
    u = [1.0 if j in active_set else 0.0 for j in range(n)]

    # x_{ij}: 1 if target i assigned to Steiner j, 0 otherwise
    assignments = solution["assignments"]
    x = [[0.0] * n for _ in range(m)]
    for i_str, j_val in assignments.items():
        i = int(i_str)
        j = int(j_val)
        if 0 <= i < m and 0 <= j < n:
            x[i][j] = 1.0

    # y_{jk}: 1 if edge (j,k) in Steiner tree, 0 otherwise (j < k)
    edges = solution.get("steiner_tree_edges", [])
    y = {}
    for edge in edges:
        j, k = int(edge[0]), int(edge[1])
        jj, kk = min(j, k), max(j, k)
        y[(jj, kk)] = 1.0

    def record_violation(constraint_idx, lhs_val, rhs_val, message):
        if constraint_idx == 5 or constraint_idx == 1:
            # Equality: violation = |LHS - RHS|
            violation_amount = abs(lhs_val - rhs_val)
        elif constraint_idx in (3, 4, 2):
            # <= : violation = max(LHS - RHS, 0)
            violation_amount = max(lhs_val - rhs_val, 0.0)
        elif constraint_idx in (7, 8):
            # >= : violation = max(RHS - LHS, 0)
            violation_amount = max(rhs_val - lhs_val, 0.0)
        elif constraint_idx == 6:
            # <= : violation = max(LHS - RHS, 0)
            violation_amount = max(lhs_val - rhs_val, 0.0)
        elif constraint_idx == 9:
            # = (binary domain): violation = distance from {0,1}
            violation_amount = abs(lhs_val - rhs_val)
        elif constraint_idx == 10:
            # = (objective consistency): violation = |reported - recomputed|
            violation_amount = abs(lhs_val - rhs_val)
        else:
            violation_amount = abs(lhs_val - rhs_val)

        if violation_amount > tol:
            violated_constraints.add(constraint_idx)
            violations.append(message)
            normalizer = max(abs(rhs_val), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ===== Constraint (1): sum_j x_{ij} = 1 for each i in M =====
    for i in range(m):
        lhs = sum(x[i][j] for j in range(n))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            record_violation(1, lhs, rhs,
                             f"Constraint (1): Target {i} assigned to "
                             f"{lhs:.6f} Steiner nodes (should be exactly 1)")

    # ===== Constraint (2): x_{ij} <= u_j for all i, j =====
    for i in range(m):
        for j in range(n):
            lhs = x[i][j]
            rhs = u[j]
            if lhs - rhs > tol:
                record_violation(2, lhs, rhs,
                                 f"Constraint (2): Target {i} assigned to "
                                 f"inactive Steiner node {j} "
                                 f"(x[{i}][{j}]={lhs:.6f}, u[{j}]={rhs:.6f})")

    # ===== Constraint (3): y_{jk} <= u_j for j < k =====
    for (jj, kk), y_val in y.items():
        lhs = y_val
        rhs = u[jj]
        if lhs - rhs > tol:
            record_violation(3, lhs, rhs,
                             f"Constraint (3): Steiner edge ({jj},{kk}) "
                             f"present but Steiner node {jj} inactive "
                             f"(y={lhs:.6f}, u[{jj}]={rhs:.6f})")

    # ===== Constraint (4): y_{jk} <= u_k for j < k =====
    for (jj, kk), y_val in y.items():
        lhs = y_val
        rhs = u[kk]
        if lhs - rhs > tol:
            record_violation(4, lhs, rhs,
                             f"Constraint (4): Steiner edge ({jj},{kk}) "
                             f"present but Steiner node {kk} inactive "
                             f"(y={lhs:.6f}, u[{kk}]={rhs:.6f})")

    # ===== Constraint (5): sum y_{jk} = sum u_j - 1 =====
    lhs_5 = sum(y.values())
    rhs_5 = sum(u) - 1.0
    violation_5 = abs(lhs_5 - rhs_5)
    if violation_5 > tol:
        record_violation(5, lhs_5, rhs_5,
                         f"Constraint (5): Number of Steiner tree edges "
                         f"({lhs_5:.6f}) != number of active Steiner nodes "
                         f"minus 1 ({rhs_5:.6f})")

    # ===== Constraint (6): Anti-cycle (subtour elimination) =====
    # The active Steiner nodes connected by the y edges must form a tree
    # (connected and acyclic). We check:
    #   (a) connectivity among active Steiner nodes
    #   (b) no cycles (guaranteed if #edges = #active_nodes - 1 AND connected,
    #       but we also explicitly check subsets)
    num_active = int(sum(u))
    num_edges = int(round(lhs_5))

    if num_active >= 2:
        # Build adjacency for active Steiner nodes from y edges
        adj = defaultdict(set)
        for (jj, kk), y_val in y.items():
            if y_val > 0.5:
                adj[jj].add(kk)
                adj[kk].add(jj)

        # Check connectivity via BFS among active nodes
        active_list = [j for j in range(n) if u[j] > 0.5]
        if active_list:
            visited = set()
            queue = deque([active_list[0]])
            visited.add(active_list[0])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if nb not in visited and u[nb] > 0.5:
                        visited.add(nb)
                        queue.append(nb)

            # Check all active nodes are connected
            unreached = [j for j in active_list if j not in visited]
            if unreached:
                # Find connected components for the violation
                # Identify the subsets H that violate subtour elimination
                # For each disconnected component, the edges within it
                # exceed |component| - 1 relative to the full tree constraint
                # Report as a subtour/connectivity violation.
                all_visited = set()
                components = []
                for start in active_list:
                    if start in all_visited:
                        continue
                    comp = set()
                    q = deque([start])
                    comp.add(start)
                    while q:
                        nd = q.popleft()
                        for nb in adj[nd]:
                            if nb not in comp and u[nb] > 0.5:
                                comp.add(nb)
                                q.append(nb)
                    components.append(comp)
                    all_visited |= comp

                if len(components) > 1:
                    # For each component H (size >= 2), check:
                    # sum_{j<k in H} y_{jk} <= sum_{j in H\{ell}} u_j
                    # Equivalently, edges in H <= |H| - 1
                    for comp in components:
                        if len(comp) < 2:
                            continue
                        comp_list = sorted(comp)
                        edges_in_comp = 0
                        for (jj, kk), y_val in y.items():
                            if y_val > 0.5 and jj in comp and kk in comp:
                                edges_in_comp += 1
                        # For a connected component, edges_in_comp should be
                        # <= |comp| - 1. But since the overall tree is
                        # disconnected, the constraint (6) is violated for
                        # a subset H that spans components.
                        # Report the disconnected tree as violating (6).
                    # Report once overall
                    comp_sizes = [len(c) for c in components]
                    # Use the full active set as H and pick ell from the
                    # smaller component. The violation:
                    # sum y in H = num_edges, sum u in H\{ell} = num_active-1
                    # The subtour violation for a proper subset:
                    for comp in components:
                        if len(comp) >= 2:
                            comp_list = sorted(comp)
                            edges_in_comp = 0.0
                            for (jj, kk), y_val in y.items():
                                if y_val > 0.5 and jj in comp and kk in comp:
                                    edges_in_comp += y_val
                            ell = comp_list[0]
                            rhs_6 = sum(u[j] for j in comp_list if j != ell)
                            lhs_6 = edges_in_comp
                            # This component is a tree internally, but the
                            # disconnection means another component also has
                            # edges = |comp2| - 1, and total edges >
                            # total_active - 1 is caught by constraint (5).
                            # However, constraint (6) is about subsets.
                            # If components are disconnected, pick subset H
                            # as one component plus one node from another.
                    # Simplest: report disconnected active Steiner graph
                    # Use the largest component as H
                    largest_comp = max(components, key=len)
                    other_nodes = active_set - largest_comp
                    if other_nodes:
                        # Pick H = largest_comp union {one node from other}
                        extra_node = min(other_nodes)
                        H = sorted(largest_comp | {extra_node})
                        ell = extra_node
                        edges_in_H = 0.0
                        for (jj, kk), y_val in y.items():
                            if y_val > 0.5 and jj in set(H) and kk in set(H):
                                edges_in_H += y_val
                        rhs_6 = sum(u[j] for j in H if j != ell)
                        lhs_6 = edges_in_H
                        if lhs_6 - rhs_6 > tol:
                            record_violation(6, lhs_6, rhs_6,
                                             f"Constraint (6): Steiner tree is "
                                             f"disconnected ({len(components)} "
                                             f"components among active nodes). "
                                             f"Subtour elimination violated for "
                                             f"H={H}, ell={ell}")

        # Also check for cycles explicitly.
        # If edges among active nodes > |active| - 1, there's a cycle.
        edges_among_active = 0
        for (jj, kk), y_val in y.items():
            if y_val > 0.5 and u[jj] > 0.5 and u[kk] > 0.5:
                edges_among_active += 1
        if edges_among_active > num_active - 1:
            # There is a cycle. Find a cycle subset H.
            # Use DFS to find a cycle
            visited_dfs = set()
            parent_dfs = {}
            cycle_H = None
            for start in active_list:
                if start in visited_dfs:
                    continue
                stack = [(start, -1)]
                while stack:
                    nd, par = stack.pop()
                    if nd in visited_dfs:
                        # Found a cycle
                        # Trace back to find H
                        cycle_H = {nd}
                        cur = par
                        while cur != nd and cur != -1:
                            cycle_H.add(cur)
                            cur = parent_dfs.get(cur, -1)
                        if cur == nd:
                            cycle_H.add(nd)
                        break
                    visited_dfs.add(nd)
                    parent_dfs[nd] = par
                    for nb in adj[nd]:
                        if nb != par and u[nb] > 0.5:
                            stack.append((nb, nd))
                if cycle_H and len(cycle_H) >= 3:
                    break

            if cycle_H and len(cycle_H) >= 3:
                H = sorted(cycle_H)
                ell = H[0]
                edges_in_H = 0.0
                H_set = set(H)
                for (jj, kk), y_val in y.items():
                    if y_val > 0.5 and jj in H_set and kk in H_set:
                        edges_in_H += y_val
                rhs_6 = sum(u[j] for j in H if j != ell)
                lhs_6 = edges_in_H
                if lhs_6 - rhs_6 > tol:
                    record_violation(6, lhs_6, rhs_6,
                                     f"Constraint (6): Cycle detected among "
                                     f"Steiner nodes. Subtour elimination "
                                     f"violated for H={H}, ell={ell}")

    # ===== Constraint (7): x_{ij} >= 0 =====
    for i in range(m):
        for j in range(n):
            lhs = x[i][j]
            rhs = 0.0
            if rhs - lhs > tol:
                record_violation(7, lhs, rhs,
                                 f"Constraint (7): x[{i}][{j}]={lhs:.6f} < 0")

    # ===== Constraint (8): y_{jk} >= 0 =====
    for (jj, kk), y_val in y.items():
        lhs = y_val
        rhs = 0.0
        if rhs - lhs > tol:
            record_violation(8, lhs, rhs,
                             f"Constraint (8): y[{jj}][{kk}]={lhs:.6f} < 0")

    # ===== Constraint (9): u_j in {0, 1} =====
    for j in range(n):
        u_val = u[j]
        # Check that u_val is 0 or 1
        dist_to_binary = min(abs(u_val - 0.0), abs(u_val - 1.0))
        if dist_to_binary > tol:
            closest = 0.0 if abs(u_val) < abs(u_val - 1.0) else 1.0
            record_violation(9, u_val, closest,
                             f"Constraint (9): u[{j}]={u_val:.6f} is not "
                             f"binary (closest binary value: {closest:.0f})")

    # ===== Constraint (10): Objective consistency (Tier C) =====
    # Recompute the total network cost from the solution variables and reject
    # any solution whose reported objective_value disagrees. Every variable
    # the objective depends on (u_j, x_{ij}, y_{jk}) is present in the
    # solution, so a full recompute is exact.
    #   obj = sum_{i,j} c_{ij} x_{ij} + sum_{j<k} d_{jk} y_{jk} + sum_j b_j u_j
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            assignment_cost = sum(
                c[i][j] * x[i][j] for i in range(m) for j in range(n)
            )
            edge_cost = sum(
                d[jj][kk] * y_val for (jj, kk), y_val in y.items()
            )
            activation_cost = sum(b[j] * u[j] for j in range(n))
            true_obj = assignment_cost + edge_cost + activation_cost
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor; the objective
            # is a sum of Euclidean distances plus integer node weights.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(10, reported, true_obj,
                                 f"Constraint (10): Objective consistency "
                                 f"violated: reported objective_value="
                                 f"{reported} differs from recomputed total "
                                 f"cost (assignment {assignment_cost:.6f} + "
                                 f"edges {edge_cost:.6f} + activation "
                                 f"{activation_cost:.6f} = {true_obj:.6f}); "
                                 f"|diff|={abs_diff:.6g}, tol={obj_tol:.6g}")

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Steiner Tree-Star (STS) Problem."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON candidate solution file."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON feasibility result."
    )
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
        print(f"FEASIBLE: No constraint violations found.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} "
              f"constraint(s) violated: {result['violated_constraints']}")
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
