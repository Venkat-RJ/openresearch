#!/usr/bin/env python3
"""
Feasibility checker for the Symmetric Generalized Traveling Salesman Problem
(E-GTSP variant: exactly one node per cluster).

Source: Fischetti, Salazar González, Toth (1997), Operations Research 45(3):378-394

Hard constraints checked (numbered top-to-bottom from the formulation section):
  Constraint 1 — (1.2): Degree constraint
      sum_{e in delta(v)} x_e = 2 * y_v   for all v in N
  Constraint 3 — (1.4): Subtour elimination / connectivity
      The selected edges on visited nodes must form a single connected tour
      (i.e., no subtours among visited nodes).
  Constraint 4 — (1.5): Binary x
      x_e in {0, 1}                       for all e in E
  Constraint 5 — (1.6): Binary y
      y_v in {0, 1}                       for all v in N
  Constraint 6 — (1.7): Exactly-one per cluster (E-GTSP replacement for 1.3)
      sum_{v in C_h} y_v = 1              for h = 1, ..., m
  Constraint 7 — (1.1): Objective-value consistency
      reported objective_value == recomputed tour cost sum_{e in E} c_e * x_e.
      The tour fully determines x_e, so the true objective is the total cost
      of the closed tour; a reported value that disagrees is rejected
      (Tier C defense against fabricated objective values).

NOTE: This is the obj-recompute (`_new`) variant. It is a strict superset of
feasibility_check.py — it adds Constraint 7 and changes no other verdict.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'tour', 'visited_nodes')
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



def load_instance(path):
    """Load instance from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    n = data["problem_size"]["n"]
    m = data["problem_size"]["m"]

    cluster_of = {}
    clusters = {}
    for c in data["clusters"]:
        cid = c["id"]
        clusters[cid] = list(c["node_ids"])
        for nid in c["node_ids"]:
            cluster_of[nid] = cid

    vals = data["distance_matrix_upper_triangular"]["values"]
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
            dist[i][j] = vals[idx]
            dist[j][i] = vals[idx]

    return n, m, clusters, cluster_of, dist


def load_solution(path):
    """Load candidate solution from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance_path, solution_path):
    """
    Check all hard constraints of the E-GTSP formulation.

    The candidate solution provides a tour (ordered list of visited nodes).
    From the tour we derive:
      - y_v = 1 if v is in visited_nodes, 0 otherwise
      - x_e = 1 for each consecutive edge in the cyclic tour, 0 otherwise
    """
    tol = 1e-5
    eps = 1e-5

    n, m, clusters, cluster_of, dist = load_instance(instance_path)
    sol = load_solution(solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
    if _frontieror_contract_result is not None:
        return _frontieror_contract_result

    tour = sol["tour"]
    visited_nodes = sol.get("visited_nodes", tour)

    # Derive y_v from visited_nodes
    visited_set = set(visited_nodes)
    y = [0] * n
    for v in visited_set:
        if 0 <= v < n:
            y[v] = 1

    # Derive x_e from the tour (cyclic: edge between consecutive nodes, last wraps to first)
    # x_e is keyed by canonical edge (min, max)
    x = {}
    tour_len = len(tour)
    for k in range(tour_len):
        i = tour[k]
        j = tour[(k + 1) % tour_len]
        ek = (min(i, j), max(i, j))
        x[ek] = x.get(ek, 0) + 1

    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 — (1.2): sum_{e in delta(v)} x_e = 2 * y_v  for all v in N
    # =========================================================================
    for v in range(n):
        # Compute degree of v in the edge set x
        deg = 0
        for (i, j), val in x.items():
            if i == v or j == v:
                deg += val
        lhs = deg
        rhs = 2 * y[v]
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 1 (degree): node {v} has degree {lhs} but 2*y_{v}={rhs}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # =========================================================================
    # Constraint 3 — (1.4): Subtour elimination (connectivity)
    #   The tour edges on visited nodes must form a single Hamiltonian cycle.
    #   Equivalently, for every proper subset S of visited nodes with
    #   2 <= |S| <= |visited|-1, sum_{e in delta(S)} x_e >= 2.
    #   We check this via connectivity: the tour graph must be connected on
    #   visited nodes.
    # =========================================================================
    if tour_len >= 2:
        # Build adjacency from tour edges
        adj = {}
        for v in visited_set:
            adj[v] = set()
        for k in range(tour_len):
            i = tour[k]
            j = tour[(k + 1) % tour_len]
            if i in adj:
                adj[i].add(j)
            if j in adj:
                adj[j].add(i)

        # BFS from first visited node
        start = tour[0]
        seen = {start}
        queue = [start]
        while queue:
            node = queue.pop()
            for nb in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)

        # If not all visited nodes reachable, we have subtours
        unreachable = visited_set - seen
        if unreachable:
            # Find the actual subtours for reporting
            # Compute the minimum violated cut: take S = seen (a proper subset)
            # delta(S) crossing edges
            S = seen
            lhs_cut = 0
            for (i, j), val in x.items():
                if (i in S) != (j in S):
                    lhs_cut += val
            rhs_cut = 2  # required by subtour elimination
            violation_amount = max(rhs_cut - lhs_cut, 0)
            if violation_amount <= tol:
                # The subtour case: the component is disconnected,
                # so delta(S) for the smaller component has 0 crossing edges
                lhs_cut = 0
                violation_amount = float(rhs_cut)
            normalizer = max(abs(rhs_cut), eps)
            violations.append(
                f"Constraint 3 (subtour elimination): tour is disconnected; "
                f"{len(unreachable)} node(s) unreachable from tour start "
                f"(subtour components exist)"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(lhs_cut),
                "rhs": float(rhs_cut),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # =========================================================================
    # Constraint 4 — (1.5): x_e in {0, 1}  for all e in E
    # =========================================================================
    for (i, j), val in x.items():
        if abs(val - round(val)) > tol or round(val) < 0 or round(val) > 1:
            lhs = float(val)
            # closest bound
            if val > 1:
                rhs = 1.0
                violation_amount = val - 1.0
            elif val < 0:
                rhs = 0.0
                violation_amount = -val
            else:
                rhs = round(val)
                violation_amount = abs(val - rhs)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 4 (binary x): edge ({i},{j}) has x={val}, "
                f"not in {{0,1}}"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # Also check for multi-edges (edge used more than once means x > 1)
    for (i, j), val in x.items():
        if val > 1 + tol:
            lhs = float(val)
            rhs = 1.0
            violation_amount = val - 1.0
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 4 (binary x): edge ({i},{j}) used {val} times "
                f"in tour, exceeds binary bound"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # =========================================================================
    # Constraint 5 — (1.6): y_v in {0, 1}  for all v in N
    # =========================================================================
    for v in range(n):
        yv = y[v]
        if abs(yv - round(yv)) > tol or round(yv) < 0 or round(yv) > 1:
            lhs = float(yv)
            rhs = float(round(yv))
            violation_amount = abs(yv - rhs)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 5 (binary y): node {v} has y={yv}, not in {{0,1}}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # =========================================================================
    # Constraint 6 — (1.7): sum_{v in C_h} y_v = 1  for h = 1, ..., m (E-GTSP)
    # =========================================================================
    for cid in sorted(clusters.keys()):
        cnodes = clusters[cid]
        lhs = sum(y[v] for v in cnodes)
        rhs = 1
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 6 (exactly one per cluster): cluster {cid} has "
                f"sum(y)={lhs}, needs = {rhs}"
            )
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # =========================================================================
    # Additional structural checks on tour validity
    # =========================================================================
    # Check that all tour nodes are valid node IDs
    for v in tour:
        if v < 0 or v >= n:
            violations.append(
                f"Constraint 5 (binary y): tour contains invalid node {v} "
                f"(not in 0..{n-1})"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })

    # Check no intra-cluster edges (E-GTSP removes them)
    for k in range(tour_len):
        i = tour[k]
        j = tour[(k + 1) % tour_len]
        if 0 <= i < n and 0 <= j < n and cluster_of.get(i) == cluster_of.get(j):
            lhs = 1.0
            rhs = 0.0
            violation_amount = 1.0
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 4 (binary x): intra-cluster edge ({i},{j}) used "
                f"in tour (both in cluster {cluster_of[i]}); E-GTSP forbids "
                f"intra-cluster edges"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # Check tour visits exactly m nodes (one per cluster)
    if tour_len != m:
        lhs = float(tour_len)
        rhs = float(m)
        violation_amount = abs(tour_len - m)
        normalizer = max(abs(rhs), eps)
        violations.append(
            f"Constraint 6 (exactly one per cluster): tour has {tour_len} "
            f"nodes but there are {m} clusters"
        )
        violation_magnitudes.append({
            "constraint": 6,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    # Check no duplicate nodes in tour
    if len(set(tour)) != tour_len:
        from collections import Counter
        counts = Counter(tour)
        dups = {v: c for v, c in counts.items() if c > 1}
        for v, c in dups.items():
            lhs = float(c)
            rhs = 1.0
            violation_amount = c - 1.0
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 1 (degree): node {v} visited {c} times in tour "
                f"(should be at most once)"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # =========================================================================
    # Constraint 7 — (1.1): Objective-value consistency  [obj-recompute / Tier C]
    #   v(GTSP) = sum_{e in E} c_e * x_e. The tour fully determines x_e (every
    #   variable the objective depends on is present in the solution), so the
    #   true objective is the exact total cost of the closed tour. Reject a
    #   solution whose self-reported objective_value disagrees with the
    #   recomputed tour cost — this blocks LLM score-gaming exploits that
    #   fabricate the objective (e.g. obj=0 or obj=sys.float_info.max) while
    #   keeping the route technically feasible.
    #
    #   This is an ADDITIVE check: it only ever appends Constraint 7. It never
    #   touches the verdict of Constraints 1/3/4/5/6 above.
    # =========================================================================
    reported_obj = sol.get("objective_value")
    # Recompute only when the tour is structurally usable: at least 2 nodes and
    # every node a valid index into the distance matrix. Invalid node IDs are
    # already flagged by Constraint 5; guarding here avoids an IndexError and
    # keeps this check from masking or duplicating that verdict.
    tour_nodes_valid = all(isinstance(v, int) and 0 <= v < n for v in tour)
    if reported_obj is not None and tour_len >= 2 and tour_nodes_valid:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Recompute the exact closed-tour cost from x_e (the tour edges).
            true_obj = 0
            for k in range(tour_len):
                i = tour[k]
                j = tour[(k + 1) % tour_len]
                true_obj += dist[i][j]
            if math.isfinite(reported):
                abs_diff = abs(reported - true_obj)
            else:
                # A non-finite reported objective can never equal a finite
                # integer tour cost.
                abs_diff = float("inf")
            # Edge costs c_e are integers (see math_model.txt), so the true
            # tour cost is an exact integer. A 0.5 floor catches any
            # integer-magnitude misreport while tolerating solver float noise
            # (~1e-4 observed in the Gurobi reference solutions).
            tol_obj = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > tol_obj:
                lhs = float(reported) if math.isfinite(reported) else reported
                rhs = float(true_obj)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Constraint 7 (objective consistency): reported "
                    f"objective_value={reported} differs from recomputed "
                    f"tour cost sum_e(c_e*x_e)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": float(lhs) if math.isfinite(reported) else float("inf"),
                    "rhs": float(rhs),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer),
                })

    # Build result
    violated_constraint_indices = sorted(
        set(vm["constraint"] for vm in violation_magnitudes)
    )
    feasible = len(violations) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraint_indices,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for E-GTSP (Fischetti et al. 1997)"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the data instance",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the candidate solution",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to write the JSON file containing the feasibility result",
    )
    args = parser.parse_args()

    result = check_feasibility(args.instance_path, args.solution_path)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: {len(result['violations'])} violation(s) found.")
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
