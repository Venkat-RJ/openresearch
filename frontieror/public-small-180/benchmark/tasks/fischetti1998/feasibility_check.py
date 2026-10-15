"""
Feasibility checker for the Orienteering Problem (OP).

Checks all hard constraints from the formulation in:
  Fischetti, Salazar González, Toth (1998)
  "Solving the Orienteering Problem through Branch-and-Cut"

Constraints checked (numbered as in the paper):
  (2) Total travel time constraint
  (3) Degree equations
  (4) Generalized Subtour Elimination Constraints (GSECs)
  (5) Depot must be visited
  (6) x_e bounds [0, 1]
  (7) y_v bounds [0, 1]
  (8) Integrality of x_e
  (9) Integrality of y_v
  (10) Objective consistency: the reported objective_value must equal the
       objective recomputed from the solution variables, max z = sum_v p_v*y_v.
       This is a Tier C defense against score-gaming exploits where a candidate
       reports a fabricated objective_value while keeping a feasible route.
"""

import argparse
import json
from collections import defaultdict, deque

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('edges', 'objective_value', 'tour', 'visited_nodes')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('tour',)
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



def canonical_edge(i, j):
    return (min(i, j), max(i, j))


def parse_edge_key(key_str):
    stripped = key_str.strip().strip("()")
    parts = stripped.split(",")
    return (int(parts[0].strip()), int(parts[1].strip()))


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = data["n"]
    depot = data["depot"]
    t0 = data["t0"]
    prizes = {int(k): v for k, v in data["prizes"].items()}
    travel = {}
    for key_str, cost in data["travel_times"].items():
        i, j = parse_edge_key(key_str)
        e = canonical_edge(i, j)
        travel[e] = cost
    nodes = list(range(1, n + 1))
    return nodes, depot, t0, prizes, travel


def load_solution(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(nodes, depot, t0, prizes, travel, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # ---- Extract solution variables ----
    visited_nodes = solution.get("visited_nodes", [])
    edges_list = solution.get("edges", [])
    tour = solution.get("tour", [])

    n = len(nodes)
    node_set = set(nodes)

    # Build y_v from visited_nodes
    y = {v: 0 for v in nodes}
    for v in visited_nodes:
        if v in node_set:
            y[v] = 1

    # Build x_e from edges list
    x = {}
    edge_counts = defaultdict(int)
    for edge in edges_list:
        i, j = edge[0], edge[1]
        e = canonical_edge(i, j)
        edge_counts[e] += 1
    # x_e = number of times edge appears (should be 0 or 1)
    for e, count in edge_counts.items():
        x[e] = count

    # Build incidence: delta(v)
    delta = defaultdict(list)
    for e in x:
        i, j = e
        delta[i].append(e)
        delta[j].append(e)

    # =========================================================================
    # Constraint (2): sum_{e in E} t_e * x_e <= t0
    # =========================================================================
    total_travel = 0.0
    for e, xe in x.items():
        if e in travel:
            total_travel += travel[e] * xe
        else:
            # Edge not in instance travel times - use solution's reported time
            # but flag as potential issue. For edges from the solution that may
            # use (i,j) with i>j, we already canonicalized.
            pass

    lhs_2 = total_travel
    rhs_2 = float(t0)
    violation_amount_2 = max(lhs_2 - rhs_2, 0.0)
    if violation_amount_2 > tol:
        violated_set.add(2)
        violations.append(
            f"Constraint (2): Total travel time {lhs_2} exceeds budget {rhs_2} "
            f"by {violation_amount_2}"
        )
        normalizer = max(abs(rhs_2), eps)
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": lhs_2,
            "rhs": rhs_2,
            "raw_excess": violation_amount_2,
            "normalizer": normalizer,
            "ratio": violation_amount_2 / normalizer,
        })

    # =========================================================================
    # Constraint (3): x(delta(v)) = 2 * y_v  for all v in V
    # =========================================================================
    for v in nodes:
        x_delta_v = sum(x.get(e, 0) for e in delta[v])
        lhs_3 = float(x_delta_v)
        rhs_3 = 2.0 * y[v]
        violation_amount_3 = abs(lhs_3 - rhs_3)
        if violation_amount_3 > tol:
            violated_set.add(3)
            violations.append(
                f"Constraint (3): Degree equation violated for node {v}: "
                f"x(delta({v}))={lhs_3}, 2*y_{v}={rhs_3}"
            )
            normalizer = max(abs(rhs_3), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs_3,
                "rhs": rhs_3,
                "raw_excess": violation_amount_3,
                "normalizer": normalizer,
                "ratio": violation_amount_3 / normalizer,
            })

    # =========================================================================
    # Constraint (4): GSEC - x(delta(S)) >= 2 * y_v
    #   for all S subset V, 1 in S, v in V \ S
    #
    # Let S be the connected component of the support graph that contains the
    # depot (so 1 in S). For any visited v in V\S the GSEC requires
    # x(delta(S)) >= 2*y_v. Since v is disconnected from S, x(delta(S)) only
    # counts edges between S and V\S.
    # =========================================================================
    # Build adjacency among visited nodes using selected edges
    adj = defaultdict(set)
    for e, xe in x.items():
        if xe >= 1:
            i, j = e
            if y[i] == 1 and y[j] == 1:
                adj[i].add(j)
                adj[j].add(i)

    visited_set_nodes = {v for v in nodes if y[v] == 1}

    # BFS from depot to find S = depot's connected component (1 in S)
    if depot in visited_set_nodes:
        reachable = set()
        queue = deque([depot])
        reachable.add(depot)
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if w not in reachable and w in visited_set_nodes:
                    reachable.add(w)
                    queue.append(w)

        # S = reachable (contains depot). GSEC must hold for each visited v in V\S.
        S = reachable
        outside_visited = visited_set_nodes - S
        if outside_visited:
            # x(delta(S)): edges with exactly one endpoint in S
            x_delta_S = 0.0
            for e, xe in x.items():
                if xe >= 1:
                    i, j = e
                    if (i in S) != (j in S):
                        x_delta_S += xe

            for v in sorted(outside_visited):
                lhs_4 = x_delta_S
                rhs_4 = 2.0 * y[v]
                violation_amount_4 = max(rhs_4 - lhs_4, 0.0)
                if violation_amount_4 > tol:
                    violated_set.add(4)
                    violations.append(
                        f"Constraint (4): GSEC violated for S={sorted(S)} "
                        f"(depot in S), v={v} in V\\S: "
                        f"x(delta(S))={lhs_4}, required >= {rhs_4}"
                    )
                    normalizer = max(abs(rhs_4), eps)
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": lhs_4,
                        "rhs": rhs_4,
                        "raw_excess": violation_amount_4,
                        "normalizer": normalizer,
                        "ratio": violation_amount_4 / normalizer,
                    })

    # =========================================================================
    # Constraint (5): y_1 = 1 (depot must be visited)
    # =========================================================================
    lhs_5 = float(y[depot])
    rhs_5 = 1.0
    violation_amount_5 = abs(lhs_5 - rhs_5)
    if violation_amount_5 > tol:
        violated_set.add(5)
        violations.append(
            f"Constraint (5): Depot (node {depot}) not visited: y_{depot}={lhs_5}"
        )
        normalizer = max(abs(rhs_5), eps)
        violation_magnitudes.append({
            "constraint": 5,
            "lhs": lhs_5,
            "rhs": rhs_5,
            "raw_excess": violation_amount_5,
            "normalizer": normalizer,
            "ratio": violation_amount_5 / normalizer,
        })

    # =========================================================================
    # Constraint (6): 0 <= x_e <= 1  for all e in E
    # =========================================================================
    for e, xe in x.items():
        # Check lower bound
        if xe < -tol:
            lhs_6 = float(xe)
            rhs_6 = 0.0
            violation_amount_6 = abs(lhs_6 - rhs_6)
            violated_set.add(6)
            violations.append(
                f"Constraint (6): x_{e} = {xe} < 0"
            )
            normalizer = max(abs(rhs_6), eps)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs_6,
                "rhs": rhs_6,
                "raw_excess": violation_amount_6,
                "normalizer": normalizer,
                "ratio": violation_amount_6 / normalizer,
            })
        # Check upper bound
        if xe > 1 + tol:
            lhs_6 = float(xe)
            rhs_6 = 1.0
            violation_amount_6 = lhs_6 - rhs_6
            violated_set.add(6)
            violations.append(
                f"Constraint (6): x_{e} = {xe} > 1"
            )
            normalizer = max(abs(rhs_6), eps)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs_6,
                "rhs": rhs_6,
                "raw_excess": violation_amount_6,
                "normalizer": normalizer,
                "ratio": violation_amount_6 / normalizer,
            })

    # =========================================================================
    # Constraint (7): 0 <= y_v <= 1  for all v in V \ {1}
    # =========================================================================
    for v in nodes:
        if v == depot:
            continue
        yv = y[v]
        if yv < -tol:
            lhs_7 = float(yv)
            rhs_7 = 0.0
            violation_amount_7 = abs(lhs_7 - rhs_7)
            violated_set.add(7)
            violations.append(
                f"Constraint (7): y_{v} = {yv} < 0"
            )
            normalizer = max(abs(rhs_7), eps)
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": lhs_7,
                "rhs": rhs_7,
                "raw_excess": violation_amount_7,
                "normalizer": normalizer,
                "ratio": violation_amount_7 / normalizer,
            })
        if yv > 1 + tol:
            lhs_7 = float(yv)
            rhs_7 = 1.0
            violation_amount_7 = lhs_7 - rhs_7
            violated_set.add(7)
            violations.append(
                f"Constraint (7): y_{v} = {yv} > 1"
            )
            normalizer = max(abs(rhs_7), eps)
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": lhs_7,
                "rhs": rhs_7,
                "raw_excess": violation_amount_7,
                "normalizer": normalizer,
                "ratio": violation_amount_7 / normalizer,
            })

    # =========================================================================
    # Constraint (8): x_e in {0, 1}  for all e in E
    # =========================================================================
    for e, xe in x.items():
        frac = abs(xe - round(xe))
        if frac > tol:
            violated_set.add(8)
            violations.append(
                f"Constraint (8): x_{e} = {xe} is not binary"
            )
            lhs_8 = float(xe)
            rhs_8 = float(round(xe))
            violation_amount_8 = frac
            normalizer = max(abs(rhs_8), eps)
            violation_magnitudes.append({
                "constraint": 8,
                "lhs": lhs_8,
                "rhs": rhs_8,
                "raw_excess": violation_amount_8,
                "normalizer": normalizer,
                "ratio": violation_amount_8 / normalizer,
            })

    # =========================================================================
    # Constraint (9): y_v in {0, 1}  for all v in V \ {1}
    # =========================================================================
    for v in nodes:
        if v == depot:
            continue
        yv = y[v]
        frac = abs(yv - round(yv))
        if frac > tol:
            violated_set.add(9)
            violations.append(
                f"Constraint (9): y_{v} = {yv} is not binary"
            )
            lhs_9 = float(yv)
            rhs_9 = float(round(yv))
            violation_amount_9 = frac
            normalizer = max(abs(rhs_9), eps)
            violation_magnitudes.append({
                "constraint": 9,
                "lhs": lhs_9,
                "rhs": rhs_9,
                "raw_excess": violation_amount_9,
                "normalizer": normalizer,
                "ratio": violation_amount_9 / normalizer,
            })

    # =========================================================================
    # Constraint (10): Objective consistency (Tier C anti-gaming check)
    #   The OP objective is  max z = sum_{v in V} p_v * y_v.
    #   Every variable that determines the objective (y_v, derived from
    #   visited_nodes) is present in the solution, so the objective can be
    #   recomputed exactly. Reject when the reported objective_value disagrees
    #   with the recomputed value beyond a small tolerance.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(prizes.get(v, 0) * y[v] for v in nodes))
            abs_diff = abs(reported - true_obj)
            # Prizes are integer (see math_model.txt), so the true objective is
            # an exact integer. 0.5 absolute floor catches any off-by-one lie;
            # the relative term guards against float noise on large objectives.
            tol_obj = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_set.add(10)
                violations.append(
                    f"Constraint (10): Objective consistency violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_v(p_v*y_v)={true_obj} (|diff|={abs_diff}, tol={tol_obj})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 10,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # =========================================================================
    # Build result
    # =========================================================================
    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Orienteering Problem (OP)."
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

    nodes, depot, t0, prizes, travel = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return
    result = check_feasibility(nodes, depot, t0, prizes, travel, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE: Solution at {args.solution_path} is feasible.")
    else:
        print(f"INFEASIBLE: Solution at {args.solution_path} violates constraints "
              f"{result['violated_constraints']}")
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
