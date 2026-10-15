"""
Feasibility checker for the Capacitated Arc-Routing Problem (CARP).
Based on the Two-Index Formulation (Equations 1-6) from:
Bode & Irnich (2012), "Cut-First Branch-and-Price-Second for the CARP",
Operations Research 60(5):1167-1182.

Constraints checked (numbered as in the paper):
  (2) Partitioning: each required edge is serviced exactly once
  (3) Subtour-elimination / connectivity: each vehicle route is connected to the depot
  (4) Parity: even degree at every node for each vehicle
  (5) Capacity: total demand per vehicle does not exceed Q
  (6) Integrality / domain: x^k_e in {0,1}, y^k_e in Z+, p^k_i in Z+
  (7) Objective consistency: reported objective_value must equal
      sum_{k,e} c^serv_e * x^k_e + sum_{k,e} c_e * y^k_e
      (Tier C defense against LLM exploits that report bogus objective values
       like 0 or sys.float_info.max while constraints technically pass.)
"""

import json
import argparse
import math
import heapq
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'routes')
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


def build_graph(instance):
    """Build graph data structures from the instance."""
    num_nodes = instance["graph"]["num_nodes"]
    depot = instance["depot"]
    capacity = instance["fleet"]["vehicle_capacity"]
    num_vehicles = instance["fleet"]["num_vehicles"]

    edges = {}
    required_edge_ids = []
    adj = defaultdict(list)

    for e in instance["edges"]:
        eid = e["edge_id"]
        edges[eid] = {
            "endpoints": tuple(e["endpoints"]),
            "cost": e["cost"],
            "is_required": e["is_required"],
            "demand": e["demand"],
            "service_cost": e["service_cost"],
        }
        u, v = e["endpoints"]
        adj[u].append((v, eid))
        adj[v].append((u, eid))
        if e["is_required"]:
            required_edge_ids.append(eid)

    return {
        "num_nodes": num_nodes,
        "depot": depot,
        "capacity": capacity,
        "num_vehicles": num_vehicles,
        "edges": edges,
        "required_edge_ids": required_edge_ids,
        "adj": adj,
    }


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # Handle no-solution cases (solver timeout, infeasible, etc.)
    obj = solution.get("objective_value")
    status_str = solution.get("status_str", "")
    if (obj is None or obj == float("inf") or
            status_str in ("NO_SOLUTION_FOUND", "INFEASIBLE", "INF_OR_UNBD")):
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": [
                f"No solution to check (status={status_str!r}, objective_value={obj!r})"
            ],
            "violation_magnitudes": [],
        }

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    graph = build_graph(instance)
    depot = graph["depot"]
    capacity = graph["capacity"]
    num_vehicles = graph["num_vehicles"]
    edges = graph["edges"]
    required_edge_ids = graph["required_edge_ids"]
    adj = graph["adj"]
    num_nodes = graph["num_nodes"]

    routes = solution.get("routes", [])

    # =========================================================================
    # Constraint (2): Partitioning — each required edge serviced exactly once
    # sum_{k in K} x^k_e = 1  for all e in E_R
    # =========================================================================
    service_count = defaultdict(int)
    for route in routes:
        for eid in route.get("serviced_edges", []):
            service_count[eid] += 1

    for eid in required_edge_ids:
        lhs = service_count[eid]
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraints_set.add(2)
            u, v = edges[eid]["endpoints"]
            if lhs == 0:
                violations.append(
                    f"Constraint (2): Required edge {eid} ({u}-{v}) is not serviced by any vehicle"
                )
            else:
                violations.append(
                    f"Constraint (2): Required edge {eid} ({u}-{v}) is serviced {lhs} times instead of exactly once"
                )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # Also check that no non-required edge is serviced
    for route in routes:
        for eid in route.get("serviced_edges", []):
            if eid not in edges:
                violated_constraints_set.add(2)
                violations.append(
                    f"Constraint (2): Edge {eid} does not exist in the instance"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": 1.0,
                    "rhs": 0.0,
                    "raw_excess": 1.0,
                    "normalizer": eps,
                    "ratio": 1.0 / eps,
                })
            elif not edges[eid]["is_required"]:
                violated_constraints_set.add(2)
                u, v = edges[eid]["endpoints"]
                violations.append(
                    f"Constraint (2): Non-required edge {eid} ({u}-{v}) is serviced"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": 1.0,
                    "rhs": 0.0,
                    "raw_excess": 1.0,
                    "normalizer": eps,
                    "ratio": 1.0 / eps,
                })

    # =========================================================================
    # Constraint (3): Subtour-elimination / connectivity
    # x^k(delta_R(S)) + y^k(delta(S)) >= 2 x^k_f
    #   for all S subset V\{d}, f in E_R(S), k in K
    #
    # Practical check: for each vehicle with serviced edges, the multigraph
    # formed by its serviced + deadheaded edges must be connected AND must
    # include the depot. This is equivalent to ensuring no subtour disconnected
    # from the depot exists.
    # =========================================================================
    for k, route in enumerate(routes):
        serviced = route.get("serviced_edges", [])
        if not serviced:
            continue

        # Build the multigraph for this vehicle
        route_adj = defaultdict(set)
        # Add serviced edges
        for eid in serviced:
            if eid in edges:
                u, v = edges[eid]["endpoints"]
                route_adj[u].add(v)
                route_adj[v].add(u)
        # Add deadheaded edges
        for dh in route.get("deadheaded_edges", []):
            dh_eid = dh["edge_id"]
            if dh_eid in edges and dh.get("times", 0) > 0:
                u, v = edges[dh_eid]["endpoints"]
                route_adj[u].add(v)
                route_adj[v].add(u)

        # All nodes that appear in this vehicle's route
        route_nodes = set()
        for eid in serviced:
            if eid in edges:
                u, v = edges[eid]["endpoints"]
                route_nodes.add(u)
                route_nodes.add(v)
        for dh in route.get("deadheaded_edges", []):
            dh_eid = dh["edge_id"]
            if dh_eid in edges and dh.get("times", 0) > 0:
                u, v = edges[dh_eid]["endpoints"]
                route_nodes.add(u)
                route_nodes.add(v)

        # Check connectivity from depot
        if depot not in route_nodes:
            # Vehicle has serviced edges but route does not visit depot
            violated_constraints_set.add(3)
            violations.append(
                f"Constraint (3): Vehicle {k} has serviced edges but its route does not include the depot (node {depot})"
            )
            # Use a violation magnitude based on the number of serviced edges
            lhs_val = 0.0
            rhs_val = 2.0
            raw = rhs_val - lhs_val
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": raw,
                "normalizer": normalizer,
                "ratio": raw / normalizer,
            })
        else:
            # BFS from depot
            visited = set()
            queue = [depot]
            visited.add(depot)
            while queue:
                node = queue.pop(0)
                for nb in route_adj[node]:
                    if nb not in visited and nb in route_nodes:
                        visited.add(nb)
                        queue.append(nb)

            unreachable = route_nodes - visited
            if unreachable:
                # Find which serviced edges are in disconnected components
                disconnected_serviced = []
                for eid in serviced:
                    if eid in edges:
                        u, v = edges[eid]["endpoints"]
                        if u in unreachable or v in unreachable:
                            disconnected_serviced.append(eid)

                if disconnected_serviced:
                    violated_constraints_set.add(3)
                    violations.append(
                        f"Constraint (3): Vehicle {k} has a disconnected subtour; "
                        f"serviced edges {disconnected_serviced} are not connected to the depot"
                    )
                    lhs_val = 0.0
                    rhs_val = 2.0
                    raw = rhs_val - lhs_val
                    normalizer = max(abs(rhs_val), eps)
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": raw,
                        "normalizer": normalizer,
                        "ratio": raw / normalizer,
                    })

    # =========================================================================
    # Constraint (4): Parity — even degree at every node for each vehicle
    # x^k(delta_R(i)) + y^k(delta(i)) = 2 p^k_i   for all i in V, k in K
    #
    # This means the total edge traversals (service + deadhead) incident to
    # each node must be even for each vehicle.
    # =========================================================================
    for k, route in enumerate(routes):
        serviced = route.get("serviced_edges", [])
        deadheaded = route.get("deadheaded_edges", [])

        # Count degree at each node: serviced edges count 1 traversal each,
        # deadheaded edges count 'times' traversals each
        node_degree = defaultdict(int)
        for eid in serviced:
            if eid in edges:
                u, v = edges[eid]["endpoints"]
                node_degree[u] += 1
                node_degree[v] += 1
        for dh in deadheaded:
            dh_eid = dh["edge_id"]
            times = dh.get("times", 0)
            if dh_eid in edges and times > 0:
                u, v = edges[dh_eid]["endpoints"]
                node_degree[u] += times
                node_degree[v] += times

        for node, deg in node_degree.items():
            if deg % 2 != 0:
                lhs = float(deg)
                # rhs should be an even number; the nearest even number
                rhs = float(2 * round(deg / 2))
                violation_amount = abs(lhs - rhs)
                if violation_amount > tol:
                    violated_constraints_set.add(4)
                    violations.append(
                        f"Constraint (4): Vehicle {k}, node {node} has odd degree {deg} "
                        f"(parity violated)"
                    )
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
    # Constraint (5): Capacity — q^T x^k <= Q for all k in K
    # Total demand of serviced edges per vehicle must not exceed capacity.
    # =========================================================================
    for k, route in enumerate(routes):
        serviced = route.get("serviced_edges", [])
        total_demand = 0.0
        for eid in serviced:
            if eid in edges:
                total_demand += edges[eid]["demand"]

        lhs = total_demand
        rhs = float(capacity)
        violation_amount = max(lhs - rhs, 0.0)
        if violation_amount > tol:
            violated_constraints_set.add(5)
            violations.append(
                f"Constraint (5): Vehicle {k} demand {lhs} exceeds capacity {rhs} "
                f"(excess {violation_amount})"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (6): Integrality and domain
    # x^k in {0,1}^{|E_R|}, y^k in Z+^{|E|}, p^k in Z+^{|V|}
    #
    # Check:
    # - x^k_e values are 0 or 1 (each required edge serviced at most once per vehicle)
    # - y^k_e values are nonneg integers (deadhead counts)
    # - p^k_i values are nonneg integers (implied by parity check above)
    # =========================================================================
    for k, route in enumerate(routes):
        # Check x^k: each required edge should be serviced at most once per vehicle
        serviced = route.get("serviced_edges", [])
        edge_counts_per_vehicle = defaultdict(int)
        for eid in serviced:
            edge_counts_per_vehicle[eid] += 1

        for eid, count in edge_counts_per_vehicle.items():
            if count > 1:
                lhs = float(count)
                rhs = 1.0
                violation_amount = abs(lhs - rhs)
                if violation_amount > tol:
                    violated_constraints_set.add(6)
                    if eid in edges:
                        u, v = edges[eid]["endpoints"]
                        violations.append(
                            f"Constraint (6): Vehicle {k} services edge {eid} ({u}-{v}) "
                            f"{count} times (x^k_e must be binary)"
                        )
                    else:
                        violations.append(
                            f"Constraint (6): Vehicle {k} services unknown edge {eid} "
                            f"{count} times (x^k_e must be binary)"
                        )
                    normalizer = max(abs(rhs), eps)
                    violation_magnitudes.append({
                        "constraint": 6,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": violation_amount,
                        "normalizer": normalizer,
                        "ratio": violation_amount / normalizer,
                    })

        # Check y^k: deadhead counts must be nonneg integers
        for dh in route.get("deadheaded_edges", []):
            times = dh.get("times", 0)
            if not isinstance(times, int) or times < 0:
                lhs = float(times)
                rhs = 0.0
                violation_amount = max(-lhs, 0.0) if isinstance(times, (int, float)) else 1.0
                if violation_amount > tol or not isinstance(times, int):
                    violated_constraints_set.add(6)
                    violations.append(
                        f"Constraint (6): Vehicle {k}, deadheaded edge {dh.get('edge_id', '?')} "
                        f"has invalid times={times} (must be nonneg integer)"
                    )
                    normalizer = max(abs(rhs), eps)
                    violation_magnitudes.append({
                        "constraint": 6,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": max(violation_amount, 1.0),
                        "normalizer": normalizer,
                        "ratio": max(violation_amount, 1.0) / normalizer,
                    })

    # =========================================================================
    # Constraint (7): Objective consistency (Tier C exploit defense)
    # Recompute the objective from the solution variables and reject if it
    # disagrees with the reported objective_value. The CARP objective (1) is
    #   obj = sum_{k,e in E_R} c^serv_e * x^k_e + sum_{k,e in E} c_e * y^k_e
    # which from the solution is
    #   sum over routes of (sum of service_cost over serviced_edges
    #                       + sum of cost*times over deadheaded_edges).
    # =========================================================================
    try:
        reported_f = float(obj)
    except (TypeError, ValueError):
        reported_f = None

    if reported_f is not None and math.isfinite(reported_f):
        recomputed = 0.0
        for route in routes:
            for eid in route.get("serviced_edges", []):
                if eid in edges:
                    recomputed += float(edges[eid]["service_cost"])
            for dh in route.get("deadheaded_edges", []):
                dh_eid = dh["edge_id"]
                times = dh.get("times", 0)
                if dh_eid in edges and isinstance(times, (int, float)) and times > 0:
                    recomputed += float(edges[dh_eid]["cost"]) * float(times)

        # 0.1% relative tolerance with 1e-3 absolute floor. CARP costs are
        # typically integers, so this is comfortably tight without being flaky
        # on Gurobi's small float roundoff.
        obj_tol = max(1e-3, 1e-3 * abs(recomputed))
        abs_diff = abs(reported_f - recomputed)
        if abs_diff > obj_tol:
            violated_constraints_set.add(7)
            violations.append(
                f"Constraint (7): Reported objective_value={reported_f} differs from "
                f"recomputed sum_k(c^serv*x^k + c*y^k)={recomputed} "
                f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
            )
            normalizer = max(abs(recomputed), eps)
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": float(reported_f),
                "rhs": float(recomputed),
                "raw_excess": float(abs_diff),
                "normalizer": float(normalizer),
                "ratio": float(abs_diff / normalizer),
            })
    elif reported_f is None or not math.isfinite(reported_f):
        # Non-numeric or non-finite reported obj is itself an objective
        # consistency violation.
        violated_constraints_set.add(7)
        violations.append(
            f"Constraint (7): Reported objective_value={obj!r} is not a finite number"
        )
        violation_magnitudes.append({
            "constraint": 7,
            "lhs": float("nan"),
            "rhs": 0.0,
            "raw_excess": 1.0,
            "normalizer": eps,
            "ratio": 1.0 / eps,
        })

    # =========================================================================
    # Build result
    # =========================================================================
    violated_list = sorted(violated_constraints_set)
    feasible = len(violated_list) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_list,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CARP solutions (Bode & Irnich 2012)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON solution file"
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON feasibility result"
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

    if result["feasible"] is None:
        print(f"NO SOLUTION - {result['violations'][0] if result['violations'] else 'No solution to check'}")
    elif result["feasible"]:
        print(f"FEASIBLE - No constraint violations found.")
    else:
        print(f"INFEASIBLE - Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
