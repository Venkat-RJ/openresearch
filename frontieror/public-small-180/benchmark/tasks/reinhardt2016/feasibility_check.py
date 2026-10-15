#!/usr/bin/env python3
"""
Feasibility checker for the ESVRPTW (Edge Set Cost Vehicle Routing Problem
with Time Windows) using the Direct Path Model (DPM) formulation.

Based on: Reinhardt, Jepsen, & Pisinger (2016),
"The Edge Set Cost of the Vehicle Routing Problem with Time Windows",
Transportation Science, 50(2), 694-707.

Checks all DPM constraints (15)-(27) against a given solution.
"""

import json
import argparse
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('edge_sets_used', 'objective_value', 'routes')
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
    with open(path) as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-4
    violations = []

    # Handle null / infeasible solutions – when the solver found no solution,
    # feasibility is indeterminate (null), not false.
    status = solution.get("status", solution.get("status_name", ""))
    status_lower = str(status).lower() if status else ""
    has_no_solution = (
        solution.get("objective_value") is None
        or "infeasible" in status_lower
        or "no_solution" in status_lower
        or status in ("infeasible_or_no_solution", None)
    )
    if has_no_solution:
        return {
            "feasible": None,
            "violated_constraints": ["no_solution"],
            "violations": ["No feasible solution provided"],
            "objective_value": None,
            "recomputed_objective": None,
        }

    nodes = instance["nodes"]
    edge_sets = instance["edge_sets"]
    dist = instance["distance_matrix"]
    num_customers = instance["num_customers"]
    vehicle_capacity = instance["vehicle_capacity"]
    schedule_horizon = instance["schedule_horizon"]

    depot = 0
    customers = set(range(1, num_customers + 1))

    node_data = {}
    for nd in nodes:
        node_data[nd["id"]] = nd

    # Build edge set membership
    edge_set_arcs = {}
    edge_set_customers = {}
    edge_set_cost = {}
    for es in edge_sets:
        r = es["id"]
        edge_set_cost[r] = es["access_cost"]
        edge_set_arcs[r] = set()
        edge_set_customers[r] = set()
        for edge in es["edges"]:
            u, v = edge[0], edge[1]
            edge_set_arcs[r].add((u, v))
            edge_set_arcs[r].add((v, u))
            if u != depot:
                edge_set_customers[r].add(u)
            if v != depot:
                edge_set_customers[r].add(v)

    # Map arcs to edge sets
    arc_to_edge_set = {}
    for es in edge_sets:
        r = es["id"]
        for edge in es["edges"]:
            u, v = edge[0], edge[1]
            arc_to_edge_set[(u, v)] = r
            arc_to_edge_set[(v, u)] = r

    routes = solution.get("routes", [])
    edge_sets_used = set(solution.get("edge_sets_used", []))

    # Domain check: every node id in any route's sequence/arcs must exist
    # in node_data; every node key in visit_times must be a valid node;
    # every edge_sets_used id must exist in instance edge_sets.
    valid_node_ids = set(node_data.keys())
    valid_edge_set_ids = set(edge_set_arcs.keys())
    for ridx, route in enumerate(routes):
        for node in route.get("sequence", []):
            if node not in valid_node_ids:
                violations.append(
                    f"Domain: route {ridx} sequence node {node!r} "
                    f"not a valid node id"
                )
        for arc in route.get("arcs", []):
            for node in arc[:2]:
                if node not in valid_node_ids:
                    violations.append(
                        f"Domain: route {ridx} arc node {node!r} "
                        f"not a valid node id"
                    )
        for node_key in route.get("visit_times", {}):
            try:
                nid = int(node_key)
            except (ValueError, TypeError):
                nid = None
            if nid not in valid_node_ids:
                violations.append(
                    f"Domain: route {ridx} visit_times key {node_key!r} "
                    f"not a valid node id"
                )
    for r in edge_sets_used:
        if r not in valid_edge_set_ids:
            violations.append(
                f"Domain: edge_sets_used contains invalid id {r!r}"
            )

    # If domain violations exist, abort downstream checks that index into
    # node_data with potentially invalid ids.
    if violations:
        return {
            "feasible": False,
            "violated_constraints": ["domain"],
            "violations": violations,
            "objective_value": solution.get("objective_value"),
            "recomputed_objective": None,
        }

    # Check: number of used routes must not exceed number of vehicles |K|
    num_vehicles = instance.get("num_vehicles")
    if num_vehicles is not None and len(routes) > num_vehicles:
        violations.append(
            f"Number of used routes ({len(routes)}) exceeds number of vehicles ({num_vehicles})"
        )

    # Check: all customers visited exactly once
    visited_customers = set()
    for route in routes:
        seq = route.get("sequence", [])
        route_customers = [n for n in seq if n != depot]
        for c in route_customers:
            if c in visited_customers:
                violations.append(f"Customer {c} visited more than once")
            visited_customers.add(c)

    missing = customers - visited_customers
    if missing:
        violations.append(f"Customers not visited: {sorted(missing)}")

    extra = visited_customers - customers
    if extra:
        violations.append(f"Invalid nodes visited: {sorted(extra)}")

    # Check each route
    total_arc_cost = 0.0
    needed_edge_sets = set()

    for ridx, route in enumerate(routes):
        seq = route.get("sequence", [])
        route_customers = [n for n in seq if n != depot]

        # Route must start and end at depot
        if len(seq) < 2:
            violations.append(f"Route {ridx}: too short (length {len(seq)})")
            continue
        if seq[0] != depot:
            violations.append(f"Route {ridx}: does not start at depot")
        if seq[-1] != depot:
            violations.append(f"Route {ridx}: does not end at depot")

        # Capacity check
        total_demand = sum(node_data[c]["demand"] for c in route_customers)
        if total_demand > vehicle_capacity + tol:
            violations.append(
                f"Route {ridx}: demand {total_demand} exceeds capacity {vehicle_capacity}"
            )

        # Time window and travel time check
        current_time = node_data[depot]["time_window_start"]
        for k in range(len(seq) - 1):
            i, j = seq[k], seq[k + 1]
            travel_time = dist[i][j] + node_data[i]["service_time"]
            arrival = current_time + travel_time
            total_arc_cost += dist[i][j]

            # Track needed edge sets
            if (i, j) in arc_to_edge_set:
                needed_edge_sets.add(arc_to_edge_set[(i, j)])

            if j == depot:
                # Return to depot: check within schedule horizon
                if arrival > schedule_horizon + tol:
                    violations.append(
                        f"Route {ridx}: return to depot at {arrival:.2f} "
                        f"exceeds horizon {schedule_horizon}"
                    )
            else:
                # Check time window
                tw_start = node_data[j]["time_window_start"]
                tw_end = node_data[j]["time_window_end"]
                arrival = max(arrival, tw_start)  # wait if early
                if arrival > tw_end + tol:
                    violations.append(
                        f"Route {ridx}: arrival at node {j} at {arrival:.2f} "
                        f"exceeds time window [{tw_start}, {tw_end}]"
                    )
                current_time = arrival

    # Domain check on submitted visit_times t_i^v: each value must be
    # non-negative and lie within the node's time window [a_i, b_i] (paper
    # constraint 22). Lower bound is 0 (start of horizon); upper bound is
    # the per-node tw_end. We also flag negatives.
    for ridx, route in enumerate(routes):
        vt = route.get("visit_times", {}) if isinstance(route, dict) else {}
        for node_key, t_val in vt.items():
            try:
                node_id = int(node_key)
            except (ValueError, TypeError):
                node_id = node_key
            if not isinstance(t_val, (int, float)):
                violations.append(
                    f"Route {ridx}: visit_times[{node_key}]={t_val!r} "
                    f"is not numeric (domain violation)")
                continue
            if t_val < -tol:
                violations.append(
                    f"Route {ridx}: visit_times[{node_key}]={t_val} < 0 "
                    f"(non-negativity violated)")
            if node_id in node_data:
                tw_lo = node_data[node_id].get("time_window_start")
                tw_hi = node_data[node_id].get("time_window_end")
                if tw_lo is not None and t_val < tw_lo - tol:
                    violations.append(
                        f"Route {ridx}: visit_times[{node_key}]={t_val} < "
                        f"time_window_start {tw_lo} (Eq 22 lower-bound)")
                if tw_hi is not None and t_val > tw_hi + tol:
                    violations.append(
                        f"Route {ridx}: visit_times[{node_key}]={t_val} > "
                        f"time_window_end {tw_hi} (Eq 22 upper-bound)")

    # Check edge sets: all needed sets must be in edge_sets_used
    for r in needed_edge_sets:
        if r not in edge_sets_used:
            violations.append(
                f"Edge set {r} is needed but not marked as used"
            )

    # Recompute objective
    total_edge_set_cost = sum(edge_set_cost[r] for r in edge_sets_used)
    recomputed_obj = total_arc_cost + total_edge_set_cost

    reported_obj = solution.get("objective_value")
    if reported_obj is not None and abs(recomputed_obj - reported_obj) > 1.0:
        violations.append(
            f"Objective mismatch: reported {reported_obj}, "
            f"recomputed {recomputed_obj:.4f}"
        )

    violated_constraints = []
    if missing:
        violated_constraints.append("visit_all_customers")
    for v in violations:
        if "capacity" in v.lower():
            violated_constraints.append("capacity")
        if "time window" in v.lower() or "horizon" in v.lower():
            violated_constraints.append("time_windows")
        if "edge set" in v.lower():
            violated_constraints.append("edge_set_linking")
        if "more than once" in v.lower():
            violated_constraints.append("visit_once")

    violated_constraints = list(set(violated_constraints))

    return {
        "feasible": len(violations) == 0,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "objective_value": reported_obj,
        "recomputed_objective": round(recomputed_obj, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for ESVRPTW (Reinhardt et al. 2016, DPM)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"] is None:
        print("Solution is INDETERMINATE (no solution from solver).")
    elif result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print("Solution is INFEASIBLE.")
        print(f"Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
