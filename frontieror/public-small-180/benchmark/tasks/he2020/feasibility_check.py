#!/usr/bin/env python3
"""
Feasibility checker for VRPTW-CNC (Vehicle Routing Problem with Time Windows
and Convex Node Costs).

Checks all hard constraints from the mathematical formulation:
  Constraint 1 (1a): Time window feasibility
  Constraint 2 (1b): Travel time precedence
  Constraint 3:      Vehicle capacity
  Constraint 4 (3b): Each customer visited exactly once (set partitioning)
  Constraint 5 (3c): Fleet size limit
  Constraint 6 (3d): Non-negativity and integrality of route usage
  Constraint 7:      Objective consistency (Tier C defense). Reported
                     objective_value must equal the recomputed total cost
                     (routing cost over all arcs + sum of customer
                     inconvenience costs rho*(t_i - preferred_time_i)^2)
                     within a small tolerance.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'routes', 'schedule')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('routes', 'schedule')
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


TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    n = instance["num_customers"]
    K = instance["num_vehicles"]
    Q = instance["vehicle_capacity"]
    dist = instance["distance_matrix"]  # (n+2) x (n+2)
    depot_end_id = instance["depot_end"]["id"]  # n+1

    # Build vertex info: time windows, demands
    tw = {}  # vertex_id -> (e, l)
    demand = {}  # vertex_id -> q
    service = {}  # vertex_id -> service duration

    # Depot start
    tw[0] = tuple(instance["depot"]["time_window"])
    demand[0] = 0
    service[0] = 0.0

    # Customer-level data for obj recomputation
    preferred = {}  # cid -> preferred service time

    # Customers
    for c in instance["customers"]:
        cid = c["id"]
        tw[cid] = tuple(c["time_window"])
        demand[cid] = c["demand"]
        service[cid] = float(c.get("service_time", instance.get("service_time", 0.0)))
        preferred[cid] = float(c["preferred_time"])

    # Depot end
    tw[depot_end_id] = tuple(instance["depot_end"]["time_window"])
    demand[depot_end_id] = 0
    service[depot_end_id] = 0.0

    routes = solution["routes"]
    schedule = solution["schedule"]
    # Schedule keys are strings; convert to int->float mapping
    sched = {int(k): float(v) for k, v in schedule.items()}

    violations = []
    violation_magnitudes = []
    violated_set = set()

    def record(constraint_idx, msg, lhs, rhs, raw_excess):
        violated_set.add(constraint_idx)
        violations.append(msg)
        normalizer = max(abs(rhs), EPS)
        ratio = raw_excess / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": ratio,
        })

    expected_schedule_ids = set(range(1, n + 1))
    if set(sched) != expected_schedule_ids:
        record(
            1,
            "Schedule must contain exactly one service start for every customer; "
            f"missing={sorted(expected_schedule_ids - set(sched))}, "
            f"unexpected={sorted(set(sched) - expected_schedule_ids)}",
            float(len(sched)),
            float(n),
            float(max(1, len(set(sched) ^ expected_schedule_ids))),
        )

    # ------------------------------------------------------------------
    # Constraint 4 (3b): Each customer visited exactly once
    # ------------------------------------------------------------------
    visit_count = {}
    for ridx, route in enumerate(routes):
        for node in route:
            if node == 0 or node == depot_end_id:
                continue
            visit_count[node] = visit_count.get(node, 0) + 1

    for cid in range(1, n + 1):
        cnt = visit_count.get(cid, 0)
        if abs(cnt - 1) > TOL:
            lhs_val = float(cnt)
            rhs_val = 1.0
            raw = abs(lhs_val - rhs_val)
            if cnt == 0:
                record(4, f"Customer {cid} is not visited by any route", lhs_val, rhs_val, raw)
            else:
                record(4, f"Customer {cid} is visited {cnt} times (expected exactly 1)", lhs_val, rhs_val, raw)

    # ------------------------------------------------------------------
    # Constraint 5 (3c): Fleet size — number of routes <= K
    # ------------------------------------------------------------------
    # Count non-empty routes (routes with at least one customer)
    num_routes_used = len(routes)
    if num_routes_used - K > TOL:
        lhs_val = float(num_routes_used)
        rhs_val = float(K)
        raw = lhs_val - rhs_val
        record(5, f"Number of routes ({num_routes_used}) exceeds fleet size K={K}", lhs_val, rhs_val, raw)

    # ------------------------------------------------------------------
    # Constraint 6 (3d): Integrality — each route used integer number of times
    # (In the solution format, each listed route is used once, so this is
    # inherently satisfied. We check for completeness.)
    # ------------------------------------------------------------------
    # Routes are listed explicitly; each is used exactly once (integer, >=0).
    # Nothing to violate here in the given solution format.

    # ------------------------------------------------------------------
    # Per-route constraints
    # ------------------------------------------------------------------
    for ridx, route in enumerate(routes):
        # Route: [0, c1, c2, ..., cm, depot_end_id]
        if len(route) < 2:
            continue

        customers_in_route = [v for v in route if v != 0 and v != depot_end_id]

        # --------------------------------------------------------------
        # Constraint 3: Capacity — sum of demands <= Q
        # --------------------------------------------------------------
        total_demand = sum(demand.get(v, 0) for v in route)
        if total_demand - Q > TOL:
            lhs_val = float(total_demand)
            rhs_val = float(Q)
            raw = lhs_val - rhs_val
            record(3, f"Route {ridx}: total demand {total_demand} exceeds capacity Q={Q}", lhs_val, rhs_val, raw)

        # Skip empty routes (depot-to-depot) for time-based checks
        if not customers_in_route:
            continue

        # Determine schedule times for all nodes in route
        # For customers: from the solution schedule
        # For depot start (0): pick latest feasible time, t_0 = min(l_0, t_first - tau)
        # For depot end (n+1): t_{n+1} = max(e_{n+1}, t_last + tau)

        route_times = {}
        for v in route:
            if v == 0 or v == depot_end_id:
                continue
            if v in sched:
                route_times[v] = sched[v]

        # Depot start time: set as late as possible while satisfying (1a) and (1b)
        first_cust = route[1]
        if first_cust != depot_end_id and first_cust in route_times:
            tau_0_first = dist[0][first_cust]
            t_0 = min(tw[0][1], route_times[first_cust] - tau_0_first)
            t_0 = max(tw[0][0], t_0)
        else:
            t_0 = tw[0][0]
        route_times[0] = t_0

        # Depot end time: set as early as possible
        last_cust = route[-2]
        if last_cust != 0 and last_cust in route_times:
            tau_last_end = dist[last_cust][depot_end_id]
            t_end = max(
                tw[depot_end_id][0],
                route_times[last_cust] + service[last_cust] + tau_last_end,
            )
        else:
            t_end = tw[depot_end_id][0]
        route_times[depot_end_id] = t_end

        # --------------------------------------------------------------
        # Constraint 1 (1a): Time windows — t_i in [e_i, l_i]
        # --------------------------------------------------------------
        for v in route:
            if v not in route_times:
                continue
            t_v = route_times[v]
            e_v, l_v = tw[v]
            # Check lower bound: t_v >= e_v
            if e_v - t_v > TOL:
                raw = e_v - t_v
                record(1, f"Route {ridx}, node {v}: service time {t_v:.4f} < earliest {e_v} (time window [{e_v}, {l_v}])",
                       t_v, e_v, raw)
            # Check upper bound: t_v <= l_v
            if t_v - l_v > TOL:
                raw = t_v - l_v
                record(1, f"Route {ridx}, node {v}: service time {t_v:.4f} > latest {l_v} (time window [{e_v}, {l_v}])",
                       t_v, l_v, raw)

        # --------------------------------------------------------------
        # Constraint 2 (1b): Travel time precedence
        #   t_{i-1} + service_{r_{i-1}} + tau_{r_{i-1}, r_i} <= t_i
        # --------------------------------------------------------------
        for idx in range(1, len(route)):
            prev = route[idx - 1]
            curr = route[idx]
            if prev not in route_times or curr not in route_times:
                continue
            tau = dist[prev][curr]
            t_prev = route_times[prev]
            t_curr = route_times[curr]
            lhs_val = t_prev + service[prev] + tau
            rhs_val = t_curr
            excess = lhs_val - rhs_val
            if excess > TOL:
                record(2, f"Route {ridx}, arc ({prev}->{curr}): arrival {lhs_val:.4f} > scheduled {rhs_val:.4f} (service {service[prev]:.1f} + travel {tau:.1f})",
                       lhs_val, rhs_val, excess)

    # ------------------------------------------------------------------
    # Constraint 7: Objective consistency (Tier C defense against
    # score-gaming exploits where reported objective_value is fabricated).
    #
    # Total cost = sum over arcs of routing cost
    #            + sum over customers of inconvenience cost
    #                f_i(t_i) = rho * (t_i - preferred_time_i)^2
    # (Depot inconvenience contributions are zero by definition.)
    #
    # All variables required by this formula (routes + per-customer
    # service start times) are present in the solution, so a full
    # recompute is possible. Skip the check only if the program did not
    # report an objective_value at all.
    # ------------------------------------------------------------------
    reported_obj_raw = solution.get("objective_value")
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and not math.isnan(reported_obj):
            rho = float(instance.get("inconvenience_cost_factor_rho", 0.0))

            routing_cost = 0.0
            for route in routes:
                for idx in range(1, len(route)):
                    u, v = route[idx - 1], route[idx]
                    try:
                        routing_cost += float(dist[u][v])
                    except (IndexError, TypeError):
                        # Out-of-range vertex (will be caught elsewhere);
                        # ignore for the obj recompute.
                        pass

            inconvenience_cost = 0.0
            for cid, t_v in sched.items():
                if cid in preferred:
                    inconvenience_cost += rho * (t_v - preferred[cid]) ** 2

            true_obj = routing_cost + inconvenience_cost
            abs_diff = abs(reported_obj - true_obj)
            tol = max(1e-6, 5e-6 * abs(true_obj))
            if abs_diff > tol:
                record(
                    7,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported_obj} differs from recomputed total cost "
                    f"(routing={routing_cost} + inconvenience={inconvenience_cost} "
                    f"= {true_obj}); |diff|={abs_diff:.3g}, tol={tol:.3g}",
                    reported_obj, true_obj, abs_diff,
                )

    # Build result
    violated_constraints = sorted(violated_set)
    feasible = len(violated_constraints) == 0

    # Deduplicate violation messages per constraint for summary
    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for VRPTW-CNC")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
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
        print("FEASIBLE: All constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
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
