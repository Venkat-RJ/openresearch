"""
Feasibility checker for the Time Window Assignment Vehicle Routing Problem (TWAVRP).

Checks all hard constraints from the mathematical formulation in Spliet & Gabor (2014).

Constraints (numbered top-to-bottom from the formulation section of math_model.txt):
  Constraint 1: Objective function (not a constraint to check)
  Constraint 2: Each customer visited exactly once per scenario
  Constraint 3: Service time >= endogenous TW start (y_v)
  Constraint 4: Service time <= endogenous TW end (y_v + w_v)
  Constraint 5: Binary route variables (x_r^omega in {0,1})
  Constraint 6: Endogenous TW start domain (y_v in [s_v, e_v - w_v])
  Constraint 7: Objective consistency (reported objective_value must equal the
                probability-weighted sum of route arc costs recomputed from the
                solution's routes and the instance distance matrix).

Additionally, route feasibility conditions from the model are checked as implicit
hard constraints embedded within the route definitions:
  - (i)   Capacity constraint per route per scenario
  - (ii)  Exogenous time window constraints
  - (iii) Travel time consistency (arrival at j >= departure from i + travel time)
These are checked as sub-parts of Constraint 5 (a route variable being 1 means
the route must be feasible).
"""

import argparse
import json
import math

TOL = 1e-5
EPS_NORM = 1e-5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_violation(constraint, lhs, rhs, raw_excess, description):
    normalizer = max(abs(rhs), EPS_NORM)
    ratio = raw_excess / normalizer
    return {
        "constraint": constraint,
        "lhs": lhs,
        "rhs": rhs,
        "raw_excess": raw_excess,
        "normalizer": normalizer,
        "ratio": ratio,
        "description": description,
    }


def check_feasibility(instance, solution):
    violations = []
    violation_magnitudes = []
    violated_set = set()

    schema_errors = []
    if not isinstance(solution, dict):
        schema_errors.append("solution must be a JSON object")
    else:
        if "objective_value" not in solution or solution["objective_value"] is None:
            schema_errors.append("missing required solution field: objective_value")
        objective = solution.get("objective_value")
        if (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(float(objective))
        ):
            schema_errors.append("objective_value must be a finite number")
        for field in ("routes", "time_windows"):
            if (
                field not in solution
                or not isinstance(solution[field], dict)
                or not solution[field]
            ):
                schema_errors.append(f"{field} must be a non-empty JSON object")
    if schema_errors:
        return {
            "feasible": False,
            "violated_constraints": ["solution_schema"],
            "violations": schema_errors,
            "violation_magnitudes": [],
        }

    n = instance["num_customers"]
    depot_node = 0
    end_depot = n + 1
    customers = {c["id"]: c for c in instance["customers"]}
    scenarios = instance["scenarios"]
    n_scenarios = len(scenarios)
    dist = instance["distance_matrix"]
    Q = instance["vehicle_capacity"]

    # Parse solution
    time_windows = solution["time_windows"]  # str(v) -> [y_v, y_v + w_v]
    routes_by_scenario = solution["routes"]  # str(scenario_1based) -> list of route dicts

    # Extract y_v values (endogenous TW start) and w_v
    y = {}
    for c in instance["customers"]:
        v = c["id"]
        sv = str(v)
        if sv in time_windows:
            y[v] = time_windows[sv][0]
        else:
            y[v] = None

    w = {}
    for c in instance["customers"]:
        v = c["id"]
        w[v] = c["endogenous_time_window_width"]

    # Build exogenous time windows
    exo_tw = {}
    exo_tw[0] = instance["depot"]["exogenous_time_window"]
    for c in instance["customers"]:
        exo_tw[c["id"]] = c["exogenous_time_window"]
    exo_tw[end_depot] = instance["depot"]["exogenous_time_window"]

    # =========================================================================
    # Constraint 6: y_v in [s_v, e_v - w_v] for all v in V'
    # This is two bounds:
    #   6a: y_v >= s_v   (>= constraint)
    #   6b: y_v <= e_v - w_v  (<= constraint)
    # =========================================================================
    for c in instance["customers"]:
        v = c["id"]
        sv_exo = c["exogenous_time_window"][0]  # s_v
        ev_exo = c["exogenous_time_window"][1]  # e_v
        wv = c["endogenous_time_window_width"]  # w_v
        yv = y[v]

        if yv is None:
            violations.append(f"Customer {v}: endogenous TW start y_{v} not specified in solution")
            violated_set.add(6)
            violation_magnitudes.append(make_violation(6, 0, sv_exo, sv_exo, f"Customer {v}: y_{v} missing"))
            continue

        # 6a: y_v >= s_v
        excess_6a = sv_exo - yv  # how much RHS exceeds LHS for >= constraint
        if excess_6a > TOL:
            violations.append(
                f"Constraint 6: Customer {v} endogenous TW start y_{v}={yv:.4f} < s_{v}={sv_exo}"
            )
            violated_set.add(6)
            violation_magnitudes.append(make_violation(6, yv, sv_exo, excess_6a,
                f"Customer {v}: y_{v}={yv:.4f} < s_{v}={sv_exo}"))

        # 6b: y_v <= e_v - w_v
        upper = ev_exo - wv
        excess_6b = yv - upper  # how much LHS exceeds RHS for <= constraint
        if excess_6b > TOL:
            violations.append(
                f"Constraint 6: Customer {v} endogenous TW start y_{v}={yv:.4f} > e_{v}-w_{v}={upper:.4f}"
            )
            violated_set.add(6)
            violation_magnitudes.append(make_violation(6, yv, upper, excess_6b,
                f"Customer {v}: y_{v}={yv:.4f} > e_{v}-w_{v}={upper:.4f}"))

    # =========================================================================
    # Per-scenario checks
    # =========================================================================
    for om_idx in range(n_scenarios):
        scenario = scenarios[om_idx]
        om_key = str(om_idx + 1)  # scenarios are 1-indexed in solution
        demands = scenario["demands"]  # list indexed by customer_index (0-based for customers 1..n)

        # Get demand map: customer_id -> demand
        dem = {}
        for ci, c in enumerate(instance["customers"]):
            dem[c["id"]] = demands[ci]

        # Get routes for this scenario
        if om_key not in routes_by_scenario:
            # No routes for this scenario - all customers unvisited
            for c in instance["customers"]:
                v = c["id"]
                violations.append(
                    f"Constraint 2: Customer {v} not visited in scenario {om_idx+1} (no routes)"
                )
                violated_set.add(2)
                violation_magnitudes.append(make_violation(2, 0, 1, 1.0,
                    f"Customer {v} visit count=0 in scenario {om_idx+1}"))
            continue

        sc_routes = routes_by_scenario[om_key]

        # Count visits per customer and collect service times
        visit_count = {c["id"]: 0 for c in instance["customers"]}
        service_time_sum = {c["id"]: 0.0 for c in instance["customers"]}

        for ri, route_info in enumerate(sc_routes):
            route = route_info["route"]  # list of node ids: [0, ..., n+1]
            arrival_times = route_info["arrival_times"]  # str(node_id) -> arrival time

            # =====================================================================
            # Constraint 5 (route feasibility): Check that each used route is feasible
            # A route x_r^omega = 1 must be a valid route satisfying conditions (i)-(iii)
            # =====================================================================

            # Identify customer nodes (exclude depot 0 and end depot n+1)
            cust_in_route = [node for node in route if 1 <= node <= n]

            # --- Condition (i): Capacity constraint ---
            route_load = sum(dem.get(node, 0) for node in cust_in_route)
            excess_cap = route_load - Q  # LHS - RHS for <= constraint
            if excess_cap > TOL:
                violations.append(
                    f"Constraint 5 (capacity): Scenario {om_idx+1} route {ri+1} "
                    f"load={route_load} exceeds capacity Q={Q}"
                )
                violated_set.add(5)
                violation_magnitudes.append(make_violation(5, route_load, Q, excess_cap,
                    f"Scenario {om_idx+1} route {ri+1}: load={route_load} > Q={Q}"))

            # --- Condition (ii): Exogenous time window constraints ---
            for node in route:
                if node == 0 or node == end_depot:
                    continue
                at_key = str(node)
                if at_key not in arrival_times:
                    continue
                arr = arrival_times[at_key]
                s_exo, e_exo = exo_tw[node]

                # arr >= s_v (exogenous)
                excess_early = s_exo - arr
                if excess_early > TOL:
                    violations.append(
                        f"Constraint 5 (exogenous TW): Scenario {om_idx+1} route {ri+1} "
                        f"customer {node} arrival={arr:.4f} < exogenous start={s_exo}"
                    )
                    violated_set.add(5)
                    violation_magnitudes.append(make_violation(5, arr, s_exo, excess_early,
                        f"Scenario {om_idx+1} route {ri+1} customer {node}: "
                        f"arrival={arr:.4f} < exo_start={s_exo}"))

                # arr <= e_v (exogenous)
                excess_late = arr - e_exo
                if excess_late > TOL:
                    violations.append(
                        f"Constraint 5 (exogenous TW): Scenario {om_idx+1} route {ri+1} "
                        f"customer {node} arrival={arr:.4f} > exogenous end={e_exo}"
                    )
                    violated_set.add(5)
                    violation_magnitudes.append(make_violation(5, arr, e_exo, excess_late,
                        f"Scenario {om_idx+1} route {ri+1} customer {node}: "
                        f"arrival={arr:.4f} > exo_end={e_exo}"))

            # --- Condition (iii): Travel time consistency ---
            # Service at j >= service at i + t_{ij} when j visited directly after i
            for k in range(len(route) - 1):
                i_node = route[k]
                j_node = route[k + 1]
                at_i_key = str(i_node)
                at_j_key = str(j_node)

                if at_i_key in arrival_times and at_j_key in arrival_times:
                    arr_i = arrival_times[at_i_key]
                    arr_j = arrival_times[at_j_key]
                    travel = dist[i_node][j_node]
                    earliest_j = arr_i + travel

                    # arr_j >= arr_i + t_{ij}  (>= constraint)
                    excess_travel = earliest_j - arr_j
                    if excess_travel > TOL:
                        violations.append(
                            f"Constraint 5 (travel time): Scenario {om_idx+1} route {ri+1} "
                            f"arc ({i_node}->{j_node}): arrival at {j_node}={arr_j:.4f} "
                            f"< departure from {i_node} ({arr_i:.4f}) + travel ({travel:.4f}) "
                            f"= {earliest_j:.4f}"
                        )
                        violated_set.add(5)
                        violation_magnitudes.append(make_violation(5, arr_j, earliest_j, excess_travel,
                            f"Scenario {om_idx+1} route {ri+1}: "
                            f"arr({j_node})={arr_j:.4f} < arr({i_node})+t={earliest_j:.4f}"))

            # Accumulate visit counts and service times for constraint 2/3/4 checks
            for node in cust_in_route:
                visit_count[node] += 1
                at_key = str(node)
                if at_key in arrival_times:
                    service_time_sum[node] += arrival_times[at_key]

        # =====================================================================
        # Constraint 2: Each customer visited exactly once per scenario
        # sum_r a_r^v x_r^omega = 1 for all v in V', omega in Omega
        # =====================================================================
        for c in instance["customers"]:
            v = c["id"]
            count = visit_count[v]
            rhs = 1
            violation_amount = abs(count - rhs)
            if violation_amount > TOL:
                violations.append(
                    f"Constraint 2: Customer {v} visited {count} time(s) in scenario {om_idx+1}"
                )
                violated_set.add(2)
                violation_magnitudes.append(make_violation(2, count, rhs, violation_amount,
                    f"Customer {v} visit_count={count} in scenario {om_idx+1}"))

        # =====================================================================
        # Constraint 3: Service time >= y_v
        # sum_r t_r^v x_r^omega >= y_v for all v in V', omega in Omega
        # =====================================================================
        for c in instance["customers"]:
            v = c["id"]
            if y[v] is None:
                continue
            if visit_count[v] == 0:
                continue  # already caught by constraint 2
            lhs = service_time_sum[v]
            rhs_val = y[v]
            # >= constraint: violation if RHS > LHS
            excess = rhs_val - lhs
            if excess > TOL:
                violations.append(
                    f"Constraint 3: Scenario {om_idx+1} customer {v} "
                    f"service_time={lhs:.4f} < y_{v}={rhs_val:.4f}"
                )
                violated_set.add(3)
                violation_magnitudes.append(make_violation(3, lhs, rhs_val, excess,
                    f"Scenario {om_idx+1} customer {v}: service_time={lhs:.4f} < y_{v}={rhs_val:.4f}"))

        # =====================================================================
        # Constraint 4: Service time <= y_v + w_v
        # sum_r t_r^v x_r^omega <= y_v + w_v for all v in V', omega in Omega
        # =====================================================================
        for c in instance["customers"]:
            v = c["id"]
            if y[v] is None:
                continue
            if visit_count[v] == 0:
                continue  # already caught by constraint 2
            lhs = service_time_sum[v]
            rhs_val = y[v] + w[v]
            # <= constraint: violation if LHS > RHS
            excess = lhs - rhs_val
            if excess > TOL:
                violations.append(
                    f"Constraint 4: Scenario {om_idx+1} customer {v} "
                    f"service_time={lhs:.4f} > y_{v}+w_{v}={rhs_val:.4f}"
                )
                violated_set.add(4)
                violation_magnitudes.append(make_violation(4, lhs, rhs_val, excess,
                    f"Scenario {om_idx+1} customer {v}: service_time={lhs:.4f} > y_{v}+w_{v}={rhs_val:.4f}"))

    # =========================================================================
    # Constraint 7: Objective consistency (Tier C defense).
    # Recompute obj = sum_omega p_omega * sum_r c_r where c_r is the sum of arc
    # costs along route r in scenario omega (from instance distance matrix).
    # All variables required (routes per scenario, probabilities, distance
    # matrix) are present in the solution + instance, so a full recompute is
    # possible. Reject when the reported objective_value differs beyond a 0.1%
    # relative tolerance with a 1e-3 absolute floor.
    # =========================================================================
    reported_obj_raw = solution.get("objective_value")
    try:
        reported_obj = float(reported_obj_raw) if reported_obj_raw is not None else None
    except (TypeError, ValueError):
        reported_obj = None

    if reported_obj is not None and not (math.isnan(reported_obj) or math.isinf(reported_obj)):
        true_obj = 0.0
        for scenario in scenarios:
            sc_id = scenario["scenario_id"]
            p_omega = float(scenario["probability"])
            sc_routes = routes_by_scenario.get(str(sc_id), [])
            scenario_cost = 0.0
            for route_info in sc_routes:
                route = route_info.get("route", [])
                for k in range(len(route) - 1):
                    i_node = route[k]
                    j_node = route[k + 1]
                    scenario_cost += float(dist[i_node][j_node])
            true_obj += p_omega * scenario_cost

        abs_diff = abs(reported_obj - true_obj)
        tol_obj = max(1e-3, 1e-3 * abs(true_obj))
        if abs_diff > tol_obj:
            violations.append(
                f"Constraint 7: Objective consistency violated: reported objective_value="
                f"{reported_obj} differs from recomputed sum_omega p_omega * sum_r c_r="
                f"{true_obj} (|diff|={abs_diff:.3g}, tol={tol_obj:.3g})"
            )
            violated_set.add(7)
            violation_magnitudes.append(make_violation(7, reported_obj, true_obj, abs_diff,
                f"Reported obj={reported_obj} vs recomputed={true_obj} (|diff|={abs_diff:.3g})"))
    elif reported_obj_raw is not None:
        # Non-finite or unparseable reported objective is itself a violation.
        violations.append(
            f"Constraint 7: Objective consistency violated: reported objective_value="
            f"{reported_obj_raw!r} is not a finite real number"
        )
        violated_set.add(7)
        violation_magnitudes.append(make_violation(7, 0.0, 0.0, float("inf"),
            f"Reported obj={reported_obj_raw!r} not a finite real number"))

    # Build result
    # Remove description from violation_magnitudes for output
    output_magnitudes = []
    for vm in violation_magnitudes:
        output_magnitudes.append({
            "constraint": vm["constraint"],
            "lhs": vm["lhs"],
            "rhs": vm["rhs"],
            "raw_excess": vm["raw_excess"],
            "normalizer": vm["normalizer"],
            "ratio": vm["ratio"],
        })

    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": output_magnitudes,
    }
    return result


def _write_error_result(result_path, message):
    result = {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": [message],
        "violation_magnitudes": [],
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for TWAVRP (Spliet & Gabor 2014)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    try:
        instance = load_json(args.instance_path)
        solution = load_json(args.solution_path)
        result = check_feasibility(instance, solution)
        with open(args.result_path, 'w') as f:
            json.dump(result, f, indent=2)
    except Exception as exc:
        result = _write_error_result(
            args.result_path,
            f"checker could not parse or validate solution: {type(exc).__name__}: {exc}",
        )

    if result["feasible"]:
        print(f"FEASIBLE - No constraint violations found.")
    else:
        print(f"INFEASIBLE - {len(result['violated_constraints'])} constraint(s) violated: "
              f"{result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
