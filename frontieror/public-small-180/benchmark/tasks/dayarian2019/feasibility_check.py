#!/usr/bin/env python3
"""
Feasibility checker for the Caterer Production-Routing Problem (CPRP).

Checks each hard constraint from the mathematical formulation in:
  Dayarian & Desaulniers (2019), Transportation Science.

Constraints (numbered as in the paper):
  (2)  Each customer visited exactly once
  (3)  Production capacity per subperiod
  (4)  Production due time linking
  (5)  Driver availability
  (6)  Employee availability
  (7)  Vehicle availability tracking
  (8)  Workstation availability tracking
  (9)  Route variable domain (binary)
  (10) Shift variable domain (integer >= 0)
  (11) Production quantity domain (>= 0)
  (12) Vehicle count domain
  (13) Workstation count domain

Released certificate extension:
  - ``customers`` is the actual visit order of every trip;
  - ``production_staffing`` contains the integer production, setup,
    workstation, and shift-start variables omitted by the earlier export;
  - (16) recomputes the complete objective from those decisions.

This makes the checker a verifier of a complete feasible witness instead of a
necessary-condition filter with a loose objective interval.
"""

import argparse
import json
import math
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = (
    'objective_value',
    'production_schedule',
    'production_staffing',
    'routes',
)
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('production_staffing',)
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
    violations = []
    violation_magnitudes = []

    # Handle null / empty solutions
    if solution.get("objective_value") is None:
        violations.append("Solution has no objective value (no feasible solution found)")
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": 0.0,
            "rhs": 1.0,
            "raw_excess": 1.0,
            "normalizer": 1.0,
            "ratio": 1.0,
        })
        return {
            "feasible": False,
            "violated_constraints": [2],
            "violations": violations,
            "violation_magnitudes": violation_magnitudes,
        }

    params = instance["global_parameters"]
    orders = instance["orders"]
    products = instance["products"]
    periods = instance["periods"]
    subperiods = instance["subperiods"]
    prod_windows = instance["production_windows"]
    travel_time = instance["travel_time_matrix"]

    num_orders = instance["num_orders"]
    num_products = instance["num_products"]
    num_periods = params["num_periods"]
    num_subperiods = params["num_subperiods"]
    n_V = instance["num_vehicles"]
    n_D = instance["num_drivers"]
    n_W = instance["num_workstations"]
    n_E = instance["num_production_employees"]
    Q = instance["vehicle_capacity"]
    sp_per_period = params["subperiods_per_period"]
    tau_P = params["preparation_time_minutes"]
    tau_S = params["service_time_minutes"]
    m_T = params["max_trips_per_route"]
    l_bar = params["max_shift_route_length_periods"]
    l_D = params["min_paid_time_minutes"]
    period_len = params["period_length_minutes"]
    horizon_start = params["horizon_start_minutes"]

    routes = solution.get("routes", [])
    prod_schedule = solution.get("production_schedule", [])

    violated_set = set()

    def record_violation(constraint_idx, msg, lhs, rhs, op="<="):
        """Record a constraint violation."""
        if op == "<=":
            raw = lhs - rhs
        elif op == ">=":
            raw = rhs - lhs
        elif op == "=":
            raw = abs(lhs - rhs)
        else:
            raw = abs(lhs - rhs)
        normalizer = max(abs(rhs), EPS)
        ratio = raw / normalizer
        violated_set.add(constraint_idx)
        violations.append(msg)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(raw),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })

    # =========================================================================
    # Build helper data structures from the solution
    # =========================================================================

    # Count visits per customer from routes
    visit_count = defaultdict(int)
    # Map: customer -> list of (route_idx, trip_idx, start_period)
    customer_delivery = defaultdict(list)

    for route in routes:
        for trip in route["trips"]:
            for cust in trip["customers"]:
                visit_count[cust] += 1
                customer_delivery[cust].append({
                    "vehicle": route["vehicle"],
                    "trip_idx": trip["trip"],
                    "start_period": trip["start_period"],
                })

    # Build production quantities: q[i][p][u] = quantity
    q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for entry in prod_schedule:
        i = entry["order"]
        p = entry["product"]
        u = entry["subperiod"]
        q[i][p][u] += entry["quantity"]

    # Valid subperiods per order per product
    U_ip = {}
    for i in range(num_orders):
        U_ip[i] = {}
        for p_str, p_data in prod_windows[str(i)].items():
            U_ip[i][int(p_str)] = set(p_data["valid_subperiod_ids"])

    # U^p: union of U_i^p
    U_p = {}
    for p in range(num_products):
        sp_set = set()
        for i in range(num_orders):
            if p in U_ip[i]:
                sp_set.update(U_ip[i][p])
        U_p[p] = sorted(sp_set)

    # Orders active in each subperiod for each product
    O_up = {}
    for p in range(num_products):
        for u in U_p[p]:
            O_up[p, u] = [i for i in range(num_orders)
                          if p in U_ip[i] and u in U_ip[i][p]]

    # Parse the explicit production/shift certificate.  The old release only
    # stored q, which was insufficient to verify constraints (3), (6), (8),
    # (10), and (13), or to bind the production part of the objective.
    staffing = solution.get("production_staffing", {})
    if not isinstance(staffing, dict):
        record_violation(
            10,
            "production_staffing must be a JSON object",
            0.0, 1.0, "="
        )
        staffing = {}

    def _integer_value(value, label, upper_bound):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            record_violation(10, f"{label} must be an integer", 1.0, 0.0, "=")
            return 0
        numeric = float(value)
        if not math.isfinite(numeric) or abs(numeric - round(numeric)) > TOL:
            record_violation(
                10, f"{label}={value!r} is not integral", numeric, round(numeric), "="
            )
            return 0
        integer = int(round(numeric))
        if integer < 0 or integer > upper_bound:
            record_violation(
                10,
                f"{label}={integer} is outside [0, {upper_bound}]",
                integer, upper_bound if integer > upper_bound else 0, "<=" if integer > upper_bound else ">="
            )
        return integer

    def _indexed_staffing(field, value_key, expected_keys, upper_bound):
        entries = staffing.get(field, [])
        if not isinstance(entries, list):
            record_violation(10, f"production_staffing.{field} must be a list", 0.0, 1.0, "=")
            entries = []
        values = {}
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                record_violation(10, f"{field}[{position}] must be an object", 0.0, 1.0, "=")
                continue
            p = entry.get("product")
            u = entry.get("subperiod")
            key = (p, u)
            if (
                isinstance(p, bool)
                or isinstance(u, bool)
                or not isinstance(p, int)
                or not isinstance(u, int)
                or key not in expected_keys
            ):
                record_violation(
                    10,
                    f"{field}[{position}] has invalid product/subperiod key {key!r}",
                    1.0, 0.0, "="
                )
                continue
            if key in values:
                record_violation(10, f"{field} contains duplicate key {key}", 2.0, 1.0, "<=")
                continue
            values[key] = _integer_value(
                entry.get(value_key), f"{field}{key}.{value_key}", upper_bound
            )
        missing = expected_keys - set(values)
        extra_count = len(set(values) - expected_keys)
        if missing or extra_count:
            record_violation(
                10,
                f"{field} must contain each valid product/subperiod exactly once; "
                f"missing={sorted(missing)[:12]}",
                len(values), len(expected_keys), "="
            )
        return {key: values.get(key, 0) for key in expected_keys}

    expected_pu = {(p, u) for p in range(num_products) for u in U_p[p]}
    f_staff = _indexed_staffing(
        "employees_by_product_subperiod", "employees", expected_pu, n_E
    )
    beta_staff = _indexed_staffing(
        "setups_by_product_subperiod", "setups", expected_pu, n_E
    )

    def _period_vector(field, upper_bound):
        raw = staffing.get(field, [])
        if not isinstance(raw, list) or len(raw) != num_periods:
            record_violation(
                10,
                f"production_staffing.{field} must have exactly {num_periods} values",
                len(raw) if isinstance(raw, list) else 0, num_periods, "="
            )
            raw = raw if isinstance(raw, list) else []
        return [
            _integer_value(
                raw[t] if t < len(raw) else 0,
                f"production_staffing.{field}[{t}]",
                upper_bound,
            )
            for t in range(num_periods)
        ]

    w_staff = _period_vector("workstations_by_period", n_W)
    alpha_staff = _period_vector("shift_starts_by_period", n_E)

    # Feasible trip start periods for each order
    T_i = {}
    for o in orders:
        i = o["order_id"]
        tw_end = o["time_window_end"]
        T_i[i] = []
        for per in periods:
            t = per["period_id"]
            if per["start_time"] + tau_P + travel_time[0][i + 1] <= tw_end:
                T_i[i].append(t)

    # Products with positive demand for each order
    P_i = {}
    for o in orders:
        i = o["order_id"]
        P_i[i] = [int(pk) for pk in o["demands"].keys() if o["demands"][pk] > 0]

    # =========================================================================
    # Constraint (2): Each customer visited exactly once
    #   sum_{r in R} a_{ir} theta_r = 1, for all i in O
    # =========================================================================
    for i in range(num_orders):
        lhs = visit_count[i]
        rhs = 1
        if abs(lhs - rhs) > TOL:
            record_violation(
                2,
                f"Order {i} visited {lhs} time(s) instead of exactly 1",
                lhs, rhs, "="
            )

    # =========================================================================
    # Constraint (3): Production capacity per subperiod
    #   sum_{i in O_u^p} q_i^{pu} <= eta^p * sum_{s in S} f_s^{pu} psi_s
    #
    # The released certificate stores f^{pu}, so the checker verifies the
    # actual assigned capacity rather than the former n_E upper bound.
    # =========================================================================
    for p in range(num_products):
        eta = products[p]["production_rate_per_subperiod"]
        for u in U_p[p]:
            orders_in_u = O_up[p, u]
            total_prod = sum(q[i][p].get(u, 0.0) for i in orders_in_u)
            capacity = eta * f_staff[p, u]
            if total_prod - capacity > TOL:
                record_violation(
                    3,
                    f"Production capacity exceeded for product {p}, subperiod {u}: "
                    f"produced {total_prod:.4f} > capacity {capacity:.4f} "
                    f"(eta={eta} * f={f_staff[p, u]})",
                    total_prod, capacity, "<="
                )

    # =========================================================================
    # Constraint (4): Production due time linking
    #   sum_{u in U_i^p, e^u <= b^t} q_i^{pu} >= d_i^p * g_{ir}^t * theta_r
    #
    # For the customer's actual delivery trip start period t*, all production
    # for that order's product must be completed (subperiod end <= period start)
    # before the trip departs.
    # =========================================================================
    for i in range(num_orders):
        if not customer_delivery[i]:
            continue
        for p in P_i[i]:
            p_str = str(p)
            demand = orders[i]["demands"][p_str]
            valid_sps = prod_windows[str(i)][p_str]["valid_subperiod_ids"]

            # For each delivery of this order, check the due time linking
            for deliv in customer_delivery[i]:
                t = deliv["start_period"]
                if t not in T_i[i]:
                    continue
                period_start = periods[t]["start_time"]
                # Sum production in subperiods where end_time <= period_start
                prod_before = sum(
                    q[i][p].get(u, 0.0)
                    for u in valid_sps
                    if subperiods[u]["end_time"] <= period_start
                )
                rhs_val = demand  # g_{ir}^t * theta_r = 1 for the actual delivery
                if rhs_val - prod_before > TOL:
                    record_violation(
                        4,
                        f"Production due time violated for order {i}, product {p}, "
                        f"trip start period {t}: produced before departure = "
                        f"{prod_before:.4f} < demand {rhs_val}",
                        prod_before, rhs_val, ">="
                    )

    # =========================================================================
    # Constraint (5): Driver availability
    #   sum_{r in R} theta_r <= n_D
    # =========================================================================
    num_routes_used = len(routes)
    if num_routes_used - n_D > TOL:
        record_violation(
            5,
            f"Driver availability exceeded: {num_routes_used} routes used > "
            f"{n_D} drivers available",
            num_routes_used, n_D, "<="
        )

    # =========================================================================
    # Constraint (6): Employee availability
    #   sum_{s in S} psi_s <= n_E
    #
    # Count the certified production and setup employees active in each
    # subperiod, exactly as in the compact released model.
    # =========================================================================
    for u in range(num_subperiods):
        employees_needed = 0
        for p in range(num_products):
            employees_needed += f_staff.get((p, u), 0)
            for u_prime in range(max(0, u - params["setup_length_subperiods"] + 1), u + 1):
                employees_needed += beta_staff.get((p, u_prime), 0)
        if employees_needed - n_E > TOL:
            record_violation(
                6,
                f"Employee availability exceeded in subperiod {u}: "
                f"need {employees_needed:.0f} employees > {n_E} available",
                employees_needed, n_E, "<="
            )

    # =========================================================================
    # Constraint (7): Vehicle availability tracking
    #   v^t in [0, n_V], tracking vehicles in use per period
    #
    # In each period, the number of vehicles simultaneously in use (on active
    # trips) must not exceed n_V.
    # =========================================================================
    for t_idx in range(num_periods):
        per = periods[t_idx]
        t_start = per["start_time"]
        t_end = per["end_time"]
        vehicles_in_use = 0
        for route in routes:
            for trip in route["trips"]:
                trip_depart = periods[trip["start_period"]]["start_time"]
                # Estimate trip return: trip departs, goes to customers, returns
                # A vehicle is "in use" from its route start to route end
                # Conservatively, a vehicle is used during any period that overlaps
                # with any of its trips
                if trip_depart < t_end:
                    # Trip starts before this period ends
                    # Check if trip could still be active during this period
                    # The trip started at trip_depart, it needs at least prep time
                    # For a simpler check: count distinct vehicles active
                    vehicles_in_use += 1
                    break  # Count each vehicle only once per period
        if vehicles_in_use - n_V > TOL:
            record_violation(
                7,
                f"Vehicle availability exceeded in period {t_idx}: "
                f"{vehicles_in_use} vehicles in use > {n_V} available",
                vehicles_in_use, n_V, "<="
            )

    # =========================================================================
    # Constraint (8): Workstation availability tracking
    #   w^t in [0, n_W], tracking workstations in use per period
    #
    # In every subperiod, w^t must cover all certified production and setup
    # employees, matching constraint (47) in the compact model.
    # =========================================================================
    for t_idx in range(num_periods):
        sp_start_id = t_idx * sp_per_period
        sp_end_id = sp_start_id + sp_per_period
        for u in range(sp_start_id, sp_end_id):
            employees_active = 0
            for p in range(num_products):
                employees_active += f_staff.get((p, u), 0)
                for u_prime in range(
                    max(0, u - params["setup_length_subperiods"] + 1), u + 1
                ):
                    employees_active += beta_staff.get((p, u_prime), 0)
            if employees_active - w_staff[t_idx] > TOL:
                record_violation(
                    8,
                    f"Certified workstations insufficient in period {t_idx}, "
                    f"subperiod {u}: need {employees_active}, "
                    f"reported w={w_staff[t_idx]}",
                    employees_active, w_staff[t_idx], "<="
                )

    # =========================================================================
    # Constraint (9): Route variable domain - theta_r in {0,1}
    # In compact solution, routes are either selected or not.
    # Check route feasibility conditions from the paper:
    #   - Vehicle capacity per trip
    #   - Time windows
    #   - Max trips per route
    #   - Route duration <= l_bar periods
    #
    # ``customers`` is the certified visit order.  Earlier artifacts exported
    # only an unordered assignment set, which made exact route-cost checking
    # impossible.
    # =========================================================================

    def simulate_trip(customer_sequence, trip_start_time):
        """Simulate a trip with a given customer visit sequence.
        Return feasibility, actual return time, the compact model's route-end
        lower bound, travel time, and violations.

        The compact model defines route_end as at least each customer's
        arrival + service + direct return-to-depot time, so that exact bound is
        tracked separately from the physical return after the final customer.
        Each violation is (message, lhs, rhs, op).
        """
        trip_violations = []
        current_time = trip_start_time + tau_P
        prev_loc = 0  # depot
        travel_total = 0.0
        route_end_bounds = []

        for c in customer_sequence:
            c_loc = c + 1
            travel = travel_time[prev_loc][c_loc]
            travel_total += travel
            arrival = current_time + travel
            tw_start = orders[c]["time_window_start"]
            tw_end = orders[c]["time_window_end"]
            service_start = max(arrival, tw_start)

            if service_start - tw_end > TOL:
                trip_violations.append((
                    f"Time window violated for order {c}: "
                    f"arrival {service_start:.2f} > window end {tw_end}",
                    service_start, tw_end, "<="
                ))

            current_time = service_start + tau_S
            route_end_bounds.append(
                service_start + tau_S + travel_time[c_loc][0]
            )
            prev_loc = c_loc

        return_travel = travel_time[prev_loc][0]
        travel_total += return_travel
        trip_return_time = current_time + return_travel
        model_end_bound = max(route_end_bounds, default=trip_start_time)
        return (
            len(trip_violations) == 0,
            trip_return_time,
            model_end_bound,
            travel_total,
            trip_violations,
        )

    total_route_travel_time = 0.0
    total_driver_paid_time = 0.0

    for route in routes:
        v = route["vehicle"]
        trip_list = route["trips"]

        # Check max trips per route
        if len(trip_list) > m_T:
            record_violation(
                9,
                f"Route for vehicle {v} has {len(trip_list)} trips > "
                f"max {m_T} trips per route",
                len(trip_list), m_T, "<="
            )

        trip_ids = [trip.get("trip") for trip in trip_list]
        if trip_ids != list(range(len(trip_list))):
            record_violation(
                9,
                f"Route for vehicle {v} must list the active trip prefix in "
                f"order; got trip ids {trip_ids}",
                1.0, 0.0, "="
            )

        prev_trip_period = None
        route_first_start = None
        route_last_end = None
        route_travel_time = 0.0

        for trip in trip_list:
            customers = trip["customers"]
            start_period = trip["start_period"]

            if start_period < 0 or start_period >= num_periods:
                record_violation(
                    9,
                    f"Route vehicle {v}, trip {trip['trip']}: "
                    f"start_period {start_period} out of range [0, {num_periods-1}]",
                    start_period, num_periods - 1, "<="
                )
                continue

            trip_start_time = periods[start_period]["start_time"]

            route_first_start = (
                trip_start_time
                if route_first_start is None
                else min(route_first_start, trip_start_time)
            )

            # Vehicle capacity per trip
            total_space = sum(orders[c]["space_requirement"] for c in customers)
            if total_space - Q > TOL:
                record_violation(
                    9,
                    f"Vehicle capacity exceeded on route vehicle {v}, "
                    f"trip {trip['trip']}: space {total_space} > capacity {Q}",
                    total_space, Q, "<="
                )

            # Verify the reported customer order and retain exact objective
            # components from it.
            _, _, model_end_bound, trip_travel_time, trip_viols = simulate_trip(
                customers, trip_start_time
            )

            for msg, lhs, rhs, op in trip_viols:
                record_violation(
                    9,
                    f"Route vehicle {v}, trip {trip['trip']}: {msg}",
                    lhs, rhs, op
                )

            route_last_end = (
                model_end_bound
                if route_last_end is None
                else max(route_last_end, model_end_bound)
            )
            route_travel_time += trip_travel_time

            # Multi-trip: In the period-discretized model, trip k+1 must start
            # at a strictly later period than trip k.
            if prev_trip_period is not None:
                if start_period <= prev_trip_period:
                    record_violation(
                        9,
                        f"Multi-trip period ordering violated on route vehicle {v}: "
                        f"trip {trip['trip']} starts at period {start_period} but "
                        f"previous trip starts at period {prev_trip_period} "
                        f"(must be strictly later)",
                        float(prev_trip_period), float(start_period), "<="
                    )

            prev_trip_period = start_period

        # Route duration
        if route_first_start is not None and route_last_end is not None:
            route_duration = route_last_end - route_first_start
            max_duration = l_bar * period_len
            if route_duration - max_duration > TOL:
                record_violation(
                    9,
                    f"Route duration exceeded for vehicle {v}: "
                    f"{route_duration:.2f} min > max {max_duration} min",
                    route_duration, max_duration, "<="
                )
            total_driver_paid_time += max(route_duration, l_D)
            total_route_travel_time += route_travel_time

    # =========================================================================
    # Constraint (10): Shift variable domain - psi_s >= 0, integer
    # Validate the compact f/beta/w/alpha certificate, including setup
    # continuity, shift starts, minimum paid time, and employee availability.
    # =========================================================================
    sigma = params["setup_length_subperiods"]
    for p in range(num_products):
        sps = U_p[p]
        for idx, u in enumerate(sps):
            prev_f = (
                f_staff[p, sps[idx - 1]]
                if idx > 0 and sps[idx - 1] == u - 1
                else 0
            )
            prev_beta = beta_staff.get((p, u - sigma), 0)
            setup_rhs = prev_beta if idx == 0 else prev_f + prev_beta
            if f_staff[p, u] - setup_rhs > TOL:
                record_violation(
                    10,
                    f"Setup continuity violated for product {p}, subperiod {u}: "
                    f"f={f_staff[p, u]} > previous f + setup={setup_rhs}",
                    f_staff[p, u], setup_rhs, "<="
                )

    for t in range(num_periods):
        shift_start_rhs = w_staff[t] if t == 0 else w_staff[t] - w_staff[t - 1]
        if shift_start_rhs - alpha_staff[t] > TOL:
            record_violation(
                10,
                f"Shift-start count insufficient in period {t}: "
                f"alpha={alpha_staff[t]} < required {shift_start_rhs}",
                alpha_staff[t], shift_start_rhs, ">="
            )

    min_paid_periods = params["min_paid_time_subperiods"] // sp_per_period
    for t in range(num_periods):
        paid_floor = sum(
            alpha_staff[t - j]
            for j in range(1, min(t + 1, min_paid_periods) + 1)
            if t - j >= 0
        )
        if paid_floor - w_staff[t] > TOL:
            record_violation(
                10,
                f"Minimum paid-time coverage violated in period {t}: "
                f"w={w_staff[t]} < {paid_floor}",
                w_staff[t], paid_floor, ">="
            )

    total_shift_starts = sum(alpha_staff)
    if total_shift_starts - n_E > TOL:
        record_violation(
            6,
            f"Employee availability exceeded: total shift starts "
            f"{total_shift_starts} > {n_E}",
            total_shift_starts, n_E, "<="
        )

    # =========================================================================
    # Constraint (11): Production quantity domain - q_i^{pu} >= 0
    # Also check that production only occurs in valid subperiods.
    # Variable domain: order, product, subperiod indices MUST be in range.
    # =========================================================================
    for entry in prod_schedule:
        i = entry["order"]
        p = entry["product"]
        u = entry["subperiod"]
        qty = entry["quantity"]

        if qty < -TOL:
            record_violation(
                11,
                f"Negative production quantity for order {i}, product {p}, "
                f"subperiod {u}: {qty:.4f}",
                qty, 0.0, ">="
            )

        # Domain checks: indices must be in range
        if not (isinstance(i, int) and 0 <= i < num_orders):
            record_violation(
                11,
                f"Production schedule entry has out-of-range order id {i!r} "
                f"(valid: 0..{num_orders - 1})",
                float(i) if isinstance(i, (int, float)) else -1.0,
                float(num_orders - 1), "<="
            )
            continue
        if not (isinstance(p, int) and 0 <= p < num_products):
            record_violation(
                11,
                f"Production schedule entry has out-of-range product id {p!r} "
                f"(valid: 0..{num_products - 1})",
                float(p) if isinstance(p, (int, float)) else -1.0,
                float(num_products - 1), "<="
            )
            continue
        if not (isinstance(u, int) and 0 <= u < num_subperiods):
            record_violation(
                11,
                f"Production schedule entry has out-of-range subperiod id {u!r} "
                f"(valid: 0..{num_subperiods - 1})",
                float(u) if isinstance(u, (int, float)) else -1.0,
                float(num_subperiods - 1), "<="
            )
            continue

        # Check valid subperiod
        if p in U_ip.get(i, {}):
            if u not in U_ip[i][p]:
                record_violation(
                    11,
                    f"Production for order {i}, product {p} in invalid "
                    f"subperiod {u} (not in valid set)",
                    1.0, 0.0, "<="
                )
        else:
            # i is in range and p is in range, but order i doesn't demand product p
            record_violation(
                11,
                f"Production for order {i}, product {p} but order "
                f"has no demand for this product",
                1.0, 0.0, "<="
            )

    # =========================================================================
    # Constraint (12): Vehicle count domain - 0 <= v^t <= n_V, integer
    # Count maximum simultaneous vehicles in use across all periods.
    # A vehicle is in use from its first trip start to its last trip return.
    # =========================================================================
    vehicles_used_set = set()
    for route in routes:
        vehicles_used_set.add(route["vehicle"])

    total_vehicles = len(vehicles_used_set)
    if total_vehicles - n_V > TOL:
        record_violation(
            12,
            f"Total vehicles used ({total_vehicles}) exceeds available ({n_V})",
            total_vehicles, n_V, "<="
        )

    # Check vehicle IDs are valid
    for vid in vehicles_used_set:
        if vid < 0 or vid >= n_V:
            record_violation(
                12,
                f"Vehicle ID {vid} out of valid range [0, {n_V - 1}]",
                vid, n_V - 1, "<="
            )

    # =========================================================================
    # Constraint (13): Workstation count domain - 0 <= w^t <= n_W, integer
    # Bounds and integrality are enforced while parsing the certificate; the
    # coverage inequalities were checked under constraint (8).
    # =========================================================================
    # (Already covered in constraint 8 checking)

    # =========================================================================
    # Additional check: Total production meets demand (Constraint 15, redundant
    # but we check it as it is a necessary condition for feasibility)
    # =========================================================================
    for i in range(num_orders):
        for p in P_i[i]:
            p_str = str(p)
            demand = orders[i]["demands"][p_str]
            valid_sps = prod_windows[str(i)][p_str]["valid_subperiod_ids"]
            total_produced = sum(q[i][p].get(u, 0.0) for u in valid_sps)
            if demand - total_produced > TOL:
                # This is implied by constraint (4) but we report as constraint 4
                # since insufficient total production means due-time linking fails
                record_violation(
                    4,
                    f"Total production insufficient for order {i}, product {p}: "
                    f"produced {total_produced:.4f} < demand {demand}",
                    total_produced, demand, ">="
                )

    # =========================================================================
    # Constraint (16): exact objective-value consistency
    #
    # Objective (1): driver salary + vehicle operating cost + employee salary
    # + setup cost.  Every component is now derived from the explicit route
    # order and production staffing certificate.
    # =========================================================================
    gamma_D = float(params["driver_salary_per_minute"])
    gamma_E = float(params["employee_salary_per_subperiod"])
    gamma_S = float(params["setup_cost"])
    gamma_V = float(params["vehicle_operating_cost_per_minute"])
    driver_cost = gamma_D * total_driver_paid_time
    vehicle_cost = gamma_V * total_route_travel_time
    employee_cost = gamma_E * sp_per_period * sum(w_staff)
    setup_cost = gamma_S * sum(beta_staff.values())
    recomputed_obj = driver_cost + vehicle_cost + employee_cost + setup_cost
    reported = float(solution["objective_value"])
    objective_tolerance = max(1e-4, 1e-6 * max(1.0, abs(recomputed_obj)))
    if abs(reported - recomputed_obj) > objective_tolerance:
        record_violation(
            16,
            f"Objective consistency violated: reported={reported:.10g}, "
            f"recomputed={recomputed_obj:.10g} "
            f"(driver={driver_cost:.10g}, vehicle={vehicle_cost:.10g}, "
            f"employee={employee_cost:.10g}, setup={setup_cost:.10g}, "
            f"tol={objective_tolerance:.3g})",
            reported, recomputed_obj, "="
        )

    # =========================================================================
    # Compile results
    # =========================================================================
    violated_constraints = sorted(violated_set)
    feasible = len(violated_constraints) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the CPRP model")
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

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")
    print(f"Result written to: {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
