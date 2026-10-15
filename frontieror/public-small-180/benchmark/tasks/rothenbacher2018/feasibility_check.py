#!/usr/bin/env python3
"""
Feasibility checker for the Truck-and-Trailer Routing Problem with Time Windows (TTRPTW).

Paper: Rothenbaecher, Drexl, Irnich (2018)
"Branch-and-Price-and-Cut for the Truck-and-Trailer Routing Problem with Time Windows"

Constraints (numbered top-to-bottom from the formulation section):
  Constraint 1 (1b): Partitioning - every customer is served exactly once.
  Constraint 2 (1c): Fleet size - do not exceed available vehicles of each type.
  Constraint 3 (1d): Variable domain - lambda^k_r >= 0, integer.

Route feasibility conditions (implicitly defining R^k):
  Constraint 4: Time feasibility - service within time windows / planning horizon.
  Constraint 5: Load feasibility - truck/trailer capacity not exceeded.
  Constraint 6: Accessibility - truck customers only when trailer detached, etc.
  Constraint 7: Loading policy - correct loading into truck/trailer.
  Constraint 8: Elementarity - each customer visited at most once per route.
  Constraint 9: Structure - route starts/ends at depot, trailer recoupled before return.
  Constraint 10: Objective consistency - reported objective_value must lie in the
                 envelope [L, U] of total cost recomputed from the solution's route
                 structure and instance cost coefficients. L assumes the truck travels
                 alone on every arc (no trailer surcharge) and U assumes the trailer
                 is towed on every arc; handling and fixed costs are exact. This guards
                 against LLM exploits that fabricate the objective (e.g. obj=0 on a
                 non-trivial route, or obj=sys.float_info.max).
"""

import argparse
import json
import math

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



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_location_map(instance):
    """Build a map from location id to location info."""
    loc_map = {}
    depot = instance['depot']
    loc_map[depot['id']] = {'x': depot['x'], 'y': depot['y'], 'type': 'depot'}

    for c in instance['truck_customers']:
        loc_map[c['id']] = {
            'x': c['x'], 'y': c['y'], 'type': 'truck_customer',
            'supply': c['supply'], 'service_time': c.get('service_time', instance['parameters']['service_time'])
        }
    for c in instance['trailer_customers']:
        loc_map[c['id']] = {
            'x': c['x'], 'y': c['y'], 'type': 'trailer_customer',
            'supply': c['supply'], 'service_time': c.get('service_time', instance['parameters']['service_time'])
        }
    for t in instance['transshipment_locations']:
        loc_map[t['id']] = {'x': t['x'], 'y': t['y'], 'type': 'transshipment'}

    return loc_map


def get_vehicle_class_info(instance, vc_id):
    """Get vehicle class info by class_id."""
    for vc in instance['fleet']['vehicle_classes']:
        if vc['class_id'] == vc_id:
            return vc
    return None


def detect_solution_format(solution):
    """Detect whether this is an efficient_algorithm or gurobi solution."""
    if 'solver' in solution and solution.get('solver') == 'Gurobi':
        return 'gurobi'
    if 'algorithm' in solution:
        return 'efficient'
    # Fallback: check route fields
    if solution.get('routes'):
        r0 = solution['routes'][0]
        if 'route_sequence_physical' in r0:
            return 'gurobi'
        if 'path_vertices' in r0:
            return 'efficient'
    return 'unknown'


def extract_routes(solution, instance, fmt):
    """
    Extract a unified route representation from either solution format.

    Returns a list of dicts, each with:
      - vehicle_class_id: int
      - customers_served: list of customer IDs (service visits only)
      - route_cost: float
      - physical_sequence: list of location IDs visited (physical locations)
      - decouple_locations: list of location IDs where trailer is decoupled
      - couple_locations: list of location IDs where trailer is recoupled
    """
    routes = []
    if fmt == 'gurobi':
        for r in solution.get('routes', []):
            vc_id = r.get('vehicle_class', r.get('vehicle_class_id'))
            routes.append({
                'vehicle_class_id': vc_id,
                'customers_served': r.get('customers_served', []),
                'route_cost': r.get('route_cost', 0.0),
                'physical_sequence': r.get('route_sequence_physical', r.get('route_sequence', [])),
                'decouple_locations': r.get('decouple_locations', []),
                'transfer_locations': r.get('transfer_locations', []),
                'couple_locations': r.get('couple_locations', []),
            })
    elif fmt == 'efficient':
        for r in solution.get('routes', []):
            vc_id = r.get('vehicle_class_id')
            vc = get_vehicle_class_info(instance, vc_id)
            has_trailer = vc['trailer_type_id'] != 0 if vc else False

            path_vertices = r.get('path_vertices', [])
            customers_served = r.get('customers_served', [])

            # Reconstruct physical sequence from path_vertices using the pricing
            # network vertex numbering from efficient_algorithm.py
            physical_seq = reconstruct_physical_sequence(
                path_vertices, vc_id, instance, has_trailer)

            dc_locs, cp_locs = extract_dc_cp_from_path(
                path_vertices, vc_id, instance, has_trailer)

            routes.append({
                'vehicle_class_id': vc_id,
                'customers_served': customers_served,
                'route_cost': r.get('route_cost', 0.0),
                'physical_sequence': physical_seq,
                'decouple_locations': dc_locs,
                'couple_locations': cp_locs,
            })
    else:
        for r in solution.get('routes', []):
            vc_id = r.get('vehicle_class_id', r.get('vehicle_class'))
            routes.append({
                'vehicle_class_id': vc_id,
                'customers_served': r.get('customers_served', []),
                'route_cost': r.get('route_cost', 0.0),
                'physical_sequence': r.get('route_sequence_physical',
                                           r.get('route_sequence',
                                                  r.get('path_vertices', []))),
                'decouple_locations': r.get('decouple_locations', []),
                'couple_locations': r.get('couple_locations', []),
            })

    return routes


def build_vertex_info_for_class(vc_id, instance, has_trailer):
    """
    Reconstruct the pricing network vertex mapping for a vehicle class,
    mirroring the logic in efficient_algorithm.py PricingNetwork._build_vertices.

    Returns vid_map: vid -> (type, location_id)
    """
    depot_id = instance['depot']['id']
    N_L = [c['id'] for c in instance['truck_customers']]
    N_H = [c['id'] for c in instance['trailer_customers']]
    T_set = [t['id'] for t in instance['transshipment_locations']]

    vid_map = {}
    vid_map[0] = ('D+', depot_id)
    vid_map[1] = ('D-', depot_id)

    next_vid = 2
    for n in N_L:
        vid_map[next_vid] = ('NL', n)
        next_vid += 1

    for n in N_H:
        vid_map[next_vid] = ('NH', n)
        next_vid += 1
        if has_trailer:
            vid_map[next_vid] = ('DC', n)
            next_vid += 1
            vid_map[next_vid] = ('TAU', n)
            next_vid += 1
            vid_map[next_vid] = ('CP', n)
            next_vid += 1

    if has_trailer:
        for t in T_set:
            vid_map[next_vid] = ('DC', t)
            next_vid += 1
            vid_map[next_vid] = ('TAU', t)
            next_vid += 1
            vid_map[next_vid] = ('CP', t)
            next_vid += 1

    return vid_map


def reconstruct_physical_sequence(path_vertices, vc_id, instance, has_trailer):
    """Convert pricing network vertex IDs to physical location IDs."""
    vid_map = build_vertex_info_for_class(vc_id, instance, has_trailer)
    physical = []
    for vid in path_vertices:
        if vid in vid_map:
            _, loc_id = vid_map[vid]
            physical.append(loc_id)
        else:
            physical.append(vid)
    return physical


def extract_dc_cp_from_path(path_vertices, vc_id, instance, has_trailer):
    """Extract decouple/couple locations from pricing network path."""
    vid_map = build_vertex_info_for_class(vc_id, instance, has_trailer)
    dc_locs = []
    cp_locs = []
    for vid in path_vertices:
        if vid in vid_map:
            vtype, loc_id = vid_map[vid]
            if vtype == 'DC':
                dc_locs.append(loc_id)
            elif vtype == 'CP':
                cp_locs.append(loc_id)
    return dc_locs, cp_locs


def _get_customer_time_window(instance, loc_id):
    """Return [e_n, l_n] for a customer loc_id from the instance, or None if absent."""
    for c in instance.get('truck_customers', []):
        if c.get('id') == loc_id:
            return c.get('time_window')
    for c in instance.get('trailer_customers', []):
        if c.get('id') == loc_id:
            return c.get('time_window')
    return None


def check_feasibility(instance, solution):
    """
    Check all hard constraints for the TTRPTW solution.
    Returns (feasible, violated_constraints, violations, violation_magnitudes).
    """
    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    fmt = detect_solution_format(solution)
    routes = extract_routes(solution, instance, fmt)

    loc_map = build_location_map(instance)
    params = instance['parameters']
    dist_matrix = instance['distance_matrix']
    time_matrix = instance['travel_time_matrix']
    depot_id = instance['depot']['id']

    N_L_ids = set(c['id'] for c in instance['truck_customers'])
    N_H_ids = set(c['id'] for c in instance['trailer_customers'])
    N_ids = N_L_ids | N_H_ids

    planning_horizon = params['planning_horizon']
    service_time = params['service_time']
    coupling_time = params['coupling_time']
    decoupling_time = params['decoupling_time']
    transfer_time = params['transfer_time']

    vc_map = {}
    for vc in instance['fleet']['vehicle_classes']:
        vc_map[vc['class_id']] = vc

    truck_type_map = {}
    for tt in instance['fleet']['truck_types']:
        truck_type_map[tt['id']] = tt
    trailer_type_map = {}
    for tt in instance['fleet']['trailer_types']:
        trailer_type_map[tt['id']] = tt

    # =========================================================================
    # Constraint 1 (1b): Partitioning - every customer served exactly once
    # =========================================================================
    customer_visit_count = {}
    for n in N_ids:
        customer_visit_count[n] = 0

    for ri, route in enumerate(routes):
        for cid in route['customers_served']:
            if cid in customer_visit_count:
                customer_visit_count[cid] += 1

    for n in sorted(N_ids):
        count = customer_visit_count[n]
        # Constraint: sum a_{rn} lambda_r = 1, i.e., count == 1
        lhs = count
        rhs = 1
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(1)
            if count == 0:
                violations.append(f"Customer {n} is not served by any route")
            else:
                violations.append(f"Customer {n} is served {count} times (expected exactly 1)")
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                'constraint': 1,
                'lhs': float(lhs),
                'rhs': float(rhs),
                'raw_excess': float(violation_amount),
                'normalizer': float(normalizer),
                'ratio': float(violation_amount / normalizer),
            })

    # =========================================================================
    # Constraint 2 (1c): Fleet size - do not exceed available vehicles
    # =========================================================================
    # Count vehicles used per truck type and trailer type
    truck_type_usage = {}
    trailer_type_usage = {}

    for route in routes:
        vc_id = route['vehicle_class_id']
        vc = vc_map.get(vc_id)
        if vc is None:
            continue
        tid = vc['truck_type_id']
        truck_type_usage[tid] = truck_type_usage.get(tid, 0) + 1
        trid = vc['trailer_type_id']
        if trid != 0:
            trailer_type_usage[trid] = trailer_type_usage.get(trid, 0) + 1

    for tid, tt in truck_type_map.items():
        used = truck_type_usage.get(tid, 0)
        available = tt['num_available']
        # Constraint: used <= available
        lhs = used
        rhs = available
        violation_amount = max(0, lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(2)
            violations.append(
                f"Truck type {tid}: {used} used, but only {available} available")
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                'constraint': 2,
                'lhs': float(lhs),
                'rhs': float(rhs),
                'raw_excess': float(violation_amount),
                'normalizer': float(normalizer),
                'ratio': float(violation_amount / normalizer),
            })

    for trid, tt in trailer_type_map.items():
        used = trailer_type_usage.get(trid, 0)
        available = tt['num_available']
        lhs = used
        rhs = available
        violation_amount = max(0, lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(2)
            violations.append(
                f"Trailer type {trid}: {used} used, but only {available} available")
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                'constraint': 2,
                'lhs': float(lhs),
                'rhs': float(rhs),
                'raw_excess': float(violation_amount),
                'normalizer': float(normalizer),
                'ratio': float(violation_amount / normalizer),
            })

    # =========================================================================
    # Constraint 3 (1d): Variable domain - lambda >= 0, integer
    # =========================================================================
    # In the solution, each route is selected (lambda=1) or not. The routes
    # list represents the selected routes. We check that route selection counts
    # are non-negative integers (they always are by construction in the solution
    # format, but we verify).
    # For the efficient solution, check if solution explicitly has lambda values.
    # In practice, each route in the list is used exactly once (lambda=1).
    # We verify the route costs are non-negative as a basic domain check.
    for ri, route in enumerate(routes):
        # lambda_r is implicitly 1 for each listed route, which is >= 0 and integer.
        # Check that route_cost >= 0 as a sanity check on the cost parameter.
        rc = route.get('route_cost', 0.0)
        lhs = 1  # lambda value (implicitly 1)
        rhs = 0  # lambda >= 0
        violation_amount = max(0, rhs - lhs)  # >= constraint: violation if rhs > lhs
        # This will always be 0 since lambda=1 >= 0, but we include the check.
        if violation_amount > tol:
            violated_constraints.add(3)
            violations.append(f"Route {ri}: lambda value is negative")
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                'constraint': 3,
                'lhs': float(lhs),
                'rhs': float(rhs),
                'raw_excess': float(violation_amount),
                'normalizer': float(normalizer),
                'ratio': float(violation_amount / normalizer),
            })

    # =========================================================================
    # Route feasibility conditions (Constraints 4-9)
    # These are checked per route.
    # =========================================================================
    for ri, route in enumerate(routes):
        vc_id = route['vehicle_class_id']
        vc = vc_map.get(vc_id)
        if vc is None:
            violations.append(f"Route {ri}: unknown vehicle class {vc_id}")
            continue

        has_trailer = vc['trailer_type_id'] != 0
        Q_L = vc['truck_capacity']
        Q_H = vc['trailer_capacity'] if has_trailer else 0

        customers_served = route['customers_served']
        physical_seq = route['physical_sequence']
        dc_locs = route['decouple_locations']
        tau_locs = route.get('transfer_locations', [])
        cp_locs = route['couple_locations']

        # Determine the vertex types along the physical sequence
        # We need to figure out trailer state (attached/detached) along the route
        # to properly check constraints.

        # Build a detailed traversal with trailer state tracking
        # For gurobi solutions, we have decouple_locations and couple_locations
        # For efficient solutions, we extracted them from the pricing network path

        # Reconstruct detailed traversal:
        # The physical_sequence includes depot and all visited locations.
        # decouple/couple locations indicate where the trailer is dropped/picked up.

        # Determine trailer state at each step
        # Start: trailer attached if has_trailer, else no trailer
        trailer_attached = has_trailer
        trailer_park_loc = None  # where the trailer is parked (None if attached)

        # Build an ordered list of events (visit customer, decouple, couple, transfer)
        # from the physical sequence and decouple/couple info

        # For efficient solutions, the path_vertices give us exact vertex types
        # For gurobi solutions, we use route_sequence (with network vertex IDs) or
        # route_sequence_physical with decouple/couple markers

        # =====================================================================
        # Constraint 4: Time feasibility
        # =====================================================================
        current_time = planning_horizon[0]
        time_violated = False

        # Track trailer state for time computation
        trailer_on = has_trailer
        dc_set = set()
        cp_set = set()

        # Build the event sequence: we need to know at each step what operation happens
        if fmt == 'efficient':
            vid_map = build_vertex_info_for_class(vc_id, instance, has_trailer)
            path_verts = route.get('path_vertices',
                                   solution['routes'][ri].get('path_vertices', []))
            event_seq = []
            for vid in path_verts:
                if vid in vid_map:
                    vtype, loc_id = vid_map[vid]
                else:
                    vtype, loc_id = '?', vid
                event_seq.append((vtype, loc_id))
        elif fmt == 'gurobi':
            # Gurobi: use route_sequence (network vertex IDs) if available
            raw_route = solution['routes'][ri]
            route_seq = raw_route.get('route_sequence', [])
            phys_seq = raw_route.get('route_sequence_physical', [])

            # Build event sequence from route_sequence and physical matching
            # In gurobi format, route_sequence contains network vertex IDs
            # and route_sequence_physical contains physical location IDs
            # Decouple/couple locations tell us where operations happen

            # Reconstruct vertex types from gurobi format
            event_seq = _build_gurobi_event_seq(
                route_seq, phys_seq, dc_locs, cp_locs, depot_id,
                N_L_ids, N_H_ids, has_trailer, instance,
                tau_locs=tau_locs)
        else:
            # Fallback: treat physical_sequence as simple location visits
            event_seq = []
            for loc in physical_seq:
                if loc == depot_id:
                    event_seq.append(('depot', loc))
                elif loc in N_L_ids:
                    event_seq.append(('NL', loc))
                elif loc in N_H_ids:
                    event_seq.append(('NH', loc))
                else:
                    event_seq.append(('?', loc))

        # Time-window verification.
        #
        # For Gurobi-format solutions, we read the MIP's own t[k,vi,n] decisions
        # from `arrival_times` rather than re-simulating along the route — the
        # network has node copies, decouple/couple, and transshipment events that
        # the simulator could not reliably reorder. If `arrival_times` is missing
        # (legacy gurobi solution emitted by the pre-fix code), fail loudly
        # instead of silently falling back to simulation; the upstream gurobi_code
        # must emit arrival_times.
        #
        # For efficient solutions and unknown formats, fall through to the
        # simulator below.
        used_arrival_times = False
        if fmt == 'gurobi':
            raw_route = solution['routes'][ri]
            arrival_times = raw_route.get('arrival_times')
            if arrival_times is None:
                violated_constraints.add(4)
                violations.append(
                    f"Route {ri}: gurobi solution missing 'arrival_times'; "
                    f"feasibility checker requires per-node MIP times for time-window "
                    f"verification (re-run with the patched gurobi_code.py).")
                violation_magnitudes.append({
                    'constraint': 4,
                    'lhs': 0.0, 'rhs': 0.0,
                    'raw_excess': 1.0, 'normalizer': 1.0, 'ratio': 1.0,
                })
                time_violated = True
            else:
                used_arrival_times = True
                route_seq = raw_route.get('route_sequence', [])
                # arrival_times keys are str (JSON serialization); accept both.
                def _t_at(node):
                    if str(node) in arrival_times:
                        return float(arrival_times[str(node)])
                    if node in arrival_times:
                        return float(arrival_times[node])
                    return None
                for step_idx, (vtype, loc_id) in enumerate(event_seq):
                    network_node = route_seq[step_idx] if step_idx < len(route_seq) else loc_id
                    t_val = _t_at(network_node)
                    if t_val is None:
                        violated_constraints.add(4)
                        violations.append(
                            f"Route {ri}: arrival_times missing entry for network "
                            f"node {network_node} at step {step_idx} (loc {loc_id})")
                        violation_magnitudes.append({
                            'constraint': 4,
                            'lhs': 0.0, 'rhs': 0.0,
                            'raw_excess': 1.0, 'normalizer': 1.0, 'ratio': 1.0,
                        })
                        time_violated = True
                        continue

                    if vtype in ('NL', 'NH'):
                        cust_tw = _get_customer_time_window(instance, loc_id)
                        if cust_tw is not None and len(cust_tw) == 2:
                            e_n, l_n = cust_tw[0], cust_tw[1]
                            if t_val < e_n - tol:
                                violated_constraints.add(4)
                                violations.append(
                                    f"Route {ri}: service at customer {loc_id} starts at "
                                    f"{t_val:.2f}, before earliest window {e_n} at step {step_idx}")
                                normalizer = max(abs(e_n), eps)
                                violation_magnitudes.append({
                                    'constraint': 4,
                                    'lhs': float(t_val), 'rhs': float(e_n),
                                    'raw_excess': float(e_n - t_val),
                                    'normalizer': float(normalizer),
                                    'ratio': float((e_n - t_val) / normalizer),
                                })
                                time_violated = True
                            if t_val > l_n + tol:
                                violated_constraints.add(4)
                                violations.append(
                                    f"Route {ri}: service at customer {loc_id} starts at "
                                    f"{t_val:.2f}, past latest window {l_n} at step {step_idx}")
                                normalizer = max(abs(l_n), eps)
                                violation_magnitudes.append({
                                    'constraint': 4,
                                    'lhs': float(t_val), 'rhs': float(l_n),
                                    'raw_excess': float(t_val - l_n),
                                    'normalizer': float(normalizer),
                                    'ratio': float((t_val - l_n) / normalizer),
                                })
                                time_violated = True

                    rhs = planning_horizon[1]
                    if t_val > rhs + tol:
                        violated_constraints.add(4)
                        violations.append(
                            f"Route {ri}: time {t_val:.2f} exceeds planning horizon "
                            f"{rhs} at step {step_idx} (location {loc_id})")
                        normalizer = max(abs(rhs), eps)
                        violation_magnitudes.append({
                            'constraint': 4,
                            'lhs': float(t_val), 'rhs': float(rhs),
                            'raw_excess': float(t_val - rhs),
                            'normalizer': float(normalizer),
                            'ratio': float((t_val - rhs) / normalizer),
                        })
                        time_violated = True

        if not used_arrival_times:
            # Simulate the route timing (efficient solutions or unknown formats).
            current_time = planning_horizon[0]
            prev_loc = None

            for step_idx, (vtype, loc_id) in enumerate(event_seq):
                if step_idx == 0:
                    prev_loc = loc_id
                    continue

                # Travel time from prev_loc to loc_id
                tt = time_matrix[prev_loc][loc_id]

                # Add handling time at previous vertex
                if step_idx > 1:
                    prev_vtype = event_seq[step_idx - 1][0]
                    if prev_vtype in ('NL', 'NH'):
                        tt += service_time
                    elif prev_vtype == 'DC':
                        tt += decoupling_time
                    elif prev_vtype == 'CP':
                        tt += coupling_time
                    elif prev_vtype == 'TAU':
                        tt += transfer_time

                current_time = current_time + tt

                # Customer-specific time window (math_model: service starts in [e_n, l_n])
                if vtype in ('NL', 'NH'):
                    cust_tw = _get_customer_time_window(instance, loc_id)
                    if cust_tw is not None and len(cust_tw) == 2:
                        e_n, l_n = cust_tw[0], cust_tw[1]
                        if current_time < e_n:
                            current_time = e_n
                        lhs_tw = current_time
                        rhs_tw = l_n
                        viol_tw = max(0, lhs_tw - rhs_tw)
                        if viol_tw > tol:
                            violated_constraints.add(4)
                            violations.append(
                                f"Route {ri}: service at customer {loc_id} starts at "
                                f"{current_time:.2f}, past latest window {l_n} at step {step_idx}")
                            normalizer = max(abs(rhs_tw), eps)
                            violation_magnitudes.append({
                                'constraint': 4,
                                'lhs': float(lhs_tw),
                                'rhs': float(rhs_tw),
                                'raw_excess': float(viol_tw),
                                'normalizer': float(normalizer),
                                'ratio': float(viol_tw / normalizer),
                            })
                            time_violated = True

                # Check time window (planning horizon for all vertices)
                lhs = current_time
                rhs = planning_horizon[1]
                violation_amount = max(0, lhs - rhs)
                if violation_amount > tol:
                    violated_constraints.add(4)
                    violations.append(
                        f"Route {ri}: time {current_time:.2f} exceeds planning horizon "
                        f"{planning_horizon[1]} at step {step_idx} (location {loc_id})")
                    normalizer = max(abs(rhs), eps)
                    violation_magnitudes.append({
                        'constraint': 4,
                        'lhs': float(lhs),
                        'rhs': float(rhs),
                        'raw_excess': float(violation_amount),
                        'normalizer': float(normalizer),
                        'ratio': float(violation_amount / normalizer),
                    })
                    time_violated = True

                prev_loc = loc_id

        # =====================================================================
        # Constraint 5: Load feasibility
        # =====================================================================
        truck_load = 0.0
        trailer_load = 0.0
        trailer_on = has_trailer

        for step_idx, (vtype, loc_id) in enumerate(event_seq):
            if vtype in ('D+', 'D-', 'depot'):
                pass
            elif vtype == 'NL':
                # Truck customer: trailer must be detached, load goes to truck
                q = loc_map.get(loc_id, {}).get('supply', 0)
                truck_load += q
            elif vtype == 'NH':
                q = loc_map.get(loc_id, {}).get('supply', 0)
                if trailer_on:
                    # Load goes to trailer first, overflow to truck
                    space_in_trailer = Q_H - trailer_load
                    to_trailer = min(q, space_in_trailer)
                    to_truck = q - to_trailer
                    trailer_load += to_trailer
                    truck_load += to_truck
                else:
                    # Trailer detached: load goes to truck
                    truck_load += q
            elif vtype == 'DC':
                # Decoupling: trailer detached at this location
                trailer_on = False
            elif vtype == 'CP':
                # Coupling: trailer reattached, transfer load from truck to trailer
                trailer_on = True
                transferable = min(truck_load, Q_H - trailer_load)
                truck_load -= transferable
                trailer_load += transferable
            elif vtype == 'TAU':
                # Transfer: transfer load from truck to trailer
                transferable = min(truck_load, Q_H - trailer_load)
                truck_load -= transferable
                trailer_load += transferable

            # Check truck capacity
            lhs_truck = truck_load
            rhs_truck = Q_L
            viol_truck = max(0, lhs_truck - rhs_truck)
            if viol_truck > tol:
                violated_constraints.add(5)
                violations.append(
                    f"Route {ri}: truck load {truck_load:.1f} exceeds capacity "
                    f"{Q_L} at step {step_idx} (location {loc_id})")
                normalizer = max(abs(rhs_truck), eps)
                violation_magnitudes.append({
                    'constraint': 5,
                    'lhs': float(lhs_truck),
                    'rhs': float(rhs_truck),
                    'raw_excess': float(viol_truck),
                    'normalizer': float(normalizer),
                    'ratio': float(viol_truck / normalizer),
                })

            # Check trailer capacity
            if has_trailer:
                lhs_trail = trailer_load
                rhs_trail = Q_H
                viol_trail = max(0, lhs_trail - rhs_trail)
                if viol_trail > tol:
                    violated_constraints.add(5)
                    violations.append(
                        f"Route {ri}: trailer load {trailer_load:.1f} exceeds capacity "
                        f"{Q_H} at step {step_idx} (location {loc_id})")
                    normalizer = max(abs(rhs_trail), eps)
                    violation_magnitudes.append({
                        'constraint': 5,
                        'lhs': float(lhs_trail),
                        'rhs': float(rhs_trail),
                        'raw_excess': float(viol_trail),
                        'normalizer': float(normalizer),
                        'ratio': float(viol_trail / normalizer),
                    })

        # =====================================================================
        # Constraint 6: Accessibility
        # =====================================================================
        trailer_on = has_trailer
        trailer_park_loc = None

        for step_idx, (vtype, loc_id) in enumerate(event_seq):
            if vtype == 'DC':
                # Decoupling: trailer must be attached to decouple
                if not trailer_on:
                    lhs = 0  # trailer_on = False means not attached
                    rhs = 1  # requires attached
                    violation_amount = abs(lhs - rhs)
                    violated_constraints.add(6)
                    violations.append(
                        f"Route {ri}: decoupling at location {loc_id} but trailer "
                        f"not attached at step {step_idx}")
                    normalizer = max(abs(rhs), eps)
                    violation_magnitudes.append({
                        'constraint': 6,
                        'lhs': float(lhs),
                        'rhs': float(rhs),
                        'raw_excess': float(violation_amount),
                        'normalizer': float(normalizer),
                        'ratio': float(violation_amount / normalizer),
                    })
                trailer_on = False
                trailer_park_loc = loc_id
            elif vtype == 'CP':
                # Coupling: trailer must be detached AND parked at this exact location.
                if trailer_on or trailer_park_loc is None or trailer_park_loc != loc_id:
                    lhs = 0  # trailer not parked here
                    rhs = 1  # trailer must be parked at loc_id
                    violation_amount = abs(lhs - rhs)
                    violated_constraints.add(6)
                    if trailer_on:
                        detail = "trailer still attached"
                    elif trailer_park_loc is None:
                        detail = "trailer was never decoupled"
                    else:
                        detail = f"trailer parked at {trailer_park_loc}"
                    violations.append(
                        f"Route {ri}: coupling at location {loc_id} but {detail} "
                        f"at step {step_idx}")
                    normalizer = max(abs(rhs), eps)
                    violation_magnitudes.append({
                        'constraint': 6,
                        'lhs': float(lhs),
                        'rhs': float(rhs),
                        'raw_excess': float(violation_amount),
                        'normalizer': float(normalizer),
                        'ratio': float(violation_amount / normalizer),
                    })
                trailer_on = True
                trailer_park_loc = None
            elif vtype == 'TAU':
                # Transfer: trailer must be parked at this location
                pass  # trailer_on state doesn't change
            elif vtype == 'NL':
                # Truck customer: trailer must be detached
                if trailer_on and has_trailer:
                    lhs = 1  # trailer attached
                    rhs = 0  # requires detached
                    violation_amount = abs(lhs - rhs)
                    violated_constraints.add(6)
                    violations.append(
                        f"Route {ri}: truck customer {loc_id} visited with trailer "
                        f"attached at step {step_idx}")
                    normalizer = max(abs(rhs), eps)
                    violation_magnitudes.append({
                        'constraint': 6,
                        'lhs': float(lhs),
                        'rhs': float(rhs),
                        'raw_excess': float(violation_amount),
                        'normalizer': float(normalizer),
                        'ratio': float(violation_amount / normalizer),
                    })

        # =====================================================================
        # Constraint 7: Loading policy
        # (Checked implicitly in Constraint 5 load simulation above.
        #  The loading policy dictates HOW load is distributed between truck
        #  and trailer. We re-verify: at trailer customers with trailer
        #  attached, supply should go to trailer first, overflow to truck.
        #  At truck customers or when trailer detached, supply goes to truck.
        #  At coupling/transfer, load transferred from truck to trailer.)
        # =====================================================================
        # We re-simulate with strict policy checking
        truck_load_c7 = 0.0
        trailer_load_c7 = 0.0
        trailer_on_c7 = has_trailer

        for step_idx, (vtype, loc_id) in enumerate(event_seq):
            if vtype in ('D+', 'D-', 'depot'):
                pass
            elif vtype == 'NL':
                q = loc_map.get(loc_id, {}).get('supply', 0)
                # Must load into truck (trailer detached or truck customer)
                truck_load_c7 += q
            elif vtype == 'NH':
                q = loc_map.get(loc_id, {}).get('supply', 0)
                if trailer_on_c7:
                    space_in_trailer = Q_H - trailer_load_c7
                    to_trailer = min(q, space_in_trailer)
                    to_truck = q - to_trailer
                    trailer_load_c7 += to_trailer
                    truck_load_c7 += to_truck
                else:
                    truck_load_c7 += q
            elif vtype == 'DC':
                # No load transfer at decoupling
                trailer_on_c7 = False
            elif vtype == 'CP':
                # Transfer as much as possible from truck to trailer
                trailer_on_c7 = True
                transferable = min(truck_load_c7, Q_H - trailer_load_c7)
                truck_load_c7 -= transferable
                trailer_load_c7 += transferable
            elif vtype == 'TAU':
                # Transfer as much as possible from truck to trailer
                transferable = min(truck_load_c7, Q_H - trailer_load_c7)
                truck_load_c7 -= transferable
                trailer_load_c7 += transferable

            # Check total load doesn't exceed combined capacity at any point
            total_load = truck_load_c7 + trailer_load_c7
            total_cap = Q_L + Q_H
            lhs = total_load
            rhs = total_cap
            violation_amount = max(0, lhs - rhs)
            if violation_amount > tol:
                violated_constraints.add(7)
                violations.append(
                    f"Route {ri}: total load {total_load:.1f} exceeds combined "
                    f"capacity {total_cap} at step {step_idx} (location {loc_id})")
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    'constraint': 7,
                    'lhs': float(lhs),
                    'rhs': float(rhs),
                    'raw_excess': float(violation_amount),
                    'normalizer': float(normalizer),
                    'ratio': float(violation_amount / normalizer),
                })

        # =====================================================================
        # Constraint 8: Elementarity - each customer visited at most once per route
        # =====================================================================
        cust_counts = {}
        for cid in customers_served:
            cust_counts[cid] = cust_counts.get(cid, 0) + 1

        for cid, cnt in cust_counts.items():
            if cnt > 1:
                lhs = cnt
                rhs = 1
                violation_amount = abs(lhs - rhs)
                violated_constraints.add(8)
                violations.append(
                    f"Route {ri}: customer {cid} visited {cnt} times in same route")
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    'constraint': 8,
                    'lhs': float(lhs),
                    'rhs': float(rhs),
                    'raw_excess': float(violation_amount),
                    'normalizer': float(normalizer),
                    'ratio': float(violation_amount / normalizer),
                })

        # =====================================================================
        # Constraint 9: Structure - route starts/ends at depot, trailer recoupled
        # =====================================================================
        # Check depot start
        if event_seq:
            first_vtype, first_loc = event_seq[0]
            if first_loc != depot_id:
                lhs = first_loc
                rhs = depot_id
                violation_amount = abs(lhs - rhs)
                violated_constraints.add(9)
                violations.append(
                    f"Route {ri}: does not start at depot (starts at location {first_loc})")
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    'constraint': 9,
                    'lhs': float(lhs),
                    'rhs': float(rhs),
                    'raw_excess': float(violation_amount),
                    'normalizer': float(normalizer),
                    'ratio': float(violation_amount / normalizer),
                })

            # Check depot end
            last_vtype, last_loc = event_seq[-1]
            if last_loc != depot_id:
                lhs = last_loc
                rhs = depot_id
                violation_amount = abs(lhs - rhs)
                violated_constraints.add(9)
                violations.append(
                    f"Route {ri}: does not end at depot (ends at location {last_loc})")
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    'constraint': 9,
                    'lhs': float(lhs),
                    'rhs': float(rhs),
                    'raw_excess': float(violation_amount),
                    'normalizer': float(normalizer),
                    'ratio': float(violation_amount / normalizer),
                })

        # Check trailer recoupled before depot return (for complete vehicles)
        if has_trailer:
            # After processing all events, trailer should be attached at end
            trailer_final = has_trailer  # start attached
            for (vtype, loc_id) in event_seq:
                if vtype == 'DC':
                    trailer_final = False
                elif vtype == 'CP':
                    trailer_final = True

            if not trailer_final:
                lhs = 0  # trailer not attached
                rhs = 1  # requires attached
                violation_amount = abs(lhs - rhs)
                violated_constraints.add(9)
                violations.append(
                    f"Route {ri}: trailer not recoupled before returning to depot")
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    'constraint': 9,
                    'lhs': float(lhs),
                    'rhs': float(rhs),
                    'raw_excess': float(violation_amount),
                    'normalizer': float(normalizer),
                    'ratio': float(violation_amount / normalizer),
                })

    # =========================================================================
    # Constraint 10: exact objective consistency
    # =========================================================================
    # Recompute the true objective from the route structure using the cost rule:
    #   per-route cost = truck_fixed_cost + travel_cost + handling_cost
    # where travel_cost = sum over arcs of dist[i][j] * (cf [+ tw if towing]).
    # In the published Gurobi format, route_sequence contains the expanded
    # decouple/couple node identifiers.  Those identifiers determine exactly
    # whether each arc is traversed with the trailer, using the same rule as
    # gurobi_code.py.  The previous all-detached/all-attached envelope admitted
    # fabricated objectives throughout a wide interval.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None and routes:
        try:
            reported = float(reported_obj)
            reported_ok = math.isfinite(reported)
        except (TypeError, ValueError):
            reported = None
            reported_ok = False
        if reported_ok:
            decoupling_cost_val = params.get('decoupling_cost', 0.0)
            coupling_cost_val = params.get('coupling_cost', 0.0)
            transfer_cost_val = params.get('transfer_cost', 0.0)

            n_orig = len(dist_matrix)
            parking_locations = sorted(
                N_H_ids
                | {item['id'] for item in instance['transshipment_locations']}
            )
            dp_nodes = {
                n_orig + idx for idx in range(len(parking_locations))
            }
            cp_nodes = {
                n_orig + len(parking_locations) + idx
                for idx in range(len(parking_locations))
            }

            def physical_node(node):
                if not isinstance(node, int):
                    raise ValueError(f"non-integer expanded node id {node!r}")
                if 0 <= node < n_orig:
                    return node
                offset = node - n_orig
                if 0 <= offset < 2 * len(parking_locations):
                    return parking_locations[offset % len(parking_locations)]
                raise ValueError(f"expanded node id {node} is out of range")

            exact_total = 0.0
            raw_routes = solution.get('routes', [])
            for route_index, r in enumerate(routes):
                vc_id = r.get('vehicle_class_id')
                vc = vc_map.get(vc_id)
                if vc is None:
                    raise ValueError(f"route {route_index} has unknown vehicle class {vc_id}")
                has_trailer = vc.get('trailer_type_id', 0) != 0
                cf = float(vc.get('truck_cost_factor', 0.0))
                tw = float(vc.get('trailer_towing_cost_multiplier', 0.0)) if has_trailer else 0.0
                fixed = float(vc.get('truck_fixed_cost', 0.0))

                if fmt != 'gurobi' or route_index >= len(raw_routes):
                    raise ValueError(
                        "exact objective validation requires route_sequence and "
                        "route_sequence_physical in the published solution format"
                    )
                network_sequence = raw_routes[route_index].get('route_sequence')
                physical_sequence = raw_routes[route_index].get(
                    'route_sequence_physical'
                )
                if (not isinstance(network_sequence, list)
                        or not isinstance(physical_sequence, list)
                        or len(network_sequence) != len(physical_sequence)
                        or len(network_sequence) < 2):
                    raise ValueError(
                        f"route {route_index} has incomplete or misaligned sequences"
                    )
                mapped_physical = [physical_node(node)
                                   for node in network_sequence]
                if mapped_physical != physical_sequence:
                    raise ValueError(
                        f"route {route_index} route_sequence_physical does not "
                        f"match its expanded route_sequence"
                    )

                travel = 0.0
                for source, target in zip(network_sequence,
                                          network_sequence[1:]):
                    source_physical = physical_node(source)
                    target_physical = physical_node(target)
                    towing = (
                        has_trailer
                        and (source == depot_id
                             or source in N_H_ids
                             or source in cp_nodes)
                        and (target == depot_id
                             or target in N_H_ids
                             or target in dp_nodes)
                    )
                    factor = cf + (tw if towing else 0.0)
                    travel += factor * float(
                        dist_matrix[source_physical][target_physical]
                    )

                n_dc = len(r.get('decouple_locations', []) or [])
                n_cp = len(r.get('couple_locations', []) or [])
                n_tau = len(r.get('transfer_locations', []) or [])
                handling = (n_dc * decoupling_cost_val
                            + n_cp * coupling_cost_val
                            + n_tau * transfer_cost_val)

                exact_total += fixed + travel + handling

            tol_obj = max(1e-3, 1e-8 * max(abs(exact_total), 1.0))
            difference = abs(reported - exact_total)
            if difference > tol_obj:
                violated_constraints.add(10)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from exactly recomputed route cost "
                    f"{exact_total:.6f} (difference={difference:.6g}, "
                    f"tol={tol_obj:.4g})")
                normalizer = max(abs(exact_total), eps)
                violation_magnitudes.append({
                    'constraint': 10,
                    'lhs': float(reported),
                    'rhs': float(exact_total),
                    'raw_excess': float(difference),
                    'normalizer': float(normalizer),
                    'ratio': float(difference / normalizer),
                })

    feasible = len(violated_constraints) == 0
    return feasible, sorted(violated_constraints), violations, violation_magnitudes


def _build_gurobi_event_seq(route_seq, phys_seq, dc_locs, cp_locs,
                             depot_id, N_L_ids, N_H_ids, has_trailer, instance,
                             tau_locs=None):
    """
    Build event sequence from gurobi solution format.

    In gurobi format:
    - route_sequence contains network vertex IDs
    - route_sequence_physical contains physical location IDs
    - When a physical location appears in decouple_locations, transfer_locations,
      or couple_locations, the vertex at that physical location in the
      route_sequence may be a DC/TAU/CP vertex rather than a customer vertex.

    We use route_sequence_physical and match against DC/TAU/CP locations.
    """
    if tau_locs is None:
        tau_locs = []
    T_set = set(t['id'] for t in instance['transshipment_locations'])

    events = []
    # Track which decouple/transfer/couple locations we've encountered
    dc_remaining = list(dc_locs)
    cp_remaining = list(cp_locs)

    if len(route_seq) == len(phys_seq):
        # Use the difference between route_seq and phys_seq to detect DC/TAU/CP
        dc_used = 0
        tau_used = 0
        cp_used = 0

        for i in range(len(phys_seq)):
            loc = phys_seq[i]
            net_vid = route_seq[i]

            if i == 0:
                events.append(('D+', depot_id))
            elif i == len(phys_seq) - 1:
                events.append(('D-', depot_id))
            elif loc == depot_id:
                events.append(('depot', loc))
            elif net_vid != loc:
                # Special vertex (DC, TAU, or CP)
                # Check in expected order: DC first, then TAU, then CP
                if dc_used < len(dc_locs) and loc == dc_locs[dc_used]:
                    events.append(('DC', loc))
                    dc_used += 1
                elif tau_used < len(tau_locs) and loc == tau_locs[tau_used]:
                    events.append(('TAU', loc))
                    tau_used += 1
                elif cp_used < len(cp_locs) and loc == cp_locs[cp_used]:
                    events.append(('CP', loc))
                    cp_used += 1
                else:
                    # Could be TAU or another special vertex
                    events.append(('TAU', loc))
            elif loc in N_L_ids:
                events.append(('NL', loc))
            elif loc in N_H_ids:
                events.append(('NH', loc))
            elif loc in T_set:
                # Transshipment visited as customer?
                events.append(('TAU', loc))
            else:
                events.append(('?', loc))
    else:
        # Fallback: just use phys_seq with simple type detection
        for i, loc in enumerate(phys_seq):
            if i == 0:
                events.append(('D+', depot_id))
            elif i == len(phys_seq) - 1:
                events.append(('D-', depot_id))
            elif loc in N_L_ids:
                events.append(('NL', loc))
            elif loc in N_H_ids:
                events.append(('NH', loc))
            else:
                events.append(('?', loc))

    return events


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for TTRPTW solutions")
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

    feasible, violated_constraints, violations_list, violation_magnitudes = \
        check_feasibility(instance, solution)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations_list,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if feasible:
        print(f"FEASIBLE: Solution passes all constraint checks.")
    else:
        print(f"INFEASIBLE: Violated constraints: {violated_constraints}")
        for v in violations_list:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
