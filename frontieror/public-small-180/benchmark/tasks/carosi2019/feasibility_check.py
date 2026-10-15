#!/usr/bin/env python3
"""
Feasibility checker for ITTVS (Integrated Timetabling and Vehicle Scheduling) solutions.

Checks hard constraints from the MILP formulation (equations 2-6) in:
  Carosi, Frangioni, Galli, Girardi, Vallese (2019)
  "A matheuristic for integrated timetabling and vehicle scheduling"
  Transportation Research Part B 127, 99-124.

Constraint numbering (top to bottom in formulation section, skipping objective):
  Constraint 1 = eq(2): TT flow conservation
  Constraint 2 = eq(3): y binary integrality
  Constraint 3 = eq(4): VS flow conservation (circulation)
  Constraint 4 = eq(5): VS capacity bounds (0 <= x <= u)
  Constraint 5 = eq(6): Linking (TT selection = VS selection)
  Constraint 6 = eq(1): Objective-value consistency (Tier C anti-exploit) -
    the program's reported objective_value must equal the recomputed
    alpha * (VS arc costs) + sum_d (TT arc costs) evaluated on the
    provided vs_flows and the selected-trip chain in each direction
    (within a 0.1% relative tolerance).
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('num_vehicles', 'objective_value', 'selected_trips', 'tt_arcs_used', 'vs_flows')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('num_vehicles', 'tt_arcs_used')
_FRONTIEROR_DERIVED_FIELD_RULES = (('num_vehicles', 'mapping_value', 'vs_flows', 'VS_O_plus-->VS_O_minus'),)
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



# ============================================================
# Tolerance constants
# ============================================================
TOL = 1e-5
EPS = 1e-5


# ============================================================
# Instance helpers
# ============================================================

def get_headway_params_minutes(instance, tw_index):
    """Return (ideal, min, max) headway in minutes for a time window."""
    for tw in instance['time_windows']:
        if tw['index'] == tw_index:
            return (
                tw['ideal_headway_seconds'] / 60.0,
                tw['min_headway_seconds'] / 60.0,
                tw['max_headway_seconds'] / 60.0,
            )
    return None


def get_stopping_params(instance, terminal, tw_index):
    """Return (min_dwell, max_dwell) in minutes for terminal at time window."""
    for entry in instance.get('stopping_times', {}).get(terminal, []):
        if entry['time_window_index'] == tw_index:
            return entry['min_stopping_time_minutes'], entry['max_stopping_time_minutes']
    return 0, 999999


def get_pull_params(instance, terminal, tw_index):
    """Return (pull_out, pull_in) times in minutes for terminal at time window."""
    for entry in instance.get('pull_in_out_times', {}).get(terminal, []):
        if entry['time_window_index'] == tw_index:
            return entry['pull_out_time_minutes'], entry['pull_in_time_minutes']
    return 0, 0


def get_depot_min_dwell(instance, tw_index):
    """Return minimum depot dwell time in minutes for time window."""
    for entry in instance.get('stopping_times', {}).get('O', []):
        if entry['time_window_index'] == tw_index:
            return entry['min_stopping_time_minutes']
    return 0


def add_violation(violated_set, violations, violation_magnitudes,
                  constraint_idx, message, lhs, rhs, operator='eq'):
    """Record a constraint violation with normalized magnitude."""
    if operator == 'geq':
        # LHS >= RHS required; violation if LHS < RHS
        violation_amount = rhs - lhs
    elif operator == 'leq':
        # LHS <= RHS required; violation if LHS > RHS
        violation_amount = lhs - rhs
    else:
        # Equality
        violation_amount = abs(lhs - rhs)

    if violation_amount <= TOL:
        return  # Not actually violated

    violated_set.add(constraint_idx)
    violations.append(message)
    normalizer = max(abs(rhs), EPS)
    violation_magnitudes.append({
        'constraint': constraint_idx,
        'lhs': float(lhs),
        'rhs': float(rhs),
        'raw_excess': float(violation_amount),
        'normalizer': float(normalizer),
        'ratio': float(violation_amount / normalizer),
    })


# ============================================================
# Main feasibility check
# ============================================================

def check_feasibility(instance, solution):
    violations = []
    violation_magnitudes = []
    violated_set = set()

    # --- Parse instance ---
    all_trips = instance['potential_trips']
    trip_map = {t['id']: t for t in all_trips}
    selected_ids = solution.get('selected_trips', [])
    selected_set = set(selected_ids)

    dir_infos = instance['directions']
    dir_ids = [d['pattern_id'] for d in dir_infos]

    # Group potential trips by direction
    trips_by_dir = {}
    for t in all_trips:
        trips_by_dir.setdefault(t['pattern_id'], []).append(t)

    # Selected trips per direction, sorted by main-stop arrival time
    sel_by_dir = {}
    for d in dir_ids:
        sel_by_dir[d] = sorted(
            [t for t in trips_by_dir.get(d, []) if t['id'] in selected_set],
            key=lambda t: t['main_stop_arrival_time_minutes']
        )

    # Determine initial / final trip sets
    init_tws = instance.get('initial_trip_time_windows', {})
    final_tws = instance.get('final_trip_time_windows', {})

    def is_initial(trip, direction):
        if direction not in init_tws:
            return True  # all trips admissible if not specified
        return trip['time_window_index'] == init_tws[direction]

    def is_final(trip, direction):
        if direction not in final_tws:
            return True
        return trip['time_window_index'] == final_tws[direction]

    # ==========================================================
    # Constraint 1  (eq 2): TT flow conservation
    #   For each direction d, the selected trips must form a
    #   valid unit-flow path source -> t1 -> ... -> tk -> sink
    #   in the TT compatibility graph G_d^TT.
    # ==========================================================
    for d in dir_ids:
        sel = sel_by_dir[d]

        # (a) At least one trip must be selected per direction
        #     (source deficit = -1, sink deficit = +1; need a path)
        if len(sel) == 0:
            add_violation(
                violated_set, violations, violation_magnitudes,
                constraint_idx=1,
                message=(f"Constraint 1 (TT flow conservation): No trips selected "
                         f"for direction {d}; cannot route unit flow from source to sink"),
                lhs=0.0, rhs=-1.0, operator='eq'
            )
            continue

        # (b) First selected trip must be an admissible initial trip
        #     (source arc must exist)
        first_trip = sel[0]
        if not is_initial(first_trip, d):
            add_violation(
                violated_set, violations, violation_magnitudes,
                constraint_idx=1,
                message=(f"Constraint 1 (TT flow conservation): First selected trip "
                         f"{first_trip['id']} in direction {d} is in time window "
                         f"{first_trip['time_window_index']}, not in initial window "
                         f"{init_tws.get(d)}; no source arc exists"),
                lhs=0.0, rhs=1.0, operator='eq'
            )

        # (c) Last selected trip must be an admissible final trip
        #     (sink arc must exist)
        last_trip = sel[-1]
        if not is_final(last_trip, d):
            add_violation(
                violated_set, violations, violation_magnitudes,
                constraint_idx=1,
                message=(f"Constraint 1 (TT flow conservation): Last selected trip "
                         f"{last_trip['id']} in direction {d} is in time window "
                         f"{last_trip['time_window_index']}, not in final window "
                         f"{final_tws.get(d)}; no sink arc exists"),
                lhs=0.0, rhs=1.0, operator='eq'
            )

        # (d) Consecutive selected trips must have feasible headway
        #     Arc (i, j) exists iff l_min <= a(j)-a(i) <= l_max
        for idx in range(1, len(sel)):
            ti = sel[idx - 1]
            tj = sel[idx]
            hw = (tj['main_stop_arrival_time_minutes']
                  - ti['main_stop_arrival_time_minutes'])
            hw_params = get_headway_params_minutes(instance, ti['time_window_index'])
            if hw_params is None:
                continue
            _, l_min, l_max = hw_params

            if hw < l_min - TOL:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=1,
                    message=(f"Constraint 1 (TT flow conservation): Direction {d}, "
                             f"headway between trip {ti['id']} and {tj['id']} is "
                             f"{hw:.4f} min, below minimum {l_min:.4f} min; "
                             f"no TT arc exists"),
                    lhs=hw, rhs=l_min, operator='geq'
                )
            elif hw > l_max + TOL:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=1,
                    message=(f"Constraint 1 (TT flow conservation): Direction {d}, "
                             f"headway between trip {ti['id']} and {tj['id']} is "
                             f"{hw:.4f} min, above maximum {l_max:.4f} min; "
                             f"no TT arc exists"),
                    lhs=hw, rhs=l_max, operator='leq'
                )

    # ==========================================================
    # Constraint 2  (eq 3): y binary integrality
    #   y_{n,m}^d in {0, 1}
    #   Automatically satisfied when solution is a list of
    #   selected trip IDs.  Check that IDs are valid.
    # ==========================================================
    for tid in selected_ids:
        if tid not in trip_map:
            add_violation(
                violated_set, violations, violation_magnitudes,
                constraint_idx=2,
                message=(f"Constraint 2 (binary integrality): Selected trip ID {tid} "
                         f"does not exist in the set of potential trips"),
                lhs=1.0, rhs=0.0, operator='eq'
            )

    # Check for duplicates (each trip arc can carry at most flow 1)
    if len(selected_ids) != len(selected_set):
        dup_counts = {}
        for tid in selected_ids:
            dup_counts[tid] = dup_counts.get(tid, 0) + 1
        for tid, cnt in dup_counts.items():
            if cnt > 1:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=2,
                    message=(f"Constraint 2 (binary integrality): Trip {tid} appears "
                             f"{cnt} times in selected_trips (must be 0 or 1)"),
                    lhs=float(cnt), rhs=1.0, operator='leq'
                )

    # ==========================================================
    # Constraint 3  (eq 4): VS flow conservation (circulation)
    # Constraint 4  (eq 5): VS capacity bounds  0 <= x <= u
    #
    # Per project rule, the checker rigorously verifies the original
    # math-model VS variables `vs_flows` provided in the solution
    # (paper Eq 4-5) rather than running a heuristic re-assignment.
    # If `vs_flows` is missing, rigorous verification cannot be
    # performed and constraint 3 is flagged as unverifiable.
    # ==========================================================
    import ast as _ast
    from collections import defaultdict
    valid_selected = [trip_map[tid] for tid in selected_ids
                      if tid in trip_map]

    def _parse_node(s):
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            try:
                return _ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return s
        return s

    raw_flows = solution.get('vs_flows', None)
    O_minus = 'VS_O_minus'
    O_plus = 'VS_O_plus'
    max_fleet = instance.get('max_fleet')
    if max_fleet is None:
        max_fleet = instance.get('objective_function', {}).get('max_fleet')

    parsed_arcs = None  # populated below; reused for objective recomputation

    if not raw_flows:
        # Required field missing — rigorous verification impossible.
        add_violation(
            violated_set, violations, violation_magnitudes,
            constraint_idx=3,
            message=("Constraint 3 (VS flow conservation): solution is "
                     "missing the required `vs_flows` field; cannot "
                     "rigorously verify the vehicle schedule (paper "
                     "Eq 4-5) without it."),
            lhs=0.0, rhs=1.0, operator='eq'
        )
    else:
        # Parse arcs and accumulate per-node in/out flows.
        flow_in = defaultdict(float)
        flow_out = defaultdict(float)
        arcs_used = []  # (from_node, to_node, flow)
        for key, val in raw_flows.items():
            try:
                f_str, t_str = key.split('-->')
            except ValueError:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=3,
                    message=f"Malformed vs_flows key {key!r} (expected 'from-->to')",
                    lhs=0.0, rhs=1.0, operator='eq'
                )
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=4,
                    message=f"Non-numeric vs_flows value for {key!r}: {val!r}",
                    lhs=0.0, rhs=1.0, operator='eq'
                )
                continue
            f_node = _parse_node(f_str)
            t_node = _parse_node(t_str)
            arcs_used.append((f_node, t_node, v))
            flow_in[t_node] += v
            flow_out[f_node] += v

        parsed_arcs = arcs_used  # share with the objective-recompute block

        # ---- Constraint 3 (Eq 4): flow conservation at each non-depot node ----
        all_nodes = set(flow_in) | set(flow_out)
        for node in all_nodes:
            if node == O_minus or node == O_plus:
                continue  # depot circulation closes via the return arc
            in_v = flow_in[node]
            out_v = flow_out[node]
            if abs(in_v - out_v) > TOL:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=3,
                    message=(f"Constraint 3 (VS flow conservation, Eq 4): "
                             f"node {node!r} has incoming flow {in_v:.6g} "
                             f"!= outgoing flow {out_v:.6g}"),
                    lhs=float(in_v), rhs=float(out_v), operator='eq'
                )

        # ---- Constraint 4 (Eq 5): per-arc capacity ----
        for f_node, t_node, v in arcs_used:
            if v < -TOL:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=4,
                    message=(f"Constraint 4 (VS capacity, Eq 5): arc "
                             f"{f_node!r}->{t_node!r} has negative flow {v:.6g}"),
                    lhs=float(v), rhs=0.0, operator='geq'
                )
            if f_node == O_plus and t_node == O_minus:
                # Return arc capacity = max_fleet
                if max_fleet is not None and v > max_fleet + TOL:
                    add_violation(
                        violated_set, violations, violation_magnitudes,
                        constraint_idx=4,
                        message=(f"Constraint 4 (VS capacity, Eq 5): return arc "
                                 f"{O_plus}->{O_minus} flow {v:.6g} exceeds "
                                 f"max_fleet {max_fleet}"),
                        lhs=float(v), rhs=float(max_fleet), operator='leq'
                    )
            else:
                # All other arcs have unit capacity
                if v > 1.0 + TOL:
                    add_violation(
                        violated_set, violations, violation_magnitudes,
                        constraint_idx=4,
                        message=(f"Constraint 4 (VS capacity, Eq 5): arc "
                                 f"{f_node!r}->{t_node!r} flow {v:.6g} > 1"),
                        lhs=float(v), rhs=1.0, operator='leq'
                    )

        # ---- Selected-trip ↔ trip-arc flow consistency ----
        # Each selected trip i must have its trip arc (i, 'start') -> (i, 'end')
        # carry exactly 1 unit of flow.
        flow_pair = {(f, t): v for f, t, v in arcs_used}
        for tid in selected_ids:
            if tid not in trip_map:
                continue
            trip_arc_flow = flow_pair.get(((tid, 'start'), (tid, 'end')), 0.0)
            if abs(trip_arc_flow - 1.0) > TOL:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=4,
                    message=(f"Constraint 4 (VS trip-arc binding): selected "
                             f"trip {tid} has trip arc flow {trip_arc_flow:.6g} "
                             f"!= 1"),
                    lhs=float(trip_arc_flow), rhs=1.0, operator='eq'
                )

        # ---- Fleet-size bound (Eq 5 / max_fleet) ----
        if max_fleet is not None:
            fleet_size = flow_out.get(O_minus, 0.0)
            if fleet_size > max_fleet + TOL:
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=4,
                    message=(f"Constraint 4 (VS capacity, Eq 5): fleet size "
                             f"{fleet_size:.6g} (out of {O_minus}) exceeds "
                             f"max_fleet {max_fleet}"),
                    lhs=float(fleet_size), rhs=float(max_fleet), operator='leq'
                )

    # ==========================================================
    # Constraint 5  (eq 6): Linking constraints
    #   sum_{(n,m) in B(i)} y_{n,m}^{d(i)} = x_{i^-, i^+}
    #   for all i in T.
    #
    #   A selected trip must be reachable in the TT graph
    #   (have a valid incoming arc from source or a predecessor).
    #   If not, y_entering_i = 0 but x_trip_i = 1 → violation.
    # ==========================================================
    for d in dir_ids:
        sel = sel_by_dir[d]
        if len(sel) == 0:
            continue

        for idx, tj in enumerate(sel):
            # Check if tj has a valid TT predecessor among earlier selected
            # trips, OR if tj is reachable from source (initial trip).
            has_valid_incoming = False

            # Source arc: exists if tj is an initial trip
            if is_initial(tj, d):
                has_valid_incoming = True

            # Predecessor arcs: (ti, tj) exists if l_min <= hw <= l_max
            if not has_valid_incoming:
                for k in range(idx):
                    ti = sel[k]
                    hw = (tj['main_stop_arrival_time_minutes']
                          - ti['main_stop_arrival_time_minutes'])
                    hw_params = get_headway_params_minutes(
                        instance, ti['time_window_index']
                    )
                    if hw_params is None:
                        continue
                    _, l_min, l_max = hw_params
                    if l_min - TOL <= hw <= l_max + TOL:
                        has_valid_incoming = True
                        break

            if not has_valid_incoming:
                # y entering tj = 0 (no valid TT arc) but x trip = 1
                add_violation(
                    violated_set, violations, violation_magnitudes,
                    constraint_idx=5,
                    message=(f"Constraint 5 (linking): Trip {tj['id']} in direction "
                             f"{d} is selected (x=1) but has no valid incoming TT "
                             f"arc (y=0); linking constraint violated"),
                    lhs=0.0, rhs=1.0, operator='eq'
                )

        # Also check: last selected trip must have a valid outgoing arc
        # (to sink if final, or to a successor)
        last = sel[-1]
        if not is_final(last, d):
            # Check if there's a valid successor among selected trips
            # that leads eventually to a final trip
            has_valid_outgoing = False
            # Since last is the last selected trip by arrival time,
            # it cannot have a successor. It must be final.
            add_violation(
                violated_set, violations, violation_magnitudes,
                constraint_idx=5,
                message=(f"Constraint 5 (linking): Last selected trip {last['id']} "
                         f"in direction {d} is not a final trip and has no "
                         f"successor; flow cannot reach sink"),
                lhs=0.0, rhs=1.0, operator='eq'
            )

    # ==========================================================
    # Constraint 6  (eq 1): Objective-value consistency  (Tier C anti-exploit)
    #   reported_obj  ==  alpha * VS_cost + sum_d TT_cost_d
    #
    # Full recompute is possible because the solution carries every
    # variable the objective depends on:
    #   - x_{u,v} appears in `vs_flows`  (arc-by-arc flow values)
    #   - y_{d,u,v} is implicit in the sorted selected_trips chain
    #     per direction (the TT path source -> t1 -> ... -> tk -> sink,
    #     which is the path the constraint checks above validate)
    #
    # Arc costs come directly from the math model:
    #   VS trip arc      (i^-, i^+)         cost 0
    #   VS in-line arc   (i^+, j^-)         cost = st(j) - et(i) - min_dwell_en(i)
    #   VS out-line arc  (i^+, j^-)         cost = pullin_en(i) + pullout_sn(j)
    #   VS start arc     (O^-, i^-)         cost = pullout_sn(i)
    #   VS end arc       (i^+, O^+)         cost = pullin_en(i)
    #   VS return arc    (O^+, O^-)         cost = M  (fleet-size proxy)
    #   TT trip-pair arc (i, j)             cost = (a(j) - a(i) - ideal_hw)^2  (quadratic)
    #   TT source/sink arcs                 cost 0
    #
    # We skip this check (silent, no extra violation) when the data
    # needed to recompute is unavailable: missing `vs_flows`
    # (constraint 3 already fires above) or missing/non-numeric
    # `objective_value`.  In every other case we recompute exactly
    # and reject if the reported value disagrees by more than the
    # 0.1% relative + 1e-3 absolute tolerance.
    # ==========================================================
    reported_obj_raw = solution.get('objective_value')
    reported_obj = None
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None

    if parsed_arcs is not None and reported_obj is not None:
        alpha = instance.get('objective_function', {}).get('alpha', 1.0)
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 1.0
        M_param = instance.get('M')
        if M_param is None:
            M_param = instance.get('objective_function', {}).get('M')
        if M_param is None:
            M_param = 10000  # default used by gurobi_code.py:148
        try:
            M_param = float(M_param)
        except (TypeError, ValueError):
            M_param = 10000.0

        # ---- VS cost from parsed_arcs ----
        vs_cost = 0.0
        for f_node, t_node, v in parsed_arcs:
            try:
                cost = _vs_arc_cost(instance, trip_map, f_node, t_node,
                                    O_minus, O_plus, M_param)
            except Exception:
                cost = None
            if cost is None:
                # Unknown arc shape — skip cost contribution for it.
                continue
            vs_cost += cost * v

        # ---- TT cost from the sorted-selected-trip chain ----
        # Per the constraint checks above, the chain source -> t1 -> ... -> tk
        # -> sink IS the TT path, so its cost is the sum of (i, j) arc penalties
        # for consecutive (ti, tj).  Source/sink arcs have cost 0.
        tt_cost = 0.0
        for d in dir_ids:
            sel = sel_by_dir[d]
            for idx in range(1, len(sel)):
                ti = sel[idx - 1]
                tj = sel[idx]
                hw_params = get_headway_params_minutes(
                    instance, ti['time_window_index']
                )
                if hw_params is None:
                    continue
                ideal, _, _ = hw_params
                actual = (tj['main_stop_arrival_time_minutes']
                          - ti['main_stop_arrival_time_minutes'])
                tt_cost += float((actual - ideal) ** 2)

        recomputed_obj = alpha * vs_cost + tt_cost
        abs_diff = abs(reported_obj - recomputed_obj)
        tol_obj = max(1e-3, 1e-3 * abs(recomputed_obj))

        if abs_diff > tol_obj:
            # Inline recording (not via add_violation) so we can apply a
            # 0.1% relative tolerance rather than the helper's hard 1e-5
            # absolute floor.
            violated_set.add(6)
            msg = (f"Constraint 6 (objective consistency, Eq 1): reported "
                   f"objective_value={reported_obj!r} differs from "
                   f"recomputed alpha*VS_cost + TT_cost = "
                   f"{alpha}*{vs_cost} + {tt_cost} = {recomputed_obj} "
                   f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})")
            violations.append(msg)
            normalizer = max(abs(recomputed_obj), EPS)
            violation_magnitudes.append({
                'constraint': 6,
                'lhs': float(reported_obj),
                'rhs': float(recomputed_obj),
                'raw_excess': float(abs_diff),
                'normalizer': float(normalizer),
                'ratio': float(abs_diff / normalizer),
            })

    # ==========================================================
    # Build result
    # ==========================================================
    feasible = len(violated_set) == 0
    # Deduplicate violated_constraints and aggregate violation messages
    unique_violated = sorted(violated_set)
    # Aggregate violation messages per constraint index
    aggregated_msgs = []
    seen_msgs = set()
    for msg in violations:
        if msg not in seen_msgs:
            aggregated_msgs.append(msg)
            seen_msgs.add(msg)

    return {
        'feasible': feasible,
        'violated_constraints': unique_violated,
        'violations': aggregated_msgs,
        'violation_magnitudes': violation_magnitudes if not feasible else [],
    }


# ============================================================
# VS arc-cost helper for objective recomputation
# ============================================================

def _vs_arc_cost(instance, trip_map, f_node, t_node,
                 O_minus, O_plus, M_param):
    """Return the math-model arc cost for a VS arc, or None if the
    arc shape is unrecognized (in which case the cost is omitted
    from the recomputed objective; the constraint check fires the
    appropriate violation separately if the arc is illegal).
    """
    # Return arc (O^+ -> O^-)
    if f_node == O_plus and t_node == O_minus:
        return float(M_param)
    # Start arc (O^- -> (i, 'start'))
    if f_node == O_minus and isinstance(t_node, tuple) \
            and len(t_node) == 2 and t_node[1] == 'start':
        tid = t_node[0]
        tr = trip_map.get(tid)
        if tr is None:
            return None
        _, _, pullout, _ = _terminal_params(
            instance, tr['start_terminal'], tr['time_window_index']
        )
        return float(pullout)
    # End arc ((i, 'end') -> O^+)
    if t_node == O_plus and isinstance(f_node, tuple) \
            and len(f_node) == 2 and f_node[1] == 'end':
        tid = f_node[0]
        tr = trip_map.get(tid)
        if tr is None:
            return None
        _, _, _, pullin = _terminal_params(
            instance, tr['end_terminal'], tr['time_window_index']
        )
        return float(pullin)
    # Trip arc ((i, 'start') -> (i, 'end'))
    if (isinstance(f_node, tuple) and isinstance(t_node, tuple)
            and len(f_node) == 2 and len(t_node) == 2
            and f_node[1] == 'start' and t_node[1] == 'end'
            and f_node[0] == t_node[0]):
        return 0.0
    # In-line / out-line compatibility arcs ((i, 'end') -> (j, 'start'))
    if (isinstance(f_node, tuple) and isinstance(t_node, tuple)
            and len(f_node) == 2 and len(t_node) == 2
            and f_node[1] == 'end' and t_node[1] == 'start'):
        i_tid, j_tid = f_node[0], t_node[0]
        ti = trip_map.get(i_tid)
        tj = trip_map.get(j_tid)
        if ti is None or tj is None:
            return None
        en_i = ti['end_terminal']
        sn_j = tj['start_terminal']
        wait = tj['departure_time_minutes'] - ti['arrival_time_minutes']
        if en_i == sn_j:
            # In-line compatibility: cost = wait - min_dwell_en(i)
            min_dwell, _, _, _ = _terminal_params(
                instance, en_i, ti['time_window_index']
            )
            return float(wait - min_dwell)
        # Out-line compatibility: cost = pullin_en(i) + pullout_sn(j)
        _, _, _, pullin_en_i = _terminal_params(
            instance, en_i, ti['time_window_index']
        )
        _, _, pullout_sn_j, _ = _terminal_params(
            instance, sn_j, tj['time_window_index']
        )
        return float(pullin_en_i + pullout_sn_j)
    return None


def _terminal_params(instance, terminal, tw_index):
    """Return (min_dwell, max_dwell, pull_out, pull_in) for a terminal
    in a given time window, matching gurobi_code.build_vs_graph_pure's
    parameter lookup.  Missing entries default to zero (consistent with
    gurobi_code's fallback)."""
    min_dwell, max_dwell = 0, 9999
    for entry in instance.get('stopping_times', {}).get(terminal, []):
        if entry['time_window_index'] == tw_index:
            min_dwell = entry['min_stopping_time_minutes']
            max_dwell = entry['max_stopping_time_minutes']
            break
    pull_out, pull_in = 0, 0
    for entry in instance.get('pull_in_out_times', {}).get(terminal, []):
        if entry['time_window_index'] == tw_index:
            pull_out = entry['pull_out_time_minutes']
            pull_in = entry['pull_in_time_minutes']
            break
    return min_dwell, max_dwell, pull_out, pull_in


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Feasibility checker for ITTVS solutions (Carosi et al. 2019)'
    )
    parser.add_argument('--instance_path', required=True,
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True,
                        help='Path to the candidate solution JSON file')
    parser.add_argument('--result_path', required=True,
                        help='Path to write the feasibility result JSON file')
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result['feasible'] else "INFEASIBLE"
    print(f"Feasibility: {status}")
    if not result['feasible']:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
