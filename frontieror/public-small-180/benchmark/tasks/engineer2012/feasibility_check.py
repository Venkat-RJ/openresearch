#!/usr/bin/env python3
"""
Feasibility Checker for Single-Product Maritime Inventory Routing.

Checks hard constraints from the mathematical formulation in Engineer et al. (2012):
  "A Branch-Price-and-Cut Algorithm for Single-Product Maritime Inventory Routing"

Constraints checked (numbered as in the paper formulation):
  (1) Inventory balance at supply ports
  (2) Inventory balance at demand ports
  (3) Storage capacity bounds: 0 <= I_{j,t} <= Q_{j,t}
  (4) Convexity: each vessel executes exactly one voyage (sum lambda = 1)
  (5) Non-negativity: lambda >= 0
  (6) Integrality of load/discharge decisions (z in {0,1})
  (7) Flow conservation (path from source to sink)
  (8) Load/discharge only if node is visited
  (9) Load/discharge quantity bounds: F_min * z <= |f| <= F_max * z
  (10) Vessel inventory tracking along path
  (11) Vessel inventory capacity: 0 <= F_n <= U_n
  (12) Arc integrality
  (13) Load/discharge decision integrality (z in {0,1})

Usage:
  python feasibility_check.py --instance_path instance_1.json \
      --solution_path efficient_solution_1.json \
      --result_path efficient_feasi_result_1.json
"""

import argparse
import ast
import json
import math
import sys
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('cost_breakdown', 'objective_value', 'port_inventories', 'vessel_operations', 'vessel_routes')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('cost_breakdown', 'objective_value', 'port_inventories', 'vessel_routes')
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


def parse_instance(data):
    """Parse instance data into a structured format."""
    T = data["planning_horizon"]
    vessels = {}
    for v in data["vessels"]:
        vessels[v["id"]] = {
            "id": v["id"],
            "capacity": v["capacity"],
            "avail_start": v["availability_start"],
            "avail_end": v["availability_end"],
            "initial_load": v.get("initial_load", 0),
            "transport_cost_per_day": v["transport_cost_per_day"],
            "demurrage_cost_per_day": v["demurrage_cost_per_day"],
        }

    supply_ports = {}
    for sp in data["supply_ports"]:
        supply_ports[sp["port_id"]] = {
            "port_id": sp["port_id"],
            "type": "supply",
            "initial_inventory": sp["initial_inventory"],
            "rates": sp["production_rate"],
            "storage_capacity": sp["storage_capacity"],
            "F_min": sp["F_min"],
            "F_max": sp["F_max"],
            "unit_cost": sp["unit_procurement_cost"],
        }

    demand_ports = {}
    for dp in data["demand_ports"]:
        demand_ports[dp["port_id"]] = {
            "port_id": dp["port_id"],
            "type": "demand",
            "initial_inventory": dp["initial_inventory"],
            "rates": dp["consumption_rate"],
            "storage_capacity": dp["storage_capacity"],
            "F_min": dp["F_min"],
            "F_max": dp["F_max"],
            "unit_cost": dp["unit_supply_revenue"],
        }

    travel_times = {}
    for key, val in data["travel_times"].items():
        a, b = key.split("->")
        travel_times[(a, b)] = val

    draft_limits = {}
    for key, val in data.get("draft_limits", {}).items():
        a, b = key.split("->")
        draft_limits[(a, b)] = val

    return T, vessels, supply_ports, demand_ports, travel_times, draft_limits


def parse_solution(sol_data, vessels_info):
    """Parse the serialized operation witness for every vessel."""
    vessel_visits = {}
    operations = sol_data["vessel_operations"]
    for vid in vessels_info:
        visits = []
        for op in operations.get(vid, []):
            visits.append({
                "port": op["port"],
                "time": int(op["time"]),
                "load_discharge": float(op["quantity"]),
                "type": op["type"],
            })
        visits.sort(key=lambda visit: visit["time"])
        vessel_visits[vid] = visits
    return vessel_visits


def parse_route_node(serialized):
    """Parse the tuple representation emitted by gurobi_code.py."""
    node = ast.literal_eval(serialized)
    if not isinstance(node, tuple) or len(node) != 2:
        raise ValueError(f"invalid route node {serialized!r}")
    return node


def check_feasibility(instance_data, solution_data):
    """Check all hard constraints and return violations."""
    tol = 1e-5
    eps = 1e-5

    T, vessels, supply_ports, demand_ports, travel_times, draft_limits = \
        parse_instance(instance_data)

    vessel_visits = parse_solution(solution_data, vessels)

    all_ports = {}
    all_ports.update(supply_ports)
    all_ports.update(demand_ports)

    violations = []
    violation_magnitudes = []

    def add_violation(constraint_idx, message, lhs, rhs, violation_amount):
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violations.append((constraint_idx, message))
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio,
        })

    # =====================================================================
    # Constraints (7), (8), (10), and (12): validate the complete serialized
    # source-to-sink path, arc timing, operation support, and commodity flow.
    # Route entries are unordered in the JSON, so first reconstruct the path.
    # =====================================================================
    expected_vessels = set(vessels)
    route_witness = solution_data["vessel_routes"]
    operation_witness = solution_data["vessel_operations"]
    if set(route_witness) != expected_vessels:
        add_violation(
            7,
            "vessel_routes keys do not exactly cover the instance vessels",
            float(len(route_witness)), float(len(expected_vessels)), 1.0,
        )
    if set(operation_witness) != expected_vessels:
        add_violation(
            8,
            "vessel_operations keys do not exactly cover the instance vessels",
            float(len(operation_witness)), float(len(expected_vessels)), 1.0,
        )

    transport_cost = 0.0
    demurrage_cost = 0.0
    for vid, vessel in vessels.items():
        source = ("src", vid)
        sink = ("snk", vid)
        serialized_arcs = route_witness.get(vid, [])
        parsed_arcs = []
        route_nodes = set()

        for arc_position, arc in enumerate(serialized_arcs):
            try:
                tail = parse_route_node(arc["tail"])
                head = parse_route_node(arc["head"])
                arc_type = arc["type"]
                reported_cost = float(arc["cost"])
                product_flow = float(arc["product_flow"])
                if not all(math.isfinite(value) for value in
                           (reported_cost, product_flow)):
                    raise ValueError("arc cost/product_flow is not finite")
            except (KeyError, TypeError, ValueError, SyntaxError) as error:
                add_violation(
                    12,
                    f"Vessel {vid} route arc {arc_position} is malformed: {error}",
                    1.0, 0.0, 1.0,
                )
                continue

            def is_port_time(node):
                return (
                    node[0] in all_ports
                    and isinstance(node[1], int)
                    and not isinstance(node[1], bool)
                    and vessel["avail_start"] <= node[1] <= vessel["avail_end"]
                )

            canonical_cost = None
            valid_arc = True
            if arc_type == "source":
                valid_arc = (
                    tail == source
                    and is_port_time(head)
                    and head[1] == vessel["avail_start"]
                )
                canonical_cost = 0.0
            elif arc_type == "sink":
                valid_arc = is_port_time(tail) and head == sink
                canonical_cost = 0.0
            elif arc_type == "demurrage":
                valid_arc = (
                    is_port_time(tail)
                    and is_port_time(head)
                    and tail[0] == head[0]
                    and head[1] == tail[1] + 1
                )
                canonical_cost = float(vessel["demurrage_cost_per_day"])
            elif arc_type == "travel":
                travel_time = travel_times.get((tail[0], head[0]))
                valid_arc = (
                    is_port_time(tail)
                    and is_port_time(head)
                    and tail[0] != head[0]
                    and travel_time is not None
                    and head[1] == tail[1] + travel_time
                )
                if travel_time is not None:
                    canonical_cost = (
                        float(vessel["transport_cost_per_day"])
                        * float(travel_time)
                    )
            else:
                valid_arc = False

            if not valid_arc or canonical_cost is None:
                add_violation(
                    7,
                    f"Vessel {vid} contains an invalid {arc_type!r} arc "
                    f"{tail!r} -> {head!r}",
                    1.0, 0.0, 1.0,
                )
                continue
            if abs(reported_cost - canonical_cost) > tol:
                add_violation(
                    7,
                    f"Vessel {vid} arc {tail!r}->{head!r} reports cost "
                    f"{reported_cost}, expected {canonical_cost}",
                    reported_cost, canonical_cost,
                    abs(reported_cost - canonical_cost),
                )
            if product_flow < -tol:
                add_violation(
                    10,
                    f"Vessel {vid} has negative product flow on "
                    f"{tail!r}->{head!r}",
                    product_flow, 0.0, -product_flow,
                )

            endpoint_limits = []
            for node in (tail, head):
                if is_port_time(node):
                    endpoint_limits.append(min(
                        vessel["capacity"],
                        draft_limits.get((vid, node[0]), vessel["capacity"]),
                    ))
            flow_limit = min(endpoint_limits, default=vessel["capacity"])
            if product_flow > flow_limit + tol:
                add_violation(
                    11,
                    f"Vessel {vid} product flow {product_flow} exceeds arc "
                    f"capacity {flow_limit} on {tail!r}->{head!r}",
                    product_flow, flow_limit, product_flow - flow_limit,
                )

            parsed_arcs.append({
                "tail": tail,
                "head": head,
                "type": arc_type,
                "flow": product_flow,
            })
            route_nodes.update((tail, head))
            if arc_type == "travel":
                transport_cost += canonical_cost
            elif arc_type == "demurrage":
                demurrage_cost += canonical_cost

        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for arc in parsed_arcs:
            outgoing[arc["tail"]].append(arc)
            incoming[arc["head"]].append(arc)
        structural_error = False
        for node in route_nodes | {source, sink}:
            expected_in = 0 if node == source else 1
            expected_out = 0 if node == sink else 1
            if len(incoming[node]) != expected_in or len(outgoing[node]) != expected_out:
                structural_error = True
                add_violation(
                    7,
                    f"Vessel {vid} route degree is invalid at {node!r}: "
                    f"in={len(incoming[node])}, out={len(outgoing[node])}",
                    float(len(incoming[node]) + len(outgoing[node])),
                    float(expected_in + expected_out), 1.0,
                )

        if not structural_error:
            traversed = []
            current = source
            while current != sink and len(traversed) <= len(parsed_arcs):
                arc = outgoing[current][0]
                traversed.append(arc)
                current = arc["head"]
            if current != sink or len(traversed) != len(parsed_arcs):
                add_violation(
                    7,
                    f"Vessel {vid} selected arcs do not form one complete "
                    "source-to-sink path",
                    float(len(traversed)), float(len(parsed_arcs)), 1.0,
                )

        operation_by_node = {}
        for visit in vessel_visits[vid]:
            node = (visit["port"], visit["time"])
            if node in operation_by_node:
                continue  # Constraint 6 records the duplicate below.
            operation_by_node[node] = visit["load_discharge"]
            expected_type = (
                "load" if visit["port"] in supply_ports else "discharge"
            )
            if visit["type"] != expected_type:
                add_violation(
                    8,
                    f"Vessel {vid} operation at {node!r} has type "
                    f"{visit['type']!r}, expected {expected_type!r}",
                    1.0, 0.0, 1.0,
                )
            if node not in outgoing:
                add_violation(
                    8,
                    f"Vessel {vid} operation at {node!r} is not supported by "
                    "a selected route visit",
                    1.0, 0.0, 1.0,
                )

        if len(outgoing[source]) == 1:
            source_flow = outgoing[source][0]["flow"]
            if abs(source_flow - vessel["initial_load"]) > tol:
                add_violation(
                    10,
                    f"Vessel {vid} source flow {source_flow} does not match "
                    f"initial load {vessel['initial_load']}",
                    source_flow, vessel["initial_load"],
                    abs(source_flow - vessel["initial_load"]),
                )
        for node in route_nodes - {source, sink}:
            if len(incoming[node]) != 1 or len(outgoing[node]) != 1:
                continue
            incoming_flow = incoming[node][0]["flow"]
            outgoing_flow = outgoing[node][0]["flow"]
            operation = operation_by_node.get(node, 0.0)
            flow_difference = abs(incoming_flow + operation - outgoing_flow)
            if flow_difference > 1e-3:
                add_violation(
                    10,
                    f"Vessel {vid} commodity flow is inconsistent at "
                    f"{node!r}: incoming {incoming_flow} + operation "
                    f"{operation} != outgoing {outgoing_flow}",
                    incoming_flow + operation, outgoing_flow, flow_difference,
                )

    # =====================================================================
    # Compute net load/discharge at each (port, time) across all vessels
    # f_total[port_id][t] = sum over vessels of f_{j,t}^{v,r}
    # For supply ports: f > 0 means loading (removing from port inventory)
    # For demand ports: f < 0 means discharging (adding to port inventory)
    # In the model: I_{j,t} = I_{j,t-1} + b_{j,t} - sum_v f_{j,t}
    # =====================================================================
    f_total = defaultdict(lambda: defaultdict(float))
    for vid, visits in vessel_visits.items():
        for visit in visits:
            f_total[visit["port"]][visit["time"]] += visit["load_discharge"]

    # =====================================================================
    # Compute port inventories
    # =====================================================================
    port_inventory = {}  # port_inventory[port_id][t] for t=0..T
    for pid, port in all_ports.items():
        inv = [0.0] * (T + 1)
        inv[0] = port["initial_inventory"]
        for t in range(1, T + 1):
            if port["type"] == "supply":
                # Constraint (1): I_{j,t} = I_{j,t-1} + b_{j,t} - sum_v f_{j,t}
                inv[t] = inv[t - 1] + port["rates"][t - 1] - f_total[pid][t]
            else:
                # Constraint (2): I_{j,t} = I_{j,t-1} - b_{j,t} - sum_v f_{j,t}
                # For demand: f < 0 for discharge, so -f > 0 adds to inventory
                inv[t] = inv[t - 1] - port["rates"][t - 1] - f_total[pid][t]
        port_inventory[pid] = inv

    # The reported inventory trajectory is a derived witness and must cover
    # every port/time exactly; otherwise a candidate could alter operations
    # while leaving an unrelated inventory summary in the JSON.
    reported_inventories = solution_data["port_inventories"]
    if set(reported_inventories) != set(all_ports):
        add_violation(
            1,
            "port_inventories keys do not exactly cover all instance ports",
            float(len(reported_inventories)), float(len(all_ports)), 1.0,
        )
    for pid in all_ports:
        entries = reported_inventories.get(pid, [])
        by_time = {}
        for entry in entries:
            time_value = int(entry["time"])
            if time_value in by_time:
                add_violation(
                    1,
                    f"Port {pid} reports inventory twice at t={time_value}",
                    2.0, 1.0, 1.0,
                )
            by_time[time_value] = float(entry["inventory"])
        expected_times = set(range(1, T + 1))
        if set(by_time) != expected_times:
            add_violation(
                1,
                f"Port {pid} inventory times do not exactly cover 1..{T}",
                float(len(by_time)), float(T), 1.0,
            )
        for time_value in expected_times & set(by_time):
            difference = abs(
                by_time[time_value] - port_inventory[pid][time_value]
            )
            if difference > 1e-3:
                add_violation(
                    1,
                    f"Port {pid} reported inventory {by_time[time_value]} at "
                    f"t={time_value}, expected {port_inventory[pid][time_value]}",
                    by_time[time_value], port_inventory[pid][time_value],
                    difference,
                )

    # =====================================================================
    # Constraint (1): Inventory balance at supply ports
    # I_{j,t} = I_{j,t-1} + b_{j,t} - sum_v f_{j,t}
    # This is an equality constraint. We check it by verifying the computed
    # inventory is consistent (it's computed directly, so the balance holds
    # by construction). The real check is whether the inventory stays within
    # bounds (Constraint 3). However, we still verify the balance equation
    # holds as an equality for supply ports.
    # =====================================================================
    for pid, port in supply_ports.items():
        for t in range(1, T + 1):
            I_prev = port_inventory[pid][t - 1]
            b_jt = port["rates"][t - 1]
            f_sum = f_total[pid][t]
            lhs = port_inventory[pid][t]
            rhs_val = I_prev + b_jt - f_sum
            diff = abs(lhs - rhs_val)
            if diff > tol:
                add_violation(
                    1,
                    f"Supply port {pid} inventory balance violated at t={t}: "
                    f"I[t]={lhs:.4f} != I[t-1]+b-f = {rhs_val:.4f}",
                    lhs, rhs_val, diff
                )

    # =====================================================================
    # Constraint (2): Inventory balance at demand ports
    # I_{j,t} = I_{j,t-1} - b_{j,t} - sum_v f_{j,t}
    # =====================================================================
    for pid, port in demand_ports.items():
        for t in range(1, T + 1):
            I_prev = port_inventory[pid][t - 1]
            b_jt = port["rates"][t - 1]
            f_sum = f_total[pid][t]
            lhs = port_inventory[pid][t]
            rhs_val = I_prev - b_jt - f_sum
            diff = abs(lhs - rhs_val)
            if diff > tol:
                add_violation(
                    2,
                    f"Demand port {pid} inventory balance violated at t={t}: "
                    f"I[t]={lhs:.4f} != I[t-1]-b-f = {rhs_val:.4f}",
                    lhs, rhs_val, diff
                )

    # =====================================================================
    # Constraint (3): Storage capacity bounds: 0 <= I_{j,t} <= Q_{j,t}
    # This is two inequalities: I_{j,t} >= 0 and I_{j,t} <= Q_{j,t}
    # =====================================================================
    for pid, port in all_ports.items():
        for t in range(1, T + 1):
            inv = port_inventory[pid][t]
            Q_jt = port["storage_capacity"][t - 1]

            # Lower bound: I_{j,t} >= 0  =>  LHS = -I_{j,t}, RHS = 0 (<=)
            # Or equivalently: 0 <= I_{j,t}, violation if I_{j,t} < 0
            if inv < -tol:
                violation_amount = -inv  # how much below 0
                add_violation(
                    3,
                    f"Port {pid} inventory below 0 at t={t}: "
                    f"I={inv:.4f} < 0",
                    inv, 0.0, violation_amount
                )

            # Upper bound: I_{j,t} <= Q_{j,t}
            if inv > Q_jt + tol:
                violation_amount = inv - Q_jt
                add_violation(
                    3,
                    f"Port {pid} inventory exceeds capacity at t={t}: "
                    f"I={inv:.4f} > Q={Q_jt:.4f}",
                    inv, Q_jt, violation_amount
                )

    # =====================================================================
    # Constraint (4): Convexity - each vessel has exactly one voyage
    # sum_{r in R_v} lambda^{v,r} = 1
    # In the realized solution, each vessel has exactly one schedule.
    # This is satisfied by construction since the solution specifies
    # one schedule per vessel. We just verify each vessel is present.
    # =====================================================================
    for vid in vessels:
        # Each vessel should appear in the solution (even if idle)
        if vid not in vessel_visits:
            add_violation(
                4,
                f"Vessel {vid} has no voyage assigned (convexity constraint)",
                0.0, 1.0, 1.0
            )

    # =====================================================================
    # Constraint (5): Non-negativity: lambda >= 0
    # Trivially satisfied since the solution represents a realized (integral)
    # voyage selection with lambda = 1 for the chosen voyage.
    # No check needed.
    # =====================================================================

    # =====================================================================
    # Constraint (6): Integrality on load/discharge decisions
    # sum_r z_{j,t}^{v,r} lambda^{v,r} in {0,1}
    # In realized solution, each vessel visits each (port, time) at most once.
    # Check: no vessel visits the same (port, time) more than once.
    # =====================================================================
    for vid, visits in vessel_visits.items():
        pt_count = defaultdict(int)
        for visit in visits:
            key = (visit["port"], visit["time"])
            pt_count[key] += 1
        for (port, time), count in pt_count.items():
            if count > 1:
                add_violation(
                    6,
                    f"Vessel {vid} visits ({port}, t={time}) {count} times "
                    f"(integrality requires at most 1)",
                    float(count), 1.0, float(count - 1)
                )

    # =====================================================================
    # Constraint (9): Load/discharge quantity bounds
    # F_n^{min} * z_n <= f_n <= F_n^{max} * z_n
    # For supply: F_min <= f <= F_max (f > 0 for loading)
    # For demand: -F_max <= f <= -F_min (f < 0 for discharging)
    # So |f| must be in [F_min, F_max] when the port is visited (z=1)
    # =====================================================================
    for vid, visits in vessel_visits.items():
        for visit in visits:
            pid = visit["port"]
            t = visit["time"]
            f_val = visit["load_discharge"]
            port = all_ports[pid]
            F_min = port["F_min"]
            F_max = port["F_max"]
            abs_f = abs(f_val)

            if port["type"] == "supply":
                # f >= F_min when visiting
                if f_val < F_min - tol:
                    violation_amount = F_min - f_val
                    add_violation(
                        9,
                        f"Vessel {vid} at supply port {pid} t={t}: "
                        f"load={f_val:.4f} < F_min={F_min}",
                        f_val, F_min, violation_amount
                    )
                # f <= F_max when visiting
                if f_val > F_max + tol:
                    violation_amount = f_val - F_max
                    add_violation(
                        9,
                        f"Vessel {vid} at supply port {pid} t={t}: "
                        f"load={f_val:.4f} > F_max={F_max}",
                        f_val, F_max, violation_amount
                    )
            else:
                # Demand port: f <= 0, |f| >= F_min, |f| <= F_max
                if f_val > tol:
                    violation_amount = f_val
                    add_violation(
                        9,
                        f"Vessel {vid} at demand port {pid} t={t}: "
                        f"load_discharge={f_val:.4f} > 0 (should be <= 0)",
                        f_val, 0.0, violation_amount
                    )
                if abs_f < F_min - tol:
                    violation_amount = F_min - abs_f
                    add_violation(
                        9,
                        f"Vessel {vid} at demand port {pid} t={t}: "
                        f"|discharge|={abs_f:.4f} < F_min={F_min}",
                        abs_f, F_min, violation_amount
                    )
                if abs_f > F_max + tol:
                    violation_amount = abs_f - F_max
                    add_violation(
                        9,
                        f"Vessel {vid} at demand port {pid} t={t}: "
                        f"|discharge|={abs_f:.4f} > F_max={F_max}",
                        abs_f, F_max, violation_amount
                    )

    # =====================================================================
    # Constraint (11): Vessel inventory capacity: 0 <= F_n <= U_n
    # Track vessel load along its path. U_n = min(Q_v, draft_limit(v,j))
    # The vessel starts with initial_load and accumulates f values.
    #
    # Per math_model.txt constraint (11): F_n is the vessel inventory
    # ENTERING node n (before loading/discharging at n). This is defined
    # by constraint (10): F_n = sum_{e=(n',n)} x_e (F_{n'} + f_{n'}).
    # Equivalently, from the LP subproblem constraint (14):
    #   0 <= sum_{i=0}^{k-1} f_{n_i} <= U_{n_k}  for k=1,...,K
    # which bounds the cumulative load BEFORE operations at node n_k.
    # =====================================================================
    for vid, visits in vessel_visits.items():
        if not visits:
            continue
        vessel = vessels[vid]
        Q_v = vessel["capacity"]
        load = vessel["initial_load"]
        for i, visit in enumerate(visits):
            pid = visit["port"]
            t = visit["time"]
            f_val = visit["load_discharge"]

            # Draft limit: U_n bounds the inventory ENTERING the node
            dl = draft_limits.get((vid, pid), Q_v)
            U_n = min(Q_v, dl)

            # Check 0 <= load <= U_n BEFORE loading/discharging at this node
            if load < -tol:
                violation_amount = -load
                add_violation(
                    11,
                    f"Vessel {vid} has negative load={load:.4f} entering "
                    f"{pid} at t={t}",
                    load, 0.0, violation_amount
                )
            if load > U_n + tol:
                violation_amount = load - U_n
                add_violation(
                    11,
                    f"Vessel {vid} load={load:.4f} exceeds capacity "
                    f"U_n={U_n:.4f} entering {pid} at t={t}",
                    load, U_n, violation_amount
                )

            # Update load AFTER checking the constraint
            load += f_val

    # =====================================================================
    # Objective and cost summary: every component is derivable from the route
    # and operation witnesses, so bind both cost_breakdown and objective_value
    # to those decisions instead of trusting candidate-reported numbers.
    # =====================================================================
    procurement_cost = sum(
        supply_ports[visit["port"]]["unit_cost"]
        * visit["load_discharge"]
        for visits in vessel_visits.values()
        for visit in visits
        if visit["port"] in supply_ports
    )
    supply_revenue = sum(
        demand_ports[visit["port"]]["unit_cost"]
        * (-visit["load_discharge"])
        for visits in vessel_visits.values()
        for visit in visits
        if visit["port"] in demand_ports
    )
    expected_costs = {
        "transport_cost": transport_cost,
        "demurrage_cost": demurrage_cost,
        "procurement_cost": procurement_cost,
        "supply_revenue": supply_revenue,
    }
    expected_costs["total_cost"] = (
        transport_cost + demurrage_cost + procurement_cost - supply_revenue
    )
    cost_breakdown = solution_data["cost_breakdown"]
    if set(cost_breakdown) != set(expected_costs):
        add_violation(
            14,
            "cost_breakdown fields do not exactly match the five documented "
            "cost components",
            float(len(cost_breakdown)), float(len(expected_costs)), 1.0,
        )
    for component, expected_value in expected_costs.items():
        if component not in cost_breakdown:
            continue
        reported_value = float(cost_breakdown[component])
        difference = abs(reported_value - expected_value)
        component_tolerance = max(1e-3, 1e-7 * abs(expected_value))
        if not math.isfinite(reported_value) or difference > component_tolerance:
            add_violation(
                14,
                f"cost_breakdown.{component}={reported_value} does not match "
                f"the decision-derived value {expected_value}",
                reported_value, expected_value, difference,
            )

    reported_objective = float(solution_data["objective_value"])
    objective_difference = abs(
        reported_objective - expected_costs["total_cost"]
    )
    objective_tolerance = max(
        1e-3, 1e-7 * abs(expected_costs["total_cost"])
    )
    if objective_difference > objective_tolerance:
        add_violation(
            14,
            f"objective_value={reported_objective} does not match the "
            f"decision-derived total {expected_costs['total_cost']}",
            reported_objective, expected_costs["total_cost"],
            objective_difference,
        )

    # =====================================================================
    # Compile results
    # =====================================================================
    violated_set = sorted(set(c for c, _ in violations))
    violation_messages = []
    # Aggregate messages per constraint
    msgs_by_constraint = defaultdict(list)
    for c, msg in violations:
        msgs_by_constraint[c].append(msg)
    for c in violated_set:
        msgs = msgs_by_constraint[c]
        if len(msgs) <= 3:
            violation_messages.extend(msgs)
        else:
            violation_messages.extend(msgs[:2])
            violation_messages.append(
                f"... and {len(msgs) - 2} more violation(s) for constraint {c}"
            )

    feasible = len(violated_set) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_set,
        "violations": violation_messages,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Maritime Inventory Routing")
    parser.add_argument("--instance_path", required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to JSON solution file")
    parser.add_argument("--result_path", required=True,
                        help="Path to write JSON feasibility result")
    args = parser.parse_args()

    instance_data = load_json(args.instance_path)
    solution_data = load_json(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(solution_data)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return

    result = check_feasibility(instance_data, solution_data)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"{status}: {args.solution_path}")
    if not result["feasible"]:
        print(f"  Violated constraints: {result['violated_constraints']}")
        for msg in result["violations"][:5]:
            print(f"  - {msg}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
