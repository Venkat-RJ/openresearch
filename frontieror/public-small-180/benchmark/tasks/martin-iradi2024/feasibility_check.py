#!/usr/bin/env python3
"""
Feasibility checker for the Multi-Port Continuous Berth Allocation Problem (MCBAP).

Checks all hard constraints from the MIP formulation (paper Equations 3-19),
numbered as Constraint 1-17 counting from top to bottom in the formulation,
plus an additional Constraint 18 that recomputes the objective from solution
variables and rejects when the reported objective_value disagrees (Tier C
defense against LLM score-gaming exploits).

Constraint mapping:
  1  (Eq 3):  Quay space limits:       x_i^c + l_i <= L^p
  2  (Eq 4):  Left-right positioning:   x_i^c + l_i <= x_j^{c'} + L^p(1 - sigma)
  3  (Eq 5):  Time ordering:            y_i^c + h_i^c <= y_j^{c'} + M(1 - delta)
  4  (Eq 6):  Non-overlapping:          sigma+sigma+delta+delta >= 1
  5  (Eq 7):  Arrival time linking:     y_i^c + h_i^c + travel = a_i^{c+1}
  6  (Eq 8):  Berthing after arrival:   a_i^c <= y_i^c
  7  (Eq 9):  Earliest start time:      EST_i^c <= y_i^c
  8  (Eq 10): Delay over EFT:           y_i^c + h_i^c - EFT <= d_i^c
  9  (Eq 11): Delay over LFT:           y_i^c + h_i^c - LFT <= u_i^c
  10 (Eq 12): Handling time:            (1 + beta*r)*h0 = h
  11 (Eq 13): Position deviation right: x - x0 <= r
  12 (Eq 14): Position deviation left:  x0 - x <= r
  13 (Eq 15): One speed per leg:        sum_s v_{i,s}^c = 1
  14 (Eq 16): Non-negativity x, y:      x, y >= 0
  15 (Eq 17): Non-negativity a,h,d,u,r: a,h,d,u,r >= 0
  16 (Eq 18): Binary speed domain:      v_{i,s}^c in {0,1}
  17 (Eq 19): Binary sigma/delta:       sigma, delta in {0,1}
  18 (Obj):   Objective envelope:       reported objective_value must lie
                                        between a tight lower bound recomputed
                                        from solution variables and a loose
                                        per-instance worst-case upper bound
"""

import argparse
import json
import math
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'ships')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('objective_value', 'ships')
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



def main():
    parser = argparse.ArgumentParser(description="MCBAP feasibility checker")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON")
    parser.add_argument("--result_path", required=True, help="Path to write result JSON")
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

    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # --- Extract instance data ---
    beta = instance["handling_time_deviation_factor_beta_per_meter"]
    ports = {p["id"]: p for p in instance["ports"]}
    distances = instance["distances_nautical_miles"]
    travel_times = instance["travel_time_hours_per_nm_by_speed"]
    speeds = instance["speeds_knots"]

    # -------------------------------------------------------------------------
    # Empty solution guard: if solver returned no incumbent (objective_value
    # is None and ships list empty/missing), do NOT vacuously return feasible
    # — solution structure REQUIRES ships routing every port_visit. Mark
    # feasi=False with a "no candidate" violation. (Earlier versions returned
    # feasi=True because subsequent loops iterated over an empty ships list.)
    # -------------------------------------------------------------------------
    if solution.get("objective_value") is None and not solution.get("ships"):
        result = {
            "feasible": False,
            "violated_constraints": [1],
            "violations": [
                "Empty solution (objective_value is None and ships list empty)"
                " — no candidate routing to verify; constraint 1 (every "
                "instance ship/port_visit must have a route) cannot be "
                "satisfied."
            ],
            "violation_magnitudes": [{
                "constraint": 1, "lhs": 0.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            }],
        }
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("INFEASIBLE: Empty solution")
        return

    # --- Build solution lookups ---
    sol_ships = {s["ship_id"]: s for s in solution["ships"]}
    sol_visits = {}
    for s in solution["ships"]:
        for pv in s["port_visits"]:
            sol_visits[(s["ship_id"], pv["port_call_index"])] = pv

    inst_ships = {s["id"]: s for s in instance["ships"]}
    inst_visits = {}
    for s in instance["ships"]:
        for pv in s["port_visits"]:
            inst_visits[(s["id"], pv["port_call_index"])] = pv

    def add_violation(constraint_idx, msg, lhs, rhs, operator="<="):
        if operator == "<=":
            violation_amount = lhs - rhs
        elif operator == ">=":
            violation_amount = rhs - lhs
        elif operator == "=":
            violation_amount = abs(lhs - rhs)
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            violated_constraints.add(constraint_idx)
            violations.append(msg)
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # A dictionary comprehension would otherwise hide duplicate ship/call
    # entries, while all subsequent loops silently skip missing entries.  Make
    # the serialized witness an exact representation of the instance before
    # checking any of its numerical constraints.
    expected_ship_ids = [s["id"] for s in instance["ships"]]
    reported_ship_ids = [s["ship_id"] for s in solution["ships"]]
    if (len(reported_ship_ids) != len(set(reported_ship_ids))
            or set(reported_ship_ids) != set(expected_ship_ids)):
        missing = sorted(set(expected_ship_ids) - set(reported_ship_ids))
        extra = sorted(set(reported_ship_ids) - set(expected_ship_ids))
        add_violation(
            1,
            f"Ship witness mismatch: missing={missing}, extra={extra}, "
            f"reported_entries={len(reported_ship_ids)}, "
            f"unique_entries={len(set(reported_ship_ids))}",
            0.0,
            1.0,
            "=",
        )

    expected_visit_keys = set(inst_visits)
    reported_visit_keys = []
    for sol_ship in solution["ships"]:
        sid = sol_ship["ship_id"]
        reported_visit_keys.extend(
            (sid, pv["port_call_index"])
            for pv in sol_ship.get("port_visits", [])
        )
    if (len(reported_visit_keys) != len(set(reported_visit_keys))
            or set(reported_visit_keys) != expected_visit_keys):
        missing = sorted(expected_visit_keys - set(reported_visit_keys))
        extra = sorted(set(reported_visit_keys) - expected_visit_keys)
        add_violation(
            1,
            f"Port-call witness mismatch: missing={missing[:20]}, "
            f"extra={extra[:20]}, reported_entries={len(reported_visit_keys)}, "
            f"expected_entries={len(expected_visit_keys)}",
            0.0,
            1.0,
            "=",
        )

    # --- Gather all berth occupations per port (for non-overlapping checks) ---
    port_occupations = defaultdict(list)

    for s in instance["ships"]:
        ship_id = s["id"]
        if ship_id not in sol_ships:
            continue
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            port_id = pv_inst["port_id"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            port_occupations[port_id].append({
                "label": f"Ship {ship_id}",
                "ship_id": ship_id,
                "call_idx": c,
                "x": sol_pv["berthing_position"],
                "length": s["length_meters"],
                "y_start": sol_pv["start_time"],
                "y_end": sol_pv["start_time"] + sol_pv["handling_time"],
                "is_external": False,
            })

    for ext_port in instance["external_ships"]:
        port_id = ext_port["port_id"]
        for ext_s in ext_port["ships"]:
            port_occupations[port_id].append({
                "label": f"External {ext_s['id']} at {port_id}",
                "ship_id": f"ext_{port_id}_{ext_s['id']}",
                "call_idx": 0,
                "x": ext_s["berthing_position"],
                "length": ext_s["length_meters"],
                "y_start": ext_s["start_time_hours"],
                "y_end": ext_s["start_time_hours"] + ext_s["handling_time_hours"],
                "is_external": True,
            })

    # =========================================================================
    # Constraint 1 (Eq 3): x_i^c + l_i <= L^p
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            port_id = pv_inst["port_id"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            x = sol_pv["berthing_position"]
            l = s["length_meters"]
            L_p = ports[port_id]["quay_length_meters"]
            add_violation(
                1,
                f"Ship {ship_id} call {c} at {port_id}: x({x})+l({l})={x+l} > L_p({L_p})",
                x + l, L_p, "<="
            )

    # Also check external ships
    for ext_port in instance["external_ships"]:
        port_id = ext_port["port_id"]
        L_p = ports[port_id]["quay_length_meters"]
        for ext_s in ext_port["ships"]:
            x = ext_s["berthing_position"]
            l = ext_s["length_meters"]
            add_violation(
                1,
                f"External ship {ext_s['id']} at {port_id}: x({x})+l({l})={x+l} > L_p({L_p})",
                x + l, L_p, "<="
            )

    # =========================================================================
    # Constraints 2-4 (Eqs 4-6): Non-overlapping
    # Constraints 2 & 3 define sigma/delta via big-M (always satisfiable).
    # Constraint 4 requires at least one ordering to hold.
    # We derive sigma/delta from the solution and check constraint 4.
    # =========================================================================
    for port_id, occs in port_occupations.items():
        L_p = ports[port_id]["quay_length_meters"]
        for i_idx in range(len(occs)):
            for j_idx in range(i_idx + 1, len(occs)):
                oi = occs[i_idx]
                oj = occs[j_idx]

                # Derive binary ordering variables
                sigma_ij = 1 if oi["x"] + oi["length"] <= oj["x"] + tol else 0
                sigma_ji = 1 if oj["x"] + oj["length"] <= oi["x"] + tol else 0
                delta_ij = 1 if oi["y_end"] <= oj["y_start"] + tol else 0
                delta_ji = 1 if oj["y_end"] <= oi["y_start"] + tol else 0

                lhs_sum = sigma_ij + sigma_ji + delta_ij + delta_ji
                if lhs_sum < 1:
                    add_violation(
                        4,
                        f"{oi['label']} call {oi['call_idx']} and {oj['label']} call {oj['call_idx']} "
                        f"overlap at {port_id} "
                        f"(space: [{oi['x']:.1f},{oi['x']+oi['length']:.1f}] vs "
                        f"[{oj['x']:.1f},{oj['x']+oj['length']:.1f}], "
                        f"time: [{oi['y_start']:.1f},{oi['y_end']:.1f}] vs "
                        f"[{oj['y_start']:.1f},{oj['y_end']:.1f}])",
                        float(lhs_sum), 1.0, ">="
                    )

    # =========================================================================
    # Constraint 5 (Eq 7): Arrival time linking
    # y_i^c + h_i^c + sum_s v_{i,s}^c * Theta_s * Delta^{rho_c, rho_{c+1}} = a_i^{c+1}
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        sol_ship = sol_ships.get(ship_id)
        if sol_ship is None:
            continue
        pvs = sorted(s["port_visits"], key=lambda pv: pv["port_call_index"])
        for idx in range(len(pvs) - 1):
            pv_curr = pvs[idx]
            pv_next = pvs[idx + 1]
            c = pv_curr["port_call_index"]
            c_next = pv_next["port_call_index"]

            sol_curr = sol_visits.get((ship_id, c))
            sol_next = sol_visits.get((ship_id, c_next))
            if sol_curr is None or sol_next is None:
                continue

            # Find speed selection for this leg
            speed_knots = None
            for ss in sol_ship["speed_selections"]:
                if ss["from_call"] == c and ss["to_call"] == c_next:
                    speed_knots = ss["speed_knots"]
                    break

            if speed_knots is None:
                add_violation(
                    13, f"Ship {ship_id}: no speed selection from call {c} to {c_next}",
                    0.0, 1.0, ">="
                )
                continue

            from_port = pv_curr["port_id"]
            to_port = pv_next["port_id"]
            dist_key = f"{from_port}_{to_port}"
            distance = distances.get(dist_key, 0)
            theta_s = travel_times.get(str(speed_knots), 0)
            travel_time = theta_s * distance

            lhs = sol_curr["start_time"] + sol_curr["handling_time"] + travel_time
            rhs = sol_next["arrival_time"]
            add_violation(
                5,
                f"Ship {ship_id}: y({sol_curr['start_time']})+h({sol_curr['handling_time']})"
                f"+travel({travel_time:.4f})={lhs:.4f} != arrival({rhs})",
                lhs, rhs, "="
            )

    # =========================================================================
    # Constraint 6 (Eq 8): a_i^c <= y_i^c
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            a = sol_pv["arrival_time"]
            y = sol_pv["start_time"]
            add_violation(
                6, f"Ship {ship_id} call {c}: arrival({a}) > start_time({y})",
                a, y, "<="
            )

    # =========================================================================
    # Constraint 7 (Eq 9): EST_i^c <= y_i^c
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            est = pv_inst["earliest_start_time_hours"]
            y = sol_pv["start_time"]
            add_violation(
                7, f"Ship {ship_id} call {c}: EST({est}) > start_time({y})",
                est, y, "<="
            )

    # =========================================================================
    # Constraint 8 (Eq 10): y_i^c + h_i^c - EFT_i^c <= d_i^c
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            y = sol_pv["start_time"]
            h = sol_pv["handling_time"]
            eft = pv_inst["expected_finish_time_hours"]
            d = sol_pv["delay"]
            lhs = y + h - eft
            add_violation(
                8, f"Ship {ship_id} call {c}: y+h-EFT={lhs:.6f} > delay({d})",
                lhs, d, "<="
            )

    # =========================================================================
    # Constraint 9 (Eq 11): y_i^c + h_i^c - LFT_i^c <= u_i^c
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            y = sol_pv["start_time"]
            h = sol_pv["handling_time"]
            lft = pv_inst["latest_finish_time_hours"]
            u = sol_pv["lft_violation"]
            lhs = y + h - lft
            add_violation(
                9, f"Ship {ship_id} call {c}: y+h-LFT={lhs:.6f} > lft_violation({u})",
                lhs, u, "<="
            )

    # =========================================================================
    # Constraint 10 (Eq 12): (1 + beta * r) * h0 = h
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            h0 = pv_inst["min_handling_time_hours"]
            r = sol_pv["position_deviation"]
            h = sol_pv["handling_time"]
            lhs = (1 + beta * r) * h0
            add_violation(
                10,
                f"Ship {ship_id} call {c}: (1+{beta}*{r})*{h0}={lhs:.6f} != h({h})",
                lhs, h, "="
            )

    # =========================================================================
    # Constraint 11 (Eq 13): x - x0 <= r
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            x = sol_pv["berthing_position"]
            x0 = pv_inst["ideal_berthing_position"]
            r = sol_pv["position_deviation"]
            add_violation(
                11,
                f"Ship {ship_id} call {c}: x({x})-x0({x0})={x-x0:.6f} > r({r})",
                x - x0, r, "<="
            )

    # =========================================================================
    # Constraint 12 (Eq 14): x0 - x <= r
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            x = sol_pv["berthing_position"]
            x0 = pv_inst["ideal_berthing_position"]
            r = sol_pv["position_deviation"]
            add_violation(
                12,
                f"Ship {ship_id} call {c}: x0({x0})-x({x})={x0-x:.6f} > r({r})",
                x0 - x, r, "<="
            )

    # =========================================================================
    # Constraint 13 (Eq 15): Exactly one speed per sailing leg
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        sol_ship = sol_ships.get(ship_id)
        if sol_ship is None:
            continue
        pvs = sorted(s["port_visits"], key=lambda pv: pv["port_call_index"])
        for idx in range(len(pvs) - 1):
            c = pvs[idx]["port_call_index"]
            c_next = pvs[idx + 1]["port_call_index"]
            count = sum(
                1 for ss in sol_ship["speed_selections"]
                if ss["from_call"] == c and ss["to_call"] == c_next
            )
            if count != 1:
                add_violation(
                    13,
                    f"Ship {ship_id}: {count} speed selections from call {c} to {c_next} (expected 1)",
                    float(count), 1.0, "="
                )

    # =========================================================================
    # Constraint 14 (Eq 16): x_i^c >= 0, y_i^c >= 0
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            x = sol_pv["berthing_position"]
            y = sol_pv["start_time"]
            if x < -tol:
                add_violation(
                    14, f"Ship {ship_id} call {c}: x({x}) < 0",
                    0.0, x, "<="
                )
            if y < -tol:
                add_violation(
                    14, f"Ship {ship_id} call {c}: y({y}) < 0",
                    0.0, y, "<="
                )

    # =========================================================================
    # Constraint 15 (Eq 17): a, h, d, u, r >= 0
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        for pv_inst in s["port_visits"]:
            c = pv_inst["port_call_index"]
            sol_pv = sol_visits.get((ship_id, c))
            if sol_pv is None:
                continue
            var_checks = [
                ("arrival_time", sol_pv["arrival_time"]),
                ("handling_time", sol_pv["handling_time"]),
                ("delay", sol_pv["delay"]),
                ("lft_violation", sol_pv["lft_violation"]),
                ("position_deviation", sol_pv["position_deviation"]),
            ]
            for var_name, val in var_checks:
                if val < -tol:
                    add_violation(
                        15, f"Ship {ship_id} call {c}: {var_name}({val}) < 0",
                        0.0, val, "<="
                    )

    # =========================================================================
    # Constraint 16 (Eq 18): Speed variables are binary (valid speed chosen)
    # =========================================================================
    for s in instance["ships"]:
        ship_id = s["id"]
        sol_ship = sol_ships.get(ship_id)
        if sol_ship is None:
            continue
        for ss in sol_ship["speed_selections"]:
            spd = ss["speed_knots"]
            if spd not in speeds:
                add_violation(
                    16,
                    f"Ship {ship_id}: speed {spd} from call {ss['from_call']} "
                    f"to {ss['to_call']} not in allowed speeds {speeds}",
                    0.0, 1.0, ">="
                )

    # =========================================================================
    # Constraint 17 (Eq 19): sigma/delta binary - trivially satisfied
    # (Derived from solution; always binary by construction)
    # =========================================================================
    # No check needed - sigma/delta are not decision variables in the solution.

    # =========================================================================
    # Constraint 18 (Obj): exact objective consistency.
    # Every cost-bearing variable is serialized: call timing/slack values and
    # one speed for every sailing leg.  With the exact ship/call coverage check
    # above, the objective is therefore a value, not an envelope.
    # =========================================================================
    reported_raw = solution.get("objective_value")
    reported_obj = None
    if reported_raw is not None:
        try:
            reported_obj = float(reported_raw)
        except (TypeError, ValueError):
            reported_obj = None

    if reported_obj is not None:
        cp = instance.get("cost_parameters", {})
        I_c = cp.get("waiting_cost_usd_per_hour", 0)
        H_c = cp.get("handling_cost_usd_per_hour", 0)
        D_c = cp.get("delay_cost_usd_per_hour", 0)
        U_c = cp.get("lft_penalty_usd", 0)
        F_c = cp.get("fuel_price_usd_per_tonne", 0)
        true_obj = 0.0
        for inst_s in instance["ships"]:
            ship_id = inst_s["id"]
            sol_ship = sol_ships.get(ship_id)
            if sol_ship is None:
                continue
            inst_pvs_by_call = {pv["port_call_index"]: pv for pv in inst_s["port_visits"]}
            fuel_by_speed = inst_s.get("fuel_consumption_per_nm_by_speed", {})

            for pv in sol_ship.get("port_visits", []):
                a = pv.get("arrival_time", 0.0)
                y = pv.get("start_time", 0.0)
                h = pv.get("handling_time", 0.0)
                d = pv.get("delay", 0.0)
                u = pv.get("lft_violation", 0.0)
                true_obj += I_c * (y - a) + H_c * h + D_c * d + U_c * u

            for ss in sol_ship.get("speed_selections", []):
                spd = ss.get("speed_knots")
                c = ss.get("from_call")
                c_next = ss.get("to_call")
                if spd is None or c is None or c_next is None:
                    continue
                pv_from = inst_pvs_by_call.get(c)
                pv_to = inst_pvs_by_call.get(c_next)
                if pv_from is None or pv_to is None:
                    continue
                dist_key = f"{pv_from['port_id']}_{pv_to['port_id']}"
                distance = distances.get(dist_key, 0)
                gamma = fuel_by_speed.get(str(spd))
                if gamma is None:
                    try:
                        gamma = fuel_by_speed.get(f"{float(spd):.1f}")
                    except (TypeError, ValueError):
                        gamma = None
                if gamma is None:
                    continue
                true_obj += F_c * gamma * distance

        obj_tol = max(1e-3, 1e-8 * max(abs(true_obj), 1.0))
        if not math.isfinite(reported_obj) or abs(reported_obj - true_obj) > obj_tol:
            add_violation(
                18,
                f"Objective consistency violated: reported objective_value="
                f"{reported_obj} differs from the fully recomputed cost "
                f"{true_obj} (absolute difference="
                f"{abs(reported_obj - true_obj):.6g}, tol={obj_tol:.6g})",
                reported_obj, true_obj, "="
            )

    # --- Build output ---
    unique_violated = sorted(violated_constraints)

    result = {
        "feasible": len(violated_constraints) == 0,
        "violated_constraints": unique_violated,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
        if violation_magnitudes
        else [],
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"{status}: {len(unique_violated)} constraint(s) violated, "
          f"{len(violation_magnitudes)} violation instance(s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
