#!/usr/bin/env python3
"""
Gurobi MIP formulation for the Multi-Port Continuous Berth Allocation Problem (MCBAP).

Based on: Martin-Iradi, Pacino, Ropke (2023) - "An adaptive large neighborhood search
heuristic for the multi-port continuous berth allocation problem"

Implements the MIP formulation from Section 3.1 (Equations 2-19).
"""

import argparse
import json
import sys
import math
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
import time
# Walk up from this file's directory to find repo root (containing scripts/).
_GUROBI_CODE_START_TIME = time.time()
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass


def load_instance(path):
    """Load a problem instance from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def build_and_solve(instance, time_limit):
    """Build and solve the MCBAP MIP model using Gurobi."""

    # -------------------------------------------------------------------------
    # Extract instance data
    # -------------------------------------------------------------------------
    cost_params = instance["cost_parameters"]
    F_cost = cost_params["fuel_price_usd_per_tonne"]
    H_cost = cost_params["handling_cost_usd_per_hour"]
    D_cost = cost_params["delay_cost_usd_per_hour"]
    I_cost = cost_params["waiting_cost_usd_per_hour"]
    U_cost = cost_params["lft_penalty_usd"]

    beta = instance["handling_time_deviation_factor_beta_per_meter"]

    # Ports
    ports = {p["id"]: p for p in instance["ports"]}
    port_ids = [p["id"] for p in instance["ports"]]
    quay_lengths = {p["id"]: p["quay_length_meters"] for p in instance["ports"]}

    # Distances between ports
    distances = {}
    for key, val in instance["distances_nautical_miles"].items():
        parts = key.split("_")
        p1, p2 = parts[0], parts[1]
        distances[(p1, p2)] = val

    # Speeds
    speeds = instance["speeds_knots"]
    speed_strs = [str(s) for s in speeds]
    travel_time_per_nm = instance["travel_time_hours_per_nm_by_speed"]

    # Ships to optimize (N*)
    ships = instance["ships"]

    # External ships (N_bar) - fixed
    external_ships_by_port = {}
    for ext_group in instance["external_ships"]:
        pid = ext_group["port_id"]
        external_ships_by_port[pid] = ext_group["ships"]

    # -------------------------------------------------------------------------
    # Build index structures
    # -------------------------------------------------------------------------

    # For each optimized ship i, build its port visits
    # ship_visits[(i, c)] = dict with port_id, ideal_pos, h0, EST, EFT, LFT, etc.
    ship_visits = {}
    ship_data = {}
    for ship in ships:
        sid = ship["id"]
        ship_data[sid] = ship
        for pv in ship["port_visits"]:
            c = pv["port_call_index"]
            ship_visits[(sid, c)] = {
                "port_id": pv["port_id"],
                "ideal_pos": pv["ideal_berthing_position"],
                "h0": pv["min_handling_time_hours"],
                "h_max": pv["max_handling_time_hours"],
                "EST": pv["earliest_start_time_hours"],
                "EFT": pv["expected_finish_time_hours"],
                "LFT": pv["latest_finish_time_hours"],
            }

    # N_p: set of (ship_id, call_index) visiting port p (optimized ships)
    # Also build external ship visits at each port
    opt_visits_at_port = {pid: [] for pid in port_ids}
    for (sid, c), info in ship_visits.items():
        opt_visits_at_port[info["port_id"]].append((sid, c))

    ext_visits_at_port = {pid: [] for pid in port_ids}
    for pid in port_ids:
        if pid in external_ships_by_port:
            for es in external_ships_by_port[pid]:
                ext_visits_at_port[pid].append(es)

    # All visits at port (for non-overlap constraints): combine optimized and external
    # External ships are identified by ("ext", port_id, ext_id)
    # Optimized ships by (ship_id, call_index)

    # For each optimized ship, get max call index
    ship_max_call = {}
    for ship in ships:
        sid = ship["id"]
        ship_max_call[sid] = max(pv["port_call_index"] for pv in ship["port_visits"])

    # -------------------------------------------------------------------------
    # Create Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("MCBAP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # -------------------------------------------------------------------------
    # Decision variables for optimized ships
    # -------------------------------------------------------------------------
    # x[i,c]: leftmost berthing position
    x = {}
    # y[i,c]: start time of berthing
    y = {}
    # h[i,c]: handling time
    h = {}
    # a[i,c]: arrival time
    a = {}
    # d[i,c]: delay over EFT
    d = {}
    # u[i,c]: delay over LFT
    u = {}
    # r[i,c]: distance from ideal position
    r = {}

    for (sid, c), info in ship_visits.items():
        L_p = quay_lengths[info["port_id"]]
        l_i = ship_data[sid]["length_meters"]

        x[sid, c] = model.addVar(lb=0, ub=L_p - l_i, name=f"x_{sid}_{c}")
        y[sid, c] = model.addVar(lb=info["EST"], name=f"y_{sid}_{c}")
        h[sid, c] = model.addVar(lb=info["h0"], name=f"h_{sid}_{c}")
        a[sid, c] = model.addVar(lb=0, name=f"a_{sid}_{c}")
        d[sid, c] = model.addVar(lb=0, name=f"d_{sid}_{c}")
        u[sid, c] = model.addVar(lb=0, name=f"u_{sid}_{c}")
        r[sid, c] = model.addVar(lb=0, name=f"r_{sid}_{c}")

    # v[i,s,c]: speed selection binary (for non-last calls)
    v = {}
    for ship in ships:
        sid = ship["id"]
        max_c = ship_max_call[sid]
        fuel_per_nm = ship["fuel_consumption_per_nm_by_speed"]
        for c_idx in range(1, max_c):  # c in C_i \ {c_i}
            for s_str in speed_strs:
                v[sid, s_str, c_idx] = model.addVar(
                    vtype=GRB.BINARY, name=f"v_{sid}_{s_str}_{c_idx}"
                )

    # sigma[i,c,j,c']: ship i left of ship j at port p
    # delta[i,c,j,c']: ship i finishes before ship j starts at port p
    sigma = {}
    delta_var = {}

    for pid in port_ids:
        # Collect all visits at this port (optimized + external)
        all_visits = []
        for (sid, c) in opt_visits_at_port[pid]:
            all_visits.append(("opt", sid, c))
        for es in ext_visits_at_port[pid]:
            all_visits.append(("ext", pid, es["id"]))

        # For each pair (i != j), create sigma and delta variables
        for idx_a in range(len(all_visits)):
            for idx_b in range(len(all_visits)):
                if idx_a == idx_b:
                    continue
                va = all_visits[idx_a]
                vb = all_visits[idx_b]
                key = (va, vb, pid)
                sigma[key] = model.addVar(vtype=GRB.BINARY, name=f"sigma_{va}_{vb}_{pid}")
                delta_var[key] = model.addVar(vtype=GRB.BINARY, name=f"delta_{va}_{vb}_{pid}")

    model.update()

    # -------------------------------------------------------------------------
    # Helper: get position, start time, handling time, length for any visit
    # -------------------------------------------------------------------------
    def get_visit_vars(visit_tuple):
        """Return (x_var_or_val, y_var_or_val, h_var_or_val, length) for a visit."""
        if visit_tuple[0] == "opt":
            sid, c = visit_tuple[1], visit_tuple[2]
            return x[sid, c], y[sid, c], h[sid, c], ship_data[sid]["length_meters"]
        else:
            # External ship
            pid_ext = visit_tuple[1]
            eid = visit_tuple[2]
            for es in external_ships_by_port[pid_ext]:
                if es["id"] == eid:
                    return (es["berthing_position"], es["start_time_hours"],
                            es["handling_time_hours"], es["length_meters"])
            raise ValueError(f"External ship not found: {visit_tuple}")

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------

    # Constraint (3): Ships berth within available quay space
    for (sid, c), info in ship_visits.items():
        l_i = ship_data[sid]["length_meters"]
        L_p = quay_lengths[info["port_id"]]
        model.addConstr(x[sid, c] + l_i <= L_p, name=f"quay_{sid}_{c}")

    # Constraints (4), (5), (6): Non-overlapping for each port
    for pid in port_ids:
        all_visits = []
        for (sid, c) in opt_visits_at_port[pid]:
            all_visits.append(("opt", sid, c))
        for es in ext_visits_at_port[pid]:
            all_visits.append(("ext", pid, es["id"]))

        L_p = quay_lengths[pid]

        for idx_a in range(len(all_visits)):
            for idx_b in range(len(all_visits)):
                if idx_a == idx_b:
                    continue
                va = all_visits[idx_a]
                vb = all_visits[idx_b]
                key_ab = (va, vb, pid)

                x_a, y_a, h_a, l_a = get_visit_vars(va)
                x_b, y_b, h_b, l_b = get_visit_vars(vb)

                # Constraint (4): x_i^c + l_i <= x_j^{c'} + L_p*(1 - sigma)
                model.addConstr(
                    x_a + l_a <= x_b + L_p * (1 - sigma[key_ab]),
                    name=f"space_{va}_{vb}_{pid}"
                )

                # Constraint (5): y_i^c + h_i^c <= y_j^{c'} + M*(1 - delta)
                # Big-M: use a large value based on planning horizon
                # INFERRED ASSUMPTION: M is set to the planning horizon in hours,
                # as the paper states "M can be limited to the latest finish time
                # of the pair of ships". We use planning_horizon_hours as a safe upper bound.
                M_val = instance.get("planning_horizon_hours", 10000)
                model.addConstr(
                    y_a + h_a <= y_b + M_val * (1 - delta_var[key_ab]),
                    name=f"time_{va}_{vb}_{pid}"
                )

        # Constraint (6): sigma_ij + sigma_ji + delta_ij + delta_ji >= 1
        # Only for i < j (to avoid redundant constraints)
        for idx_a in range(len(all_visits)):
            for idx_b in range(idx_a + 1, len(all_visits)):
                va = all_visits[idx_a]
                vb = all_visits[idx_b]
                key_ab = (va, vb, pid)
                key_ba = (vb, va, pid)
                model.addConstr(
                    sigma[key_ab] + sigma[key_ba] +
                    delta_var[key_ab] + delta_var[key_ba] >= 1,
                    name=f"nonoverlap_{va}_{vb}_{pid}"
                )

    # Constraint (7): Arrival time linking via sailing speed
    for ship in ships:
        sid = ship["id"]
        max_c = ship_max_call[sid]
        fuel_per_nm = ship["fuel_consumption_per_nm_by_speed"]
        route = ship["route"]

        for c_idx in range(1, max_c):  # c in C_i \ {c_i}
            next_c = c_idx + 1
            port_from = ship_visits[(sid, c_idx)]["port_id"]
            port_to = ship_visits[(sid, next_c)]["port_id"]
            dist = distances[(port_from, port_to)]

            travel_expr = gp.LinExpr()
            for s_str in speed_strs:
                theta_s = travel_time_per_nm[s_str]
                travel_expr += v[sid, s_str, c_idx] * theta_s * dist

            # y_i^c + h_i^c + travel_time = a_i^{c+1}
            model.addConstr(
                y[sid, c_idx] + h[sid, c_idx] + travel_expr == a[sid, next_c],
                name=f"arrival_{sid}_{c_idx}"
            )

    # Constraint (8): Berthing starts after arrival
    for (sid, c), info in ship_visits.items():
        model.addConstr(a[sid, c] <= y[sid, c], name=f"arr_before_berth_{sid}_{c}")

    # For the first port call, arrival time = EST (or could be earlier)
    # INFERRED ASSUMPTION: For the first port call (c=1), arrival time a_i^1
    # is not linked by constraint (7). We set a_i^1 <= y_i^1 via constraint (8)
    # and a_i^1 >= 0 via non-negativity. The waiting cost will push a_i^1 close to y_i^1.
    # Actually, for the first call, a_i^1 is the ship's arrival time at the first port.
    # The paper doesn't explicitly define it; we assume a_i^1 = EST_i^1 as the earliest
    # the ship could arrive, so waiting cost = y_i^1 - EST_i^1.
    for ship in ships:
        sid = ship["id"]
        first_c = 1
        est = ship_visits[(sid, first_c)]["EST"]
        # INFERRED ASSUMPTION: a_i^1 = EST_i^1 for the first port call.
        # This means the ship arrives exactly at the earliest start time.
        # The paper doesn't provide a separate arrival time for the first visit.
        model.addConstr(a[sid, first_c] == est, name=f"first_arrival_{sid}")

    # Constraint (9): Berthing starts after earliest start time
    # (Already enforced by lb on y variables)

    # Constraint (10): Delay over expected finish time
    for (sid, c), info in ship_visits.items():
        model.addConstr(
            y[sid, c] + h[sid, c] - info["EFT"] <= d[sid, c],
            name=f"delay_{sid}_{c}"
        )

    # Constraint (11): Delay over latest finish time
    for (sid, c), info in ship_visits.items():
        model.addConstr(
            y[sid, c] + h[sid, c] - info["LFT"] <= u[sid, c],
            name=f"lft_delay_{sid}_{c}"
        )

    # Constraint (12): Handling time depends on deviation from ideal position
    # h_i^c = (1 + beta * r^{i,c}) * h_0^{i,c}
    # => h_i^c = h_0 + beta * h_0 * r^{i,c}  (linear since h_0 is a parameter)
    for (sid, c), info in ship_visits.items():
        h0 = info["h0"]
        model.addConstr(
            h[sid, c] == h0 + beta * h0 * r[sid, c],
            name=f"handling_{sid}_{c}"
        )

    # Constraints (13), (14): Position deviation
    for (sid, c), info in ship_visits.items():
        x0 = info["ideal_pos"]
        model.addConstr(x[sid, c] - x0 <= r[sid, c], name=f"dev_right_{sid}_{c}")
        model.addConstr(x0 - x[sid, c] <= r[sid, c], name=f"dev_left_{sid}_{c}")

    # Constraint (15): Exactly one speed chosen per sailing leg
    for ship in ships:
        sid = ship["id"]
        max_c = ship_max_call[sid]
        for c_idx in range(1, max_c):
            model.addConstr(
                gp.quicksum(v[sid, s_str, c_idx] for s_str in speed_strs) == 1,
                name=f"one_speed_{sid}_{c_idx}"
            )

    # -------------------------------------------------------------------------
    # Objective function (Equation 2)
    # -------------------------------------------------------------------------
    obj = gp.LinExpr()

    for ship in ships:
        sid = ship["id"]
        max_c = ship_max_call[sid]
        fuel_per_nm = ship["fuel_consumption_per_nm_by_speed"]
        route = ship["route"]

        for pv in ship["port_visits"]:
            c = pv["port_call_index"]
            # Waiting cost: I * (y_i^c - a_i^c)
            obj += I_cost * (y[sid, c] - a[sid, c])
            # Handling cost: H * h_i^c
            obj += H_cost * h[sid, c]
            # Delay cost: D * d_i^c
            obj += D_cost * d[sid, c]
            # LFT penalty: U * u_i^c
            obj += U_cost * u[sid, c]

        # Fuel cost for sailing legs
        for c_idx in range(1, max_c):
            next_c = c_idx + 1
            port_from = ship_visits[(sid, c_idx)]["port_id"]
            port_to = ship_visits[(sid, next_c)]["port_id"]
            dist = distances[(port_from, port_to)]

            for s_str in speed_strs:
                gamma_s = fuel_per_nm[s_str]
                # Fuel cost = F * gamma_s * distance * v_{i,s}^c
                obj += F_cost * gamma_s * dist * v[sid, s_str, c_idx]

    model.setObjective(obj, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    solution = {}

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["status"] = model.Status
        solution["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None

        ship_solutions = []
        for ship in ships:
            sid = ship["id"]
            max_c = ship_max_call[sid]
            ship_sol = {"ship_id": sid, "port_visits": []}

            for pv in ship["port_visits"]:
                c = pv["port_call_index"]
                pv_sol = {
                    "port_call_index": c,
                    "port_id": pv["port_id"],
                    "berthing_position": x[sid, c].X,
                    "start_time": y[sid, c].X,
                    "handling_time": h[sid, c].X,
                    "arrival_time": a[sid, c].X,
                    "delay": d[sid, c].X,
                    "lft_violation": u[sid, c].X,
                    "position_deviation": r[sid, c].X,
                }
                ship_sol["port_visits"].append(pv_sol)

            # Speed selections
            speed_selections = []
            for c_idx in range(1, max_c):
                for s_str in speed_strs:
                    if v[sid, s_str, c_idx].X > 0.5:
                        speed_selections.append({
                            "from_call": c_idx,
                            "to_call": c_idx + 1,
                            "speed_knots": float(s_str)
                        })
                        break
            ship_sol["speed_selections"] = speed_selections
            ship_solutions.append(ship_sol)

        solution["ships"] = ship_solutions
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["ships"] = []

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIP solver for the Multi-Port Continuous Berth Allocation Problem"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    solution = build_and_solve(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.2f}")
    else:
        print("No feasible solution found within time limit.")


if __name__ == "__main__":
    main()
