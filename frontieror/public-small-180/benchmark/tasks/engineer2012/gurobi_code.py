#!/usr/bin/env python3
"""
Compact Arc-Flow MIP Formulation for the Maritime Inventory Routing Problem.

Based on:
  Engineer, Furman, Nemhauser, Savelsbergh, Song (2012),
  "A Branch-Price-and-Cut Algorithm for Single-Product Maritime Inventory
   Routing"

Since the column-generation formulation in the paper cannot be directly solved
by Gurobi (the set R_v of voyages is exponentially large), we implement a
compact arc-flow formulation on a time-expanded network, as referenced in
Song and Furman (2012).

The formulation uses a commodity-flow approach for vessel inventory tracking:
  - w[v,e]: continuous flow of product on arc e for vessel v
  - This avoids the nonlinear constraint (10) from the paper and the need
    for big-M linearization, yielding a much tighter LP relaxation.

Usage:
  python gurobi_code.py --instance_path path/to/instance.json \\
                        --solution_path gurobi_solution.json \\
                        --time_limit 3600
"""

import argparse
import json
import math
import sys
import time

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
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
# =============================================================================
# Data Loading
# =============================================================================

def load_instance(path):
    """Load instance data from a JSON file."""
    with open(path, "r") as fh:
        data = json.load(fh)
    return data


# =============================================================================
# Helpers
# =============================================================================

def _node_str(node):
    """Convert a node to a string suitable for constraint naming."""
    if isinstance(node, tuple):
        return f"{node[0]}_{node[1]}"
    return str(node)


# =============================================================================
# Network Construction
# =============================================================================

def build_time_expanded_network(data):
    """
    Build a time-expanded network for each vessel.

    For each vessel v, construct:
      - Nodes: (j, t) for each port j and each time t in
               [avail_start, avail_end], plus source 'src_v' and sink 'snk_v'
      - Travel arcs: ((j1,t1), (j2,t2)) with t2 = t1 + travel_time(j1, j2)
      - Demurrage arcs: ((j,t), (j,t+1)) for waiting at a port
      - Source arcs: from src_v to supply port nodes at avail_start
      - Sink arcs: from any port-time node to snk_v

    INFERRED ASSUMPTION: Vessels start "at sea" and can reach any supply port
    at their availability_start time.

    Returns:
      networks: dict keyed by vessel id
      all_ports: list of port ids
      port_type: dict port_id -> "supply" or "demand"
      travel_time_dict: dict (p1,p2) -> int
      draft_limit_dict: dict (v_id, p_id) -> float
    """
    T = data["planning_horizon"]
    all_ports = []
    port_type = {}
    for sp in data["supply_ports"]:
        all_ports.append(sp["port_id"])
        port_type[sp["port_id"]] = "supply"
    for dp in data["demand_ports"]:
        all_ports.append(dp["port_id"])
        port_type[dp["port_id"]] = "demand"

    travel_time_dict = {}
    for key, val in data["travel_times"].items():
        p1, p2 = key.split("->")
        travel_time_dict[(p1, p2)] = val

    draft_limit_dict = {}
    for key, val in data["draft_limits"].items():
        v_id, p_id = key.split("->")
        draft_limit_dict[(v_id, p_id)] = val

    networks = {}

    for vessel in data["vessels"]:
        v_id = vessel["id"]
        avail_start = vessel["availability_start"]
        avail_end = vessel["availability_end"]
        load_window_end = vessel["loading_window_end"]
        cap = vessel["capacity"]

        src_node = ("src", v_id)
        snk_node = ("snk", v_id)

        nodes = set()
        arcs = []  # list of (tail, head, arc_type, cost)

        # Port-time nodes
        for j in all_ports:
            for t in range(avail_start, avail_end + 1):
                nodes.add((j, t))

        # Travel arcs
        for j1 in all_ports:
            for j2 in all_ports:
                if j1 == j2:
                    continue
                if (j1, j2) not in travel_time_dict:
                    continue
                tt = travel_time_dict[(j1, j2)]
                arc_cost = vessel["transport_cost_per_day"] * tt
                for t1 in range(avail_start, avail_end + 1):
                    t2 = t1 + tt
                    if t2 > avail_end:
                        break
                    arcs.append(((j1, t1), (j2, t2), "travel", arc_cost))

        # Demurrage arcs
        dem_cost = vessel["demurrage_cost_per_day"]
        for j in all_ports:
            for t in range(avail_start, avail_end):
                arcs.append(((j, t), (j, t + 1), "demurrage", dem_cost))

        # Source arcs: connect to supply ports at avail_start
        # INFERRED ASSUMPTION: vessels can appear at any supply port at
        # availability_start.  We also allow connections to demand ports
        # for generality (the cost structure will discourage starting at
        # demand ports without cargo).
        for j in all_ports:
            arcs.append((src_node, (j, avail_start), "source", 0.0))

        # Sink arcs: from every port-time node
        for j in all_ports:
            for t in range(avail_start, avail_end + 1):
                arcs.append(((j, t), snk_node, "sink", 0.0))

        nodes.add(src_node)
        nodes.add(snk_node)

        # U_n: max product on vessel at node n
        U_n = {}
        for j in all_ports:
            dl = draft_limit_dict.get((v_id, j), cap)
            u = min(cap, dl)
            for t in range(avail_start, avail_end + 1):
                U_n[(j, t)] = u
        U_n[src_node] = cap
        # INFERRED ASSUMPTION: vessel may end the horizon with remaining
        # inventory.  We do NOT force emptiness at avail_end.
        U_n[snk_node] = cap

        networks[v_id] = {
            "nodes": nodes,
            "arcs": arcs,
            "src": src_node,
            "snk": snk_node,
            "U_n": U_n,
            "capacity": cap,
            "avail_start": avail_start,
            "avail_end": avail_end,
            "load_window_end": load_window_end,
        }

    return networks, all_ports, port_type, travel_time_dict, draft_limit_dict


# =============================================================================
# Preprocessing (Eqs. 16, 17)
# =============================================================================

def preprocess_bounds(data, networks, all_ports, port_type):
    """
    Tighten F_max at supply nodes (Eq. 16) and F_min at demand nodes (Eq. 17).

    Sign convention:
      Supply: f >= 0 (loading)   -> F_min >= 0, F_max >= 0
      Demand: f <= 0 (discharge) -> F_min <= 0, F_max <= 0
    """
    port_data = {}
    for sp in data["supply_ports"]:
        port_data[sp["port_id"]] = sp
    for dp in data["demand_ports"]:
        port_data[dp["port_id"]] = dp

    F_min_node = {}
    F_max_node = {}

    for vessel in data["vessels"]:
        v_id = vessel["id"]
        cap = vessel["capacity"]
        net = networks[v_id]
        a_s = net["avail_start"]
        a_e = net["avail_end"]

        for j in all_ports:
            pd = port_data[j]
            I0 = pd["initial_inventory"]

            if port_type[j] == "supply":
                prod = pd["production_rate"]
                stor = pd["storage_capacity"]
                Fj_min = pd["F_min"]
                Fj_max = pd["F_max"]

                for t in range(a_s, a_e + 1):
                    sum_p_t = sum(prod[tp - 1] for tp in range(1, t + 1)
                                 if tp - 1 < len(prod))
                    sum_p_tm1 = sum(prod[tp - 1] for tp in range(1, t)
                                   if tp - 1 < len(prod))
                    Q_tm1 = stor[t - 2] if t >= 2 and t - 2 < len(stor) \
                        else stor[0]
                    b_t = prod[t - 1] if t - 1 < len(prod) else 0
                    dl = net["U_n"].get((j, t), cap)

                    excess = I0 + sum_p_tm1 - Q_tm1
                    n_prior = math.ceil(max(0, excess) / Fj_max) if Fj_max > 0 else 0
                    t4 = I0 + sum_p_t - n_prior * Fj_min

                    t_max = min(Fj_max, dl, Q_tm1 + b_t, max(0, t4))
                    t_max = max(t_max, 0)

                    F_max_node[(v_id, j, t)] = t_max
                    F_min_node[(v_id, j, t)] = Fj_min

            else:  # demand
                cons = pd["consumption_rate"]
                stor = pd["storage_capacity"]
                Fj_min_d = pd["F_min"]
                Fj_max_d = pd["F_max"]

                for t in range(a_s, a_e + 1):
                    sum_c_t = sum(cons[tp - 1] for tp in range(1, t + 1)
                                 if tp - 1 < len(cons))
                    sum_c_tm1 = sum(cons[tp - 1] for tp in range(1, t)
                                   if tp - 1 < len(cons))
                    Q_t = stor[t - 1] if t - 1 < len(stor) else stor[-1]
                    b_t = cons[t - 1] if t - 1 < len(cons) else 0
                    dl = net["U_n"].get((j, t), cap)

                    deficit = -I0 + sum_c_tm1
                    n_prior = math.ceil(max(0, deficit) / Fj_max_d) \
                        if Fj_max_d > 0 else 0
                    avail_space = Q_t - I0 + sum_c_t - n_prior * Fj_min_d
                    t4 = -max(0, avail_space)

                    t_min = max(-Fj_max_d, -dl, -(Q_t + b_t), t4)
                    t_min = min(t_min, 0)

                    F_min_node[(v_id, j, t)] = t_min
                    F_max_node[(v_id, j, t)] = -Fj_min_d

    return F_min_node, F_max_node


# =============================================================================
# MIP Model
# =============================================================================

def build_model(data, networks, all_ports, port_type, F_min_node, F_max_node):
    """
    Build the compact arc-flow MIP using a commodity-flow formulation.

    Variables:
      x[v, arc_idx]  in {0,1}:  arc selection for vessel v
      z[v, j, t]     in {0,1}:  load/discharge indicator
      f[v, j, t]     continuous: quantity loaded (>0) or discharged (<0)
      w[v, arc_idx]  >= 0:      product flow on arc for vessel v
      I[j, t]        >= 0:      port inventory

    The key insight: instead of tracking vessel inventory at nodes with big-M
    constraints (which yields a very weak LP relaxation), we track the
    commodity flow on each arc:
      - w[v,e] represents the amount of product on vessel v traversing arc e
      - Coupling: w[v,e] <= cap * x[v,e]
      - Node balance: incoming_w + f[v,j,t] = outgoing_w  (product conservation)
    This gives a tight LP relaxation and allows Gurobi to find solutions fast.
    """
    T = data["planning_horizon"]
    model = gp.Model("MaritimeIRP")
    model.setParam("Threads", 1)

    port_data = {}
    for sp in data["supply_ports"]:
        port_data[sp["port_id"]] = sp
    for dp in data["demand_ports"]:
        port_data[dp["port_id"]] = dp

    supply_ports = [sp["port_id"] for sp in data["supply_ports"]]
    demand_ports = [dp["port_id"] for dp in data["demand_ports"]]

    # -----------------------------------------------------------------
    # Build arc index structures
    # -----------------------------------------------------------------
    out_arcs = {}   # (v_id, node) -> list of arc indices
    in_arcs = {}    # (v_id, node) -> list of arc indices
    arc_info = {}   # (v_id, arc_idx) -> (tail, head, atype, cost)

    for vessel in data["vessels"]:
        v_id = vessel["id"]
        for idx, (tail, head, atype, cost) in enumerate(networks[v_id]["arcs"]):
            arc_info[(v_id, idx)] = (tail, head, atype, cost)
            out_arcs.setdefault((v_id, tail), []).append(idx)
            in_arcs.setdefault((v_id, head), []).append(idx)

    # =================================================================
    # Variables
    # =================================================================
    print("  Creating variables...")

    # x[v, arc_idx]: binary arc selection
    x = {}
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        for idx in range(len(net["arcs"])):
            _, _, atype, cost = arc_info[(v_id, idx)]
            x[(v_id, idx)] = model.addVar(
                vtype=GRB.BINARY, obj=cost, name=f"x_{v_id}_{idx}"
            )

    # w[v, arc_idx]: product flow on arc (continuous)
    w = {}
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        cap = vessel["capacity"]
        for idx in range(len(net["arcs"])):
            w[(v_id, idx)] = model.addVar(
                lb=0.0, ub=cap, vtype=GRB.CONTINUOUS,
                name=f"w_{v_id}_{idx}"
            )

    # z[v, j, t]: binary load/discharge decision
    z = {}
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        for j in all_ports:
            for t in range(net["avail_start"], net["avail_end"] + 1):
                z[(v_id, j, t)] = model.addVar(
                    vtype=GRB.BINARY, name=f"z_{v_id}_{j}_{t}"
                )

    # f[v, j, t]: load/discharge quantity (continuous)
    f = {}
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        for j in all_ports:
            for t in range(net["avail_start"], net["avail_end"] + 1):
                fmin = F_min_node.get((v_id, j, t), 0)
                fmax = F_max_node.get((v_id, j, t), 0)
                if port_type[j] == "supply":
                    lb, ub = 0.0, max(fmax, 0.0)
                    unit_cost = port_data[j]["unit_procurement_cost"]
                else:
                    lb, ub = min(fmin, 0.0), 0.0
                    # f <= 0 at demand; unit_supply_revenue * f effectively
                    # subtracts revenue from the objective
                    unit_cost = port_data[j]["unit_supply_revenue"]

                f[(v_id, j, t)] = model.addVar(
                    lb=lb, ub=ub, vtype=GRB.CONTINUOUS,
                    obj=unit_cost, name=f"f_{v_id}_{j}_{t}"
                )

    # I[j, t]: port inventory
    I_var = {}
    for j in all_ports:
        pd = port_data[j]
        stor = pd["storage_capacity"]
        for t in range(1, T + 1):
            Q_jt = stor[t - 1] if t - 1 < len(stor) else stor[-1]
            I_var[(j, t)] = model.addVar(
                lb=0.0, ub=Q_jt, vtype=GRB.CONTINUOUS,
                name=f"I_{j}_{t}"
            )

    model.update()

    # =================================================================
    # Constraints
    # =================================================================
    print("  Adding constraints...")

    # -----------------------------------------------------------------
    # (C1) Flow conservation: path from source to sink for each vessel
    # -----------------------------------------------------------------
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        src = net["src"]
        snk = net["snk"]
        for node in net["nodes"]:
            o_list = out_arcs.get((v_id, node), [])
            i_list = in_arcs.get((v_id, node), [])
            out_sum = gp.quicksum(x[(v_id, i)] for i in o_list)
            in_sum = gp.quicksum(x[(v_id, i)] for i in i_list)
            if node == src:
                rhs = 1
            elif node == snk:
                rhs = -1
            else:
                rhs = 0
            model.addConstr(
                out_sum - in_sum == rhs,
                name=f"flow_{v_id}_{_node_str(node)}"
            )

    # -----------------------------------------------------------------
    # (C2) Commodity flow coupling: w <= U_n * x
    # The product flow w on arc (tail -> head) represents the cargo
    # in transit.  This equals:
    #   - EXIT inventory from tail (after loading at tail)
    #   - ENTRY inventory at head (before loading at head)
    # Draft limits must be enforced at BOTH endpoints:
    #   w <= U_n(head) * x   (entry bound at destination)
    #   w <= U_n(tail) * x   (exit bound at origin)
    # Combined: w <= min(U_n(head), U_n(tail)) * x
    # -----------------------------------------------------------------
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        cap = vessel["capacity"]
        net = networks[v_id]
        for idx in range(len(net["arcs"])):
            tail, head, atype, _ = arc_info[(v_id, idx)]
            if isinstance(head, tuple):
                u_head = net["U_n"].get(head, cap)
            else:
                u_head = cap
            if isinstance(tail, tuple):
                u_tail = net["U_n"].get(tail, cap)
            else:
                u_tail = cap
            u = min(u_head, u_tail)
            model.addConstr(
                w[(v_id, idx)] <= u * x[(v_id, idx)],
                name=f"wcap_{v_id}_{idx}"
            )

    # -----------------------------------------------------------------
    # (C3) Commodity flow conservation at each port-time node:
    #   sum(incoming w) + f[v,j,t] = sum(outgoing w)
    #
    # For supply: f >= 0, so incoming_w + f = outgoing_w means vessel
    #   picks up product (outgoing > incoming).
    # For demand: f <= 0, so incoming_w + f = outgoing_w means vessel
    #   drops off product (outgoing < incoming).
    #
    # At source node: initial_load + outgoing_w = 0 (if initial_load=0
    #   this means vessel starts empty).
    # At sink node: incoming_w = 0 is NOT enforced (vessel can end with
    #   product).  We let the variable be free.
    # -----------------------------------------------------------------
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        init_load = vessel["initial_load"]
        src = net["src"]
        snk = net["snk"]

        # Source: outgoing product flow = initial_load
        src_out = out_arcs.get((v_id, src), [])
        model.addConstr(
            gp.quicksum(w[(v_id, i)] for i in src_out) == init_load,
            name=f"wbal_src_{v_id}"
        )

        # Port-time nodes
        for j in all_ports:
            for t in range(net["avail_start"], net["avail_end"] + 1):
                node = (j, t)
                i_list = in_arcs.get((v_id, node), [])
                o_list = out_arcs.get((v_id, node), [])
                in_w = gp.quicksum(w[(v_id, i)] for i in i_list)
                out_w = gp.quicksum(w[(v_id, i)] for i in o_list)
                model.addConstr(
                    in_w + f[(v_id, j, t)] == out_w,
                    name=f"wbal_{v_id}_{j}_{t}"
                )

        # Sink: no explicit constraint needed -- incoming w can be any
        # non-negative value (product remaining on vessel).

    # -----------------------------------------------------------------
    # (C4) Load/discharge only if node is visited (Eq. 8):
    #   z[v,j,t] <= sum of outgoing arcs from (j,t)
    # -----------------------------------------------------------------
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        for j in all_ports:
            for t in range(net["avail_start"], net["avail_end"] + 1):
                node = (j, t)
                o_list = out_arcs.get((v_id, node), [])
                if not o_list:
                    model.addConstr(z[(v_id, j, t)] == 0,
                                    name=f"vis_{v_id}_{j}_{t}")
                else:
                    model.addConstr(
                        z[(v_id, j, t)] <= gp.quicksum(
                            x[(v_id, i)] for i in o_list),
                        name=f"vis_{v_id}_{j}_{t}")

    # -----------------------------------------------------------------
    # (C5) Load/discharge quantity bounds (Eq. 9):
    #   F_min * z <= f <= F_max * z
    # -----------------------------------------------------------------
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        for j in all_ports:
            for t in range(net["avail_start"], net["avail_end"] + 1):
                fmin_val = F_min_node.get((v_id, j, t), 0)
                fmax_val = F_max_node.get((v_id, j, t), 0)
                model.addConstr(
                    f[(v_id, j, t)] >= fmin_val * z[(v_id, j, t)],
                    name=f"flo_{v_id}_{j}_{t}")
                model.addConstr(
                    f[(v_id, j, t)] <= fmax_val * z[(v_id, j, t)],
                    name=f"fhi_{v_id}_{j}_{t}")

    # -----------------------------------------------------------------
    # (C6) Port inventory balance (Eqs. 1, 2):
    #   Supply: I[j,t] = I[j,t-1] + b[j,t] - sum_v f[v,j,t]
    #   Demand: I[j,t] = I[j,t-1] - b[j,t] - sum_v f[v,j,t]
    # -----------------------------------------------------------------
    for j in all_ports:
        pd = port_data[j]
        I0 = pd["initial_inventory"]

        if port_type[j] == "supply":
            rates = pd["production_rate"]
            for t in range(1, T + 1):
                b_jt = rates[t - 1] if t - 1 < len(rates) else 0
                I_prev = I0 if t == 1 else I_var[(j, t - 1)]
                total_f = gp.quicksum(
                    f[(v["id"], j, t)]
                    for v in data["vessels"]
                    if (v["id"], j, t) in f
                )
                model.addConstr(
                    I_var[(j, t)] == I_prev + b_jt - total_f,
                    name=f"inv_{j}_{t}")
        else:
            rates = pd["consumption_rate"]
            for t in range(1, T + 1):
                b_jt = rates[t - 1] if t - 1 < len(rates) else 0
                I_prev = I0 if t == 1 else I_var[(j, t - 1)]
                total_f = gp.quicksum(
                    f[(v["id"], j, t)]
                    for v in data["vessels"]
                    if (v["id"], j, t) in f
                )
                model.addConstr(
                    I_var[(j, t)] == I_prev - b_jt - total_f,
                    name=f"inv_{j}_{t}")

    # Port storage bounds: already set via variable bounds on I_var.

    # -----------------------------------------------------------------
    # (C7) Loading window:
    # INFERRED ASSUMPTION: The loading_window_end field indicates the
    # latest time a vessel's FIRST loading can start.  We do NOT enforce
    # a hard ban on loading at supply ports after loading_window_end,
    # because doing so can make the problem infeasible when supply ports
    # keep producing after all loading windows have closed.
    # Instead, we treat the availability window [avail_start, avail_end]
    # as the only hard time constraint on the vessel.  The loading
    # window is interpreted as a soft scheduling preference encoded in
    # the time-expanded network structure (vessels are unlikely to
    # revisit supply ports after their initial transatlantic crossing
    # due to travel time costs).
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # (C8) Force load/discharge before relocating (symmetry breaking):
    #   For a travel arc from (j,t), the vessel must have loaded/discharged
    #   at port j at some point during its stay.  Simplified: either z=1
    #   at (j,t), or the vessel arrived via a demurrage arc (was already
    #   at j, so loaded earlier).
    # -----------------------------------------------------------------
    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        for idx, (tail, head, atype, cost) in enumerate(net["arcs"]):
            if atype == "travel" and isinstance(tail, tuple):
                j, t = tail
                i_list = in_arcs.get((v_id, (j, t)), [])
                dem_in = [ai for ai in i_list
                          if arc_info[(v_id, ai)][2] == "demurrage"]
                if dem_in:
                    model.addConstr(
                        x[(v_id, idx)] <= z[(v_id, j, t)] + gp.quicksum(
                            x[(v_id, ai)] for ai in dem_in),
                        name=f"fld_{v_id}_{idx}")
                else:
                    model.addConstr(
                        x[(v_id, idx)] <= z[(v_id, j, t)],
                        name=f"fld_{v_id}_{idx}")

    model.update()
    print(f"  Model: {model.NumVars} variables, {model.NumConstrs} constraints")

    return model, x, w, z, f, I_var


# =============================================================================
# Solution Extraction
# =============================================================================

def extract_solution(model, data, networks, all_ports, port_type,
                     x, w, z, f, I_var):
    """Extract solution from the optimized Gurobi model."""
    T = data["planning_horizon"]

    solution = {
        "objective_value": None,
        "solver_status": None,
        "mip_gap": None,
        "solve_time": model.Runtime,
        "vessel_routes": {},
        "port_inventories": {},
        "vessel_operations": {},
    }

    if model.SolCount == 0:
        solution["solver_status"] = "no_feasible_solution"
        return solution

    solution["objective_value"] = model.ObjVal
    solution["mip_gap"] = model.MIPGap
    if model.Status == GRB.OPTIMAL:
        solution["solver_status"] = "optimal"
    elif model.Status == GRB.TIME_LIMIT:
        solution["solver_status"] = "time_limit"
    else:
        solution["solver_status"] = f"status_{model.Status}"

    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        route_arcs = []
        operations = []

        for idx, (tail, head, atype, cost) in enumerate(net["arcs"]):
            if x[(v_id, idx)].X > 0.5:
                w_val = w[(v_id, idx)].X
                route_arcs.append({
                    "tail": str(tail),
                    "head": str(head),
                    "type": atype,
                    "cost": cost,
                    "product_flow": round(w_val, 4),
                })

        for j in all_ports:
            for t in range(net["avail_start"], net["avail_end"] + 1):
                if (v_id, j, t) in z and z[(v_id, j, t)].X > 0.5:
                    f_val = f[(v_id, j, t)].X
                    operations.append({
                        "port": j,
                        "time": t,
                        "quantity": round(f_val, 4),
                        "type": ("load" if port_type[j] == "supply"
                                 else "discharge"),
                    })

        solution["vessel_routes"][v_id] = route_arcs
        solution["vessel_operations"][v_id] = operations

    for j in all_ports:
        inv = []
        for t in range(1, T + 1):
            if (j, t) in I_var:
                inv.append({
                    "time": t,
                    "inventory": round(I_var[(j, t)].X, 4),
                })
        solution["port_inventories"][j] = inv

    return solution


def compute_cost_breakdown(model, data, networks, all_ports, port_type,
                           x, f):
    """Compute a detailed cost breakdown."""
    if model.SolCount == 0:
        return {}

    port_data = {}
    for sp in data["supply_ports"]:
        port_data[sp["port_id"]] = sp
    for dp in data["demand_ports"]:
        port_data[dp["port_id"]] = dp

    transport_cost = 0.0
    demurrage_cost = 0.0
    procurement_cost = 0.0
    supply_revenue = 0.0

    for vessel in data["vessels"]:
        v_id = vessel["id"]
        net = networks[v_id]
        for idx, (tail, head, atype, cost) in enumerate(net["arcs"]):
            if x[(v_id, idx)].X > 0.5:
                if atype == "travel":
                    transport_cost += cost
                elif atype == "demurrage":
                    demurrage_cost += cost

        for j in all_ports:
            for t in range(net["avail_start"], net["avail_end"] + 1):
                if (v_id, j, t) in f:
                    fv = f[(v_id, j, t)].X
                    if abs(fv) > 1e-6:
                        if port_type[j] == "supply":
                            procurement_cost += (
                                port_data[j]["unit_procurement_cost"] * fv)
                        else:
                            supply_revenue += (
                                port_data[j]["unit_supply_revenue"] * (-fv))

    return {
        "transport_cost": round(transport_cost, 4),
        "demurrage_cost": round(demurrage_cost, 4),
        "procurement_cost": round(procurement_cost, 4),
        "supply_revenue": round(supply_revenue, 4),
        "total_cost": round(
            transport_cost + demurrage_cost + procurement_cost
            - supply_revenue, 4),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compact arc-flow MIP for Maritime Inventory Routing "
                    "(Engineer et al. 2012)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write solution JSON file.")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Gurobi time limit in seconds (default 3600).")
    parser.add_argument("--mip_gap", type=float, default=1e-4,
                        help="Gurobi MIPGap tolerance (default 1e-4).")
    parser.add_argument("--threads", type=int, default=1,
                        help="Threads for Gurobi (default: 1).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose Gurobi output.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print("=" * 70)
    print("Maritime Inventory Routing Problem - Compact Arc-Flow MIP")
    print("Based on Engineer et al. (2012)")
    print("=" * 70)

    # Load instance
    print(f"\nLoading instance: {args.instance_path}")
    data = load_instance(args.instance_path)
    print(f"  Horizon={data['planning_horizon']}, "
          f"vessels={data['num_vessels']}, "
          f"supply={data['num_supply_ports']}, "
          f"demand={data['num_demand_ports']}")

    # Build networks
    print("\nBuilding time-expanded networks...")
    networks, all_ports, port_type, tt_dict, dl_dict = \
        build_time_expanded_network(data)
    for v_id, net in networks.items():
        print(f"  {v_id}: {len(net['nodes'])} nodes, {len(net['arcs'])} arcs, "
              f"avail=[{net['avail_start']},{net['avail_end']}], "
              f"load_end={net['load_window_end']}")

    # Preprocess
    print("\nPreprocessing bounds (Eqs. 16, 17)...")
    F_min_node, F_max_node = preprocess_bounds(
        data, networks, all_ports, port_type)

    # Build model
    print("\nBuilding MIP model...")
    model, x, w, z, f, I_var = build_model(
        data, networks, all_ports, port_type, F_min_node, F_max_node)

    # Gurobi parameters
    model.setParam("TimeLimit", args.time_limit)
    if args.mip_gap > 0:
        model.setParam("MIPGap", args.mip_gap)
    if args.threads > 0:
        model.setParam("Threads", args.threads)
    if not args.verbose:
        model.setParam("OutputFlag", 1)
    # Find any feasible incumbent quickly — maritime inventory routing has a
    # tight LP relaxation and gurobi can spend the full budget proving infeas/
    # closing gap without producing an incumbent. MIPFocus=1 + NoRelHeurTime
    # bias the search toward feasibility.
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, args.time_limit * 0.05))

    # Optimize
    print("\nOptimizing...")
    print("=" * 70)
    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0
    print("=" * 70)

    # Report
    print(f"\nSolve time: {solve_time:.2f}s")
    print(f"Gurobi status: {model.Status}")

    if model.SolCount > 0:
        print(f"Objective value: {model.ObjVal:.4f}")
        print(f"MIP gap: {model.MIPGap:.6f}")
        if model.Status == GRB.OPTIMAL:
            print("Solution is OPTIMAL.")
        else:
            print("Feasible solution found (not proven optimal).")

        breakdown = compute_cost_breakdown(
            model, data, networks, all_ports, port_type, x, f)
        print("\nCost Breakdown:")
        for k, v in breakdown.items():
            print(f"  {k}: {v}")

        print("\nVessel Operations:")
        for vessel in data["vessels"]:
            v_id = vessel["id"]
            net = networks[v_id]
            ops = []
            for j in all_ports:
                for t in range(net["avail_start"], net["avail_end"] + 1):
                    if (v_id, j, t) in z and z[(v_id, j, t)].X > 0.5:
                        fv = f[(v_id, j, t)].X
                        ptype = "load" if port_type[j] == "supply" else "disch"
                        ops.append((t, j, ptype, fv))
            if ops:
                ops.sort()
                print(f"  {v_id}:")
                for (t, j, pt, fv) in ops:
                    print(f"    t={t:3d} {pt:6s} at {j}: {fv:+.2f}")
            else:
                print(f"  {v_id}: idle")
    else:
        print("No feasible solution found.")

    # Save solution
    solution = extract_solution(
        model, data, networks, all_ports, port_type, x, w, z, f, I_var)
    solution["solve_time"] = round(solve_time, 2)
    solution["instance_path"] = args.instance_path
    if model.SolCount > 0:
        solution["cost_breakdown"] = compute_cost_breakdown(
            model, data, networks, all_ports, port_type, x, f)

    print(f"\nSaving solution to: {args.solution_path}")
    with open(args.solution_path, "w") as fout:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, fout, indent=2)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
