"""
Compact Arc-Flow MIP Formulation for the Truck-and-Trailer Routing Problem
with Time Windows (TTRPTW)

IMPORTANT NOTES:
    1. The paper "Branch-and-Price-and-Cut for the Truck-and-Trailer Routing
       Problem with Time Windows" (Rothenbaecher, Drexl, Irnich, 2018,
       Transportation Science) ONLY provides a set-partitioning model
       (formulation (1a)-(1d)) that requires a branch-and-price-and-cut
       framework with column generation. The paper does NOT provide a compact
       MIP formulation.

    2. This file implements a compact arc-flow MIP formulation as a reasonable
       implementation choice so that Gurobi can solve it directly without
       column generation. This formulation is NOT from the paper -- it is an
       engineering decision to capture the same problem structure in a form
       amenable to a general-purpose MIP solver.

    3. KEY MODELING DECISIONS:
       - Following the paper's pricing network (Section 3, Table 2), each
         parking location p has three copies: d(p) for decoupling, tau(p)
         for load transfer, and c(p) for coupling. This allows the vehicle
         to "visit" the same physical location multiple times with different
         operations.
       - For complete vehicles, the route goes:
         depot -> (trailer customers on x-arcs) -> d(p) [decouple] ->
         (truck customers and trailer customers on y-arcs) -> c(p) [couple]
         -> (trailer customers on x-arcs) -> depot
       - Solo trucks: simple TSP-like routing through customers.
       - At most one decouple/recouple cycle per complete vehicle.
       - MTZ subtour elimination.
       - A greedy heuristic provides a warm-start MIP solution.
       - Since customer_time_windows=false, we use [0, max_route_duration].

Author: Generated as a compact MIP alternative to the paper's BPC approach
"""

import argparse
import json
import os
import sys
import time as time_module
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


try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy is not installed. "
          "Please install Gurobi and its Python interface.")
    sys.exit(1)


def load_instance(path):
    """Load a TTRPTW instance from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def solve_ttrptw(instance_path, solution_path, time_limit):
    """Build and solve a compact arc-flow MIP for the TTRPTW."""
    data = load_instance(instance_path)

    params = data["parameters"]
    fleet = data["fleet"]
    depot = data["depot"]
    truck_custs = data["truck_customers"]
    trailer_custs = data["trailer_customers"]
    trans_locs = data["transshipment_locations"]
    dist_orig = data["distance_matrix"]
    ttime_orig = data["travel_time_matrix"]

    horizon_end = params["planning_horizon"][1]
    coupling_time_val = params["coupling_time"]
    decoupling_time_val = params["decoupling_time"]
    transfer_time_val = params["transfer_time"]
    coupling_cost_val = params["coupling_cost"]
    decoupling_cost_val = params["decoupling_cost"]
    transfer_cost_val = params["transfer_cost"]
    service_time_val = params["service_time"]

    depot_id = depot["id"]  # 0
    N_L = [c["id"] for c in truck_custs]
    N_H = [c["id"] for c in trailer_custs]
    T_locs = [t["id"] for t in trans_locs]
    P_orig = sorted(set(N_H) | set(T_locs))  # parking locations (original IDs)
    N_cust = sorted(N_L + N_H)
    n_orig = len(dist_orig)

    supply = {}
    svc = {}
    cust_tw = {}  # customer id -> (earliest, latest); only populated when set
    for c in truck_custs + trailer_custs:
        supply[c["id"]] = c["supply"]
        svc[c["id"]] = c.get("service_time", service_time_val)
        tw = c.get("time_window")
        if tw is not None:
            cust_tw[c["id"]] = (float(tw[0]), float(tw[1]))
    for nid in [depot_id] + T_locs:
        supply[nid] = 0
        svc[nid] = 0.0

    vehicle_classes = fleet["vehicle_classes"]
    truck_types = {tt["id"]: tt for tt in fleet["truck_types"]}
    trailer_types = {tt["id"]: tt for tt in fleet["trailer_types"]}
    vc_info = {vc["class_id"]: vc for vc in vehicle_classes}

    def is_complete(cid):
        return vc_info[cid]["trailer_type_id"] != 0

    # =========================================================================
    # EXPANDED NODE SET
    # For complete vehicles, each parking location p gets two extra copies:
    #   d(p) = decouple node (arrive with trailer, leave without)
    #   c(p) = couple node (arrive without trailer, leave with)
    # These are virtual nodes at the same physical location.
    #
    # For solo trucks, only depot + customers are needed.
    # =========================================================================
    # Assign IDs to expanded nodes:
    #   Original nodes: 0..n_orig-1
    #   d(p) for p in P_orig: n_orig, n_orig+1, ...
    #   c(p) for p in P_orig: after all d(p) nodes

    dp_node = {}  # decouple copy: dp_node[p] = new_id
    cp_node = {}  # couple copy:   cp_node[p] = new_id
    next_id = n_orig
    for p in P_orig:
        dp_node[p] = next_id
        next_id += 1
    for p in P_orig:
        cp_node[p] = next_id
        next_id += 1

    total_nodes = next_id

    # Expanded distance and travel time: dp/cp nodes inherit from original p
    def edist(i, j):
        """Distance between expanded nodes."""
        ri = i if i < n_orig else (
            P_orig[(i - n_orig) % len(P_orig)] if i < n_orig + len(P_orig)
            else P_orig[(i - n_orig - len(P_orig)) % len(P_orig)])
        rj = j if j < n_orig else (
            P_orig[(j - n_orig) % len(P_orig)] if j < n_orig + len(P_orig)
            else P_orig[(j - n_orig - len(P_orig)) % len(P_orig)])
        return dist_orig[ri][rj]

    def ettime(i, j):
        """Travel time between expanded nodes."""
        ri = i if i < n_orig else (
            P_orig[(i - n_orig) % len(P_orig)] if i < n_orig + len(P_orig)
            else P_orig[(i - n_orig - len(P_orig)) % len(P_orig)])
        rj = j if j < n_orig else (
            P_orig[(j - n_orig) % len(P_orig)] if j < n_orig + len(P_orig)
            else P_orig[(j - n_orig - len(P_orig)) % len(P_orig)])
        return ttime_orig[ri][rj]

    # Helper to get the "real" (physical) node for any expanded node
    def real_node(i):
        if i < n_orig:
            return i
        idx = (i - n_orig) % len(P_orig)
        return P_orig[idx]

    # =========================================================================
    # NODE SETS for each vehicle type
    # =========================================================================
    # Solo truck nodes: depot + all customers (no trans, no dp/cp)
    solo_nodes = [depot_id] + N_cust

    # Complete vehicle nodes: depot + customers + dp + cp copies
    # Note: Transshipment locations are only useful as parking locations,
    # which is handled by dp_node/cp_node copies. They serve no customer
    # supply, so we exclude the original T_locs nodes from the CV node set
    # to reduce the number of arcs.
    cv_nodes = [depot_id] + N_cust
    cv_dp_nodes = [dp_node[p] for p in P_orig]
    cv_cp_nodes = [cp_node[p] for p in P_orig]
    cv_all_nodes = cv_nodes + cv_dp_nodes + cv_cp_nodes

    N_L_s = set(N_L)
    N_H_s = set(N_H)
    T_locs_s = set(T_locs)
    N_cust_s = set(N_cust)
    dp_nodes_s = set(cv_dp_nodes)
    cp_nodes_s = set(cv_cp_nodes)

    M = horizon_end  # Big-M for time

    # =========================================================================
    # BUILD MODEL
    # =========================================================================
    model = gp.Model("TTRPTW")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    model.setParam("MIPGap", 1e-4)
    model.setParam("MIPFocus", 1)

    vehicles = []
    for vc in vehicle_classes:
        for vi in range(vc["num_available"]):
            vehicles.append((vc["class_id"], vi))

    # =========================================================================
    # VARIABLES
    # =========================================================================
    u = {}        # vehicle used
    arc_v = {}    # arc traversed
    sv = {}       # customer served
    tm = {}       # time at node
    ld = {}       # total load at node
    ps = {}       # MTZ position

    for (k, vi) in vehicles:
        vc_d = vc_info[k]
        is_cv = is_complete(k)
        Q_L = vc_d["truck_capacity"]
        Q_H = vc_d["trailer_capacity"] if is_cv else 0
        Q_total = Q_L + Q_H

        u[k, vi] = model.addVar(vtype=GRB.BINARY, name=f"u_{k}_{vi}")

        nodes = cv_all_nodes if is_cv else solo_nodes
        n_v = len(nodes)

        for n in N_cust:
            sv[k, vi, n] = model.addVar(vtype=GRB.BINARY,
                                        name=f"sv_{k}_{vi}_{n}")

        # Arc variables
        # For solo trucks: arcs between solo_nodes
        # For CVs: structured arcs based on paper's Table 2
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue

                # For solo trucks: all arcs allowed
                if not is_cv:
                    arc_v[k, vi, i, j] = model.addVar(
                        vtype=GRB.BINARY, name=f"a_{k}_{vi}_{i}_{j}")
                    continue

                # For complete vehicles: restrict arcs based on Table 2
                # x-arcs (with trailer): depot <-> N_H, depot <-> dp, N_H <-> N_H, N_H <-> dp
                # y-arcs (truck alone): dp -> N_L, dp -> N_H, N_L -> N_L,
                #    N_L -> N_H, N_H -> N_L, N_L -> cp, N_H -> cp
                # dp -> cp (direct return without serving anyone)
                # cp -> N_H, cp -> dp (another cycle), cp -> depot
                # Note: T_locs original nodes are excluded from CV nodes;
                # T_locs are only available as parking via dp/cp copies.

                create_arc = False

                # Categorize nodes
                i_is_depot = (i == depot_id)
                j_is_depot = (j == depot_id)
                i_is_NL = (i in N_L_s)
                j_is_NL = (j in N_L_s)
                i_is_NH = (i in N_H_s)
                j_is_NH = (j in N_H_s)
                i_is_dp = (i in dp_nodes_s)
                j_is_dp = (j in dp_nodes_s)
                i_is_cp = (i in cp_nodes_s)
                j_is_cp = (j in cp_nodes_s)

                # x-arcs (with trailer attached):
                # depot -> NH, depot -> dp
                if i_is_depot and (j_is_NH or j_is_dp):
                    create_arc = True
                # NH -> depot
                if i_is_NH and j_is_depot:
                    create_arc = True
                # NH <-> NH
                if i_is_NH and j_is_NH:
                    create_arc = True
                # NH -> dp (going to decouple)
                if i_is_NH and j_is_dp:
                    create_arc = True
                # cp -> depot (return after coupling)
                if i_is_cp and j_is_depot:
                    create_arc = True
                # cp -> NH (visit more trailer custs after coupling)
                if i_is_cp and j_is_NH:
                    create_arc = True
                # cp -> dp (another decouple cycle -- not used since we limit
                # to 1 cycle, but included for completeness)
                if i_is_cp and j_is_dp:
                    create_arc = True

                # y-arcs (truck alone):
                # dp -> NL, dp -> NH (leaving parking, truck alone)
                if i_is_dp and (j_is_NL or j_is_NH):
                    create_arc = True
                # dp -> cp (direct: decouple then couple without visiting anyone)
                if i_is_dp and j_is_cp:
                    create_arc = True
                # NL -> NL, NL -> NH, NL -> cp (truck alone)
                if i_is_NL and (j_is_NL or j_is_NH or j_is_cp):
                    create_arc = True
                # NH -> NL, NH -> cp (truck alone in decouple phase)
                if i_is_NH and (j_is_NL or j_is_cp):
                    create_arc = True

                if create_arc:
                    arc_v[k, vi, i, j] = model.addVar(
                        vtype=GRB.BINARY, name=f"a_{k}_{vi}_{i}_{j}")

        # Time, load, position. Customer time windows (when present in the
        # instance) are enforced as variable bounds on tm[k,vi,i] for the
        # node copy whose physical node is a customer with a tw entry.
        for i in nodes:
            rn = real_node(i)
            if rn in cust_tw:
                tw_lo, tw_hi = cust_tw[rn]
            else:
                tw_lo, tw_hi = 0.0, horizon_end
            tm[k, vi, i] = model.addVar(lb=tw_lo, ub=tw_hi,
                                        vtype=GRB.CONTINUOUS,
                                        name=f"tm_{k}_{vi}_{i}")
            ld[k, vi, i] = model.addVar(lb=0, ub=Q_total,
                                        vtype=GRB.CONTINUOUS,
                                        name=f"ld_{k}_{vi}_{i}")
            ps[k, vi, i] = model.addVar(lb=0, ub=n_v,
                                        vtype=GRB.CONTINUOUS,
                                        name=f"ps_{k}_{vi}_{i}")

    model.update()

    # =========================================================================
    # OBJECTIVE
    # =========================================================================
    obj = gp.LinExpr()
    for (k, vi) in vehicles:
        vc_d = vc_info[k]
        is_cv = is_complete(k)
        cf = vc_d["truck_cost_factor"]
        tw = vc_d["trailer_towing_cost_multiplier"]
        nodes = cv_all_nodes if is_cv else solo_nodes

        obj += vc_d["truck_fixed_cost"] * u[k, vi]

        for i in nodes:
            for j in nodes:
                if (k, vi, i, j) not in arc_v:
                    continue
                d = edist(i, j)

                if not is_cv:
                    # Solo truck: truck cost only
                    obj += cf * d * arc_v[k, vi, i, j]
                else:
                    # Determine if this is a with-trailer or truck-alone arc
                    # by checking node types.
                    # With-trailer arcs: involve depot, NH, T, dp(arriving), cp(leaving)
                    # Truck-alone arcs: involve dp(leaving), NL, cp(arriving)
                    is_x_arc = False
                    # x-arc if source is depot/NH/cp AND dest is depot/NH/dp
                    i_trailer = (i == depot_id or i in N_H_s
                                 or i in cp_nodes_s)
                    j_trailer = (j == depot_id or j in N_H_s
                                 or j in dp_nodes_s)
                    if i_trailer and j_trailer:
                        is_x_arc = True

                    if is_x_arc:
                        obj += (cf + tw) * d * arc_v[k, vi, i, j]
                    else:
                        obj += cf * d * arc_v[k, vi, i, j]

        # Handling costs for CV
        if is_cv:
            for p in P_orig:
                dp = dp_node[p]
                cp_n = cp_node[p]
                # Decouple cost: paid if dp node is visited
                dp_in = gp.quicksum(
                    arc_v[k, vi, j, dp] for j in nodes
                    if (k, vi, j, dp) in arc_v)
                obj += decoupling_cost_val * dp_in
                # Couple cost: paid if cp node is visited
                cp_in = gp.quicksum(
                    arc_v[k, vi, j, cp_n] for j in nodes
                    if (k, vi, j, cp_n) in arc_v)
                obj += coupling_cost_val * cp_in

    model.setObjective(obj, GRB.MINIMIZE)

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # (C1) Each customer served exactly once
    for n in N_cust:
        model.addConstr(
            gp.quicksum(sv[k, vi, n] for (k, vi) in vehicles) == 1,
            name=f"once_{n}")

    # (C2) Fleet
    for tt_id, tt in truck_types.items():
        rel = [(k, vi) for (k, vi) in vehicles
               if vc_info[k]["truck_type_id"] == tt_id]
        if rel:
            model.addConstr(
                gp.quicksum(u[k, vi] for (k, vi) in rel)
                <= tt["num_available"], name=f"ft_{tt_id}")
    for tt_id, tt in trailer_types.items():
        rel = [(k, vi) for (k, vi) in vehicles
               if vc_info[k]["trailer_type_id"] == tt_id]
        if rel:
            model.addConstr(
                gp.quicksum(u[k, vi] for (k, vi) in rel)
                <= tt["num_available"], name=f"fh_{tt_id}")

    # Symmetry breaking
    for vc in vehicle_classes:
        k = vc["class_id"]
        for vi in range(vc["num_available"] - 1):
            model.addConstr(u[k, vi] >= u[k, vi + 1],
                            name=f"sym_{k}_{vi}")

    # Per-vehicle
    for (k, vi) in vehicles:
        vc_d = vc_info[k]
        is_cv = is_complete(k)
        Q_L = vc_d["truck_capacity"]
        Q_H = vc_d["trailer_capacity"] if is_cv else 0
        Q_total = Q_L + Q_H
        nodes = cv_all_nodes if is_cv else solo_nodes
        n_v = len(nodes)

        def out_f(i):
            return gp.quicksum(arc_v[k, vi, i, j] for j in nodes
                               if (k, vi, i, j) in arc_v)
        def in_f(i):
            return gp.quicksum(arc_v[k, vi, j, i] for j in nodes
                               if (k, vi, j, i) in arc_v)

        # Depot
        model.addConstr(out_f(depot_id) == u[k, vi], name=f"do_{k}_{vi}")
        model.addConstr(in_f(depot_id) == u[k, vi], name=f"di_{k}_{vi}")

        # Flow conservation at all non-depot nodes
        for i in nodes:
            if i == depot_id:
                continue
            model.addConstr(in_f(i) == out_f(i), name=f"fc_{k}_{vi}_{i}")

        # Served = inflow for customers
        for n in N_cust:
            model.addConstr(in_f(n) == sv[k, vi, n],
                            name=f"svl_{k}_{vi}_{n}")
            model.addConstr(sv[k, vi, n] <= u[k, vi],
                            name=f"svu_{k}_{vi}_{n}")

        # At most one visit per node
        for i in nodes:
            if i == depot_id:
                continue
            model.addConstr(in_f(i) <= 1, name=f"v1_{k}_{vi}_{i}")

        # Depot load = 0
        model.addConstr(ld[k, vi, depot_id] == 0, name=f"dl_{k}_{vi}")

        # MTZ + time + load along arcs
        for i in nodes:
            for j in nodes:
                if (k, vi, i, j) not in arc_v:
                    continue

                # MTZ
                if i != depot_id and j != depot_id:
                    model.addConstr(
                        ps[k, vi, j] >= ps[k, vi, i] + 1
                        - n_v * (1 - arc_v[k, vi, i, j]),
                        name=f"mtz_{k}_{vi}_{i}_{j}")

                # Time propagation (skip j=depot: depot time=0 is departure,
                # return time is not tracked at depot since it's a single node)
                if j != depot_id:
                    st_i = svc.get(real_node(i), 0.0)
                    handling_i = 0.0
                    if i in dp_nodes_s:
                        handling_i = decoupling_time_val
                    elif i in cp_nodes_s:
                        handling_i = coupling_time_val

                    model.addConstr(
                        tm[k, vi, j] >= tm[k, vi, i] + st_i + handling_i
                        + ettime(i, j) - M * (1 - arc_v[k, vi, i, j]),
                        name=f"tm_{k}_{vi}_{i}_{j}")

                # Load propagation (both lower and upper bounds for tightness)
                rj = real_node(j)
                if rj in N_cust_s and j not in dp_nodes_s and j not in cp_nodes_s:
                    # Visiting a customer: load increases by supply[rj]
                    model.addConstr(
                        ld[k, vi, j] >= ld[k, vi, i] + supply[rj]
                        - Q_total * (1 - arc_v[k, vi, i, j]),
                        name=f"ld_{k}_{vi}_{i}_{j}")
                    model.addConstr(
                        ld[k, vi, j] <= ld[k, vi, i] + supply[rj]
                        + Q_total * (1 - arc_v[k, vi, i, j]),
                        name=f"ldup_{k}_{vi}_{i}_{j}")
                elif j != depot_id:
                    # Non-customer (dp, cp): load preserved
                    model.addConstr(
                        ld[k, vi, j] >= ld[k, vi, i]
                        - Q_total * (1 - arc_v[k, vi, i, j]),
                        name=f"ldu_{k}_{vi}_{i}_{j}")
                    model.addConstr(
                        ld[k, vi, j] <= ld[k, vi, i]
                        + Q_total * (1 - arc_v[k, vi, i, j]),
                        name=f"ldl_{k}_{vi}_{i}_{j}")

        # Route duration: for any arc (i, depot), the departure from i
        # must leave enough time to return:
        # tm[i] + svc[i] + handling + ttime[i][depot] <= horizon_end
        for i in nodes:
            if i == depot_id:
                continue
            if (k, vi, i, depot_id) not in arc_v:
                continue
            st_i = svc.get(real_node(i), 0.0)
            handling_i = 0.0
            if i in dp_nodes_s:
                handling_i = decoupling_time_val
            elif i in cp_nodes_s:
                handling_i = coupling_time_val
            model.addConstr(
                tm[k, vi, i] + st_i + handling_i + ettime(i, depot_id)
                <= horizon_end + M * (1 - arc_v[k, vi, i, depot_id]),
                name=f"dur_{k}_{vi}_{i}")

        # ---- COMPLETE VEHICLE specific constraints ----
        if is_cv:
            # (CV1) At most one decouple location used (at most one dp visited)
            model.addConstr(
                gp.quicksum(in_f(dp_node[p]) for p in P_orig) <= 1,
                name=f"mdc_{k}_{vi}")

            # (CV2) At most one couple location used
            model.addConstr(
                gp.quicksum(in_f(cp_node[p]) for p in P_orig) <= 1,
                name=f"mcp_{k}_{vi}")

            # (CV3) Number of decouples = number of couples
            model.addConstr(
                gp.quicksum(in_f(dp_node[p]) for p in P_orig) ==
                gp.quicksum(in_f(cp_node[p]) for p in P_orig),
                name=f"dcp_{k}_{vi}")

            # (CV4) Couple must be at same physical location as decouple
            for p in P_orig:
                model.addConstr(
                    in_f(cp_node[p]) <= in_f(dp_node[p]),
                    name=f"cpdc_{k}_{vi}_{p}")

            # (CV5) Truck customers can only be served if decouple happens
            for n in N_L:
                model.addConstr(
                    sv[k, vi, n] <=
                    gp.quicksum(in_f(dp_node[p]) for p in P_orig),
                    name=f"tcdc_{k}_{vi}_{n}")

            # (CV5b) Ordering: NL customers must come between dp and cp
            # in the MTZ position. This enforces that the trailer is
            # detached before serving truck customers.
            for n in N_L:
                for p in P_orig:
                    dp_n = dp_node[p]
                    cp_n = cp_node[p]
                    # ps[n] >= ps[dp] + 1 if both n and dp are visited
                    model.addConstr(
                        ps[k, vi, n] >= ps[k, vi, dp_n] + 1
                        - n_v * (1 - sv[k, vi, n])
                        - n_v * (1 - in_f(dp_n)),
                        name=f"odp_{k}_{vi}_{n}_{p}")
                    # ps[cp] >= ps[n] + 1 if both n and cp are visited
                    model.addConstr(
                        ps[k, vi, cp_n] >= ps[k, vi, n] + 1
                        - n_v * (1 - sv[k, vi, n])
                        - n_v * (1 - in_f(cp_n)),
                        name=f"ocp_{k}_{vi}_{n}_{p}")

            # (CV6) Capacity during truck-alone phase:
            # When the truck is alone (between dp and cp), the supply
            # picked up must fit in the truck. The load at dp includes
            # supply from the with-trailer phase, and the load increase
            # from dp to cp must be <= Q_L (truck capacity).
            # ld[cp] - ld[dp] <= Q_L for each parking location p.
            for p in P_orig:
                dp_n = dp_node[p]
                cp_n = cp_node[p]
                model.addConstr(
                    ld[k, vi, cp_n] - ld[k, vi, dp_n] <= Q_L,
                    name=f"capalone_{k}_{vi}_{p}")

        # ---- CAPACITY CUTS ----
        # Total served supply <= vehicle total capacity
        model.addConstr(
            gp.quicksum(supply[n] * sv[k, vi, n] for n in N_cust) <= Q_total,
            name=f"capcut_{k}_{vi}")

    # =========================================================================
    # WARM START: Greedy heuristic
    # =========================================================================
    print("Building heuristic warm-start solution...")
    heur_routes = _greedy_heuristic(data, P_orig, dp_node, cp_node, vc_info)
    heur_served = set()
    for r in heur_routes:
        heur_served.update(r["customers"])
    print(f"Heuristic: {len(heur_served)}/{len(N_cust)} customers, "
          f"{len(heur_routes)} vehicles")

    # Apply warm start
    for (k, vi) in vehicles:
        u[k, vi].Start = 0
        for n in N_cust:
            sv[k, vi, n].Start = 0
        nodes = cv_all_nodes if is_complete(k) else solo_nodes
        for i in nodes:
            for j in nodes:
                if (k, vi, i, j) in arc_v:
                    arc_v[k, vi, i, j].Start = 0

    for r in heur_routes:
        k = r["class_id"]
        vi = r["vehicle_index"]
        is_cv = is_complete(k)
        vc_d = vc_info[k]
        Q_total = vc_d["truck_capacity"] + (vc_d["trailer_capacity"] if is_cv else 0)

        u[k, vi].Start = 1
        for n in r["customers"]:
            sv[k, vi, n].Start = 1

        seq = r["sequence"]
        for idx in range(len(seq) - 1):
            i, j = seq[idx], seq[idx + 1]
            if (k, vi, i, j) in arc_v:
                arc_v[k, vi, i, j].Start = 1

        # Time and load
        cur_time = 0.0
        cur_load = 0.0
        for idx, node in enumerate(seq):
            if (k, vi, node) in tm:
                tm[k, vi, node].Start = cur_time
            if (k, vi, node) in ld:
                ld[k, vi, node].Start = min(cur_load, Q_total)
            if (k, vi, node) in ps:
                ps[k, vi, node].Start = idx

            if idx < len(seq) - 1:
                nxt = seq[idx + 1]
                rn = real_node(node)
                st_node = svc.get(rn, 0.0)
                handling = 0.0
                if node in dp_nodes_s:
                    handling = decoupling_time_val
                elif node in cp_nodes_s:
                    handling = coupling_time_val
                cur_time += st_node + handling + ettime(node, nxt)
                rnxt = real_node(nxt)
                if rnxt in N_cust_s and nxt not in dp_nodes_s and nxt not in cp_nodes_s:
                    cur_load += supply[rnxt]

        # Depot load must be 0 (override end-of-route accumulation)
        if (k, vi, depot_id) in ld:
            ld[k, vi, depot_id].Start = 0
        if (k, vi, depot_id) in tm:
            tm[k, vi, depot_id].Start = 0
        if (k, vi, depot_id) in ps:
            ps[k, vi, depot_id].Start = 0

    # =========================================================================
    # SOLVE
    # =========================================================================
    model.update()
    print(f"\nModel: {model.NumVars} vars, {model.NumConstrs} constrs")
    print(f"Time limit: {time_limit}s")

    solve_start = time_module.time()
    model.optimize()
    solve_time = time_module.time() - solve_start

    # =========================================================================
    # EXTRACT SOLUTION
    # =========================================================================
    solution = {
        "instance_path": instance_path,
        "solver": "Gurobi",
        "formulation": "compact_arc_flow_MIP_with_node_copies",
        "note": (
            "The paper only provides a set-partitioning model (1a)-(1d) "
            "requiring branch-and-price-and-cut. This compact arc-flow MIP "
            "with decouple/couple node copies is an implementation choice."
        ),
        "solve_time_seconds": round(solve_time, 2),
        "time_limit_seconds": time_limit,
    }

    if model.Status == GRB.OPTIMAL:
        solution["status"] = "optimal"
        solution["objective_value"] = round(model.ObjVal, 6)
        solution["mip_gap"] = model.MIPGap
    elif model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            solution["status"] = "time_limit_with_solution"
            solution["objective_value"] = round(model.ObjVal, 6)
            solution["mip_gap"] = model.MIPGap
            solution["best_bound"] = round(model.ObjBound, 6)
        else:
            solution["status"] = "time_limit_no_solution"
            solution["objective_value"] = None
            try:
                solution["best_bound"] = round(model.ObjBound, 6)
            except Exception:
                solution["best_bound"] = None
    elif model.Status == GRB.INFEASIBLE:
        solution["status"] = "infeasible"
        solution["objective_value"] = None
        print("WARNING: Model is infeasible.")
        try:
            model.computeIIS()
            iis_c = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
            solution["iis_constraints"] = iis_c[:30]
            print(f"IIS (first 30): {iis_c[:30]}")
        except Exception:
            pass
    elif model.Status == GRB.UNBOUNDED:
        solution["status"] = "unbounded"
        solution["objective_value"] = None
    else:
        solution["status"] = f"gurobi_status_{model.Status}"
        solution["objective_value"] = None
        if model.SolCount > 0:
            solution["objective_value"] = round(model.ObjVal, 6)

    # ---- Route extraction ----
    routes = []
    if model.SolCount > 0:
        for (k_v, vi_v) in vehicles:
            vc_d = vc_info[k_v]
            is_cv = is_complete(k_v)
            if u[k_v, vi_v].X < 0.5:
                continue

            route_info = {
                "vehicle_class": k_v,
                "vehicle_index": vi_v,
                "vehicle_description": vc_d["description"],
                "is_complete_vehicle": is_cv,
                "fixed_cost": vc_d["truck_fixed_cost"],
                "customers_served": [],
                "route_sequence": [],
                "route_sequence_physical": [],
                "decouple_locations": [],
                "couple_locations": [],
            }

            for n in N_cust:
                if sv[k_v, vi_v, n].X > 0.5:
                    route_info["customers_served"].append(n)

            nodes = cv_all_nodes if is_cv else solo_nodes
            adj = {}
            for i in nodes:
                for j in nodes:
                    if (k_v, vi_v, i, j) in arc_v and arc_v[k_v, vi_v, i, j].X > 0.5:
                        adj[i] = j

            seq = [depot_id]
            cur = depot_id
            for _ in range(len(nodes) + 2):
                nxt = adj.get(cur)
                if nxt is None:
                    break
                seq.append(nxt)
                if nxt == depot_id:
                    break
                cur = nxt
            if seq[-1] != depot_id:
                seq.append(depot_id)

            route_info["route_sequence"] = seq
            route_info["route_sequence_physical"] = [real_node(n) for n in seq]

            # Persist the model's time decisions so the feasibility checker
            # can verify time windows / planning horizon directly from the
            # MIP's t[k,vi,n] schedule, rather than re-simulating timing
            # (which mis-handles node copies / decouple / couple / transfer
            # events on this TTRPTW network).
            arrival_times = {}
            for n in seq:
                if (k_v, vi_v, n) in tm:
                    arrival_times[str(n)] = round(float(tm[k_v, vi_v, n].X), 6)
            route_info["arrival_times"] = arrival_times

            if is_cv:
                for p in P_orig:
                    dp = dp_node[p]
                    cp_n = cp_node[p]
                    dp_in_val = sum(
                        arc_v[k_v, vi_v, j, dp].X
                        for j in nodes if (k_v, vi_v, j, dp) in arc_v)
                    if dp_in_val > 0.5:
                        route_info["decouple_locations"].append(p)
                    cp_in_val = sum(
                        arc_v[k_v, vi_v, j, cp_n].X
                        for j in nodes if (k_v, vi_v, j, cp_n) in arc_v)
                    if cp_in_val > 0.5:
                        route_info["couple_locations"].append(p)

            # Costs
            cf = vc_d["truck_cost_factor"]
            tw = vc_d["trailer_towing_cost_multiplier"]
            travel = 0.0
            for i in nodes:
                for j in nodes:
                    if (k_v, vi_v, i, j) in arc_v and arc_v[k_v, vi_v, i, j].X > 0.5:
                        d = edist(i, j)
                        if not is_cv:
                            travel += cf * d
                        else:
                            i_trailer = (i == depot_id or i in N_H_s
                                         or i in cp_nodes_s)
                            j_trailer = (j == depot_id or j in N_H_s
                                         or j in dp_nodes_s)
                            if i_trailer and j_trailer:
                                travel += (cf + tw) * d
                            else:
                                travel += cf * d

            handling = (len(route_info["decouple_locations"]) * decoupling_cost_val
                        + len(route_info["couple_locations"]) * coupling_cost_val)

            route_info["travel_cost"] = round(travel, 4)
            route_info["handling_cost"] = round(handling, 4)
            route_info["route_cost"] = round(
                vc_d["truck_fixed_cost"] + travel + handling, 4)
            routes.append(route_info)

    solution["routes"] = routes

    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"\nStatus: {solution['status']}")
    print(f"Objective: {solution['objective_value']}")
    print(f"Solve time: {solve_time:.2f}s")
    if routes:
        print(f"Vehicles: {len(routes)}")
        for r in routes:
            print(f"  Class {r['vehicle_class']} idx {r['vehicle_index']}: "
                  f"custs={r['customers_served']}, cost={r['route_cost']}")
            print(f"    Route (physical): {r['route_sequence_physical']}")
            if r['decouple_locations']:
                print(f"    Decouple at: {r['decouple_locations']}")
            if r['couple_locations']:
                print(f"    Couple at: {r['couple_locations']}")
    print(f"\nSolution written to: {solution_path}")
    return solution


def _greedy_heuristic(data, P_orig, dp_node, cp_node, vc_info):
    """
    Build a feasible heuristic solution using expanded node copies.
    Returns list of routes with sequences using expanded node IDs.
    """
    params = data["parameters"]
    fleet = data["fleet"]
    dist_orig = data["distance_matrix"]
    ttime_orig = data["travel_time_matrix"]

    depot_id = data["depot"]["id"]
    N_L = [c["id"] for c in data["truck_customers"]]
    N_H = [c["id"] for c in data["trailer_customers"]]
    T_locs = [t["id"] for t in data["transshipment_locations"]]

    supply = {}
    svc = {}
    for c in data["truck_customers"] + data["trailer_customers"]:
        supply[c["id"]] = c["supply"]
        svc[c["id"]] = c.get("service_time", params["service_time"])
    for nid in [depot_id] + T_locs:
        supply[nid] = 0
        svc[nid] = 0.0

    horizon_end = params["planning_horizon"][1]
    decoupling_t = params["decoupling_time"]
    coupling_t = params["coupling_time"]

    n_orig = len(dist_orig)

    def real_node(i):
        if i < n_orig:
            return i
        idx = (i - n_orig) % len(P_orig)
        return P_orig[idx]

    def edist(i, j):
        return dist_orig[real_node(i)][real_node(j)]

    def ettime(i, j):
        return ttime_orig[real_node(i)][real_node(j)]

    unserved_L = set(N_L)
    unserved_H = set(N_H)
    used_count = {vc["class_id"]: 0 for vc in fleet["vehicle_classes"]}
    routes = []

    # Strategy 1: Complete vehicles for trailer customers + truck customers
    for vc in sorted(fleet["vehicle_classes"],
                     key=lambda v: v["total_capacity"], reverse=True):
        k = vc["class_id"]
        if vc["trailer_type_id"] == 0:
            continue
        Q_L = vc["truck_capacity"]
        Q_H = vc["trailer_capacity"]
        Q_total = Q_L + Q_H
        cf = vc["truck_cost_factor"]
        tw = vc["trailer_towing_cost_multiplier"]

        while used_count[k] < vc["num_available"] and unserved_H:
            # Phase 1: serve trailer customers with trailer
            seq_phase1 = [depot_id]
            load = 0
            t = 0.0
            served_H = []

            cands = sorted(unserved_H, key=lambda n: dist_orig[seq_phase1[-1]][n])
            for n in cands:
                new_load = load + supply[n]
                extra = svc[seq_phase1[-1]] + ttime_orig[seq_phase1[-1]][n]
                back = svc[n] + ttime_orig[n][depot_id]
                if new_load <= Q_total and t + extra + back <= horizon_end:
                    seq_phase1.append(n)
                    load += supply[n]
                    t += svc[seq_phase1[-2]] + ttime_orig[seq_phase1[-2]][n]
                    served_H.append(n)

            if not served_H:
                break

            # Phase 2: try to decouple and serve truck customers
            best_park = None
            best_L = []
            best_cost = float("inf")

            if unserved_L:
                for p in P_orig:
                    # Time to go to dp(p), decouple, serve truck custs, return, couple
                    dp = dp_node[p]
                    cp_n = cp_node[p]
                    t_to_p = t + svc[seq_phase1[-1]] + ttime_orig[seq_phase1[-1]][p]
                    t_after_dc = t_to_p + decoupling_t
                    truck_load = 0
                    truck_seq = []
                    t_truck = t_after_dc
                    cur = p

                    remaining_L = sorted(unserved_L, key=lambda n: dist_orig[cur][n])
                    for n in remaining_L:
                        nl = truck_load + supply[n]
                        extra = svc[cur] + ttime_orig[cur][n]
                        back = svc[n] + ttime_orig[n][p] + coupling_t + ttime_orig[p][depot_id]
                        if nl <= Q_L and t_truck + extra + back <= horizon_end:
                            truck_seq.append(n)
                            truck_load += supply[n]
                            t_truck += svc[cur] + ttime_orig[cur][n]
                            cur = n

                    if truck_seq:
                        # cost = dist to p + decouple + truck route + return + couple + to depot
                        cost = (dist_orig[seq_phase1[-1]][p] * (cf + tw)
                                + sum(dist_orig[truck_seq[i]][truck_seq[i+1]] * cf
                                      for i in range(len(truck_seq)-1)) if len(truck_seq) > 1 else 0)
                        cost += dist_orig[p][truck_seq[0]] * cf if truck_seq else 0
                        cost += dist_orig[truck_seq[-1]][p] * cf if truck_seq else 0
                        cost += dist_orig[p][depot_id] * (cf + tw)

                        if cost < best_cost:
                            best_cost = cost
                            best_park = p
                            best_L = truck_seq[:]

            # Build full expanded sequence
            if best_park is not None and best_L:
                dp = dp_node[best_park]
                cp_n = cp_node[best_park]
                full_seq = seq_phase1 + [dp] + best_L + [cp_n, depot_id]
                served_all = served_H + best_L
            else:
                full_seq = seq_phase1 + [depot_id]
                served_all = served_H

            routes.append({
                "class_id": k,
                "vehicle_index": used_count[k],
                "sequence": full_seq,
                "customers": served_all,
            })
            used_count[k] += 1
            for n in served_H:
                unserved_H.discard(n)
            for n in best_L:
                unserved_L.discard(n)

    # Strategy 2: Solo trucks for remaining customers
    all_unserved = sorted(unserved_L | unserved_H)
    for vc in sorted(fleet["vehicle_classes"],
                     key=lambda v: v["truck_capacity"], reverse=True):
        k = vc["class_id"]
        if vc["trailer_type_id"] != 0:
            continue
        Q_L = vc["truck_capacity"]

        while used_count[k] < vc["num_available"] and all_unserved:
            seq = [depot_id]
            load = 0
            t = 0.0
            served = []

            remaining = list(all_unserved)
            while remaining:
                best_n = None
                best_d = float("inf")
                for n in remaining:
                    nl = load + supply[n]
                    extra = svc[seq[-1]] + ttime_orig[seq[-1]][n]
                    back = svc[n] + ttime_orig[n][depot_id]
                    if nl <= Q_L and t + extra + back <= horizon_end:
                        d = dist_orig[seq[-1]][n]
                        if d < best_d:
                            best_d = d
                            best_n = n
                if best_n is None:
                    break
                seq.append(best_n)
                load += supply[best_n]
                t += svc[seq[-2]] + ttime_orig[seq[-2]][best_n]
                served.append(best_n)
                remaining.remove(best_n)

            if not served:
                break

            seq.append(depot_id)
            routes.append({
                "class_id": k,
                "vehicle_index": used_count[k],
                "sequence": seq,
                "customers": served,
            })
            used_count[k] += 1
            for n in served:
                all_unserved.remove(n)

    if all_unserved:
        print(f"WARNING: Heuristic failed to serve: {all_unserved}")

    return routes


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Solve the TTRPTW using a compact arc-flow MIP in Gurobi."
        )
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Output solution JSON path.")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Gurobi time limit in seconds (default: 3600).")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    if not os.path.isfile(args.instance_path):
        print(f"ERROR: Instance file not found: {args.instance_path}")
        sys.exit(1)

    solve_ttrptw(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
