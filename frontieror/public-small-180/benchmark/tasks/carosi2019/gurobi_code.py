"""
Gurobi MILP implementation of the ITTVS (Integrated Timetabling and Vehicle Scheduling) model.

Source: Carosi, Frangioni, Galli, Girardi, Vallese (2019)
  "A matheuristic for integrated timetabling and vehicle scheduling"
  Transportation Research Part B 127 (2019) 99-124

Mathematical model (1)-(6) as described in Section 4 of the paper.

Uses the "pure" VS compatibility graph (Section 4.2.1) for simplicity.
"""

import argparse
import json
import math

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
        pass# ============================================================
# INSTANCE LOADING
# ============================================================

def load_instance(path):
    with open(path) as f:
        raw = json.load(f)
    return convert_instance(raw)


def convert_instance(raw):
    """Convert raw instance JSON to the format expected by the solver.

    Raw instance keys:
      potential_trips[i]: id, pattern_id, direction, start_terminal, end_terminal,
                          departure_time_minutes, arrival_time_minutes, time_window_index
      time_windows[i]:    index, ideal_headway_seconds, min_headway_seconds, max_headway_seconds
      stopping_times[terminal][i]: time_window_index, min_stopping_time_minutes, max_stopping_time_minutes
      pull_in_out_times[terminal][i]: time_window_index, pull_out_time_minutes, pull_in_time_minutes
      directions[i]:      pattern_id  (used as direction identifier)
      objective_function:  alpha

    Solver expects:
      trips[i]:    id, direction (=pattern_id), a (=departure minutes), st (=departure),
                   et (=arrival), sn (=start_terminal), en (=end_terminal), time_window (=tw index)
      time_windows[i]: id (=index), headway_params[direction]->{ideal,min,max} (in minutes),
                       terminal_params[terminal]->{min_dwell,max_dwell,pullout_time,pullin_time},
                       depot_params->{min_dwell}
      directions:  list of pattern_id strings
      alpha:       from objective_function
    """
    # Build direction list (pattern_id strings)
    directions = [d['pattern_id'] for d in raw['directions']]

    # Build stopping_times and pull_in_out lookup: (terminal, tw_index) -> values
    stop_lookup = {}  # (terminal, tw_index) -> (min_stop, max_stop)
    for terminal, entries in raw['stopping_times'].items():
        for entry in entries:
            stop_lookup[(terminal, entry['time_window_index'])] = (
                entry['min_stopping_time_minutes'],
                entry['max_stopping_time_minutes'],
            )

    pull_lookup = {}  # (terminal, tw_index) -> (pull_out, pull_in)
    for terminal, entries in raw['pull_in_out_times'].items():
        for entry in entries:
            pull_lookup[(terminal, entry['time_window_index'])] = (
                entry['pull_out_time_minutes'],
                entry['pull_in_time_minutes'],
            )

    # Depot stopping times
    depot_stop_lookup = {}  # tw_index -> min_stop
    for entry in raw['stopping_times'].get('O', []):
        depot_stop_lookup[entry['time_window_index']] = entry['min_stopping_time_minutes']

    # All terminals (exclude depot 'O')
    terminals = [d['start_terminal'] for d in raw['directions']]
    terminals += [d['end_terminal'] for d in raw['directions']]
    terminals = list(set(t for t in terminals if t != 'O'))

    # Build converted time_windows
    time_windows = []
    for tw in raw['time_windows']:
        idx = tw['index']
        # Headway: same for all directions (seconds -> minutes)
        hw_ideal = tw['ideal_headway_seconds'] / 60.0
        hw_min = tw['min_headway_seconds'] / 60.0
        hw_max = tw['max_headway_seconds'] / 60.0
        headway_params = {d: {'ideal': hw_ideal, 'min': hw_min, 'max': hw_max} for d in directions}

        # Terminal params
        terminal_params = {}
        for t in terminals:
            min_dwell, max_dwell = stop_lookup.get((t, idx), (0, 9999))
            pullout, pullin = pull_lookup.get((t, idx), (0, 0))
            terminal_params[t] = {
                'min_dwell': min_dwell,
                'max_dwell': max_dwell,
                'pullout_time': pullout,
                'pullin_time': pullin,
            }

        depot_params = {'min_dwell': depot_stop_lookup.get(idx, 0)}

        time_windows.append({
            'id': idx,
            'headway_params': headway_params,
            'terminal_params': terminal_params,
            'depot_params': depot_params,
        })

    # Build trips from potential_trips
    trips = []
    for pt in raw['potential_trips']:
        trips.append({
            'id': pt['id'],
            'direction': pt['pattern_id'],
            'a': pt['main_stop_arrival_time_minutes'],  # arrival at main stop (for headway calc per paper)
            'st': pt['departure_time_minutes'],       # start time
            'et': pt['arrival_time_minutes'],          # end time
            'sn': pt['start_terminal'],                # start node (terminal)
            'en': pt['end_terminal'],                  # end node (terminal)
            'time_window': pt['time_window_index'],
        })

    # Determine initial/final trips per direction based on time window indices
    initial_trips = {}
    final_trips = {}
    init_tw = raw.get('initial_trip_time_windows', {})
    final_tw = raw.get('final_trip_time_windows', {})
    for d in directions:
        if d in init_tw:
            initial_trips[d] = [t['id'] for t in trips if t['direction'] == d and t['time_window'] == init_tw[d]]
        if d in final_tw:
            final_trips[d] = [t['id'] for t in trips if t['direction'] == d and t['time_window'] == final_tw[d]]

    alpha = raw.get('objective_function', {}).get('alpha', 1.0)
    M = raw.get('M', 10000)
    max_fleet = raw.get('max_fleet', None)

    return {
        'trips': trips,
        'directions': directions,
        'time_windows': time_windows,
        'initial_trips': initial_trips,
        'final_trips': final_trips,
        'alpha': alpha,
        'M': M,
        'max_fleet': max_fleet,
    }


# ============================================================
# PARAMETER HELPERS
# ============================================================

def get_hw_params(instance, direction, tw_id):
    """Return (ideal, min, max) headway for given direction and time window."""
    for tw in instance['time_windows']:
        if tw['id'] == tw_id:
            p = tw['headway_params'][direction]
            return p['ideal'], p['min'], p['max']
    raise ValueError(f"Time window {tw_id} not found")


def get_terminal_params(instance, terminal, tw_id):
    """Return (min_dwell, max_dwell, pullout_time, pullin_time) for terminal and time window."""
    for tw in instance['time_windows']:
        if tw['id'] == tw_id:
            p = tw['terminal_params'][terminal]
            return p['min_dwell'], p['max_dwell'], p['pullout_time'], p['pullin_time']
    raise ValueError(f"Time window {tw_id} not found")


def get_depot_min_dwell(instance, tw_id):
    """Return minimum depot dwell time for time window."""
    for tw in instance['time_windows']:
        if tw['id'] == tw_id:
            return tw['depot_params'].get('min_dwell', 0)
    return 0


# ============================================================
# PENALTY FUNCTION
# ============================================================

def headway_penalty(actual_hw, ideal_hw):
    """
    Quadratic penalty for headway deviation from ideal.

    NOT SPECIFIED IN PAPER: exact closed-form formula of the quadratic penalty.
    INFERRED ASSUMPTION: penalty = (actual_hw - ideal_hw)^2 (squared deviation in minutes^2).
    Properties: zero if actual_hw == ideal_hw; positive and nondecreasing in |actual_hw - ideal_hw|.
    """
    return float((actual_hw - ideal_hw) ** 2)


# ============================================================
# TT GRAPH CONSTRUCTION (Section 4.1)
# ============================================================

def build_tt_graph(direction, trips, instance):
    """
    Build TT compatibility graph G_d^TT for direction d.

    Nodes: N_d^TT = T_d ∪ {source_d, sink_d}
    Arcs:
      - (source_d, i) for i in T_d^ini: cost 0
      - (i, j) for i,j in T_d s.t. l_d^{h(i)} <= a(j)-a(i) <= l_bar_d^{h(i)}: cost = penalty
      - (i, sink_d) for i in T_d^fin: cost 0

    Returns:
      source, sink : node identifiers for source/sink
      nodes        : list of all nodes
      arcs         : dict (u,v) -> cost
      backward_star: dict trip_id -> list of arcs entering that node
    """
    source = ('source', direction)
    sink = ('sink', direction)

    initial_ids = set(
        instance.get('initial_trips', {}).get(direction, [t['id'] for t in trips])
    )
    final_ids = set(
        instance.get('final_trips', {}).get(direction, [t['id'] for t in trips])
    )

    arcs = {}  # (u, v) -> cost

    # Source arcs to initial trips (cost 0)
    for t in trips:
        if t['id'] in initial_ids:
            arcs[(source, t['id'])] = 0.0

    # Trip-to-trip arcs: arc (i, j) iff l_d^{h(i)} <= a(j)-a(i) <= l_bar_d^{h(i)}
    # Arc cost = penalty(a(j)-a(i), l_hat_d^{h(i)})
    sorted_trips = sorted(trips, key=lambda x: x['a'])
    for idx_i, ti in enumerate(sorted_trips):
        l_hat, l_min, l_max = get_hw_params(instance, direction, ti['time_window'])
        for idx_j in range(idx_i + 1, len(sorted_trips)):
            tj = sorted_trips[idx_j]
            hw = tj['a'] - ti['a']
            if hw > l_max:
                break  # sorted by a; all further trips exceed max headway
            if hw >= l_min:
                cost = headway_penalty(hw, l_hat)
                arcs[(ti['id'], tj['id'])] = cost

    # Final arcs from final trips to sink (cost 0)
    for t in trips:
        if t['id'] in final_ids:
            arcs[(t['id'], sink)] = 0.0

    # Backward star B(i): set of arcs entering trip node i
    # Used in linking constraint (6)
    backward_star = {t['id']: [] for t in trips}
    for (u, v), cost in arcs.items():
        if v in backward_star:
            backward_star[v].append((u, v))

    all_nodes = [source, sink] + [t['id'] for t in trips]
    return source, sink, all_nodes, arcs, backward_star


# ============================================================
# VS GRAPH CONSTRUCTION — "Pure" version (Section 4.2.1)
# ============================================================

def build_vs_graph_pure(trips, instance):
    """
    Build the "pure" VS compatibility graph G^VS (Section 4.2.1).

    Nodes: N^VS = {(i,'start'), (i,'end') : i in T} ∪ {O^-, O^+}

    Arc types:
      1. Trip arcs (i^-, i^+): cost 0, capacity 1
      2. In-line compatibility arcs (i^+, j^-) where en(i)=sn(j),
         feasible dwell: delta_n^{h(i)} <= st(j)-et(i) <= delta_bar_n^{h(i)}
         cost = st(j)-et(i)-delta_n^{h(i)} (extra waiting), capacity 1
      3. Out-line compatibility arcs (i^+, j^-) where en(i)!=sn(j),
         feasible: st(j)-et(i) >= t_{en(i),-}^{h(i)} + delta_O^{h(i)} + t_{sn(j),+}^{h(j)}
         cost = t_{en(i),-}^{h(i)} + t_{sn(j),+}^{h(j)}, capacity 1
      4. Start arcs (O^-, i^-): cost = t_{sn(i),+}^{h(i)}, capacity 1
      5. End arcs (i^+, O^+): cost = t_{en(i),-}^{h(i)}, capacity 1
      6. Return arc (O^+, O^-): cost = M (fleet size proxy), capacity = max_fleet

    Returns:
      O_minus, O_plus : node identifiers
      all_nodes       : set of all nodes
      arcs            : dict (u,v) -> (cost, capacity)
    """
    O_minus = 'VS_O_minus'
    O_plus = 'VS_O_plus'

    M = instance.get('M', 10000)
    max_fleet = instance.get('max_fleet', None)
    cap_K = max_fleet if (max_fleet is not None and max_fleet > 0) else 100000

    alpha = instance.get('alpha', 1.0)

    arcs = {}  # (u, v) -> (cost, capacity)

    for t in trips:
        i = t['id']
        i_start = (i, 'start')  # i^-
        i_end = (i, 'end')      # i^+
        tw_id = t['time_window']
        sn, en = t['sn'], t['en']

        _, _, pullout_sn, _ = get_terminal_params(instance, sn, tw_id)
        _, _, _, pullin_en = get_terminal_params(instance, en, tw_id)

        # 1. Trip arc (i^-, i^+): cost 0, capacity 1
        arcs[(i_start, i_end)] = (0.0, 1)

        # 4. Start arc (O^-, i^-): cost = t_{sn(i),+}^{h(i)}, capacity 1
        arcs[(O_minus, i_start)] = (float(pullout_sn), 1)

        # 5. End arc (i^+, O^+): cost = t_{en(i),-}^{h(i)}, capacity 1
        arcs[(i_end, O_plus)] = (float(pullin_en), 1)

    # 2 & 3. Compatibility arcs between trips
    # Optimized: sort trips by start time and use early termination.
    # Pre-compute terminal params to avoid repeated lookups.
    tp_cache = {}
    def get_tp_cached(terminal, tw_id):
        key = (terminal, tw_id)
        if key not in tp_cache:
            tp_cache[key] = get_terminal_params(instance, terminal, tw_id)
        return tp_cache[key]

    dd_cache = {}
    def get_dd_cached(tw_id):
        if tw_id not in dd_cache:
            dd_cache[tw_id] = get_depot_min_dwell(instance, tw_id)
        return dd_cache[tw_id]

    sorted_by_st = sorted(trips, key=lambda t: t['st'])

    for idx_i, ti in enumerate(sorted_by_st):
        i = ti['id']
        tw_i = ti['time_window']
        en_i = ti['en']
        _, _, _, pullin_en_i = get_tp_cached(en_i, tw_i)
        min_dwell_i, max_dwell_i, _, _ = get_tp_cached(en_i, tw_i)
        depot_dwell_i = get_dd_cached(tw_i)

        # Only consider trips j where st(j) > et(i) (j starts after i ends)
        # and wait = st(j) - et(i) is within a reasonable window.
        # Max useful wait: max_dwell at any terminal, or large deadhead time
        max_useful_wait = max(max_dwell_i, pullin_en_i + depot_dwell_i + 60)

        for idx_j in range(idx_i + 1, len(sorted_by_st)):
            tj = sorted_by_st[idx_j]
            wait = tj['st'] - ti['et']

            if wait < 0:
                continue  # j starts before i ends
            if wait > max_useful_wait:
                break  # sorted by st; all further trips have even larger wait

            j = tj['id']
            tw_j = tj['time_window']
            sn_j = tj['sn']
            _, _, pullout_sn_j, _ = get_tp_cached(sn_j, tw_j)

            if en_i == sn_j:
                # 2. In-line compatibility: en(i) = sn(j)
                # Condition: delta_{en(i)}^{h(i)} <= st(j)-et(i) <= delta_bar_{en(i)}^{h(i)}
                if min_dwell_i <= wait <= max_dwell_i:
                    extra_wait = float(wait - min_dwell_i)
                    arc_key = ((i, 'end'), (j, 'start'))
                    arcs[arc_key] = (extra_wait, 1)
            else:
                # 3. Out-line compatibility: en(i) != sn(j)
                # Condition: st(j)-et(i) >= t_{en(i),-}^{h(i)} + delta_O^{h(i)} + t_{sn(j),+}^{h(j)}
                min_time = pullin_en_i + depot_dwell_i + pullout_sn_j
                if wait >= min_time:
                    cost = float(pullin_en_i + pullout_sn_j)
                    arc_key = ((i, 'end'), (j, 'start'))
                    arcs[arc_key] = (cost, 1)

    # 6. Return arc (O^+, O^-): cost = M (fleet size proxy), capacity = K
    arcs[(O_plus, O_minus)] = (float(M), cap_K)

    all_nodes = set()
    for (u, v) in arcs:
        all_nodes.add(u)
        all_nodes.add(v)

    return O_minus, O_plus, all_nodes, arcs


# ============================================================
# MAIN SOLVER
# ============================================================

def solve_ittvs(instance, time_limit):
    """
    Solve the ITTVS MILP (1)-(6) using Gurobi.

    Decision variables:
      y_{d,u,v} in {0,1}: TT arc selection (constraint 3)
      x_{u,v} >= 0 (continuous): VS arc flow (automatically integer at optimum
                                  due to total unimodularity, as noted in paper)

    Returns solution dict with objective_value and other info.
    """
    trips = instance['trips']
    directions = instance['directions']
    trips_by_dir = {d: [t for t in trips if t['direction'] == d] for d in directions}
    alpha = instance.get('alpha', 1.0)

    # ---- Build graphs ----
    tt_data = {}
    for d in directions:
        src, snk, nodes, arcs, bstar = build_tt_graph(d, trips_by_dir[d], instance)
        tt_data[d] = {
            'source': src, 'sink': snk,
            'nodes': nodes, 'arcs': arcs, 'backward_star': bstar
        }

    vs_ominus, vs_oplus, vs_nodes, vs_arcs = build_vs_graph_pure(trips, instance)

    # ---- Create Gurobi model ----
    model = gp.Model("ITTVS")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPFocus", 1)    # CPXPARAM_Emphasis_MIP=1 analogue (from paper Section 7.2)
    model.setParam("RINS", 0)        # CPXPARAM_MIP_Strategy_LBHeur=1 analogue

    # ---- Decision variables ----

    # y_{d,u,v} in {0,1} for TT arcs (constraint 3)
    y = {}
    for d in directions:
        for (u, v) in tt_data[d]['arcs']:
            y[(d, u, v)] = model.addVar(
                vtype=GRB.BINARY, name=f"y_{d}_{u}_{v}"
            )

    # x_{u,v} >= 0 continuous for VS arcs (constraint 5)
    # Note: x need not be declared integer due to total unimodularity (paper Note 1)
    x = {}
    for (u, v), (cost, cap) in vs_arcs.items():
        x[(u, v)] = model.addVar(
            lb=0.0, ub=float(cap), vtype=GRB.CONTINUOUS, name=f"x_{u}_{v}"
        )

    model.update()

    # ---- Objective (1): min alpha*c*x + sum_d c^d*y^d ----
    obj = gp.LinExpr()
    for (u, v), (cost, cap) in vs_arcs.items():
        obj += alpha * cost * x[(u, v)]
    for d in directions:
        for (u, v), cost in tt_data[d]['arcs'].items():
            obj += cost * y[(d, u, v)]
    model.setObjective(obj, GRB.MINIMIZE)

    # ---- Constraint (2): TT flow conservation ----
    # sum_{(m,n) in A_d^TT} y_{m,n}^d - sum_{(n,m) in A_d^TT} y_{n,m}^d = b_n^d
    # b_{source} = -1, b_{sink} = +1, b_{trip} = 0
    for d in directions:
        src = tt_data[d]['source']
        snk = tt_data[d]['sink']
        arcs_d = tt_data[d]['arcs']
        nodes_d = tt_data[d]['nodes']

        # Pre-build adjacency lists for efficiency
        tt_in_arcs = {}   # node -> list of (u, v)
        tt_out_arcs = {}  # node -> list of (u, v)
        for node in nodes_d:
            tt_in_arcs[node] = []
            tt_out_arcs[node] = []
        for (u, v) in arcs_d:
            if v in tt_in_arcs:
                tt_in_arcs[v].append((u, v))
            if u in tt_out_arcs:
                tt_out_arcs[u].append((u, v))

        for node in nodes_d:
            b = -1 if node == src else (1 if node == snk else 0)
            in_flow = gp.quicksum(y[(d, u, v)] for (u, v) in tt_in_arcs[node])
            out_flow = gp.quicksum(y[(d, u, v)] for (u, v) in tt_out_arcs[node])
            model.addConstr(in_flow - out_flow == b, name=f"tt_flow_{d}_{node}")

    # ---- Constraint (4): VS flow conservation (circulation) ----
    # sum_{(m,n) in A^VS} x_{m,n} - sum_{(n,m) in A^VS} x_{n,m} = 0 for all n
    # Pre-build adjacency lists for VS graph
    vs_in_arcs = {node: [] for node in vs_nodes}
    vs_out_arcs = {node: [] for node in vs_nodes}
    for (u, v) in vs_arcs:
        if v in vs_in_arcs:
            vs_in_arcs[v].append((u, v))
        if u in vs_out_arcs:
            vs_out_arcs[u].append((u, v))

    for node in vs_nodes:
        in_flow = gp.quicksum(x[(u, v)] for (u, v) in vs_in_arcs[node])
        out_flow = gp.quicksum(x[(u, v)] for (u, v) in vs_out_arcs[node])
        model.addConstr(in_flow - out_flow == 0, name=f"vs_flow_{node}")

    # ---- Constraint (6): Linking constraints ----
    # sum_{(n,m) in B(i)} y_{n,m}^{d(i)} = x_{i^-, i^+}  for all i in T
    for trip in trips:
        i = trip['id']
        d = trip['direction']
        bstar = tt_data[d]['backward_star'].get(i, [])

        tt_in = gp.LinExpr()
        for (u, v) in bstar:
            tt_in += y[(d, u, v)]

        vs_trip_key = ((i, 'start'), (i, 'end'))
        if vs_trip_key in x:
            model.addConstr(tt_in == x[vs_trip_key], name=f"link_{i}")

    # ---- Solve ----
    model.optimize()

    # ---- Extract solution ----
    solution = {}

    if model.SolCount > 0:
        solution['objective_value'] = float(model.ObjVal)
        solution['status'] = 'optimal' if model.Status == GRB.OPTIMAL else 'feasible_time_limit'
        solution['mip_gap'] = float(model.MIPGap) if model.SolCount > 0 else None

        # Selected trips (those whose trip arc carries flow)
        selected_trips = []
        for trip in trips:
            key = ((trip['id'], 'start'), (trip['id'], 'end'))
            if key in x and x[key].X > 0.5:
                selected_trips.append(trip['id'])
        solution['selected_trips'] = selected_trips

        # Number of vehicles (flow on return arc)
        ret_key = (vs_oplus, vs_ominus)
        if ret_key in x:
            solution['num_vehicles'] = int(round(x[ret_key].X))

        # TT: arcs used per direction (for timetable reconstruction)
        tt_arcs_used = {}
        for d in directions:
            tt_arcs_used[d] = []
            for (u, v) in tt_data[d]['arcs']:
                if y[(d, u, v)].X > 0.5:
                    tt_arcs_used[d].append([str(u), str(v)])
        solution['tt_arcs_used'] = tt_arcs_used

        # VS: arc flows (for vehicle schedule reconstruction)
        vs_flows = {}
        for (u, v), (cost, cap) in vs_arcs.items():
            val = x[(u, v)].X
            if val > 1e-4:
                vs_flows[f"{u}-->{v}"] = round(val, 4)
        solution['vs_flows'] = vs_flows

    else:
        solution['objective_value'] = float('inf')
        solution['status'] = 'no_feasible_solution'
        solution['selected_trips'] = []
        solution['num_vehicles'] = None

    return solution


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gurobi MILP solver for ITTVS (Carosi et al. 2019)'
    )
    parser.add_argument('--instance_path', required=True,
                        help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True,
                        help='Path where solution JSON will be written')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help='Maximum solver runtime in seconds')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    solution = solve_ittvs(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2, default=str)

    print(f"Solution written to: {args.solution_path}")
    print(f"Status: {solution.get('status')}")
    print(f"Objective value: {solution.get('objective_value')}")
    print(f"Selected trips: {solution.get('selected_trips')}")
    print(f"Num vehicles: {solution.get('num_vehicles')}")


if __name__ == '__main__':
    main()
