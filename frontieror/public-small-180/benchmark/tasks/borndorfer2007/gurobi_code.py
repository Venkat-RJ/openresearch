#!/usr/bin/env python3
"""
Gurobi MIP implementation for the Line Planning Problem (LPP).

Paper: "A Column-Generation Approach to Line Planning in Public Transport"
Authors: Borndörfer, Grötschel, Pfetsch (2007), Transportation Science 41(1), pp. 123-132.

Model (LPP):
  min  lambda * (C^T x + c^T f) + (1 - lambda) * tau^T y

  subject to:
    y(P_st) = d_st                          for all (s,t) in D        (1)
    y(P_a) - sum_{l: e(a) in l} kappa_l f_l <= 0   for all a in A    (2)
    f(L_e) <= Lambda_e                       for all e in E           (3)
    f <= F * x                                                        (4)
    x_l in {0,1}                             for all l in L           (5)
    f_l >= 0                                 for all l in L           (6)
    y_p >= 0                                 for all p in P           (7)

Since the sets L (lines) and P (passenger paths) are exponentially large,
we enumerate lines with a time/count budget and use k-shortest paths for passengers.

INFERRED ASSUMPTION: The paper uses column generation to handle the exponential
number of variables. For the direct MIP, we enumerate a bounded subset of lines
(capped at MAX_LINES to keep the model tractable) and k-shortest passenger paths.
This is necessary because full enumeration is intractable for dense networks.
**NOT SPECIFIED IN PAPER** for a direct MIP approach.
"""

import argparse
import json
import sys
import time
import heapq
from collections import defaultdict
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


try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy is required. Install Gurobi and its Python interface.")
    sys.exit(1)

# Maximum number of lines to enumerate before stopping
MAX_LINES = 50000
# Maximum time (seconds) for line enumeration
MAX_ENUM_TIME = 30


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_directed_arcs(edges_data):
    """
    Build directed arcs from undirected edges.
    Each edge e = (u,v) produces two arcs:
      arc a(e): u -> v  (arc_id = 2*e_id)
      arc a_bar(e): v -> u  (arc_id = 2*e_id + 1)
    """
    arcs = []
    arc_adj = defaultdict(list)
    for e in edges_data:
        eid = e["id"]
        u, v = e["endpoints"]
        tt = e["traveling_time_seconds"]
        fwd_id = 2 * eid
        arcs.append({"id": fwd_id, "from": u, "to": v, "edge_id": eid, "travel_time": tt})
        arc_adj[u].append((v, fwd_id))
        bwd_id = 2 * eid + 1
        arcs.append({"id": bwd_id, "from": v, "to": u, "edge_id": eid, "travel_time": tt})
        arc_adj[v].append((u, bwd_id))
    return arcs, arc_adj


def enumerate_lines_bounded(instance, edges_data, max_length, max_lines=MAX_LINES,
                            max_time=MAX_ENUM_TIME):
    """
    Enumerate feasible lines for each mode with time and count bounds.
    A line of mode i is a simple path in G_i between two terminals of T_i,
    with at most max_length edges.

    Prioritizes shorter lines first (BFS-like by depth).
    """
    lines = []
    modes = instance["modes"]
    start_time = time.time()

    for mode_idx, mode in enumerate(modes):
        terminals = set(mode["terminals"])
        mode_adj = defaultdict(list)
        for eid in mode["edge_indices"]:
            e = edges_data[eid]
            u, v = e["endpoints"]
            mode_adj[u].append((v, eid))
            mode_adj[v].append((u, eid))

        terminal_list = sorted(terminals)

        # Enumerate using iterative deepening to prioritize shorter lines
        for max_depth in range(1, max_length + 1):
            if len(lines) >= max_lines or (time.time() - start_time) > max_time:
                break
            for s in terminal_list:
                if len(lines) >= max_lines or (time.time() - start_time) > max_time:
                    break
                # DFS with depth exactly up to max_depth (but only add lines at this depth
                # that weren't found at shorter depths)
                stack = [(s, frozenset([s]), [], [s])]
                while stack:
                    if len(lines) >= max_lines or (time.time() - start_time) > max_time:
                        break
                    curr, visited, edge_path, node_path = stack.pop()
                    depth = len(edge_path)

                    # Record line if at a different terminal and at exactly this depth level
                    if curr in terminals and curr != s and curr > s and depth == max_depth:
                        lines.append({
                            "mode_idx": mode_idx,
                            "edges": list(edge_path),
                            "nodes": list(node_path),
                        })

                    if depth < max_depth:
                        for neighbor, eid in mode_adj[curr]:
                            if neighbor not in visited:
                                stack.append((
                                    neighbor,
                                    visited | frozenset([neighbor]),
                                    edge_path + [eid],
                                    node_path + [neighbor],
                                ))

    # Deduplicate lines (same set of edges)
    seen = set()
    unique_lines = []
    for line in lines:
        key = (line["mode_idx"], tuple(sorted(line["edges"])))
        if key not in seen:
            seen.add(key)
            unique_lines.append(line)

    # Post-enumeration edge coverage: ensure every edge has ≥1 line.
    # The bounded enumeration may truncate before generating a line through
    # some edges; the capacity constraint sum y - kappa*f ≤ 0 then has no f
    # variable, forcing all paths through that edge to 0 → presolve INFEAS.
    covered = set()
    for line in unique_lines:
        covered.update(line["edges"])
    uncovered_edges = [e for e in edges_data if e["id"] not in covered]
    pre_count = len(unique_lines)
    from collections import deque
    for e in uncovered_edges:
        eid = e["id"]
        u, v = e["endpoints"]
        for mode_idx, mode in enumerate(modes):
            if eid not in mode["edge_indices"]:
                continue
            terminals_set = set(mode["terminals"])
            mode_adj = defaultdict(list)
            for meid in mode["edge_indices"]:
                me = edges_data[meid]
                mu, mv = me["endpoints"]
                mode_adj[mu].append((mv, meid))
                mode_adj[mv].append((mu, meid))

            def bfs_terminal(start, exclude=None):
                q = deque([(start, [start], [])])
                vis = {start}
                while q:
                    n, npath, epath = q.popleft()
                    if n in terminals_set and n != exclude:
                        return npath, epath
                    for nb, meid in mode_adj[n]:
                        if nb not in vis:
                            vis.add(nb)
                            q.append((nb, npath + [nb], epath + [meid]))
                return None, None

            up, ue = bfs_terminal(u)
            if up is None:
                continue
            s = up[-1]
            vp, ve = bfs_terminal(v, exclude=s)
            if vp is None:
                continue
            line_edges = list(reversed(ue)) + [eid] + ve
            line_nodes = list(reversed(up)) + vp
            if len(line_edges) > max_length:
                continue
            key = (mode_idx, tuple(sorted(line_edges)))
            if key in seen:
                continue
            seen.add(key)
            unique_lines.append({
                "mode_idx": mode_idx,
                "edges": line_edges,
                "nodes": line_nodes,
            })
            break  # one line per edge is enough
    if uncovered_edges:
        added = len(unique_lines) - pre_count
        print(f"  Edge coverage: added {added} fallback lines for {len(uncovered_edges)} uncovered edges (total: {len(unique_lines)})")

    return unique_lines


def dijkstra(source, target, arc_adj, arcs, num_nodes, excluded_arcs=None):
    """Dijkstra's shortest path. Returns (dist, arc_path) or None."""
    if excluded_arcs is None:
        excluded_arcs = set()
    dist = [float('inf')] * num_nodes
    prev = [None] * num_nodes
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == target:
            path_arcs = []
            node = target
            while prev[node] is not None:
                pn, aid = prev[node]
                path_arcs.append(aid)
                node = pn
            path_arcs.reverse()
            return dist[target], path_arcs
        for v, aid in arc_adj[u]:
            if aid in excluded_arcs:
                continue
            nd = d + arcs[aid]["travel_time"]
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, aid)
                heapq.heappush(pq, (nd, v))
    return None


def k_shortest_paths(source, target, arc_adj, arcs, num_nodes, k):
    """Yen's k-shortest simple paths algorithm."""
    result = dijkstra(source, target, arc_adj, arcs, num_nodes)
    if result is None:
        return []
    _, first_path = result
    A = [first_path]
    B = []

    for ki in range(1, k):
        if not A:
            break
        last_path = A[-1]
        last_nodes = [source]
        for aid in last_path:
            last_nodes.append(arcs[aid]["to"])

        for i in range(len(last_path)):
            spur_node = last_nodes[i]
            root_path = last_path[:i]

            excluded = set()
            for p in A:
                if len(p) >= i and p[:i] == root_path and i < len(p):
                    excluded.add(p[i])

            root_nodes = set(last_nodes[:i])
            excluded_arcs = set(excluded)
            for aid_check in range(len(arcs)):
                if arcs[aid_check]["from"] in root_nodes:
                    excluded_arcs.add(aid_check)

            spur_result = dijkstra(spur_node, target, arc_adj, arcs, num_nodes, excluded_arcs)
            if spur_result is not None:
                _, spur_path = spur_result
                total_path = root_path + spur_path
                total_cost = sum(arcs[aid]["travel_time"] for aid in total_path)
                path_nodes = [source]
                for aid in total_path:
                    path_nodes.append(arcs[aid]["to"])
                if len(path_nodes) == len(set(path_nodes)):
                    heapq.heappush(B, (total_cost, total_path))

        if not B:
            break
        while B:
            cost, candidate = heapq.heappop(B)
            if candidate not in A:
                A.append(candidate)
                break
        else:
            break
    return A


def main():
    parser = argparse.ArgumentParser(description="Gurobi MIP solver for Line Planning Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path for output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Solver time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    total_start = time.time()
    print("Loading instance...")
    instance = load_instance(args.instance_path)

    lam = instance["global_parameters"]["lambda"]
    max_line_length = instance["global_parameters"]["max_line_length_edges"]
    F = instance["global_parameters"]["frequency_upper_bound_F"]
    modes = instance["modes"]
    edges_data = instance["network"]["edges"]
    od_pairs = instance["od_matrix"]
    num_nodes = instance["network"]["num_nodes"]

    arcs, arc_adj = build_directed_arcs(edges_data)
    num_arcs = len(arcs)

    # Step 1: Enumerate lines (bounded)
    print("Enumerating lines (bounded)...")
    t0 = time.time()
    lines = enumerate_lines_bounded(instance, edges_data, max_line_length)
    print(f"  Enumerated {len(lines)} lines in {time.time()-t0:.1f}s")

    if len(lines) == 0:
        print("ERROR: No feasible lines found.")
        solution = {"objective_value": None, "status": "infeasible", "error": "No lines enumerated"}
        with open(args.solution_path, 'w') as f:
            solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(solution, f, indent=2)
        return

    # Step 2: Find passenger paths (k-shortest per OD pair)
    print("Finding passenger paths...")
    t0 = time.time()
    # INFERRED ASSUMPTION: k=10 shortest paths per OD pair. **NOT SPECIFIED IN PAPER**
    # for a direct MIP. The paper uses column generation for passenger paths.
    K_PATHS = 10
    paths_by_od = {}
    for od in od_pairs:
        s, t = od["origin"], od["destination"]
        paths_by_od[(s, t)] = k_shortest_paths(s, t, arc_adj, arcs, num_nodes, K_PATHS)
    total_paths = sum(len(v) for v in paths_by_od.values())
    print(f"  Found {total_paths} passenger paths in {time.time()-t0:.1f}s")

    # Step 3: Build Gurobi model
    print("Building Gurobi model...")

    # Calculate remaining time for solver
    elapsed = time.time() - total_start
    solver_time = max(10, args.time_limit - int(elapsed))

    model = gp.Model("LPP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", solver_time)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = model.addVars(len(lines), vtype=GRB.BINARY, name="x")
    f = model.addVars(len(lines), vtype=GRB.CONTINUOUS, lb=0.0, name="f")

    all_paths = []
    path_idx_by_od = {}
    for od in od_pairs:
        key = (od["origin"], od["destination"])
        path_idx_by_od[key] = []
        for path_arcs in paths_by_od.get(key, []):
            idx = len(all_paths)
            all_paths.append((key, path_arcs))
            path_idx_by_od[key].append(idx)
    y = model.addVars(len(all_paths), vtype=GRB.CONTINUOUS, lb=0.0, name="y")

    # Objective: min lambda * (C^T x + c^T f) + (1 - lambda) * tau^T y
    obj = gp.LinExpr()
    for l_idx, line in enumerate(lines):
        mode = modes[line["mode_idx"]]
        C_l = mode["fixed_cost_per_line"]
        c_l = sum(edges_data[eid]["operating_cost"] for eid in line["edges"])
        obj += lam * C_l * x[l_idx]
        obj += lam * c_l * f[l_idx]
    for p_idx, (od_key, path_arcs) in enumerate(all_paths):
        tau_p = sum(arcs[aid]["travel_time"] for aid in path_arcs)
        obj += (1 - lam) * tau_p * y[p_idx]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (1): Flow conservation
    for od in od_pairs:
        key = (od["origin"], od["destination"])
        demand = od["demand"]
        indices = path_idx_by_od.get(key, [])
        if not indices:
            print(f"  WARNING: No paths for OD pair {key}, demand={demand}")
            continue
        model.addConstr(
            gp.quicksum(y[i] for i in indices) == demand,
            name=f"flow_{key[0]}_{key[1]}"
        )

    # Constraint (2): Capacity
    arc_to_lines = defaultdict(list)
    for l_idx, line in enumerate(lines):
        kappa = modes[line["mode_idx"]]["vehicle_capacity"]
        for eid in line["edges"]:
            arc_to_lines[2 * eid].append((l_idx, kappa))
            arc_to_lines[2 * eid + 1].append((l_idx, kappa))

    arc_to_paths = defaultdict(list)
    for p_idx, (od_key, path_arcs) in enumerate(all_paths):
        for aid in path_arcs:
            arc_to_paths[aid].append(p_idx)

    for aid in set(arc_to_paths.keys()) | set(arc_to_lines.keys()):
        path_ids = arc_to_paths.get(aid, [])
        line_entries = arc_to_lines.get(aid, [])
        if not path_ids and not line_entries:
            continue
        lhs = gp.LinExpr()
        for p_idx in path_ids:
            lhs += y[p_idx]
        for l_idx, kappa in line_entries:
            lhs -= kappa * f[l_idx]
        model.addConstr(lhs <= 0, name=f"cap_a{aid}")

    # Constraint (3): Frequency bounds
    edge_to_lines = defaultdict(list)
    for l_idx, line in enumerate(lines):
        for eid in line["edges"]:
            edge_to_lines[eid].append(l_idx)
    for e in edges_data:
        eid = e["id"]
        Lambda_e = e["edge_capacity"]
        line_ids = edge_to_lines.get(eid, [])
        if line_ids:
            model.addConstr(
                gp.quicksum(f[l_idx] for l_idx in line_ids) <= Lambda_e,
                name=f"freq_e{eid}"
            )

    # Constraint (4): Linking f <= F*x
    for l_idx in range(len(lines)):
        model.addConstr(f[l_idx] <= F * x[l_idx], name=f"link_{l_idx}")

    print(f"Model: {model.NumVars} vars, {model.NumConstrs} constrs")
    print(f"Solving with time limit {solver_time}s...")
    model.optimize()

    # Extract solution
    solution = {}
    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution["objective_value"] = obj_val
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "time_limit"
        solution["mip_gap"] = model.MIPGap if hasattr(model, 'MIPGap') else None
        solution["solve_time_seconds"] = model.Runtime

        active_lines = []
        total_fixed_cost = 0.0
        total_operating_cost = 0.0
        for l_idx, line in enumerate(lines):
            if x[l_idx].X > 0.5:
                mode = modes[line["mode_idx"]]
                active_lines.append({
                    "line_index": l_idx,
                    "mode": mode["name"],
                    "nodes": line["nodes"],
                    "edges": line["edges"],
                    "frequency": f[l_idx].X,
                })
                total_fixed_cost += mode["fixed_cost_per_line"]
                total_operating_cost += (
                    sum(edges_data[eid]["operating_cost"] for eid in line["edges"])
                    * f[l_idx].X
                )
        solution["active_lines"] = active_lines
        solution["num_active_lines"] = len(active_lines)

        total_travel_time = 0.0
        num_active_paths = 0
        active_paths = []
        # Verification aggregates: per-OD served demand and per-arc total flow.
        # These are formulation-agnostic stats (they don't expose the path-based
        # decision variables), so any alternative solver can produce them by
        # aggregating its own flows once at the end. The checker will prefer
        # these over `active_passenger_paths` when both are present.
        od_served_agg = defaultdict(float)
        arc_total_flow = defaultdict(float)
        for p_idx, (od_key, path_arcs) in enumerate(all_paths):
            if y[p_idx].X > 1e-6:
                tau_p = sum(arcs[aid]["travel_time"] for aid in path_arcs)
                total_travel_time += tau_p * y[p_idx].X
                num_active_paths += 1
                active_paths.append({
                    "origin": od_key[0],
                    "destination": od_key[1],
                    "arcs": list(path_arcs),
                    "flow": y[p_idx].X,
                })
                od_served_agg[(od_key[0], od_key[1])] += y[p_idx].X
                for aid in path_arcs:
                    arc_total_flow[aid] += y[p_idx].X

        solution["active_passenger_paths"] = active_paths
        solution["od_served"] = [
            {"origin": o, "destination": d, "flow": f}
            for (o, d), f in od_served_agg.items()
        ]
        solution["arc_total_flow"] = {str(aid): f for aid, f in arc_total_flow.items()}
        solution["total_travel_time"] = total_travel_time
        solution["scaled_travel_time"] = (1 - lam) * total_travel_time
        solution["total_fixed_cost"] = total_fixed_cost
        solution["total_operating_cost"] = total_operating_cost
        solution["scaled_line_cost"] = lam * (total_fixed_cost + total_operating_cost)
        solution["num_active_passenger_paths"] = num_active_paths

        print(f"\n=== SOLUTION ===")
        print(f"Objective value: {obj_val:.2f}")
        print(f"Active lines: {len(active_lines)}")
        print(f"Active passenger paths: {num_active_paths}")
        print(f"Total travel time: {total_travel_time:.2f}")
        print(f"Scaled travel time: {(1-lam)*total_travel_time:.2f}")
        print(f"Total fixed cost: {total_fixed_cost:.2f}")
        print(f"Total operating cost: {total_operating_cost:.2f}")
        print(f"Scaled line cost: {lam*(total_fixed_cost+total_operating_cost):.2f}")
    else:
        solution["objective_value"] = None
        solution["status"] = "infeasible" if model.Status == GRB.INFEASIBLE else "no_solution"
        solution["solve_time_seconds"] = model.Runtime
        print("No feasible solution found.")

    with open(args.solution_path, 'w') as outf:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, outf, indent=2)
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
