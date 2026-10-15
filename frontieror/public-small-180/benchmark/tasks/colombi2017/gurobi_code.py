#!/usr/bin/env python3
"""
Gurobi implementation of the DPRPP-IC (Directed Profitable Rural Postman Problem
with Incompatibility Constraints) using Formulation (A) from Colombi et al. (2017).

Branch-and-cut with dynamically separated connectivity constraints (3).
Variables z and u are relaxed to continuous [0,1] per Proposition 1.

The paper uses CPLEX 12.6.2; here we use Gurobi as the solver.
"""

import json
import argparse
import math
from collections import defaultdict

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
    with open(path) as f:
        return json.load(f)


def build_adjacency(nodes, arcs):
    """Build outgoing and incoming adjacency lists."""
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for (i, j) in arcs:
        outgoing[i].append((i, j))
        incoming[j].append((i, j))
    return outgoing, incoming


def find_weakly_connected_components(nodes_set, arc_vals, arcs, threshold):
    """
    Find weakly connected components of the subgraph induced by arcs
    with value > threshold.
    """
    # Build undirected adjacency for arcs above threshold
    adj = defaultdict(set)
    active_nodes = set()
    for (i, j) in arcs:
        if arc_vals.get((i, j), 0) > threshold + 1e-9:
            adj[i].add(j)
            adj[j].add(i)
            active_nodes.add(i)
            active_nodes.add(j)

    visited = set()
    components = []
    for node in active_nodes:
        if node in visited:
            continue
        # BFS
        comp = set()
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in adj[n]:
                if nb not in visited:
                    queue.append(nb)
        if comp:
            components.append(comp)
    return components


def solve_dprpp_ic(instance, time_limit):
    """
    Solve the DPRPP-IC using Formulation (A) with Gurobi.
    Connectivity constraints (3) are separated dynamically via callbacks.
    """
    num_nodes = instance["num_nodes"]
    depot = instance["depot"]

    # --- Build arc data ---
    arc_cost = {}
    for arc in instance["arcs"]:
        i, j, c = arc[0], arc[1], arc[2]
        arc_cost[(i, j)] = c
    all_arcs = list(arc_cost.keys())

    # --- Build profitable arc data ---
    arc_profit = {}
    for pa in instance["profitable_arcs"]:
        i, j, p = pa[0], pa[1], pa[2]
        arc_profit[(i, j)] = p
    profitable_arcs = list(arc_profit.keys())

    # --- V_I nodes (initial nodes of profitable arcs) ---
    vi_nodes = set(instance.get("VI_nodes", []))
    if not vi_nodes:
        vi_nodes = set(i for (i, j) in profitable_arcs)

    # --- Incompatibility edges ---
    strong_incomp = []
    for edge in instance["strong_incompatibilities"]:
        strong_incomp.append((edge[0], edge[1]))

    weak_incomp = []
    weak_penalty = {}

    # Compute val(q) = sum_{(q,s) in R} (p_qs - c_qs) for penalty computation
    val = defaultdict(float)
    for (i, j), p in arc_profit.items():
        c = arc_cost.get((i, j), 0)
        val[i] += (p - c)

    gamma = instance.get("generation_parameters", {}).get("gamma", 0.01)

    for edge in instance["weak_incompatibilities"]:
        i, j = edge[0], edge[1]
        if len(edge) >= 3:
            c_bar = edge[2]
        else:
            # **INFERRED ASSUMPTION**: Penalty computed as ceil(gamma * (val(i) + val(j)))
            # as described in Section 6.1 of the paper.
            c_bar = math.ceil(gamma * (val[i] + val[j]))
        weak_incomp.append((i, j))
        weak_penalty[(i, j)] = c_bar

    # --- V_bar: nodes appearing in incompatibility edges ---
    v_bar = set()
    for (i, j) in strong_incomp:
        v_bar.add(i)
        v_bar.add(j)
    for (i, j) in weak_incomp:
        v_bar.add(i)
        v_bar.add(j)

    # Map profitable arcs leaving each node
    profitable_from = defaultdict(list)
    for (i, j) in profitable_arcs:
        profitable_from[i].append((i, j))

    # Adjacency
    nodes = set(range(num_nodes))
    outgoing, incoming = build_adjacency(nodes, all_arcs)

    # ========== Build Gurobi Model ==========
    model = gp.Model("DPRPP_IC_FormA")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LazyConstraints", 1)
    # Suppress output for cleaner runs
    model.setParam("OutputFlag", 1)

    # --- Decision Variables ---
    # x[i,j] >= 0 integer: number of times arc (i,j) is traversed
    x = {}
    for (i, j) in all_arcs:
        x[(i, j)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")

    # y[i,j] in {0,1}: 1 if profitable arc (i,j) is served
    y = {}
    for (i, j) in profitable_arcs:
        y[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    # z[i] in [0,1]: 1 if at least one profitable arc leaving i is served
    # Per Proposition 1, z can be relaxed to continuous [0,1]
    z = {}
    for i in v_bar:
        z[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"z_{i}")

    # u[i,j] in [0,1]: 1 if weak incompatibility penalty is paid
    # Per Proposition 1, u can be relaxed to continuous [0,1]
    u = {}
    for (i, j) in weak_incomp:
        u[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"u_{i}_{j}")

    model.update()

    # --- Objective Function ---
    # max sum p_ij * y_ij - sum c_ij * x_ij - sum c_bar_ij * u_ij
    obj = gp.LinExpr()
    for (i, j) in profitable_arcs:
        obj += arc_profit[(i, j)] * y[(i, j)]
    for (i, j) in all_arcs:
        obj -= arc_cost[(i, j)] * x[(i, j)]
    for (i, j) in weak_incomp:
        obj -= weak_penalty[(i, j)] * u[(i, j)]
    model.setObjective(obj, GRB.MAXIMIZE)

    # --- Constraint (1): x_ij >= y_ij for (i,j) in R ---
    for (i, j) in profitable_arcs:
        model.addConstr(x[(i, j)] >= y[(i, j)], name=f"serve_{i}_{j}")

    # --- Constraint (2): flow conservation at each node ---
    for j in nodes:
        out_expr = gp.LinExpr()
        in_expr = gp.LinExpr()
        for (jj, k) in outgoing[j]:
            out_expr += x[(jj, k)]
        for (k, jj) in incoming[j]:
            in_expr += x[(k, jj)]
        model.addConstr(out_expr == in_expr, name=f"flow_{j}")

    # --- Constraint (4): y_ij <= z_i for i in V_bar, (i,j) in R ---
    for i in v_bar:
        for (ii, j) in profitable_from.get(i, []):
            model.addConstr(y[(ii, j)] <= z[i], name=f"link_{ii}_{j}")

    # --- Constraint (5): z_i + z_j <= 1 for {i,j} in E_1 ---
    for (i, j) in strong_incomp:
        if i in z and j in z:
            model.addConstr(z[i] + z[j] <= 1, name=f"strong_{i}_{j}")

    # --- Constraint (6): z_i + z_j - u_ij <= 1 for {i,j} in E_2 ---
    for (i, j) in weak_incomp:
        if i in z and j in z:
            model.addConstr(z[i] + z[j] - u[(i, j)] <= 1, name=f"weak_{i}_{j}")

    # --- Constraint (3): connectivity - separated lazily ---
    # For S ⊆ V\{0}, (k,s) ∈ R(S): sum_{(i,j) ∈ δ+(S)} x_ij >= y_ks

    # Store references for callback closure
    cb_data = {
        "x": x, "y": y,
        "all_arcs": all_arcs, "profitable_arcs": profitable_arcs,
        "outgoing": outgoing, "depot": depot, "nodes": nodes,
    }

    def connectivity_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Separate connectivity constraints on integer solutions
            x_val = {}
            for (i, j) in cb_data["all_arcs"]:
                x_val[(i, j)] = model.cbGetSolution(cb_data["x"][(i, j)])
            y_val = {}
            for (i, j) in cb_data["profitable_arcs"]:
                y_val[(i, j)] = model.cbGetSolution(cb_data["y"][(i, j)])

            _separate_and_add(model, x_val, y_val, is_integer=True)

        elif where == GRB.Callback.MIPNODE:
            # Separate connectivity constraints on fractional solutions
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            x_val = {}
            for (i, j) in cb_data["all_arcs"]:
                x_val[(i, j)] = model.cbGetNodeRel(cb_data["x"][(i, j)])
            y_val = {}
            for (i, j) in cb_data["profitable_arcs"]:
                y_val[(i, j)] = model.cbGetNodeRel(cb_data["y"][(i, j)])

            _separate_and_add(model, x_val, y_val, is_integer=False)

    def _separate_and_add(model, x_val, y_val, is_integer):
        """
        Heuristic separation for connectivity constraints (3).
        For each epsilon in {0, 0.25, 0.5}, compute connected components
        of graph induced by arcs with x > epsilon. For components not
        containing the depot, check for violated constraints.
        """
        dep = cb_data["depot"]
        out = cb_data["outgoing"]
        pa = cb_data["profitable_arcs"]
        x_vars = cb_data["x"]
        y_vars = cb_data["y"]

        cuts_added = 0
        for eps in [0.0, 0.25, 0.5]:
            components = find_weakly_connected_components(
                cb_data["nodes"], x_val, cb_data["all_arcs"], eps
            )

            for comp in components:
                if dep in comp:
                    continue

                # Check each profitable arc (k,s) with both endpoints in comp
                for (k, s) in pa:
                    if k not in comp or s not in comp:
                        continue

                    y_ks_val = y_val[(k, s)]
                    tol = 0.5 if is_integer else 1e-4
                    if y_ks_val < tol:
                        continue

                    # Compute cut value: sum x_ij for arcs leaving comp
                    cut_val = 0.0
                    cut_expr = gp.LinExpr()
                    for node_in_S in comp:
                        for (ii, jj) in out[node_in_S]:
                            if jj not in comp:
                                cut_val += x_val[(ii, jj)]
                                cut_expr += x_vars[(ii, jj)]

                    if cut_val < y_ks_val - 1e-4:
                        if is_integer:
                            model.cbLazy(cut_expr >= y_vars[(k, s)])
                        else:
                            model.cbCut(cut_expr >= y_vars[(k, s)])
                        cuts_added += 1

    # ========== Solve ==========
    model.optimize(connectivity_callback)

    # ========== Extract Solution ==========
    result = {"objective_value": None}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        served_arcs = []
        for (i, j) in profitable_arcs:
            if y[(i, j)].X > 0.5:
                served_arcs.append({
                    "from": i, "to": j,
                    "profit": arc_profit[(i, j)],
                    "cost": arc_cost.get((i, j), 0)
                })
        result["served_arcs"] = served_arcs

        tour_arcs = []
        for (i, j) in all_arcs:
            count = int(round(x[(i, j)].X))
            if count > 0:
                tour_arcs.append({
                    "from": i, "to": j,
                    "count": count,
                    "cost": arc_cost[(i, j)]
                })
        result["tour_arcs"] = tour_arcs

        result["total_profit"] = sum(
            arc_profit[(i, j)] for (i, j) in profitable_arcs if y[(i, j)].X > 0.5
        )
        result["total_travel_cost"] = sum(
            arc_cost[(i, j)] * int(round(x[(i, j)].X))
            for (i, j) in all_arcs if x[(i, j)].X > 0.5
        )
        result["total_penalty"] = sum(
            weak_penalty[(i, j)] * u[(i, j)].X
            for (i, j) in weak_incomp
        ) if weak_incomp else 0.0

        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
    else:
        # No feasible solution found: null solution (stay at depot, no profit)
        result["objective_value"] = 0.0
        result["served_arcs"] = []
        result["tour_arcs"] = []
        result["total_profit"] = 0.0
        result["total_travel_cost"] = 0.0
        result["total_penalty"] = 0.0
        result["status"] = "no_solution"
        result["mip_gap"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="DPRPP-IC Solver using Gurobi - Formulation (A) (Colombi et al. 2017)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_dprpp_ic(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Objective value: {result['objective_value']}")
    print(f"Status: {result['status']}")
    if result.get("mip_gap") is not None:
        print(f"MIP gap: {result['mip_gap']:.4f}")


if __name__ == "__main__":
    main()
