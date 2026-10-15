#!/usr/bin/env python3
"""
Gurobi implementation of the General Arc-flow Formulation with Graph Compression
for Bin Packing and Related Problems.

Based on: Brandao and Pedroso (2016) - "Bin Packing and Related Problems:
          General Arc-flow Formulation with Graph Compression"

This implements the arc-flow MIP formulation (Equations 17-22) over a compressed
graph built via Algorithm 1 (direct Step-3 construction + Step-4 compression).
"""

import argparse
import json
import sys
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
    """Load a BPP instance from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    items = []
    for item in data["items"]:
        items.append({
            "type": item["item_type"],
            "weight": item["weight"],
            "demand": item["demand"],
        })
    W = data["parameters"]["W"]
    return items, W, data


def preprocess_items(items, W):
    """
    Sort items in decreasing order by normalized weight (alpha_i = w_i / W).
    Ties broken by decreasing weight (lexicographic in 1D).
    Returns sorted items with original type mapping.
    """
    # For 1D: alpha_i = w_i / W
    items_sorted = sorted(items, key=lambda it: (-it["weight"] / W, -it["weight"]))
    return items_sorted


def build_arc_flow_graph(items, W):
    """
    Build the compressed arc-flow graph using Algorithm 1 (direct Step-3 construction).

    For 1D bin packing:
    - States: (x, i, c) where x = space used, i = current item index, c = copies used
    - Lifting: for each state, compute the highest position reachable
    - Memoization on (lifted_x, i, c)

    Returns: (arcs, vertices, source_label, target_label)
    where arcs is a list of (u, v, item_index) with item_index 0-based (0 = loss arc uses index -1 here,
    we use item_index = -1 for loss arcs to distinguish from item 0).
    """
    m = len(items)
    weights = [it["weight"] for it in items]
    demands = [it["demand"] for it in items]

    arcs = []  # list of (u, v, item_idx) where item_idx is 0-based, -1 for loss
    vertices = set()

    # Memoization table: (x, i, c) -> node_label
    dp = {}

    def highest_position(x, i, c):
        """
        Compute the highest position (Step-3 lifting) for 1D case.
        Solve: minimize W - sum_{j>=i} w_j * y_j
               s.t. sum_{j>=i} w_j * y_j <= W - x
                    y_j <= b_j for j > i (or b_i - c for j == i)
                    y_j >= 0, integer

        This is equivalent to: W - maxFill, where maxFill is the max weight
        we can pack from items i..m-1 into remaining capacity W - x.

        We solve this greedily / via a simple bounded knapsack approach.

        **INFERRED ASSUMPTION**: The paper says "solving one-dimensional knapsack
        problems" for lifting. We use a DP-based bounded knapsack solver for exactness.
        """
        remaining = W - x
        if remaining <= 0:
            return W

        # Simple DP for bounded knapsack to maximize total weight
        # Items from index i to m-1, with adjusted bounds
        sub_items = []
        for j in range(i, m):
            bound = demands[j] - c if j == i else demands[j]
            if bound > 0 and weights[j] <= remaining:
                sub_items.append((weights[j], bound))

        if not sub_items:
            return W - 0  # can't pack anything more => highest pos = W

        # DP: max weight packable into capacity 'remaining'.
        # Always use the exact bounded-knapsack DP (no greedy fallback).
        dp_knap = [0] * (remaining + 1)
        for (w, b) in sub_items:
            if b == 1:
                # 0-1 item: iterate backwards
                for cap in range(remaining, w - 1, -1):
                    dp_knap[cap] = max(dp_knap[cap], dp_knap[cap - w] + w)
            else:
                # Bounded item: binary decomposition
                k = 1
                left = b
                while left > 0:
                    take = min(k, left)
                    tw = take * w
                    for cap in range(remaining, tw - 1, -1):
                        dp_knap[cap] = max(dp_knap[cap], dp_knap[cap - tw] + tw)
                    left -= take
                    k *= 2
        max_fill = dp_knap[remaining]

        return W - max_fill

    def build(x, i, c):
        """
        Recursive graph construction (Algorithm 1).
        Returns the node label for state (x, i, c).
        """
        # Step 1: Lift state
        x = highest_position(x, i, c)

        key = (x, i, c)
        if key in dp:
            return dp[key]

        # Initialize node label to W (target)
        u = W

        up_x = None
        # Option 1: Skip current item (move to next item type)
        if i < m - 1:
            up_x = build(x, i + 1, 0)
            u = up_x

        # Option 2: Use one more copy of item i
        if c < demands[i] and x + weights[i] <= W:
            v = build(x + weights[i], i, c + 1)
            u = min(u, v - weights[i])
            # Add item arc
            arcs.append((u, v, i))
            vertices.add(u)
            vertices.add(v)
            # Add loss arc between levels if needed
            if i < m - 1 and up_x is not None and u != up_x:
                arcs.append((u, up_x, -1))  # loss arc
                vertices.add(up_x)

        dp[key] = u
        return u

    # Build graph starting from initial state
    sys.setrecursionlimit(100000)
    source = build(0, 0, 0)
    target = W

    vertices.add(source)
    vertices.add(target)

    # Add loss arcs from all internal nodes to target
    for v in list(vertices):
        if v != source and v != target:
            arcs.append((v, target, -1))

    return arcs, vertices, source, target


def apply_step4_compression(arcs, vertices, source, target, items):
    """
    Step-4 compression: relabel nodes using longest path from source.

    psi(v) = 0 if v = source
    psi(v) = max over incoming arcs (u,v,i) of {psi(u) + w_i} otherwise
    where w_i = 0 for loss arcs.
    """
    weights = [it["weight"] for it in items]

    # Build adjacency list for incoming edges
    incoming = defaultdict(list)  # node -> list of (from_node, item_idx)
    all_nodes = set()
    for (u, v, idx) in arcs:
        incoming[v].append((u, idx))
        all_nodes.add(u)
        all_nodes.add(v)

    # Build adjacency list for outgoing edges (for topological sort)
    outgoing = defaultdict(list)
    in_degree = defaultdict(int)
    for node in all_nodes:
        in_degree[node] = 0
    for (u, v, idx) in arcs:
        outgoing[u].append(v)
        in_degree[v] += 1

    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque()
    for node in all_nodes:
        if in_degree[node] == 0:
            queue.append(node)

    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in outgoing[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Compute psi (longest path from source) via topological order.
    psi = {}
    psi[source] = 0
    for node in topo_order:
        if node not in psi:
            psi[node] = 0
        for (u, idx) in incoming[node]:
            if u in psi:
                w = 0 if idx == -1 else weights[idx]
                psi[node] = max(psi[node], psi[u] + w)

    # Relabel: merge nodes with same psi value
    new_arcs = set()
    for (u, v, idx) in arcs:
        nu = psi.get(u, u)
        nv = psi.get(v, v)
        if nu != nv:  # skip self-loops
            new_arcs.add((nu, nv, idx))

    new_source = psi.get(source, 0)
    new_target = psi.get(target, target)
    new_vertices = set()
    for (u, v, idx) in new_arcs:
        new_vertices.add(u)
        new_vertices.add(v)
    new_vertices.add(new_source)
    new_vertices.add(new_target)

    return list(new_arcs), new_vertices, new_source, new_target


def solve_with_gurobi(arcs, vertices, source, target, items, W, time_limit):
    """
    Build and solve the arc-flow MIP formulation (Equations 17-22).
    """
    m_items = len(items)
    demands = [it["demand"] for it in items]

    # J = {i : b_i = 1} (0-indexed)
    J = set(i for i in range(m_items) if demands[i] == 1)

    model = gp.Model("ArcFlowBPP")

    # Solver parameters as specified in the paper (Section 6)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)
    model.setParam("Presolve", 1)
    model.setParam("Method", 2)
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 1.0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("MIPGapAbs", 1 - 1e-5)

    # Decision variables
    # z: number of bins
    z = model.addVar(vtype=GRB.INTEGER, name="z", lb=0)

    # f_{uvi}: flow on arc (u, v, i)
    f = {}
    for idx, (u, v, item_idx) in enumerate(arcs):
        ub = GRB.INFINITY
        if item_idx >= 0:
            ub = demands[item_idx]
        f[idx] = model.addVar(
            vtype=GRB.INTEGER, name=f"f_{idx}", lb=0, ub=ub
        )

    model.update()

    # Objective (17): minimize z
    model.setObjective(z, GRB.MINIMIZE)

    # Constraint (18): Flow conservation
    # For each node k: sum(inflow) - sum(outflow) = -z (source), z (target), 0 (other)
    node_in = defaultdict(list)   # node -> list of arc indices (incoming)
    node_out = defaultdict(list)  # node -> list of arc indices (outgoing)
    for idx, (u, v, item_idx) in enumerate(arcs):
        node_out[u].append(idx)
        node_in[v].append(idx)

    for node in vertices:
        inflow = gp.quicksum(f[idx] for idx in node_in.get(node, []))
        outflow = gp.quicksum(f[idx] for idx in node_out.get(node, []))
        if node == source:
            model.addConstr(inflow - outflow == -z, name=f"flow_source")
        elif node == target:
            model.addConstr(inflow - outflow == z, name=f"flow_target")
        else:
            model.addConstr(inflow - outflow == 0, name=f"flow_{node}")

    # Constraints (19) and (20): Demand satisfaction
    # Group arcs by item type
    item_arcs = defaultdict(list)  # item_idx -> list of arc indices
    for idx, (u, v, item_idx) in enumerate(arcs):
        if item_idx >= 0:
            item_arcs[item_idx].append(idx)

    for i in range(m_items):
        total_flow = gp.quicksum(f[idx] for idx in item_arcs.get(i, []))
        if i in J:
            # Constraint (20): equality
            model.addConstr(total_flow == demands[i], name=f"demand_eq_{i}")
        else:
            # Constraint (19): inequality (>=)
            model.addConstr(total_flow >= demands[i], name=f"demand_geq_{i}")

    # Solve
    model.optimize()

    # Extract solution
    obj_val = None
    bin_assignments = []

    if model.SolCount > 0:
        obj_val = model.ObjVal

        # Flow decomposition to recover bin assignments
        flow_values = {}
        for idx, (u, v, item_idx) in enumerate(arcs):
            val = round(f[idx].X)
            if val > 0:
                flow_values[idx] = val

        # Decompose flow into paths from source to target
        # Build residual adjacency
        residual_out = defaultdict(list)
        for idx in flow_values:
            u, v, item_idx = arcs[idx]
            residual_out[u].append((v, item_idx, idx))

        total_bins = round(z.X)
        for _ in range(total_bins):
            path_items = []
            current = source
            while current != target:
                found = False
                for (v, item_idx, arc_idx) in residual_out[current]:
                    if flow_values.get(arc_idx, 0) > 0:
                        flow_values[arc_idx] -= 1
                        if flow_values[arc_idx] == 0:
                            del flow_values[arc_idx]
                        if item_idx >= 0:
                            path_items.append(items[item_idx]["type"])
                        current = v
                        found = True
                        break
                if not found:
                    break
            # Append every path so len(bin_assignments) == round(z.X);
            # paths consisting only of loss arcs yield empty bins.
            bin_assignments.append(path_items)

    return obj_val, bin_assignments, model.Status


def main():
    parser = argparse.ArgumentParser(
        description="Arc-flow formulation for Bin Packing (Gurobi solver)"
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

    # Load instance
    items, W, instance_data = load_instance(args.instance_path)

    # Preprocess: sort items
    items_sorted = preprocess_items(items, W)

    # Build compressed arc-flow graph
    arcs, vertices, source, target = build_arc_flow_graph(items_sorted, W)

    # Apply Step-4 compression
    arcs, vertices, source, target = apply_step4_compression(
        arcs, vertices, source, target, items_sorted
    )

    # Solve MIP
    obj_val, bin_assignments, status = solve_with_gurobi(
        arcs, vertices, source, target, items_sorted, W, args.time_limit
    )

    # Build solution output
    solution = {
        "objective_value": obj_val,
        "status": "optimal" if status == GRB.OPTIMAL else
                  "time_limit" if status == GRB.TIME_LIMIT else
                  "feasible" if obj_val is not None else "infeasible",
        "num_bins": int(round(obj_val)) if obj_val is not None else None,
        "bins": bin_assignments,
    }

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if obj_val is not None:
        print(f"Objective value (bins used): {obj_val}")


if __name__ == "__main__":
    main()
