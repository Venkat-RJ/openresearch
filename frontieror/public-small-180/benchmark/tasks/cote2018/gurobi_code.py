"""
Gurobi implementation of the Arc-Flow formulation for the Cutting Stock Problem (CSP)
from Côté & Iori (2018) "The Meet-in-the-Middle Principle for Cutting and Packing Problems"

This implements the arc-flow model (equations 11-14) using normal patterns (eq. 15)
for the arc set construction, as described in Section 4.
"""

import json
import argparse
import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
import time
_GUROBI_CODE_START_TIME = time.time()
# Walk up from this file's directory to find repo root (containing scripts/).
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
    with open(path, 'r') as f:
        data = json.load(f)
    W = data["bin_capacity"]
    items = []
    for it in data["items"]:
        items.append((it["width"], it["demand"]))
    return W, items


def compute_normal_patterns_csp(widths, demands, W):
    """
    Compute B'_i for each item type i using eq. (15).
    Items are sorted by nonincreasing width.
    For item type i, B'_i = { x = sum_{j=1}^{i} w_j * xi_j : 0 <= x <= W - w_i,
                               xi_j in {0,1,...,d_bar_j^i} }
    where d_bar_j^i = d_j for j < i, d_bar_i^i = d_i - 1.

    We use dynamic programming (Algorithm 1 adapted for CSP).
    """
    m = len(widths)
    # B_prime[i] = set of positions for item type i
    B_prime = [set() for _ in range(m)]

    # We compute all B'_i sets using DP.
    # For each item i, we need combinations of items 1..i with modified demands.
    # We do this incrementally: maintain a DP array that tracks reachable sums
    # using items 1..i with their full demands, then for item i we use d_i - 1.

    # Actually, let's compute B'_i for each i separately using bounded knapsack DP.
    # For efficiency, we do it incrementally.

    for i in range(m):
        w_i = widths[i]
        if w_i > W:
            continue
        cap = W - w_i

        # DP: can we reach sum x using items j=0..i with demands d_bar_j^i?
        # d_bar_j^i = demands[j] for j < i, demands[i] - 1 for j = i
        # We use a boolean array reachable[0..cap]
        reachable = [False] * (cap + 1)
        reachable[0] = True

        for j in range(i + 1):
            w_j = widths[j]
            if w_j > cap:
                continue
            d_bar = demands[j] if j < i else demands[i] - 1
            if d_bar <= 0:
                continue

            # Bounded knapsack: add up to d_bar copies of w_j
            # Use binary decomposition for efficiency
            remaining = d_bar
            k = 1
            while remaining > 0:
                take = min(k, remaining)
                weight = take * w_j
                # Traverse from cap down to weight
                for x in range(cap, weight - 1, -1):
                    if reachable[x - weight]:
                        reachable[x] = True
                remaining -= take
                k *= 2

        B_prime[i] = {x for x in range(cap + 1) if reachable[x]}

    return B_prime


def build_arc_flow_model(W, widths, demands, B_prime, time_limit):
    """
    Build the arc-flow MILP (eqs. 11-14) using normal patterns.
    """
    m = len(widths)
    w_min = min(widths)

    # Collect all vertices that appear as arc endpoints
    # Item arcs: (p, p + w_i) for each item type i and p in B'_i
    item_arcs = {}  # (p, q) -> list of item types
    arc_to_items = {}  # (p, q) -> set of item type indices

    for i in range(m):
        w_i = widths[i]
        for p in B_prime[i]:
            q = p + w_i
            if q > W:
                continue
            key = (p, q)
            if key not in arc_to_items:
                arc_to_items[key] = set()
            arc_to_items[key].add(i)

    # Items with w_i > W/2: force start position to 0 only
    for i in range(m):
        if widths[i] > W / 2:
            B_prime[i] = {0} if 0 in B_prime[i] else set()

    # Rebuild item arcs after forcing large items to position 0
    item_arcs_list = []  # list of (p, q, item_type)
    item_arc_set = set()  # set of (p, q) that are item arcs

    for i in range(m):
        w_i = widths[i]
        for p in B_prime[i]:
            q = p + w_i
            if q > W:
                continue
            item_arcs_list.append((p, q, i))
            item_arc_set.add((p, q))

    # Compute B' = union of all B'_i, plus endpoints of item arcs
    all_vertices = set()
    all_vertices.add(0)
    all_vertices.add(W)
    for p, q, _ in item_arcs_list:
        all_vertices.add(p)
        all_vertices.add(q)

    # Loss arcs: unit-width, connecting consecutive vertices in B' union endpoints,
    # but only in interval [w_min, ..., W].
    # Also, a loss arc is removed if an item arc connects the same two vertices.
    sorted_vertices = sorted(all_vertices)

    loss_arcs = []
    for idx in range(len(sorted_vertices) - 1):
        p = sorted_vertices[idx]
        q = sorted_vertices[idx + 1]
        # Only in interval [w_min, W]
        if p < w_min:
            continue
        # Remove if item arc exists
        if (p, q) in item_arc_set:
            continue
        loss_arcs.append((p, q))

    # Build Gurobi model
    model = gp.Model("CSP_ArcFlow")
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)

    # Variable z: number of bins
    z = model.addVar(vtype=GRB.INTEGER, name="z", lb=0)

    # Arc variables
    # For item arcs, we group by (p, q) to allow multiple item types
    all_arcs = set()
    for p, q, _ in item_arcs_list:
        all_arcs.add((p, q))
    for p, q in loss_arcs:
        all_arcs.add((p, q))

    x = {}
    for (p, q) in all_arcs:
        x[p, q] = model.addVar(vtype=GRB.INTEGER, name=f"x_{p}_{q}", lb=0)

    model.update()

    # Objective: minimize z
    model.setObjective(z, GRB.MINIMIZE)

    # Flow conservation constraints (eq. 12)
    # For each vertex q: sum of outgoing - sum of incoming = z (q=0), -z (q=W), 0 otherwise
    for q in sorted_vertices:
        outgoing = gp.LinExpr()
        incoming = gp.LinExpr()
        for (p2, q2) in all_arcs:
            if p2 == q:
                outgoing += x[p2, q2]
            if q2 == q:
                incoming += x[p2, q2]

        if q == 0:
            model.addConstr(outgoing - incoming == z, name=f"flow_{q}")
        elif q == W:
            model.addConstr(outgoing - incoming == -z, name=f"flow_{q}")
        else:
            model.addConstr(outgoing - incoming == 0, name=f"flow_{q}")

    # Demand constraints (eq. 13)
    # For each item type i: sum of x_{q, q+w_i} over all valid q >= d_i
    for i in range(m):
        w_i = widths[i]
        demand_expr = gp.LinExpr()
        for p in B_prime[i]:
            q = p + w_i
            if (p, q) in x:
                demand_expr += x[p, q]
        model.addConstr(demand_expr >= demands[i], name=f"demand_{i}")

    return model, z, x


def solve_and_output(model, z, solution_path):
    model.optimize()

    result = {}
    if model.SolCount > 0:
        result["objective_value"] = round(model.ObjVal)
    else:
        result["objective_value"] = None
        result["status"] = "no_feasible_solution"

    result["solver_status"] = model.Status
    result["mip_gap"] = model.MIPGap if model.SolCount > 0 else None

    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CSP Arc-Flow with Gurobi")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json")
    parser.add_argument("--time_limit", type=int, default=1200)
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    W, items = load_instance(args.instance_path)

    # Sort items by nonincreasing width (as required by the arc-flow formulation)
    items.sort(key=lambda x: -x[0])
    widths = [it[0] for it in items]
    demands = [it[1] for it in items]

    # Phase 1: Compute normal patterns B'_i for each item type
    B_prime = compute_normal_patterns_csp(widths, demands, W)

    # Phase 2 & 3: Build and solve arc-flow MILP
    model, z, x = build_arc_flow_model(W, widths, demands, B_prime, args.time_limit)

    # Solve and output
    solve_and_output(model, z, args.solution_path)


if __name__ == "__main__":
    main()
