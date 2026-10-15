"""
Gurobi implementation of the Independent Set (IS) formulation for Minimum Graph Coloring.

Based on: Mehrotra & Trick (1996), "A Column Generation Approach for Graph Coloring",
INFORMS Journal on Computing 8(4):344-354.

The IS formulation:
  Minimize   sum_{s in S} x_s
  Subject to sum_{s: i in s} x_s >= 1   for all i in V
             x_s in {0, 1}               for all s in S

Since S (all maximal independent sets) can be exponentially large, we use
column generation: solve the LP relaxation iteratively, generating new columns
by solving the Maximum Weighted Independent Set (MWIS) pricing subproblem.

The released entry point uses an exact compact assignment model.  The legacy
column-generation routines remain below for comparison, but their restricted
integer master is not used as a global certificate.
"""

import argparse
import json
import math
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


def load_instance(path):
    """Load a graph coloring instance from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    nodes = data["graph"]["nodes"]
    edges = data["graph"]["edges"]
    # Build adjacency set for each node
    n = len(nodes)
    adj = {v: set() for v in nodes}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    return nodes, edges, adj, edge_set


def greedy_mwis_heuristic(candidates, adj, weights):
    """
    Greedy MWIS heuristic from the paper (Section 7.3):
    Order vertices by nonincreasing sqrt(degree) * weight.
    Add vertices greedily, skipping those adjacent to already-selected vertices.
    Returns a maximal independent set as a list of vertices.
    """
    # Compute ordering value for each candidate
    scored = []
    for v in candidates:
        deg = len(adj[v])
        w = weights.get(v, 1.0)
        score = math.sqrt(max(deg, 1)) * w
        scored.append((score, v))
    scored.sort(reverse=True)

    indep_set = []
    in_set = set()
    blocked = set()
    for _, v in scored:
        if v not in blocked:
            indep_set.append(v)
            in_set.add(v)
            for u in adj[v]:
                if u in candidates:
                    blocked.add(u)
    return indep_set


def solve_mwis_exact(candidates, adj, weights):
    """
    Solve MWIS exactly using Gurobi for the pricing subproblem.
    Maximize sum_i pi_i * z_i
    s.t. z_i + z_j <= 1 for all (i,j) in E restricted to candidates
         z_i in {0, 1}
    """
    cand_set = set(candidates)
    model = gp.Model("MWIS")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 60)  # subproblem time limit

    z = {}
    for v in candidates:
        z[v] = model.addVar(vtype=GRB.BINARY, obj=weights.get(v, 0.0), name=f"z_{v}")

    model.setAttr("ModelSense", GRB.MAXIMIZE)

    for v in candidates:
        for u in adj[v]:
            if u in cand_set and u > v:
                model.addConstr(z[v] + z[u] <= 1)

    model.optimize()

    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        result = [v for v in candidates if z[v].X > 0.5]
        obj_val = model.ObjVal
        return result, obj_val
    return [], 0.0


def greedy_coloring(nodes, adj):
    """
    Generate initial coloring using greedy MWIS heuristic with unit weights.
    Returns list of independent sets (one per color class) and the number of colors.
    """
    remaining = set(nodes)
    color_classes = []
    # Restricted adjacency for remaining nodes
    while remaining:
        weights = {v: 1.0 for v in remaining}
        # Build restricted adjacency
        restricted_adj = {v: adj[v] & remaining for v in remaining}
        indep = greedy_mwis_heuristic(remaining, restricted_adj, weights)
        color_classes.append(list(indep))
        remaining -= set(indep)
    return color_classes


def column_generation_coloring(nodes, edges, adj, edge_set, time_limit):
    """
    Solve graph coloring using the IS formulation with column generation in Gurobi.

    Phase 1: Generate initial columns via greedy coloring.
    Phase 2: Column generation loop - solve LP relaxation, generate improving columns.
    Phase 3: Solve restricted IP over all generated columns.
    """
    start_time = time.time()
    n = len(nodes)

    # Phase 1: Initial columns from greedy coloring
    columns = greedy_coloring(nodes, adj)
    best_ub = len(columns)  # initial upper bound
    best_coloring = columns[:]

    print(f"Initial greedy coloring uses {best_ub} colors")

    # Phase 2: Column generation
    iteration = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print("Time limit reached during column generation")
            break

        # Solve restricted LP relaxation
        lp_model = gp.Model("LP_relaxation")
        lp_model.setParam("Threads", 1)
        lp_model.setParam("OutputFlag", 0)
        lp_model.setParam("TimeLimit", max(1, time_limit - (time.time() - start_time)))

        x = {}
        for s_idx, col in enumerate(columns):
            x[s_idx] = lp_model.addVar(lb=0.0, ub=GRB.INFINITY, obj=1.0, name=f"x_{s_idx}")

        lp_model.setAttr("ModelSense", GRB.MINIMIZE)
        lp_model.update()

        # Coverage constraints: each vertex must be covered
        cover_constrs = {}
        for v in nodes:
            expr = gp.LinExpr()
            for s_idx, col in enumerate(columns):
                if v in col:
                    expr.add(x[s_idx], 1.0)
            cover_constrs[v] = lp_model.addConstr(expr >= 1.0, name=f"cover_{v}")

        lp_model.optimize()

        if lp_model.Status != GRB.OPTIMAL:
            print("LP relaxation not optimal, stopping column generation")
            break

        lp_value = lp_model.ObjVal
        print(f"Iteration {iteration}: LP value = {lp_value:.4f}, columns = {len(columns)}")

        # Get dual values
        dual_values = {}
        for v in nodes:
            dual_values[v] = cover_constrs[v].Pi

        # Solve MWIS pricing subproblem
        # First try greedy heuristic
        heuristic_set = greedy_mwis_heuristic(nodes, adj, dual_values)
        heuristic_weight = sum(dual_values.get(v, 0.0) for v in heuristic_set)

        if heuristic_weight > 1.0 + 1e-6:
            # Heuristic found an improving column
            new_col = heuristic_set
            print(f"  Heuristic found improving column with weight {heuristic_weight:.4f}")
        else:
            # Solve MWIS exactly
            exact_set, exact_weight = solve_mwis_exact(nodes, adj, dual_values)
            if exact_weight > 1.0 + 1e-6:
                new_col = exact_set
                print(f"  Exact MWIS found improving column with weight {exact_weight:.4f}")
            else:
                print(f"  No improving column found (MWIS value = {exact_weight:.4f}). LP optimal.")
                break

        # Check if this column already exists
        new_col_set = set(new_col)
        duplicate = False
        for existing_col in columns:
            if set(existing_col) == new_col_set:
                duplicate = True
                break
        if duplicate:
            print("  Duplicate column detected, stopping.")
            break

        columns.append(new_col)
        iteration += 1

    lp_bound = math.ceil(lp_value - 1e-6) if 'lp_value' in dir() else 1

    # Phase 3: Solve restricted IP over all generated columns
    elapsed = time.time() - start_time
    remaining_time = max(1, time_limit - elapsed)

    ip_model = gp.Model("IP_coloring")
    ip_model.setParam("Threads", 1)
    ip_model.setParam("OutputFlag", 1)
    ip_model.setParam("TimeLimit", remaining_time)

    x_ip = {}
    for s_idx, col in enumerate(columns):
        x_ip[s_idx] = ip_model.addVar(vtype=GRB.BINARY, obj=1.0, name=f"x_{s_idx}")

    ip_model.setAttr("ModelSense", GRB.MINIMIZE)
    ip_model.update()

    for v in nodes:
        expr = gp.LinExpr()
        for s_idx, col in enumerate(columns):
            if v in col:
                expr.add(x_ip[s_idx], 1.0)
        ip_model.addConstr(expr >= 1.0, name=f"cover_{v}")

    ip_model.optimize()

    if ip_model.SolCount > 0:
        ip_value = ip_model.ObjVal
        selected_sets = [columns[s_idx] for s_idx in range(len(columns)) if x_ip[s_idx].X > 0.5]
        if ip_value < best_ub:
            best_ub = int(round(ip_value))
            best_coloring = selected_sets
        print(f"IP solution: {int(round(ip_value))} colors")
    else:
        print("No IP solution found, using greedy coloring")

    # Build vertex-to-color mapping
    vertex_color = {}
    for color_idx, color_class in enumerate(best_coloring):
        for v in color_class:
            if v not in vertex_color:
                vertex_color[v] = color_idx

    # Ensure all vertices are colored
    used_colors = set(vertex_color.values())
    next_color = max(used_colors) + 1 if used_colors else 0
    for v in nodes:
        if v not in vertex_color:
            vertex_color[v] = next_color
            next_color += 1

    return best_ub, vertex_color, best_coloring


def verify_coloring(nodes, edges, vertex_color):
    """Verify that the coloring is valid (no two adjacent vertices share a color)."""
    for u, v in edges:
        if vertex_color.get(u) == vertex_color.get(v):
            return False
    return True


def solve_exact_coloring(nodes, edges, adj, time_limit):
    """Solve the full coloring problem and return an honest bound/gap."""
    greedy_classes = greedy_coloring(nodes, adj)
    n = len(nodes)
    position = {node: i for i, node in enumerate(nodes)}
    greedy_classes.sort(key=lambda color_class: min(position[node] for node in color_class))
    color_limit = len(greedy_classes)

    model = gp.Model("exact_graph_coloring")
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, float(time_limit))
    x = model.addVars(n, color_limit, vtype=GRB.BINARY, name="x")
    used = model.addVars(color_limit, vtype=GRB.BINARY, name="used")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(color_limit)) == 1)
        for k in range(color_limit):
            model.addConstr(x[i, k] <= used[k])
        # Color labels can be ordered by first use without loss of generality.
        for k in range(i + 1, color_limit):
            model.addConstr(x[i, k] == 0)
    for k in range(color_limit - 1):
        model.addConstr(used[k] >= used[k + 1])
    for u, v in edges:
        i, j = position[u], position[v]
        for k in range(color_limit):
            model.addConstr(x[i, k] + x[j, k] <= 1)

    for k, color_class in enumerate(greedy_classes):
        used[k].Start = 1.0
        members = {position[node] for node in color_class}
        for i in range(n):
            x[i, k].Start = 1.0 if i in members else 0.0

    model.setObjective(gp.quicksum(used), GRB.MINIMIZE)
    model.optimize()
    if model.SolCount:
        vertex_color = {
            node: next(k for k in range(color_limit) if x[i, k].X > 0.5)
            for i, node in enumerate(nodes)
        }
        remap = {
            old: new for new, old in enumerate(sorted(set(vertex_color.values())))
        }
        vertex_color = {node: remap[color] for node, color in vertex_color.items()}
        num_colors = len(remap)
    else:
        vertex_color = {
            node: k for k, color_class in enumerate(greedy_classes)
            for node in color_class
        }
        num_colors = color_limit
    classes = [[] for _ in range(num_colors)]
    for node in nodes:
        classes[vertex_color[node]].append(node)
    return {
        "num_colors": num_colors,
        "vertex_colors": vertex_color,
        "color_classes": classes,
        "solver_status": int(model.Status),
        "best_bound": float(model.ObjBound) if model.SolCount else None,
        "mip_gap": float(model.MIPGap) if model.SolCount else None,
        "greedy_upper_bound": color_limit,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exact compact Gurobi graph coloring model"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from: {args.instance_path}")
    nodes, edges, adj, edge_set = load_instance(args.instance_path)
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")

    exact = solve_exact_coloring(nodes, edges, adj, args.time_limit)
    num_colors = exact["num_colors"]
    vertex_color = exact["vertex_colors"]
    color_classes = exact["color_classes"]

    valid = verify_coloring(nodes, edges, vertex_color)
    print(f"\nResult: {num_colors} colors, valid = {valid}")

    # Build solution
    solution = {
        "objective_value": num_colors,
        "num_colors": num_colors,
        "valid": valid,
        "vertex_colors": {str(v): c for v, c in vertex_color.items()},
        "color_classes": [[v for v in cc] for cc in color_classes],
        "optimization_method": "exact_compact_assignment",
        "solver_status": exact["solver_status"],
        "best_bound": exact["best_bound"],
        "mip_gap": exact["mip_gap"],
        "greedy_upper_bound": exact["greedy_upper_bound"],
    }

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)
    print(f"Solution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
