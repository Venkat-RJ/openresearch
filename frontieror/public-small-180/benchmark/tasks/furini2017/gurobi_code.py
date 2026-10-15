"""
Vertex Coloring Problem (VCP) via auxiliary graph formulation.

Based on: Furini, Gabrel, Ternier (2017)
Uses the relationship chi(G) = |V| - alpha(G_A) where G_A is an auxiliary graph
and alpha(G_A) is the maximum stable set size on G_A.

The auxiliary graph G_A is constructed as:
1. Build complement graph of G
2. Orient complement edges acyclically: arc (v_i, v_j) if i < j
3. Build line graph L of oriented complement
4. Remove simplicial pair edges from L

The MSSP (Maximum Stable Set Problem) on G_A is solved via Gurobi ILP.
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
def load_instance(instance_path):
    """Load graph instance from JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    graph = data["graph"]
    n = graph["n"]
    edges = [tuple(e) for e in graph["edges"]]
    return n, edges


def build_adjacency_set(n, edges):
    """Build adjacency set for quick lookup."""
    adj = set()
    for u, v in edges:
        adj.add((u, v))
        adj.add((v, u))
    return adj


def build_auxiliary_graph(n, edges):
    """
    Build auxiliary graph G_A following Furini et al. (2017).

    Returns:
        arc_list: list of complement arcs (i, j) with i < j
        aux_edges: list of edges in G_A (pairs of arc indices)
    """
    adj_set = build_adjacency_set(n, edges)

    # Step 1-2: Build oriented complement arcs (i, j) where i < j and (i, j) not in E
    arc_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in adj_set:
                arc_list.append((i, j))

    if len(arc_list) == 0:
        return arc_list, []

    # Index arcs by their endpoints for line graph construction
    # arc_from_endpoint[v] = list of arc indices that have v as an endpoint
    arc_from_source = defaultdict(list)
    arc_from_target = defaultdict(list)
    for idx, (u, v) in enumerate(arc_list):
        arc_from_source[u].append(idx)
        arc_from_target[v].append(idx)

    # Collect all arc indices touching each vertex
    arc_at_vertex = defaultdict(list)
    for idx, (u, v) in enumerate(arc_list):
        arc_at_vertex[u].append(idx)
        arc_at_vertex[v].append(idx)

    # Step 3: Build line graph edges - two arcs are adjacent if they share an endpoint
    line_edges_set = set()
    for vertex, arc_indices in arc_at_vertex.items():
        for ii in range(len(arc_indices)):
            for jj in range(ii + 1, len(arc_indices)):
                a = arc_indices[ii]
                b = arc_indices[jj]
                key = (min(a, b), max(a, b))
                line_edges_set.add(key)

    # Step 4: Remove simplicial pair edges
    # A pair {a, b} is simplicial if:
    #   a = (u, v), b = (u, w) (same source u), and
    #   (min(v, w), max(v, w)) is also a complement arc (i.e., v and w are non-adjacent in G)
    simplicial = set()
    for u, arc_indices in arc_from_source.items():
        # All arcs from source u: (u, v1), (u, v2), ...
        targets = [(idx, arc_list[idx][1]) for idx in arc_indices]
        for ii in range(len(targets)):
            for jj in range(ii + 1, len(targets)):
                idx_a, v = targets[ii]
                idx_b, w = targets[jj]
                # Check if (min(v,w), max(v,w)) is a complement arc
                # i.e., v and w are non-adjacent in G
                if (v, w) not in adj_set and (w, v) not in adj_set and v != w:
                    key = (min(idx_a, idx_b), max(idx_a, idx_b))
                    simplicial.add(key)

    aux_edges = [e for e in line_edges_set if e not in simplicial]

    return arc_list, aux_edges


def solve_mssp(arc_list, aux_edges, time_limit):
    """
    Solve Maximum Stable Set Problem on auxiliary graph G_A using Gurobi.

    Returns:
        alpha: size of maximum stable set (or best found)
        selected: list of selected arc indices
        optimal: whether solution is proven optimal
    """
    num_vars = len(arc_list)

    if num_vars == 0:
        return 0, [], True

    model = gp.Model("MSSP")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    # Binary variables x_v for each vertex in G_A (each complement arc)
    x = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

    # Objective: maximize sum of x_v
    model.setObjective(gp.quicksum(x[v] for v in range(num_vars)), GRB.MAXIMIZE)

    # Constraints: x_u + x_v <= 1 for each edge (u, v) in G_A
    for u, v in aux_edges:
        model.addConstr(x[u] + x[v] <= 1)

    model.optimize()

    if model.SolCount == 0:
        return 0, [], False

    optimal = model.Status == GRB.OPTIMAL
    alpha = int(round(model.ObjVal))
    selected = [i for i in range(num_vars) if x[i].X > 0.5]

    return alpha, selected, optimal


class UnionFind:
    """Union-Find data structure for merging vertices into color classes."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def extract_coloring_from_stable_set(n, edges, arc_list, selected):
    """
    Extract a vertex coloring from the selected stable set arcs.

    Each selected arc (i, j) in the complement means i and j are non-adjacent in G,
    so they can share a color. We merge endpoints via union-find.

    Returns:
        coloring: dict mapping vertex -> color (0-indexed)
        valid: whether the coloring is valid
    """
    uf = UnionFind(n)
    for idx in selected:
        u, v = arc_list[idx]
        uf.union(u, v)

    # Assign colors based on connected components
    root_to_color = {}
    color_counter = 0
    coloring = {}
    for v in range(n):
        root = uf.find(v)
        if root not in root_to_color:
            root_to_color[root] = color_counter
            color_counter += 1
        coloring[v] = root_to_color[root]

    # Verify coloring validity
    adj_set = build_adjacency_set(n, edges)
    valid = True
    for u, v in edges:
        if coloring[u] == coloring[v]:
            valid = False
            break

    return coloring, valid


def dsatur_coloring(n, edges):
    """
    DSATUR greedy heuristic for vertex coloring.

    At each step, color the uncolored vertex with the highest saturation degree
    (number of distinct colors in its neighborhood). Break ties by highest
    uncolored degree.

    Returns:
        coloring: dict mapping vertex -> color (0-indexed)
        num_colors: number of colors used
    """
    adj_list = defaultdict(set)
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)

    coloring = {}
    neighbor_colors = defaultdict(set)  # vertex -> set of colors used by neighbors
    colored = [False] * n

    for _ in range(n):
        # Pick uncolored vertex with max saturation, break ties by degree
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in range(n):
            if colored[v]:
                continue
            sat = len(neighbor_colors[v])
            deg = len(adj_list[v])
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_v = v
                best_sat = sat
                best_deg = deg

        # Assign smallest available color
        used = neighbor_colors[best_v]
        c = 0
        while c in used:
            c += 1
        coloring[best_v] = c
        colored[best_v] = True

        # Update neighbor saturation
        for w in adj_list[best_v]:
            if not colored[w]:
                neighbor_colors[w].add(c)

    num_colors = max(coloring.values()) + 1 if coloring else 0
    return coloring, num_colors


def solve_k_coloring_ilp(n, edges, k, time_limit):
    """
    Solve a k-coloring assignment ILP to find a valid coloring with exactly k colors.

    Variables:
        x_{v,c} in {0,1}: vertex v gets color c
    Constraints:
        sum_c x_{v,c} = 1 for all v (each vertex gets exactly one color)
        x_{u,c} + x_{v,c} <= 1 for all (u,v) in E, for all c

    Returns:
        coloring: dict mapping vertex -> color, or None if infeasible/timeout
    """
    model = gp.Model("KColoring")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    # Variables
    x = model.addVars(n, k, vtype=GRB.BINARY, name="x")

    # Each vertex gets exactly one color
    for v in range(n):
        model.addConstr(gp.quicksum(x[v, c] for c in range(k)) == 1)

    # Adjacent vertices get different colors
    for u, v in edges:
        for c in range(k):
            model.addConstr(x[u, c] + x[v, c] <= 1)

    # Symmetry breaking: color 0 assigned to vertex 0
    if n > 0 and k > 0:
        model.addConstr(x[0, 0] == 1)

    # Feasibility problem
    model.setObjective(0, GRB.MINIMIZE)

    model.optimize()

    if model.SolCount == 0:
        return None

    coloring = {}
    for v in range(n):
        for c in range(k):
            if x[v, c].X > 0.5:
                coloring[v] = c
                break

    return coloring


def main():
    parser = argparse.ArgumentParser(
        description="Solve Vertex Coloring Problem using auxiliary graph formulation "
        "(Furini, Gabrel, Ternier 2017)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to instance JSON file"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True, help="Path to output solution JSON file"
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600, help="Time limit in seconds (default: 3600)"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    n, edges = load_instance(args.instance_path)

    # Build auxiliary graph G_A
    arc_list, aux_edges = build_auxiliary_graph(n, edges)

    # Handle edge case: complete graph (no complement edges)
    if len(arc_list) == 0:
        # Every pair is adjacent, need n colors
        chi = n
        coloring = {v: v for v in range(n)}
    else:
        # Solve MSSP on G_A
        alpha, selected, optimal = solve_mssp(arc_list, aux_edges, args.time_limit)

        # chi(G) = |V| - alpha(G_A)
        chi = n - alpha

        # Extract coloring from stable set solution
        coloring, valid = extract_coloring_from_stable_set(n, edges, arc_list, selected)

        if not valid:
            # Fallback 1: try k-coloring ILP with k = chi
            remaining_time = max(args.time_limit / 4, 60.0)
            ilp_coloring = solve_k_coloring_ilp(n, edges, chi, remaining_time)
            if ilp_coloring is not None:
                coloring = ilp_coloring
                valid = True

        if not valid:
            # Fallback 2: DSATUR greedy heuristic
            coloring, num_colors_dsatur = dsatur_coloring(n, edges)
            # If DSATUR uses more colors than chi, update chi to be safe
            # (chi from MSSP is a lower bound if optimal, but coloring must be feasible)
            if num_colors_dsatur > chi:
                chi = num_colors_dsatur

    # Build color classes
    num_colors_used = max(coloring.values()) + 1 if coloring else 0
    color_classes = defaultdict(list)
    for v in range(n):
        color_classes[coloring[v]].append(v)

    # Final validation
    adj_set = build_adjacency_set(n, edges)
    coloring_valid = True
    for u, v in edges:
        if coloring[u] == coloring[v]:
            coloring_valid = False
            break

    # Prepare solution
    solution = {
        "objective_value": chi,
        "num_colors_used": num_colors_used,
        "coloring_valid": coloring_valid,
        "coloring": coloring,
        "color_classes": {str(c): verts for c, verts in sorted(color_classes.items())},
        "auxiliary_graph": {
            "num_complement_arcs": len(arc_list),
            "num_aux_edges": len(aux_edges),
            "stable_set_size": chi and (n - chi) or 0,
        },
    }

    # Write solution
    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Chromatic number (chi): {chi}")
    print(f"Colors used in coloring: {num_colors_used}")
    print(f"Coloring valid: {coloring_valid}")
    print(f"Auxiliary graph: {len(arc_list)} arcs, {len(aux_edges)} edges")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
