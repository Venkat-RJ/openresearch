#!/usr/bin/env python3
"""
Gurobi implementation of the Edge Interdiction Clique Problem (EICP)
using the set-covering ILP formulation (4) with clique-covering inequalities (5)
from Furini et al. (2021) "A branch-and-cut algorithm for the Edge Interdiction
Clique Problem", EJOR 294, 54-69.

The formulation:
  (ILP)  min  sum_{ell in L} (ell - 1) * z_ell
  s.t.   sum_{ell in L} z_ell = 1
         [Clique-covering inequalities (5) separated via lazy callbacks]
         sum_{uv in E} w_uv <= k
         w_uv in {0,1}  for all uv in E
         z_ell in {0,1}  for all ell in L

  where L = {ell_min+1, ..., ell_max+1}

Clique-covering inequalities (5):
  sum_{e in E(K)} w_e >= sum_{ell'=ell_min+1}^{ell} f(ell', K) * z_{ell'}
  for ell in L \ {ell_max+1}, K in K_ell

These are separated on the fly using a maximum clique algorithm on the
interdicted graph at each integer solution node.
"""

import json
import argparse
import math

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
    """Load problem instance from JSON."""
    with open(path) as f:
        return json.load(f)


def build_graph(vertices, edges_list):
    """Build adjacency sets and canonical edge set."""
    adj = {v: set() for v in vertices}
    edge_set = set()
    for u, v in edges_list:
        a, b = min(u, v), max(u, v)
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((a, b))
    return adj, edge_set


def f_prop2(ell, q):
    """
    Proposition 2 (Hassan et al., 2018): minimum number of edges to remove
    from a complete graph K_q so that the clique number of the remaining
    graph is at most ell - 1.

    f(ell, K_q) = n_{alpha-1} * C(alpha-1, 2) + n_alpha * C(alpha, 2)
    where alpha = ceil(q / (ell-1))
          n_{alpha-1} = (ell-1)*alpha - q
          n_alpha = (q - n_{alpha-1}*(alpha-1)) / alpha
    """
    if ell <= 1 or q <= ell - 1:
        return 0
    alpha = math.ceil(q / (ell - 1))
    n_alpha_minus_1 = (ell - 1) * alpha - q
    n_alpha = (q - n_alpha_minus_1 * (alpha - 1)) // alpha
    result = (n_alpha_minus_1 * (alpha - 1) * (alpha - 2) // 2
              + n_alpha * alpha * (alpha - 1) // 2)
    return result


def greedy_coloring(vertices_list, adj):
    """
    Greedy sequential coloring of the subgraph induced by vertices_list.
    Returns a dict mapping vertex -> color (0-indexed).
    """
    color = {}
    for v in vertices_list:
        used = set()
        for u in adj.get(v, set()):
            if u in color:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def max_clique_bb(vertices, adj):
    """
    Branch-and-bound maximum clique algorithm with greedy coloring bound.
    Returns the maximum clique as a list of vertices.
    """
    if not vertices:
        return []

    # Degeneracy ordering: repeatedly remove vertex with minimum degree
    remaining = set(vertices)
    order = []
    deg = {v: len(adj[v] & remaining) for v in remaining}
    while remaining:
        v = min(remaining, key=lambda x: deg[x])
        order.append(v)
        remaining.remove(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1

    # Reverse so high-degeneracy vertices are processed first
    order.reverse()

    # Greedy initial clique
    best_clique = []
    for v in order:
        if all(v in adj[u] for u in best_clique):
            best_clique.append(v)

    best = list(best_clique)
    best_size = len(best)

    def bb(candidates, clique):
        nonlocal best, best_size

        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best = list(clique)
            return

        # Color candidates for upper bound
        col = greedy_coloring(candidates, adj)
        num_colors = max(col.values()) + 1 if col else 0

        if len(clique) + num_colors <= best_size:
            return

        # Sort by color, branch on highest color first
        sorted_cands = sorted(candidates, key=lambda v: col[v])

        for i in range(len(sorted_cands) - 1, -1, -1):
            v = sorted_cands[i]
            # Prune: if color index + 1 + clique size <= best, can't improve
            if len(clique) + col[v] + 1 <= best_size:
                return

            new_cands = [u for u in sorted_cands[:i] if u in adj[v]]
            clique.append(v)
            bb(new_cands, clique)
            clique.pop()

    bb(order, [])
    return best


def solve_eicp(instance_path, solution_path, time_limit):
    """Solve the EICP using the set-covering ILP with clique-covering inequalities."""
    data = load_instance(instance_path)
    vertices = data["vertices"]
    edges_list = data["edges"]
    k = data["interdiction_budget_k"]

    adj, edge_set = build_graph(vertices, edges_list)
    edges_sorted = sorted(edge_set)

    # Trivial case: can remove all edges
    if k >= len(edges_sorted):
        result = {
            "objective_value": 1,
            "interdicted_edges": [list(e) for e in edges_sorted[:k]]
        }
        with open(solution_path, 'w') as f:
            result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(result, f, indent=2)
        return

    # Compute omega(G) = maximum clique size
    omega_G_clique = max_clique_bb(vertices, adj)
    omega_G = len(omega_G_clique)

    # Trivial cases
    if omega_G <= 1:
        result = {"objective_value": max(omega_G, 1), "interdicted_edges": []}
        with open(solution_path, 'w') as f:
            result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(result, f, indent=2)
        return

    if omega_G <= 2:
        # Already minimum clique number with edges present
        result = {"objective_value": 2, "interdicted_edges": []}
        with open(solution_path, 'w') as f:
            result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(result, f, indent=2)
        return

    # ell_min = 2, ell_max = omega(G)
    ell_min = 2
    ell_max = omega_G

    # L = {ell_min+1, ..., ell_max+1}
    L = list(range(ell_min + 1, ell_max + 2))

    # ---- Build Gurobi model ----
    model = gp.Model("EICP_SetCovering")
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)
    model.setParam("LazyConstraints", 1)
    model.setParam("OutputFlag", 1)

    # Edge interdiction variables: w[e] = 1 if edge e is interdicted
    w = {}
    for e in edges_sorted:
        w[e] = model.addVar(vtype=GRB.BINARY, name=f"w_{e[0]}_{e[1]}")

    # Level indicator variables: z[ell] = 1 if clique number = ell-1
    z = {}
    for ell in L:
        z[ell] = model.addVar(vtype=GRB.BINARY, name=f"z_{ell}")

    # Objective (4a): min sum (ell-1) * z_ell
    model.setObjective(
        gp.quicksum((ell - 1) * z[ell] for ell in L), GRB.MINIMIZE
    )

    # Constraint (4b): sum z_ell = 1
    model.addConstr(gp.quicksum(z[ell] for ell in L) == 1, "sum_z")

    # Budget constraint: sum w_e <= k
    model.addConstr(gp.quicksum(w[e] for e in edges_sorted) <= k, "budget")

    # ---- Lazy constraint callback for separation of (5) ----
    def separation_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get current integer solution
            w_vals = model.cbGetSolution(w)
            z_vals = model.cbGetSolution(z)

            # Find ell* such that z[ell*] ~ 1
            ell_star = None
            for ell in L:
                if z_vals[ell] > 0.5:
                    ell_star = ell
                    break

            if ell_star is None or ell_star == L[-1]:
                # z_{ell_max+1} = 1 means no interdiction helps; no cut needed
                return

            # Build interdicted graph adjacency
            inter_adj = {v: set() for v in vertices}
            for e in edges_sorted:
                if w_vals[e] < 0.5:  # edge NOT interdicted
                    inter_adj[e[0]].add(e[1])
                    inter_adj[e[1]].add(e[0])

            # Find maximum clique in interdicted graph
            clique = max_clique_bb(vertices, inter_adj)
            q = len(clique)

            if q >= ell_star:
                # Violated clique-covering inequality detected
                # Compute edges of clique K that are in E
                clique_edges_in_E = []
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (min(clique[i], clique[j]),
                             max(clique[i], clique[j]))
                        if e in edge_set:
                            clique_edges_in_E.append(e)

                # RHS: sum_{ell'=ell_min+1}^{ell_star} f(ell', K) * z[ell']
                rhs_terms = []
                for ell_prime in range(ell_min + 1, ell_star + 1):
                    if ell_prime in z:
                        fval = f_prop2(ell_prime, q)
                        if fval > 0:
                            rhs_terms.append((fval, ell_prime))

                if clique_edges_in_E and rhs_terms:
                    model.cbLazy(
                        gp.quicksum(w[e] for e in clique_edges_in_E)
                        >= gp.quicksum(fval * z[ell_p] for fval, ell_p in rhs_terms)
                    )

    model.optimize(separation_callback)

    # ---- Extract solution ----
    obj_val = None
    interdicted = []

    if model.SolCount > 0:
        obj_val = round(model.ObjVal)
        for e in edges_sorted:
            if w[e].X > 0.5:
                interdicted.append(list(e))
    else:
        # No feasible solution found within time limit
        obj_val = omega_G

    result = {
        "objective_value": obj_val,
        "interdicted_edges": interdicted,
    }

    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve EICP using Gurobi with set-covering ILP formulation"
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

    solve_eicp(args.instance_path, args.solution_path, args.time_limit)
