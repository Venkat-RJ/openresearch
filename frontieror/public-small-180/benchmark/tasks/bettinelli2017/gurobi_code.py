"""
Knapsack Problem with Conflict Graph (KPCG) — Clique-based formulation.

Implements Model 2 from Bettinelli et al. (2017): the clique cover
formulation where conflict constraints are lifted to maximal-clique
inequalities, yielding a tighter LP relaxation than the edge formulation.

Usage:
    python gurobi_code.py --instance_path instance_1.json \
                          --solution_path gurobi_solution_1.json \
                          --time_limit 600
"""

import argparse
import json
import random
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
        pass# ------------------------------------------------------------------ #
#  Heuristic clique cover generation                                  #
# ------------------------------------------------------------------ #

def build_adjacency(num_vertices, edges):
    """Return adjacency set dict from an edge list."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def generate_clique_cover(num_vertices, edges, profits, weights):
    """
    Heuristic clique cover that guarantees every conflict edge is
    contained in at least one maximal clique.

    Algorithm (per Bettinelli et al.):
      1. Pick a random uncovered edge (i, j).
      2. Seed a clique C = {i, j}.
      3. Consider every other vertex l in non-increasing p_l / w_l
         order; add l to C if l is adjacent to every member of C.
      4. Mark all edges inside C as covered.
      5. Repeat until every edge is covered.
    """
    adj = build_adjacency(num_vertices, edges)

    # Efficiency ratios for tie-breaking / greedy extension order
    ratio = []
    for i in range(num_vertices):
        w = weights[i] if weights[i] > 0 else 1e-9
        ratio.append(profits[i] / w)

    # Vertices sorted by non-increasing efficiency (precomputed once)
    sorted_vertices = sorted(range(num_vertices), key=lambda v: -ratio[v])

    uncovered = set()
    for u, v in edges:
        uncovered.add((min(u, v), max(u, v)))

    cliques = []
    edge_list = list(uncovered)

    while uncovered:
        # Pick a random uncovered edge
        # (filter edge_list lazily)
        while edge_list:
            idx = random.randrange(len(edge_list))
            e = edge_list[idx]
            if e in uncovered:
                break
            # Swap-remove
            edge_list[idx] = edge_list[-1]
            edge_list.pop()
        else:
            break  # no uncovered edges left

        i, j = e
        clique = {i, j}

        # Extend greedily
        for l in sorted_vertices:
            if l in clique:
                continue
            if all(l in adj[m] for m in clique):
                clique.add(l)

        # Mark all intra-clique edges as covered
        members = sorted(clique)
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                a, b = members[a_idx], members[b_idx]
                key = (min(a, b), max(a, b))
                uncovered.discard(key)

        cliques.append(members)

    return cliques


# ------------------------------------------------------------------ #
#  Gurobi model                                                       #
# ------------------------------------------------------------------ #

def solve_kpcg(instance_path, solution_path, time_limit):
    # Load instance
    with open(instance_path, "r") as f:
        data = json.load(f)

    num_items = data["num_items"]
    capacity = data["capacity"]
    profits = data["items"]["profits"]
    weights = data["items"]["weights"]
    cg = data["conflict_graph"]
    num_vertices = cg["num_vertices"]
    edges = [tuple(e) for e in cg["edges"]]

    # Generate clique cover
    cliques = generate_clique_cover(num_vertices, edges, profits, weights)

    # Build model
    model = gp.Model("KPCG_clique")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # Decision variables
    x = model.addVars(num_items, vtype=GRB.BINARY, name="x")

    # Objective
    model.setObjective(
        gp.quicksum(profits[i] * x[i] for i in range(num_items)),
        GRB.MAXIMIZE,
    )

    # Capacity constraint
    model.addConstr(
        gp.quicksum(weights[i] * x[i] for i in range(num_items)) <= capacity,
        name="capacity",
    )

    # Clique constraints
    for k, clique in enumerate(cliques):
        model.addConstr(
            gp.quicksum(x[i] for i in clique) <= 1,
            name=f"clique_{k}",
        )

    model.optimize()

    # Extract solution
    objective_value = None
    selected_items = []

    if model.SolCount > 0:
        objective_value = model.ObjVal
        selected_items = [i for i in range(num_items) if x[i].X > 0.5]

    result = {
        "objective_value": objective_value,
        "selected_items": selected_items,
    }

    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Objective value : {objective_value}")
    print(f"Selected items  : {selected_items}")
    print(f"Solution written to {solution_path}")


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Solve KPCG with clique-based Gurobi formulation."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the instance JSON file.",
    )
    parser.add_argument(
        "--solution_path", type=str, default="gurobi_solution_1.json",
        help="Path for the output solution JSON file.",
    )
    parser.add_argument(
        "--time_limit", type=int, required=True,
        help="Solver time limit in seconds.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve_kpcg(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
