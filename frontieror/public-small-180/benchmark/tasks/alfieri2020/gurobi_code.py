"""
gurobi_code.py
==============
Implements the GCP-compact formulation for the Graph Coloring Problem (GCP)
using Gurobi as the MIP solver.

Paper: "Enhancing Column Generation by a Machine-Learning-Based
        Pricing Heuristic for Graph Coloring"
Authors: Shen, Sun, Li, Eberhard, Ernst (2022, AAAI)

Mathematical Model (GCP-compact):
  min   sum_{c in C} z_c                                          (1)
  s.t.  sum_{c in C} x_{i,c} = 1,       for all i in V           (2)
        x_{i,c} + x_{j,c} <= z_c,       for all (i,j) in E, c   (3)
        x_{i,c} in {0,1},               for all i in V, c in C   (4)
        z_c in {0,1},                   for all c in C            (5)
"""

import argparse
import json
import sys
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
def greedy_coloring(n: int, adj: list) -> int:
    """
    Compute an upper bound on the chromatic number via greedy coloring.
    Returns the number of colors used.
    """
    colors = [-1] * n
    for v in range(n):
        used = {colors[u] for u in adj[v] if colors[u] != -1}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return max(colors) + 1


def solve_gcp_compact(instance_path: str, solution_path: str, time_limit: int):
    """
    Solve the Graph Coloring Problem using the GCP-compact MIP formulation.
    """
    # ----------------------------------------------------------------
    # Load instance
    # ----------------------------------------------------------------
    with open(instance_path) as f:
        data = json.load(f)

    n = data["num_nodes"]
    edges = data["edges"]

    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))

    E = list(edge_set)  # undirected edge list (u < v)
    V = list(range(n))

    # ----------------------------------------------------------------
    # Determine number of colors C
    # Upper bound: greedy chromatic number.
    # NOTE (INFERRED ASSUMPTION): The paper does not specify how |C| is
    # set in the GCP-compact formulation. Using a greedy coloring upper
    # bound reduces the model size while preserving optimality.
    # ----------------------------------------------------------------
    num_colors = greedy_coloring(n, adj)
    C = list(range(num_colors))

    # ----------------------------------------------------------------
    # Build Gurobi model
    # ----------------------------------------------------------------
    model = gp.Model("GCP-compact")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i, c] = 1 if vertex i is assigned color c  (eq. 4)
    x = model.addVars(n, num_colors, vtype=GRB.BINARY, name="x")
    # z[c] = 1 if color c is used                  (eq. 5)
    z = model.addVars(num_colors, vtype=GRB.BINARY, name="z")

    # Objective: minimize number of colors used     (eq. 1)
    model.setObjective(gp.quicksum(z[c] for c in C), GRB.MINIMIZE)

    # Constraint (2): each vertex receives exactly one color
    for i in V:
        model.addConstr(
            gp.quicksum(x[i, c] for c in C) == 1,
            name=f"assign_{i}"
        )

    # Constraint (3): adjacent vertices cannot share a color;
    # if both endpoints use color c then z[c] must be 1
    for (i, j) in E:
        for c in C:
            model.addConstr(
                x[i, c] + x[j, c] <= z[c],
                name=f"conflict_{i}_{j}_{c}"
            )

    # ----------------------------------------------------------------
    # Symmetry-breaking constraints (INFERRED ASSUMPTION):
    # The paper does not specify symmetry-breaking for the compact model.
    # We impose z[0] >= z[1] >= ... >= z[num_colors-1] so that colors are
    # used in order.  This does not change the optimal objective value but
    # substantially reduces the number of equivalent optimal solutions.
    # ----------------------------------------------------------------
    for c in range(num_colors - 1):
        model.addConstr(z[c] >= z[c + 1], name=f"sym_{c}")

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------
    model.optimize()

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    if model.SolCount > 0:
        obj_val = model.ObjVal
        coloring = {}
        for i in V:
            for c in C:
                if x[i, c].X > 0.5:
                    coloring[str(i)] = c
                    break

        solution = {
            "objective_value": obj_val,
            "num_colors_used": int(round(obj_val)),
            "coloring": coloring,
            "solver_status": model.Status,
            "is_optimal": model.Status == GRB.OPTIMAL,
        }
    else:
        # No feasible solution found within time limit
        solution = {
            "objective_value": None,
            "num_colors_used": None,
            "coloring": None,
            "solver_status": model.Status,
            "is_optimal": False,
        }

    # ----------------------------------------------------------------
    # Write solution
    # ----------------------------------------------------------------
    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    print(f"Objective value: {solution['objective_value']}")
    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve GCP using GCP-compact MIP formulation via Gurobi"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON instance file"
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path where the solution JSON will be written"
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="Maximum solver runtime in seconds"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_gcp_compact(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
