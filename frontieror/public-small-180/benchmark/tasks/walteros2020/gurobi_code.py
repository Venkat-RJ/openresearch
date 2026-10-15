"""
Maximum Clique Problem — Gurobi IP Formulation

Given a simple undirected graph G = (V, E), find a clique C ⊆ V of maximum size.

IP Formulation:
    Decision variables: x_v ∈ {0, 1} for each v ∈ V (1 if vertex v is in the clique)
    Objective:   max  Σ_{v ∈ V} x_v
    Constraints: x_i + x_j ≤ 1   for all {i, j} ∈ Ē  (non-edges of G)

NOTE: The complement-based formulation enumerates all non-edges of G. For large,
sparse graphs the number of non-edges can approach n*(n-1)/2 - m, which may be
very memory-intensive. For the instances considered here (e.g. n=1000, m=5985)
this is manageable, but for significantly larger or denser instances an
alternative formulation or decomposition strategy may be necessary.
"""

import argparse
import json
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


def load_instance(instance_path: str) -> tuple:
    """Load a graph instance from a JSON file.

    Returns:
        n: number of vertices
        adj: list of sets representing adjacency sets for each vertex
    """
    with open(instance_path, "r") as f:
        data = json.load(f)

    n = data["n"]
    adj = [set() for _ in range(n)]
    for u, v in data["edges"]:
        adj[u].add(v)
        adj[v].add(u)

    return n, adj


def solve(instance_path: str, solution_path: str, time_limit: int) -> None:
    n, adj = load_instance(instance_path)

    # --- Build Gurobi model ---
    model = gp.Model("MaxClique")
    model.setParam("Threads", 1)

    # Binary decision variables: x_v = 1 if vertex v is in the clique
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective: maximise the number of selected vertices
    model.setObjective(gp.quicksum(x[v] for v in range(n)), GRB.MAXIMIZE)

    # Constraints: for every non-edge {i, j}, at most one endpoint in the clique
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                model.addConstr(x[i] + x[j] <= 1)

    # Solver parameters
    model.setParam("TimeLimit", time_limit)

    # Solve
    model.optimize()

    # --- Extract solution ---
    objective_value = None
    clique_vertices = []

    if model.SolCount > 0:
        objective_value = round(model.ObjVal)
        clique_vertices = [v for v in range(n) if x[v].X > 0.5]

    solution = {
        "objective_value": objective_value,
        "clique_size": len(clique_vertices),
        "clique_vertices": clique_vertices,
        "status": model.Status,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "runtime": model.Runtime,
    }

    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Objective value: {objective_value}")
    print(f"Clique vertices: {clique_vertices}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve the Maximum Clique problem using Gurobi."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the instance JSON file.",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path to write the solution JSON file.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Time limit for the solver in seconds (default: 3600).",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
