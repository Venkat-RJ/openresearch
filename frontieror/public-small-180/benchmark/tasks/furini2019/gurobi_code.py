"""
Gurobi implementation of the Maximum Clique Problem using ILP Formulation (3)
from: "A Branch and Bound Algorithm for the Maximum Clique Problem"
      Pardalos & Rodgers, Computers Ops Res. Vol. 19, No. 5, pp. 363-375, 1992

Formulation (3) - Integer Linear Programming:
  maximize   sum_{i=1}^{n} x_i
  subject to x_i + x_j <= 1   for all (v_i, v_j) in complement edge set \bar{E}
             x_i in {0, 1}    for all i

The constraints are over edges of the COMPLEMENT graph: pairs of vertices
that are NOT connected by an edge in the original graph G.
"""

import json
import argparse
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
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_max_clique(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    n = data["num_vertices"]
    edges = data["edges"]

    # Build adjacency set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    # Create Gurobi model
    model = gp.Model("MaxClique")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x_i = 1 if vertex i is in the clique
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective: maximize sum of x_i
    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)

    # Constraints: x_i + x_j <= 1 for all (i, j) in complement edge set
    # i.e., for all pairs (i, j) where (i, j) is NOT an edge in G
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                model.addConstr(x[i] + x[j] <= 1,
                                name=f"complement_{i}_{j}")

    # Optimize
    model.optimize()

    # Extract solution
    objective_value = 0
    clique_vertices = []

    if model.SolCount > 0:
        objective_value = round(model.ObjVal)
        clique_vertices = [i for i in range(n) if x[i].X > 0.5]

    # Build solution dictionary
    solution = {
        "objective_value": objective_value,
        "clique_size": objective_value,
        "clique_vertices": clique_vertices,
        "status": model.Status,
        "status_description": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
        }.get(model.Status, f"OTHER_{model.Status}"),
        "num_vertices": n,
        "num_edges": len(edges),
    }

    with open(solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Maximum clique size: {objective_value}")
    print(f"Clique vertices: {clique_vertices}")
    print(f"Solution written to: {solution_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve Maximum Clique Problem using Gurobi ILP (Formulation 3)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_max_clique(args.instance_path, args.solution_path, args.time_limit)
