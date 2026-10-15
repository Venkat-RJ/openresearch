"""
Maximum Weighted Clique Problem (MWCP) — Edge Formulation (Equation 20)

Formulation:
    max  sum_{v in V} w(v) * x_v
    s.t. x_u + x_v <= 1   for all (u, v) in complement(E)
         x_v in {0, 1}    for all v in V

Every pair of vertices NOT connected by an edge in G cannot both be in the
clique.  This is equivalent to requiring that the selected vertices form a
clique in the original graph.
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


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def solve_mwcp(instance, time_limit):
    graph = instance["graph"]
    num_vertices = graph["num_vertices"]
    vertices = graph["vertices"]
    edges = graph["edges"]
    weights = instance["vertex_weights"]

    # Build adjacency set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    model = gp.Model("MWCP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # Decision variables: x_v = 1 if vertex v is in the clique
    x = model.addVars(vertices, vtype=GRB.BINARY, name="x")

    # Objective: maximize total weight of selected vertices
    model.setObjective(
        gp.quicksum(weights[v] * x[v] for v in vertices), GRB.MAXIMIZE
    )

    # Constraints: for every pair (u, v) NOT in E, x_u + x_v <= 1
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            u = vertices[i]
            v = vertices[j]
            pair = (min(u, v), max(u, v))
            if pair not in edge_set:
                model.addConstr(x[u] + x[v] <= 1)

    model.optimize()

    result = {
        "objective_value": None,
        "clique_vertices": [],
        "clique_weight": None,
        "status": model.Status,
        "status_name": None,
        "mip_gap": None,
    }

    if model.Status == GRB.OPTIMAL:
        result["status_name"] = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT:
        result["status_name"] = "TIME_LIMIT"
    elif model.Status == GRB.INFEASIBLE:
        result["status_name"] = "INFEASIBLE"
    else:
        result["status_name"] = str(model.Status)

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        clique = [v for v in vertices if x[v].X > 0.5]
        result["clique_vertices"] = clique
        result["clique_weight"] = sum(weights[v] for v in clique)
        try:
            result["mip_gap"] = model.MIPGap
        except Exception:
            result["mip_gap"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve the Maximum Weighted Clique Problem using Gurobi (Edge Formulation)."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to the instance JSON file."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True, help="Path to write the solution JSON file."
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600, help="Gurobi time limit in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_mwcp(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Status: {result['status_name']}")
    print(f"Objective value: {result['objective_value']}")
    print(f"Clique size: {len(result['clique_vertices'])}")
    print(f"Clique vertices: {result['clique_vertices']}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
