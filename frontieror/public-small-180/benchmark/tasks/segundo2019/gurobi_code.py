"""
Maximum Edge-Weighted Clique Problem (MEWCP) solver using Gurobi.

Implements the F1 (Nonedge) formulation from Segundo et al. (2019):

  max  sum_{e in E} c_e * y_e

  s.t. x_u + x_v <= 1        for all {u,v} in E_bar  (non-edges)
       y_e <= x_u             for all e={u,v} in E
       y_e <= x_v             for all e={u,v} in E
       x_u in {0,1}           for all u in V
       y_e in {0,1}           for all e={u,v} in E
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


def _log_final_solution(log_path: str | None, solution: dict) -> None:
    """Ensure the selected final reference is represented in the JSONL log."""
    objective = solution.get("objective_value")
    if log_path is None or objective is None:
        return

    last_objective = None
    if _os.path.exists(log_path):
        with open(log_path, "r") as log_file:
            for line in log_file:
                if line.strip():
                    last_objective = json.loads(line).get("objective_value")

    if last_objective is not None and abs(float(last_objective) - float(objective)) <= 1e-9:
        return

    record = {
        "time": time.time() - _GUROBI_CODE_START_TIME,
        "objective_value": objective,
        "source": solution.get("status"),
    }
    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(record) + "\n")


def solve_mewcp(
    instance_path: str,
    solution_path: str,
    time_limit: float,
    log_path: str | None = None,
) -> None:
    # Load instance
    with open(instance_path, "r") as f:
        data = json.load(f)

    graph = data["graph"]
    vertices = graph["vertices"]
    edges = graph["edges"]  # list of [u, v, weight]
    n = graph["num_vertices"]

    # Build adjacency set for efficient non-edge checking
    adj = set()
    for u, v, _w in edges:
        adj.add((min(u, v), max(u, v)))

    # Create model
    model = gp.Model("MEWCP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # Decision variables
    x = model.addVars(vertices, vtype=GRB.BINARY, name="x")
    y = {}
    for u, v, w in edges:
        key = (min(u, v), max(u, v))
        if key not in y:
            y[key] = model.addVar(vtype=GRB.BINARY, name=f"y_{key[0]}_{key[1]}")

    # Objective: maximize sum of c_e * y_e
    edge_weight = {}
    for u, v, w in edges:
        key = (min(u, v), max(u, v))
        edge_weight[key] = w

    model.setObjective(
        gp.quicksum(edge_weight[key] * y[key] for key in y),
        GRB.MAXIMIZE,
    )

    # Constraints (27): x_u + x_v <= 1 for all non-edges
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            key = (min(u, v), max(u, v))
            if key not in adj:
                model.addConstr(x[u] + x[v] <= 1)

    # Constraints (6) and (7): y_e <= x_u, y_e <= x_v for all edges
    for u, v, _w in edges:
        key = (min(u, v), max(u, v))
        ye = y[key]
        model.addConstr(ye <= x[u])
        model.addConstr(ye <= x[v])

    # Keep a deterministic positive fallback separate from Gurobi's search.
    # Any positive-weight edge is itself a valid two-vertex clique.
    positive_edges = [(u, v, w) for u, v, w in edges if w > 0]
    fallback_edge = max(positive_edges, key=lambda edge: edge[2]) if positive_edges else None

    # Optimize
    model.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    status_str = status_map.get(model.Status, f"OTHER({model.Status})")

    # Extract solution
    solution = {
        "objective_value": None,
        "clique_vertices": [],
        "clique_edges": [],
        "status": None,
        "gurobi_status": status_str,
        "gurobi_incumbent_objective": model.ObjVal if model.SolCount > 0 else None,
    }

    if model.SolCount > 0 and model.ObjVal > 0:
        solution["objective_value"] = model.ObjVal
        solution["status"] = "gurobi_incumbent"
        clique_verts = [u for u in vertices if x[u].X > 0.5]
        solution["clique_vertices"] = clique_verts

        clique_set = set(clique_verts)
        for key, ye in y.items():
            if ye.X > 0.5:
                u, v = key
                if u in clique_set and v in clique_set:
                    solution["clique_edges"].append(
                        [u, v, edge_weight[key]]
                    )
    elif fallback_edge is not None:
        start_u, start_v, start_weight = fallback_edge
        solution["objective_value"] = start_weight
        solution["clique_vertices"] = [start_u, start_v]
        solution["clique_edges"] = [[start_u, start_v, start_weight]]
        solution["status"] = "zero_incumbent_fallback"
        print(
            f"Using zero-incumbent fallback: edge ({start_u}, {start_v}) "
            f"with weight {start_weight}"
        )

    # Write solution
    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)
    _log_final_solution(log_path, solution)

    print(f"Status: {status_str}")
    if solution["objective_value"] is not None:
        print(f"Objective: {solution['objective_value']}")
        print(f"Clique size: {len(solution['clique_vertices'])}")
        print(f"Clique edges: {len(solution['clique_edges'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve the Maximum Edge-Weighted Clique Problem (MEWCP) using Gurobi."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True, help="Path to write the JSON solution file."
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600, help="Gurobi time limit in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve_mewcp(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()
