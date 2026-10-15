"""
Gurobi solver for the Orienteering Problem (OP).

Implements the ILP formulation from:
  Fischetti, Salazar González, Toth (1998)
  "Solving the Orienteering Problem through Branch-and-Cut"
  INFORMS Journal on Computing 10(2):133-148

Mathematical model:
  max  sum_{v in V} p_v * y_v                          (1)
  s.t.
  sum_{e in E} t_e * x_e <= t_0                        (2)  travel time budget
  x(delta(v)) = 2 * y_v         for all v in V         (3)  degree equations
  x(delta(S)) >= 2 * y_v        for all S subset V,    (4)  GSECs (lazy)
                                 1 not in S, v in S
  y_1 = 1                                              (5)  depot visited
  x_e <= y_j  for e in delta(j), j in V\\{1}            (12) logical constraints
  x_e in {0,1}, y_v in {0,1}                            (8,9)

GSECs (4) are separated dynamically via lazy constraint callbacks.
"""

import argparse
import json
import sys
import time
from collections import defaultdict

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


# ---------------------------------------------------------------------------
# Utility: parse edge key strings like "(1,2)" into integer tuples (i, j)
# ---------------------------------------------------------------------------
def parse_edge_key(key_str):
    """Parse a string of the form '(i,j)' into a tuple of ints (i, j)."""
    stripped = key_str.strip().strip("()")
    parts = stripped.split(",")
    return (int(parts[0].strip()), int(parts[1].strip()))


def canonical_edge(i, j):
    """Return the canonical (min, max) representation for an undirected edge."""
    return (min(i, j), max(i, j))


# ---------------------------------------------------------------------------
# Load instance from JSON
# ---------------------------------------------------------------------------
def load_instance(path):
    """
    Load an OP instance from a JSON file.

    Returns:
        nodes    : list of node ids (1-indexed integers)
        depot    : depot node id (int)
        t0       : maximum travel time (int/float)
        prizes   : dict  node_id -> prize value
        edges    : list of canonical edge tuples (i, j) with i < j
        travel   : dict  (i,j) -> travel time  (i < j)
    """
    with open(path, "r") as f:
        data = json.load(f)

    n = data["n"]
    depot = data["depot"]
    t0 = data["t0"]

    # Prizes: keys are string node ids
    prizes = {int(k): v for k, v in data["prizes"].items()}

    # Travel times: keys are strings like "(1,2)"
    travel = {}
    for key_str, cost in data["travel_times"].items():
        i, j = parse_edge_key(key_str)
        e = canonical_edge(i, j)
        travel[e] = cost

    nodes = list(range(1, n + 1))
    edges = sorted(travel.keys())

    return nodes, depot, t0, prizes, edges, travel


# ---------------------------------------------------------------------------
# Build adjacency / incidence helpers
# ---------------------------------------------------------------------------
def build_incidence(nodes, edges):
    """
    Build a mapping from each node to the set of edges incident to it.

    Returns:
        delta : dict  node_id -> list of (i, j) edges incident to that node
    """
    delta = defaultdict(list)
    for (i, j) in edges:
        delta[i].append((i, j))
        delta[j].append((i, j))
    return delta


# ---------------------------------------------------------------------------
# Connected components via BFS on the support graph
# ---------------------------------------------------------------------------
def find_connected_components(adj):
    """
    Given an adjacency dict {node: set of neighbours}, find all connected
    components via BFS.

    Returns a list of sets, each set being the node set of one component.
    """
    visited = set()
    components = []

    for start in adj:
        if start in visited:
            continue
        # BFS
        component = set()
        queue = [start]
        while queue:
            v = queue.pop()
            if v in visited:
                continue
            visited.add(v)
            component.add(v)
            for w in adj[v]:
                if w not in visited:
                    queue.append(w)
        components.append(component)

    return components


# ---------------------------------------------------------------------------
# GSEC lazy-constraint callback
# ---------------------------------------------------------------------------
def gsec_callback(model, where):
    """
    Gurobi callback for lazy GSEC separation.

    When an integer feasible solution is found (MIPSOL), we:
      1. Extract current x and y values.
      2. Build the support graph (edges with x_e = 1).
      3. Find connected components.
      4. For every component S that does NOT contain the depot,
         if any y_v = 1 for v in S, add the GSEC:
           sum_{e in delta(S)} x_e >= 2 * y_v
    """
    if where != GRB.Callback.MIPSOL:
        return

    # Retrieve callback data stored on the model object
    x_vars = model._x_vars      # dict (i,j) -> Gurobi Var
    y_vars = model._y_vars      # dict node  -> Gurobi Var
    nodes = model._nodes
    edges = model._edges
    depot = model._depot
    delta = model._delta         # node -> list of incident edges

    # Get current solution values
    x_val = {}
    for e in edges:
        x_val[e] = model.cbGetSolution(x_vars[e])

    y_val = {}
    for v in nodes:
        y_val[v] = model.cbGetSolution(y_vars[v])

    # Build adjacency list for the support graph (edges with x_e ~ 1)
    adj = defaultdict(set)
    for (i, j) in edges:
        if x_val[(i, j)] > 0.5:
            adj[i].add(j)
            adj[j].add(i)

    # Also ensure all visited nodes appear in adj (isolated visited nodes)
    for v in nodes:
        if y_val[v] > 0.5 and v not in adj:
            adj[v] = set()

    if not adj:
        return

    # Find connected components
    components = find_connected_components(adj)

    # For each component not containing the depot, add GSECs if needed
    for comp in components:
        if depot in comp:
            continue

        # Check if any node in this component is visited
        visited_in_comp = [v for v in comp if y_val[v] > 0.5]
        if not visited_in_comp:
            continue

        # Compute delta(S): edges with exactly one endpoint in comp
        delta_S_edges = []
        for v in comp:
            for e in delta[v]:
                i, j = e
                other = j if i == v else i
                if other not in comp:
                    delta_S_edges.append(e)

        # Remove duplicates (an edge (i,j) with i in S, j not in S
        # will be found from both endpoints if both are in comp -- but
        # since j is NOT in comp, it's only found once from i's side.
        # However, the same edge could appear from different v in comp
        # that share the same edge to the outside.)
        delta_S_edges = list(set(delta_S_edges))

        # Build the LHS expression: sum of x_e for e in delta(S)
        lhs = gp.quicksum(x_vars[e] for e in delta_S_edges)

        # Add a GSEC for each visited node in the component
        for v in visited_in_comp:
            model.cbLazy(lhs >= 2 * y_vars[v])


# ---------------------------------------------------------------------------
# Extract tour from solution
# ---------------------------------------------------------------------------
def extract_tour(visited_nodes, used_edges, depot):
    """
    Given the set of visited nodes and used edges, reconstruct the tour
    as an ordered list of nodes starting and ending at the depot.

    Returns:
        tour : list of node ids forming the cycle, starting and ending at depot
    """
    if len(visited_nodes) <= 1:
        return [depot]

    # Build adjacency for the tour subgraph
    adj = defaultdict(list)
    for (i, j) in used_edges:
        adj[i].append(j)
        adj[j].append(i)

    # Walk the cycle starting from the depot
    tour = [depot]
    prev = None
    current = depot

    while True:
        neighbors = adj[current]
        # Pick the neighbor that is not the previous node
        next_node = None
        for nb in neighbors:
            if nb != prev:
                next_node = nb
                break

        if next_node is None or next_node == depot:
            # We've completed the cycle
            tour.append(depot)
            break

        tour.append(next_node)
        prev = current
        current = next_node

    return tour


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve_op(instance_path, solution_path, time_limit):
    """Build and solve the OP model using Gurobi."""

    print("=" * 70)
    print("Orienteering Problem -- Gurobi Solver")
    print("Fischetti, Salazar González, Toth (1998) formulation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load instance
    # ------------------------------------------------------------------
    nodes, depot, t0, prizes, edges, travel = load_instance(instance_path)
    n = len(nodes)
    m = len(edges)
    delta = build_incidence(nodes, edges)

    print(f"\nInstance loaded from: {instance_path}")
    print(f"  Nodes (n)     : {n}")
    print(f"  Edges (m)     : {m}")
    print(f"  Depot         : {depot}")
    print(f"  Time budget   : {t0}")
    print(f"  Total prize   : {sum(prizes.values())}")

    # ------------------------------------------------------------------
    # 2. Create Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model("OrienteeringProblem")
    model.setParam("Threads", 1)

    # Suppress default Gurobi output (we print our own summary)
    model.setParam("OutputFlag", 1)

    # Enable lazy constraints
    model.Params.LazyConstraints = 1

    # Set time limit
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # ------------------------------------------------------------------
    # 3. Decision variables
    # ------------------------------------------------------------------
    # x_e in {0,1} for each edge e
    x = {}
    for e in edges:
        x[e] = model.addVar(vtype=GRB.BINARY, name=f"x_{e[0]}_{e[1]}")

    # y_v in {0,1} for each node v
    y = {}
    for v in nodes:
        y[v] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}")

    model.update()

    # ------------------------------------------------------------------
    # 4. Objective: maximize total prize collected
    # ------------------------------------------------------------------
    # (1) max sum_{v in V} p_v * y_v
    model.setObjective(
        gp.quicksum(prizes[v] * y[v] for v in nodes),
        GRB.MAXIMIZE,
    )

    # ------------------------------------------------------------------
    # 5. Constraints
    # ------------------------------------------------------------------

    # (2) Travel time budget: sum_{e in E} t_e * x_e <= t_0
    model.addConstr(
        gp.quicksum(travel[e] * x[e] for e in edges) <= t0,
        name="travel_time_budget",
    )

    # (3) Degree equations: x(delta(v)) = 2 * y_v for all v in V
    for v in nodes:
        incident = delta[v]
        model.addConstr(
            gp.quicksum(x[e] for e in incident) == 2 * y[v],
            name=f"degree_{v}",
        )

    # (5) Depot must be visited: y_1 = 1
    model.addConstr(y[depot] == 1, name="depot_visited")

    # (12) Logical constraints: x_e <= y_j for all e in delta(j), j != depot
    #      These strengthen the LP relaxation and make y integrality redundant.
    for v in nodes:
        if v == depot:
            continue
        for e in delta[v]:
            model.addConstr(
                x[e] <= y[v],
                name=f"logical_{e[0]}_{e[1]}_node_{v}",
            )

    # ------------------------------------------------------------------
    # 6. Store data on the model for the callback
    # ------------------------------------------------------------------
    model._x_vars = x
    model._y_vars = y
    model._nodes = nodes
    model._edges = edges
    model._depot = depot
    model._delta = delta

    # ------------------------------------------------------------------
    # 7. Optimize with GSEC callback
    # ------------------------------------------------------------------
    print("\nStarting optimization...")
    start_time = time.time()
    model.optimize(gsec_callback)
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.2f} seconds.")

    # ------------------------------------------------------------------
    # 8. Extract and report solution
    # ------------------------------------------------------------------
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.NODE_LIMIT: "node_limit",
        GRB.SOLUTION_LIMIT: "solution_limit",
        GRB.INTERRUPTED: "interrupted",
    }
    status_str = status_map.get(model.status, f"unknown_{model.status}")

    # Check if we have a feasible solution
    if model.SolCount == 0:
        print(f"\nNo feasible solution found. Status: {status_str}")
        solution = {
            "objective_value": None,
            "visited_nodes": [],
            "edges": [],
            "tour": [],
            "total_travel_time": None,
            "total_prize": None,
            "status": status_str,
            "solver_time_seconds": round(elapsed, 3),
            "mip_gap": None,
            "node_count": int(model.NodeCount),
        }
    else:
        obj_val = model.ObjVal

        # Visited nodes
        visited_nodes = sorted([v for v in nodes if y[v].X > 0.5])

        # Used edges
        used_edges = sorted([e for e in edges if x[e].X > 0.5])

        # Total travel time
        total_travel = sum(travel[e] for e in used_edges)

        # Total prize
        total_prize = sum(prizes[v] for v in visited_nodes)

        # Reconstruct tour
        tour = extract_tour(visited_nodes, used_edges, depot)

        # MIP gap
        mip_gap = model.MIPGap if hasattr(model, "MIPGap") else None

        # Best bound
        best_bound = model.ObjBound if hasattr(model, "ObjBound") else None

        print(f"\n{'=' * 70}")
        print("SOLUTION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Status            : {status_str}")
        print(f"  Objective value   : {obj_val}")
        print(f"  Total prize       : {total_prize}")
        print(f"  Total travel time : {total_travel} (budget: {t0})")
        print(f"  Visited nodes     : {visited_nodes}")
        print(f"  Number visited    : {len(visited_nodes)}")
        print(f"  Tour              : {' -> '.join(map(str, tour))}")
        print(f"  Edges used        : {used_edges}")
        print(f"  MIP gap           : {mip_gap}")
        print(f"  Best bound        : {best_bound}")
        print(f"  Solver time       : {elapsed:.2f}s")
        print(f"  Nodes explored    : {int(model.NodeCount)}")
        print(f"{'=' * 70}")

        solution = {
            "objective_value": obj_val,
            "visited_nodes": visited_nodes,
            "edges": [list(e) for e in used_edges],
            "tour": tour,
            "total_travel_time": total_travel,
            "total_prize": total_prize,
            "status": status_str,
            "solver_time_seconds": round(elapsed, 3),
            "mip_gap": mip_gap,
            "best_bound": best_bound,
            "node_count": int(model.NodeCount),
        }

    # ------------------------------------------------------------------
    # 9. Write solution to JSON
    # ------------------------------------------------------------------
    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)
    print(f"\nSolution written to: {solution_path}")

    return solution


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Solve the Orienteering Problem (OP) using Gurobi, "
            "following the Fischetti, Salazar González, Toth (1998) formulation."
        ),
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
        default="gurobi_solution_1.json",
        help="Path to write the solution JSON file (default: gurobi_solution_1.json).",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="Solver time limit in seconds.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_op(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
