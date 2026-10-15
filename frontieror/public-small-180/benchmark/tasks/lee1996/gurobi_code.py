"""
Gurobi implementation of the Steiner Tree-Star (STS) Problem.

Based on:
  Lee, Chiu, Ryan (1996) "A Branch and Cut Algorithm for a Steiner
  Tree-Star Problem", INFORMS Journal on Computing, 8(3):194-201.

The STS problem connects m target nodes through n Steiner nodes by:
  - Assigning each target node to exactly one active Steiner node
  - Connecting active Steiner nodes via a spanning tree
  - Minimizing: assignment costs + Steiner edge costs + Steiner node fixed costs

Variables:
  x[i,j] in [0,1]  : 1 if target i connects to Steiner j
  u[j]   in {0,1}   : 1 if Steiner j is active
  y[j,k] >= 0       : 1 if edge (j,k) is in Steiner spanning tree (j < k)

Only u[j] is declared binary; integrality of x and y follows from the
polyhedral structure (Proposition 1 in the paper).
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
# Callback: Lazy subtour elimination constraints (anti-cycle)
# ---------------------------------------------------------------------------

def subtour_callback(model, where):
    """
    Lazy constraint callback for subtour elimination among Steiner nodes.

    When a candidate integer solution is found (MIPSOL), we:
      1. Identify the set of active Steiner nodes S (u_j >= 0.5).
      2. Build the subgraph of S using edges with y_{jk} >= 0.5.
      3. Find connected components of that subgraph.
      4. If there are multiple components (the tree is disconnected),
         add a subtour elimination (anti-cycle) constraint for each
         component C:
           sum_{j<k, j,k in C} y_{jk} <= |C| - 1

    We also check for fractional (LP relaxation) solutions at MIPNODE
    when the node relaxation has been solved, using the same logic with
    a relaxed threshold.
    """
    if where == GRB.Callback.MIPSOL:
        _separate_subtours(model, is_integer=True)
    elif where == GRB.Callback.MIPNODE:
        # Only separate at the root node or when status is optimal
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            _separate_subtours(model, is_integer=False)


def _separate_subtours(model, is_integer):
    """
    Core separation routine for subtour elimination constraints.

    Parameters
    ----------
    model : gurobipy.Model
        The Gurobi model (accessed via callback).
    is_integer : bool
        True if separating from an integer solution (MIPSOL),
        False if separating from a fractional LP solution (MIPNODE).
    """
    n = model._n
    u_vars = model._u
    y_vars = model._y

    # Retrieve current variable values
    if is_integer:
        u_val = [model.cbGetSolution(u_vars[j]) for j in range(n)]
        y_val = {}
        for j in range(n):
            for k in range(j + 1, n):
                y_val[j, k] = model.cbGetSolution(y_vars[j, k])
    else:
        u_val = [model.cbGetNodeRel(u_vars[j]) for j in range(n)]
        y_val = {}
        for j in range(n):
            for k in range(j + 1, n):
                y_val[j, k] = model.cbGetNodeRel(y_vars[j, k])

    # Identify active Steiner nodes
    if is_integer:
        active = [j for j in range(n) if u_val[j] > 0.5]
    else:
        # For fractional solutions, use a lower threshold
        active = [j for j in range(n) if u_val[j] > 0.01]

    if len(active) <= 1:
        return  # No subtour possible with 0 or 1 active nodes

    # Build adjacency list for active nodes using edges with positive flow
    threshold = 0.5 if is_integer else 0.01
    adj = defaultdict(set)
    active_set = set(active)
    for j in active:
        for k in active:
            if j < k and (j, k) in y_val and y_val[j, k] > threshold:
                adj[j].add(k)
                adj[k].add(j)

    # Find connected components via BFS
    visited = set()
    components = []
    for start in active:
        if start in visited:
            continue
        component = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor in active_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    # If all active nodes are in one component, no subtour violation
    if len(components) <= 1:
        # Even with one component, check if there are cycles within it
        # For integer solutions: if edges in component > |component| - 1, there's a cycle
        if is_integer and len(components) == 1:
            comp = components[0]
            comp_set = set(comp)
            edge_count = sum(
                1 for j in comp for k in comp if j < k and y_val.get((j, k), 0) > 0.5
            )
            if edge_count > len(comp) - 1:
                # There is a cycle; add the anti-cycle constraint
                expr = gp.quicksum(
                    y_vars[j, k]
                    for j in comp for k in comp if j < k
                )
                model.cbLazy(expr <= len(comp) - 1)
        return

    # Add a subtour elimination constraint for each component
    for comp in components:
        if len(comp) < 2:
            continue
        comp_set = set(comp)

        if is_integer:
            # For integer solutions, add SEC: sum y_{jk} for j,k in C <= |C| - 1
            expr = gp.quicksum(
                y_vars[j, k]
                for j in comp for k in comp if j < k
            )
            model.cbLazy(expr <= len(comp) - 1)
        else:
            # For fractional solutions, add the same constraint form
            # Also check if the constraint is actually violated
            lhs_val = sum(
                y_val.get((min(j, k), max(j, k)), 0)
                for idx_j, j in enumerate(comp)
                for k in comp[idx_j + 1:]
            )
            rhs_val = sum(u_val[j] for j in comp) - 1

            if lhs_val > rhs_val + 1e-6:
                # Pick any ell in comp for the general form:
                # sum_{j<k in H} y_{jk} <= sum_{j in H\{ell}} u_j
                ell = comp[0]
                expr_lhs = gp.quicksum(
                    y_vars[min(j, k), max(j, k)]
                    for idx_j, j in enumerate(comp)
                    for k in comp[idx_j + 1:]
                )
                expr_rhs = gp.quicksum(u_vars[j] for j in comp if j != ell)
                model.cbLazy(expr_lhs <= expr_rhs)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(instance_data, time_limit=None):
    """
    Build the Gurobi MIP model for the Steiner Tree-Star problem.

    Parameters
    ----------
    instance_data : dict
        Parsed JSON instance.
    time_limit : int or None
        Solver time limit in seconds.

    Returns
    -------
    model : gurobipy.Model
    x : dict of Gurobi variables x[i,j]
    u : list of Gurobi variables u[j]
    y : dict of Gurobi variables y[j,k] for j < k
    """
    m = instance_data["target_nodes"]["count"]
    n = instance_data["steiner_nodes"]["count"]

    c = instance_data["cost_matrices"]["assignment_costs"]["matrix"]      # m x n
    d = instance_data["cost_matrices"]["interconnection_costs"]["matrix"]  # n x n
    b = instance_data["steiner_nodes"]["node_weights"]                    # n

    model = gp.Model("SteinerTreeStar")
    model.setParam("Threads", 1)

    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------

    # x[i,j]: continuous in [0,1], 1 if target i assigned to Steiner j
    x = {}
    for i in range(m):
        for j in range(n):
            x[i, j] = model.addVar(
                lb=0.0, ub=1.0, obj=c[i][j],
                vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}"
            )

    # u[j]: binary, 1 if Steiner node j is active
    u = []
    for j in range(n):
        u.append(model.addVar(
            lb=0.0, ub=1.0, obj=b[j],
            vtype=GRB.BINARY, name=f"u_{j}"
        ))

    # y[j,k] for j < k: continuous >= 0, 1 if edge (j,k) in Steiner tree
    y = {}
    for j in range(n):
        for k in range(j + 1, n):
            y[j, k] = model.addVar(
                lb=0.0, ub=1.0, obj=d[j][k],
                vtype=GRB.CONTINUOUS, name=f"y_{j}_{k}"
            )

    model.update()

    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------

    # (1) Each target node assigned to exactly one Steiner node
    for i in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n)) == 1,
            name=f"assign_{i}"
        )

    # (2) Target can only connect to active Steiner node
    for i in range(m):
        for j in range(n):
            model.addConstr(x[i, j] <= u[j], name=f"link_{i}_{j}")

    # (3) Edge requires endpoint j to be active
    for j in range(n):
        for k in range(j + 1, n):
            model.addConstr(y[j, k] <= u[j], name=f"edge_j_{j}_{k}")

    # (4) Edge requires endpoint k to be active
    for j in range(n):
        for k in range(j + 1, n):
            model.addConstr(y[j, k] <= u[k], name=f"edge_k_{j}_{k}")

    # (5) Spanning tree: number of edges = number of active nodes - 1
    model.addConstr(
        gp.quicksum(y[j, k] for j in range(n) for k in range(j + 1, n))
        == gp.quicksum(u[j] for j in range(n)) - 1,
        name="spanning_tree"
    )

    # -----------------------------------------------------------------------
    # Valid inequalities (11) and (12) added at root
    # -----------------------------------------------------------------------

    # (11) Degree inequality type 1:
    #   sum_{ell < j} y_{ell,j} + sum_{j < k} y_{j,k} >= u_j
    for j in range(n):
        lhs = gp.quicksum(y[ell, j] for ell in range(j))
        lhs += gp.quicksum(y[j, k] for k in range(j + 1, n))
        model.addConstr(lhs >= u[j], name=f"degree1_{j}")

    # (12) Degree inequality type 2:
    #   sum_i x_{ij} + sum_{ell < j} y_{ell,j} + sum_{j < k} y_{j,k} >= 3*u_j
    for j in range(n):
        lhs = gp.quicksum(x[i, j] for i in range(m))
        lhs += gp.quicksum(y[ell, j] for ell in range(j))
        lhs += gp.quicksum(y[j, k] for k in range(j + 1, n))
        model.addConstr(lhs >= 3 * u[j], name=f"degree2_{j}")

    # -----------------------------------------------------------------------
    # Model parameters
    # -----------------------------------------------------------------------
    model.ModelSense = GRB.MINIMIZE
    model.Params.LazyConstraints = 1

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Store data on the model for callback access
    model._n = n
    model._m = m
    model._u = u
    model._y = y
    model._x = x

    return model, x, u, y


# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------

def extract_solution(model, x, u, y, m, n):
    """
    Extract the solution from a solved Gurobi model.

    Returns a dictionary suitable for JSON serialization.
    """
    sol = {}

    status = model.Status
    sol["solver_status"] = status
    sol["status_name"] = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(status, f"OTHER({status})")

    if model.SolCount == 0:
        sol["objective_value"] = None
        sol["best_bound"] = model.ObjBound if hasattr(model, "ObjBound") else None
        sol["solution_count"] = 0
        return sol

    sol["objective_value"] = model.ObjVal
    sol["best_bound"] = model.ObjBound
    sol["mip_gap"] = model.MIPGap
    sol["solution_count"] = model.SolCount
    sol["runtime_seconds"] = model.Runtime
    sol["node_count"] = int(model.NodeCount)

    # Active Steiner nodes
    active_steiners = []
    for j in range(n):
        if u[j].X > 0.5:
            active_steiners.append(j)
    sol["active_steiner_nodes"] = active_steiners

    # Target-to-Steiner assignments
    assignments = {}
    for i in range(m):
        for j in range(n):
            if x[i, j].X > 0.5:
                assignments[str(i)] = j
                break
    sol["assignments"] = assignments

    # Steiner tree edges
    edges = []
    for j in range(n):
        for k in range(j + 1, n):
            if y[j, k].X > 0.5:
                edges.append([j, k])
    sol["steiner_tree_edges"] = edges

    # Cost breakdown
    assignment_cost = sum(
        x[i, j].X * x[i, j].Obj for i in range(m) for j in range(n)
    )
    tree_cost = sum(
        y[j, k].X * y[j, k].Obj for j in range(n) for k in range(j + 1, n)
    )
    node_cost = sum(u[j].X * u[j].Obj for j in range(n))

    sol["cost_breakdown"] = {
        "assignment_cost": round(assignment_cost, 6),
        "steiner_tree_cost": round(tree_cost, 6),
        "steiner_node_cost": round(node_cost, 6),
        "total": round(assignment_cost + tree_cost + node_cost, 6),
    }

    return sol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Solve the Steiner Tree-Star (STS) Problem using Gurobi."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", type=str, default="gurobi_solution_1.json",
        help="Path for the output solution JSON file (default: gurobi_solution_1.json)."
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Maximum solver time in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)

    m = instance_data["target_nodes"]["count"]
    n = instance_data["steiner_nodes"]["count"]

    print(f"Instance: {args.instance_path}")
    print(f"  Target nodes (m): {m}")
    print(f"  Steiner nodes (n): {n}")
    if args.time_limit is not None:
        print(f"  Time limit: {args.time_limit}s")

    # Build model
    model, x, u, y = build_model(instance_data, time_limit=args.time_limit)

    # Solve with callback
    wall_start = time.time()
    model.optimize(subtour_callback)
    wall_end = time.time()

    print(f"\nSolver finished in {wall_end - wall_start:.2f}s wall-clock time.")
    print(f"  Status: {model.Status}")
    if model.SolCount > 0:
        print(f"  Objective: {model.ObjVal:.6f}")
        print(f"  Best bound: {model.ObjBound:.6f}")
        print(f"  Gap: {model.MIPGap:.6%}")
    else:
        print("  No feasible solution found.")

    # Extract and save solution
    solution = extract_solution(model, x, u, y, m, n)
    solution["instance_path"] = args.instance_path
    solution["wall_clock_seconds"] = round(wall_end - wall_start, 3)

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"\nSolution written to: {args.solution_path}")

    return 0 if model.SolCount > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
