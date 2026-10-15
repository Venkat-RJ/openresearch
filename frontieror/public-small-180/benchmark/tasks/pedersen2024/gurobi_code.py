#!/usr/bin/env python3
"""
Flow-based MIP formulation (FLOW) for the Quota Steiner Tree Problem (QSTP).

Implements the formulation from Pedersen et al. (2024), equations (23)-(30):
    min  c^T x + w^T y
    s.t.
        sum_{v in T_p} q_v * y_v >= Q                           (25)
        flow balance constraints                                 (26)
        x_a <= y_v   for all a in delta^-(v), v in T_p          (27)
        f_a <= M * x_a  for all a in A                          (28)
        y_v in {0,1}  for all v in T_p                          (29)
        x_a in {0,1}, f_a >= 0  for all a in A                  (30)

The bi-objective extension uses the weighted-sum approach (eq. 21):
    C_bar(S) = alpha * C(S) + (1-alpha) * L(S)

Reference: Pedersen et al. (2024) - "An efficient solver for large-scale
onshore wind farm siting including cable routing", EJOR 317(2), 616-630.
"""

import argparse
import json
import math
import os
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
def load_instance(instance_path):
    """Load a QSTP instance from a JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def euclidean_distance(n1, n2):
    """Compute Euclidean distance between two nodes."""
    return math.sqrt((n1["x"] - n2["x"]) ** 2 + (n1["y"] - n2["y"]) ** 2)


def build_and_solve(instance_data, time_limit):
    """
    Build and solve the flow-based MIP formulation for the QSTP.

    Parameters
    ----------
    instance_data : dict
        The parsed JSON instance data.
    time_limit : int
        Maximum solver runtime in seconds.

    Returns
    -------
    dict
        Solution dictionary with objective_value and solution details.
    """
    params = instance_data["parameters"]
    alpha = params["alpha"]
    Q = params["quota"]
    nodes = instance_data["nodes"]
    edges = instance_data["edges"]

    # Build node lookup
    node_by_id = {n["id"]: n for n in nodes}

    # Classify nodes
    T_f = [n["id"] for n in nodes if n["type"] == "substation"]
    T_p = [n["id"] for n in nodes if n["type"] == "potential_turbine"]
    steiner_nodes = [n["id"] for n in nodes if n["type"] == "steiner"]
    all_nodes = [n["id"] for n in nodes]

    # Choose an arbitrary root from fixed terminals
    # As stated in the paper (Section 3.1.2): all substations are interconnected
    # by edges of zero cost, so the choice of root does not matter.
    r = T_f[0]

    # Compute combined edge costs and vertex costs using bi-objective
    # weighted-sum approach (eq. 21):
    #   c_bar(e) = alpha * c(e) + (1-alpha) * s(e)
    #   w_bar(v) = alpha * w(v) + (1-alpha) * s^v(v)
    vertex_cost = {}
    vertex_profit = {}
    for n in nodes:
        if n["type"] == "potential_turbine":
            vertex_cost[n["id"]] = alpha * n["cost"] + (1.0 - alpha) * n["scenic_impact"]
            vertex_profit[n["id"]] = n["profit"]
        else:
            vertex_cost[n["id"]] = 0.0
            vertex_profit[n["id"]] = 0.0

    edge_cost = {}
    for e in edges:
        u, v = e["from"], e["to"]
        c_bar = alpha * e["cost"] + (1.0 - alpha) * e["scenic_impact"]
        edge_cost[(u, v)] = c_bar

    # Build directed arcs: for each undirected edge (u,v), create arcs (u,v) and (v,u)
    # Arc costs incorporate vertex-cost shifting (eq. 3):
    #   c(i,j) = c_e + w_j  if j in T_p
    #   c(i,j) = c_e        otherwise
    arcs = []
    arc_cost = {}
    for e in edges:
        u, v = e["from"], e["to"]
        c_bar = edge_cost[(u, v)]

        # Arc (u, v)
        cost_uv = c_bar + vertex_cost.get(v, 0.0) if v in vertex_cost and node_by_id[v]["type"] == "potential_turbine" else c_bar
        arcs.append((u, v))
        arc_cost[(u, v)] = cost_uv

        # Arc (v, u)
        cost_vu = c_bar + vertex_cost.get(u, 0.0) if u in vertex_cost and node_by_id[u]["type"] == "potential_turbine" else c_bar
        arcs.append((v, u))
        arc_cost[(v, u)] = cost_vu

    # Big-M value (as per paper Section 3.1.2): M = |T_f union T_p|
    M = len(T_f) + len(T_p)

    # Build Gurobi model
    model = gp.Model("QSTP_FLOW")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    # Suppress output for cleaner execution
    model.setParam("OutputFlag", 1)

    # --- Decision variables ---
    # x_a in {0,1}: arc selection
    x = {}
    for a in arcs:
        x[a] = model.addVar(vtype=GRB.BINARY, name=f"x_{a[0]}_{a[1]}")

    # y_v in {0,1}: potential terminal selection
    y = {}
    for v in T_p:
        y[v] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}")

    # f_a >= 0: flow on arcs
    f = {}
    for a in arcs:
        f[a] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_{a[0]}_{a[1]}")

    model.update()

    # --- Objective (eq. 23) ---
    # Note: In the flow-based formulation, the objective is c^T x + w^T y.
    # However, since we use vertex-cost shifting in arc costs (c(i,j) = c_e + w_j
    # for j in T_p), the vertex costs are already embedded in the arc costs.
    # The paper's flow formulation (Section 3.1.2) uses the ORIGINAL (unshifted)
    # arc costs plus explicit w^T y terms.
    # We implement it as stated: min c^T x + w^T y where c here are the
    # ORIGINAL edge costs (not shifted), and w are vertex costs.

    # Re-compute arc costs WITHOUT vertex-cost shifting for the flow formulation
    # since the formulation explicitly includes + w^T y
    arc_cost_unshifted = {}
    for e in edges:
        u, v = e["from"], e["to"]
        c_bar = alpha * e["cost"] + (1.0 - alpha) * e["scenic_impact"]
        arc_cost_unshifted[(u, v)] = c_bar
        arc_cost_unshifted[(v, u)] = c_bar

    obj = gp.LinExpr()
    for a in arcs:
        obj += arc_cost_unshifted[a] * x[a]
    for v in T_p:
        obj += vertex_cost[v] * y[v]

    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraint (25): Quota constraint ---
    model.addConstr(
        gp.quicksum(vertex_profit[v] * y[v] for v in T_p) >= Q,
        name="quota"
    )

    # --- Constraint (26): Flow balance ---
    # Build adjacency lists for incoming and outgoing arcs
    delta_minus = {v: [] for v in all_nodes}  # incoming arcs
    delta_plus = {v: [] for v in all_nodes}   # outgoing arcs
    for a in arcs:
        delta_minus[a[1]].append(a)
        delta_plus[a[0]].append(a)

    T_p_set = set(T_p)
    T_f_set = set(T_f)

    for v in all_nodes:
        flow_in = gp.quicksum(f[a] for a in delta_minus[v])
        flow_out = gp.quicksum(f[a] for a in delta_plus[v])

        if v == r:
            # Root node: no explicit constraint (it is the source of flow)
            # The root sends flow out; its balance is determined implicitly.
            continue
        elif v in T_f_set and v != r:
            # Fixed terminal (not root): must receive 1 unit of flow (eq. 26)
            model.addConstr(flow_in - flow_out == 1, name=f"flow_balance_{v}")
        elif v in T_p_set:
            # Potential terminal: receives flow = y_v (eq. 26)
            model.addConstr(flow_in - flow_out == y[v], name=f"flow_balance_{v}")
        else:
            # Steiner node: flow conservation (eq. 26)
            model.addConstr(flow_in - flow_out == 0, name=f"flow_balance_{v}")

    # --- Constraint (27): x_a <= y_v for all incoming arcs of potential terminals ---
    for v in T_p:
        for a in delta_minus[v]:
            model.addConstr(x[a] <= y[v], name=f"link_arc_terminal_{a[0]}_{a[1]}")

    # --- Constraint (28): f_a <= M * x_a ---
    for a in arcs:
        model.addConstr(f[a] <= M * x[a], name=f"flow_cap_{a[0]}_{a[1]}")

    # Optimize
    model.optimize()

    # Extract solution
    solution = {
        "instance_name": instance_data.get("instance_name", "unknown"),
        "solver": "Gurobi",
        "formulation": "FLOW (flow-based MIP)",
        "status": None,
        "objective_value": None,
        "selected_turbines": [],
        "selected_edges": [],
        "total_edge_cost": 0.0,
        "total_vertex_cost": 0.0,
        "total_profit": 0.0,
    }

    if model.SolCount > 0:
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        solution["objective_value"] = model.ObjVal

        # Extract selected turbines
        for v in T_p:
            if y[v].X > 0.5:
                solution["selected_turbines"].append(v)
                solution["total_vertex_cost"] += vertex_cost[v]
                solution["total_profit"] += vertex_profit[v]

        # Extract selected arcs (as undirected edges)
        selected_arcs = []
        for a in arcs:
            if x[a].X > 0.5:
                selected_arcs.append({"from": a[0], "to": a[1], "cost": arc_cost_unshifted[a]})
                solution["total_edge_cost"] += arc_cost_unshifted[a]

        solution["selected_edges"] = selected_arcs

        # Compute the original cost C(S) and scenic impact L(S)
        C_S = 0.0
        L_S = 0.0
        for a_info in selected_arcs:
            u, v_node = a_info["from"], a_info["to"]
            # Find the original undirected edge
            for e in edges:
                if (e["from"] == u and e["to"] == v_node) or (e["from"] == v_node and e["to"] == u):
                    C_S += e["cost"]
                    L_S += e["scenic_impact"]
                    break
        for v in solution["selected_turbines"]:
            n = node_by_id[v]
            C_S += n["cost"]
            L_S += n["scenic_impact"]

        solution["cost_C"] = C_S
        solution["scenic_impact_L"] = L_S
        solution["alpha"] = alpha

    else:
        solution["status"] = "infeasible_or_no_solution"
        solution["objective_value"] = None

    # Add solver statistics
    solution["solver_stats"] = {
        "runtime_seconds": model.Runtime,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "node_count": int(model.NodeCount),
        "status_code": model.Status,
    }

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve the Quota Steiner Tree Problem (QSTP) using Gurobi's flow-based MIP formulation."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the QSTP instance.",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path where the solution JSON file will be written.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="Maximum solver runtime in seconds.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    instance_data = load_instance(args.instance_path)

    # Solve
    solution = build_and_solve(instance_data, args.time_limit)

    # Write solution
    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.4f}")
        print(f"Status: {solution['status']}")
        print(f"Selected turbines: {solution['selected_turbines']}")
        print(f"Total profit: {solution['total_profit']:.4f}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
