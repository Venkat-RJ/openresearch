"""
Gurobi implementation of the Multistage Stochastic Self-Scheduling Unit Commitment (MSS)
Pan & Guan (2016), "Strong Formulations for the Multistage Stochastic
Self-Scheduling Unit Commitment," Operations Research 64(6):1482-1498.

Implements the deterministic equivalent MILP (formulation MSS, equations 1a-1i + 2).
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


def build_tree_data(nodes_list):
    """Parse node list into convenient dicts."""
    nodes = {}
    for n in nodes_list:
        nid = n["id"]
        nodes[nid] = {
            "time": n["time"],
            "parent": n["parent_id"],
            "children": n["children_ids"],
            "prob": n["probability"],
            "price": n["electricity_price_dollars_per_MWh"],
        }
    return nodes


def get_path_to_root(node_id, nodes):
    """Return list of node ids from root to node_id (inclusive)."""
    path = []
    cur = node_id
    while cur is not None:
        path.append(cur)
        cur = nodes[cur]["parent"]
    path.reverse()
    return path


def get_descendants(node_id, nodes):
    """Return all descendants of node_id including itself."""
    result = []
    stack = [node_id]
    while stack:
        cur = stack.pop()
        result.append(cur)
        stack.extend(nodes[cur]["children"])
    return result


def get_H_r(node_i, r, nodes):
    """
    H_r(i) = { k in descendants(i) : 0 <= t(k) - t(i) <= r - 1 }
    Used for min-up/down time constraints.
    """
    t_i = nodes[node_i]["time"]
    desc = get_descendants(node_i, nodes)
    return [k for k in desc if 0 <= nodes[k]["time"] - t_i <= r - 1]


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for MSS Unit Commitment (Pan & Guan 2016)"
    )
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", required=True, help="Path for solution JSON output")
    parser.add_argument("--time_limit", type=int, required=True, help="Solver time limit (seconds)")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    with open(args.instance_path) as f:
        inst = json.load(f)

    gen = inst["generator"]
    C_lo = gen["C_lower_MW"]
    C_up = gen["C_upper_MW"]
    V_plus = gen["V_plus_MW_per_h"]
    V_minus = gen["V_minus_MW_per_h"]
    U_bar = gen["U_bar_startup_cost_dollars"]
    # NOTE: U_underline (shutdown cost) not specified separately in paper Table 1.
    # INFERRED ASSUMPTION: set equal to U_bar (startup cost) as a symmetric default.
    U_under = gen.get("U_lower_shutdown_cost_dollars", U_bar)
    fuel_a = gen["fuel_a_dollars_per_h"]
    fuel_b = gen["fuel_b_dollars_per_MWh"]

    tree_cfg = inst["scenario_tree"]
    L = tree_cfg["L_min_up_time"]
    ell = tree_cfg["ell_min_down_time"]

    # Initial conditions
    # INFERRED ASSUMPTION: root node treated as a free decision variable;
    # initial_generator_status_y0 and initial_generation_x0_MW from instance
    # are used to fix the root's initial state before period 1.
    # The root node IS period 1, so there is no "period 0" node in the tree.
    # We interpret y0_init and x0_init as the state BEFORE the root node
    # (i.e., the state at period 0, used to compute ramp/startup at root).
    y0_init = tree_cfg.get("initial_generator_status_y0", 0)
    x0_init = tree_cfg.get("initial_generation_x0_MW", 0.0)

    nodes = build_tree_data(tree_cfg["nodes"])
    all_nodes = list(nodes.keys())
    # Non-root nodes (those with a parent)
    non_root = [i for i in all_nodes if nodes[i]["parent"] is not None]

    # f(x_i) = fuel_a * y_i + fuel_b * x_i (linear approximation)
    # fuel_a is fixed cost per period online, fuel_b is variable cost per MWh
    # INFERRED ASSUMPTION: piecewise linear fuel cost not specified in paper;
    # use linear approximation f(x) = fuel_a + fuel_b * x when online (fuel_a * y_i + fuel_b * x_i).

    model = gp.Model("MSS_UC")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("MIPGap", 1e-4)  # 0.01% optimality gap as in paper
    # Solver hints: l11 + l41 were 1h TLE no-incumbent in prior runs (1M-node
    # scenario tree → ~1M binary y vars, weak LP relaxation). MIPFocus=1
    # prioritizes finding any feasible incumbent over closing the gap, and
    # NoRelHeurTime gives gurobi a root-node heuristic budget. Other 3
    # instances (l21/l31/l51) already produce results; expected minor
    # slowdown is acceptable.
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, args.time_limit * 0.05))

    # Decision variables
    x = model.addVars(all_nodes, lb=0.0, name="x")
    y = model.addVars(all_nodes, vtype=GRB.BINARY, name="y")
    u = model.addVars(non_root, vtype=GRB.BINARY, name="u")

    # Objective: max sum_i p_i * (q_i*x_i - U_bar*u_i - U_under*(y_{i^-} - y_i + u_i) - f(x_i))
    # f(x_i) = fuel_a * y_i + fuel_b * x_i
    # For root node (no u_0): no startup/shutdown cost at root (v_0 undefined)
    # INFERRED ASSUMPTION: root node has no startup/shutdown cost since u_0 doesn't exist.
    obj = gp.LinExpr()
    for i in all_nodes:
        p_i = nodes[i]["prob"]
        q_i = nodes[i]["price"]
        par = nodes[i]["parent"]
        revenue = p_i * q_i
        fuel = p_i * fuel_b
        fixed = p_i * fuel_a
        obj += revenue * x[i] - fuel * x[i] - fixed * y[i]
        if par is not None:
            obj -= p_i * U_bar * u[i]
            obj -= p_i * U_under * (y[par] - y[i] + u[i])
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint (1f): C_lo * y_i <= x_i <= C_up * y_i
    for i in all_nodes:
        model.addConstr(x[i] >= C_lo * y[i], name=f"gen_lb_{i}")
        model.addConstr(x[i] <= C_up * y[i], name=f"gen_ub_{i}")

    # Constraint (1d): y_i - y_{i^-} <= u_i
    for i in non_root:
        par = nodes[i]["parent"]
        model.addConstr(y[i] - y[par] <= u[i], name=f"startup_def_{i}")

    # Constraint (2): u_i <= y_i and u_i <= 1 - y_{i^-}
    for i in non_root:
        par = nodes[i]["parent"]
        model.addConstr(u[i] <= y[i], name=f"u_le_y_{i}")
        model.addConstr(u[i] <= 1 - y[par], name=f"u_le_1my_par_{i}")

    # Constraint (1b): min-up time
    # y_i - y_{i^-} <= y_k  for all k in H_L(i)
    for i in non_root:
        par = nodes[i]["parent"]
        H = get_H_r(i, L, nodes)
        for k in H:
            model.addConstr(y[i] - y[par] <= y[k], name=f"minup_{i}_{k}")

    # Constraint (1c): min-down time
    # y_{i^-} - y_i <= 1 - y_k  for all k in H_ell(i)
    for i in non_root:
        par = nodes[i]["parent"]
        H = get_H_r(i, ell, nodes)
        for k in H:
            model.addConstr(y[par] - y[i] <= 1 - y[k], name=f"mindown_{i}_{k}")

    # Constraint (1g): ramp-up
    # x_i - x_{i^-} <= V^+ * y_{i^-} + C_up * (1 - y_{i^-})
    # Constraint (1h): ramp-down
    # x_{i^-} - x_i <= V^- * y_i + C_up * (1 - y_i)
    for i in non_root:
        par = nodes[i]["parent"]
        if par == 0 and nodes[0]["parent"] is None:
            # par is root node — use root's y variable
            # Ramp-up from root
            model.addConstr(
                x[i] - x[par] <= V_plus * y[par] + C_up * (1 - y[par]),
                name=f"rampup_{i}"
            )
            model.addConstr(
                x[par] - x[i] <= V_minus * y[i] + C_up * (1 - y[i]),
                name=f"rampdown_{i}"
            )
        else:
            model.addConstr(
                x[i] - x[par] <= V_plus * y[par] + C_up * (1 - y[par]),
                name=f"rampup_{i}"
            )
            model.addConstr(
                x[par] - x[i] <= V_minus * y[i] + C_up * (1 - y[i]),
                name=f"rampdown_{i}"
            )

    # Handle initial conditions: ramp constraints between initial state and root node
    # The root node is node 0 (period 1). We treat the initial state (period 0)
    # as a fixed parameter (y0_init, x0_init) not a variable.
    # INFERRED ASSUMPTION: Add ramp constraints from initial state to root.
    root_id = [i for i in all_nodes if nodes[i]["parent"] is None][0]
    # Ramp-up from initial state to root
    model.addConstr(
        x[root_id] - x0_init <= V_plus * y0_init + C_up * (1 - y0_init),
        name="rampup_init"
    )
    # Ramp-down from initial state to root
    model.addConstr(
        x0_init - x[root_id] <= V_minus * y[root_id] + C_up * (1 - y[root_id]),
        name="rampdown_init"
    )
    # Min-up/down from initial state: if generator was on before root,
    # enforce it stays on for first L-1 periods from root (approximation).
    # INFERRED ASSUMPTION: initial state constraints not fully specified; only basic ramp applied.

    model.optimize()

    # Collect solution
    solution = {}
    status = model.Status

    if model.SolCount > 0:
        obj_val = model.ObjVal
        # Sparse certificate encoding: omitted entries are exactly zero (as
        # documented in solution_schema.json and interpreted by the checker).
        # Dense maps on the released million-node trees exceed 50 MB each.
        x_sol = {str(i): x[i].X for i in all_nodes if abs(x[i].X) > 1e-9}
        y_sol = {str(i): 1 for i in all_nodes if y[i].X > 0.5}
        u_sol = {str(i): 1 for i in non_root if u[i].X > 0.5}
        solution = {
            "objective_value": obj_val,
            "status": status,
            "mip_gap": model.MIPGap if status not in [GRB.OPTIMAL] else 0.0,
            "x": x_sol,
            "y": y_sol,
            "u": u_sol,
        }
    else:
        solution = {
            "objective_value": None,
            "status": status,
            "message": "No feasible solution found within time limit.",
        }

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Objective value: {solution['objective_value']}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
