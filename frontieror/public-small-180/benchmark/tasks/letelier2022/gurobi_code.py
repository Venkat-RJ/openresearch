"""
BPPTL (Bin Packing Problem with Time Lags) - Compact Formulation (Model 2)
from Letelier et al. (2022), solved with Gurobi.

Minimize the total number of non-empty bin-period pairs subject to:
  - Each real item assigned to exactly one bin in one period
  - Bin capacity respected per (bin, period) pair
  - Time lag constraints between items
"""

import argparse
import json
import math
import sys
import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
import time
_GUROBI_CODE_START_TIME = time.time()
# Walk up from this file's directory to find repo root (containing scripts/).
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
        return json.load(f)


def solve_bpptl(instance, time_limit=None):
    params = instance["parameters"]
    T = params["T"]  # number of periods (1..T)
    W = params["W"]  # bin capacity
    L_param = params["L"]  # max bins per period ("infinity" or int)

    source_id = instance["source_item_id"]
    sink_id = instance["sink_item_id"]

    # Collect real items
    real_items = [it for it in instance["items"] if it["type"] == "treatment"]
    n = len(real_items)
    item_ids = [it["item_id"] for it in real_items]
    weights = {it["item_id"]: it["weight"] for it in real_items}

    # Number of bins: if L is "infinity", use n as upper bound
    if L_param == "infinity":
        num_bins = n
    else:
        num_bins = int(L_param)

    periods = list(range(1, T + 1))  # 1..T
    bins = list(range(num_bins))

    # Build arc list indexed by item ids (including source/sink)
    arcs = instance["arcs"]

    # ---------------------------------------------------------------
    # Preprocessing: Floyd-Warshall for longest paths
    # ---------------------------------------------------------------
    all_node_ids = [it["item_id"] for it in instance["items"]]
    id_set = set(all_node_ids)

    # Initialize distance matrix with -infinity (no path)
    INF = float("-inf")
    dist = {i: {j: INF for j in all_node_ids} for i in all_node_ids}
    for i in all_node_ids:
        dist[i][i] = 0

    for arc in arcs:
        u, v, lag = arc["from"], arc["to"], arc["lag"]
        if lag > dist[u][v]:
            dist[u][v] = lag

    # Floyd-Warshall: longest paths
    for k in all_node_ids:
        for i in all_node_ids:
            if dist[i][k] == INF:
                continue
            for j in all_node_ids:
                if dist[k][j] == INF:
                    continue
                candidate = dist[i][k] + dist[k][j]
                if candidate > dist[i][j]:
                    dist[i][j] = candidate

    # Earliest start and latest start for each real item
    # Source is at time 0, sink at time T+1
    es = {}
    ls = {}
    for i in item_ids:
        # es(i) = d(source, i)  (longest path from source to i)
        d_source_i = dist[source_id][i]
        es[i] = max(1, d_source_i) if d_source_i != INF else 1

        # ls(i) = (T+1) - d(i, sink)
        d_i_sink = dist[i][sink_id]
        ls[i] = (T + 1) - d_i_sink if d_i_sink != INF else T

        # Clamp to valid range
        es[i] = max(1, es[i])
        ls[i] = min(T, ls[i])

    # Feasible periods for each item
    feasible_periods = {}
    for i in item_ids:
        feasible_periods[i] = [p for p in periods if es[i] <= p <= ls[i]]

    # ---------------------------------------------------------------
    # Build Gurobi model
    # ---------------------------------------------------------------
    model = gp.Model("BPPTL_Model2")
    model.setParam("Threads", 1)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Variables: x[i,b,p] binary
    x = {}
    for i in item_ids:
        for b in bins:
            for p in feasible_periods[i]:
                x[i, b, p] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{b}_{p}")

    # Variables: u[b,p] binary
    u = {}
    for b in bins:
        for p in periods:
            u[b, p] = model.addVar(vtype=GRB.BINARY, name=f"u_{b}_{p}")

    model.update()

    # Objective: min sum u[b,p]
    model.setObjective(
        gp.quicksum(u[b, p] for b in bins for p in periods), GRB.MINIMIZE
    )

    # Constraint (2b): each item assigned exactly once
    for i in item_ids:
        model.addConstr(
            gp.quicksum(
                x[i, b, p] for b in bins for p in feasible_periods[i]
            )
            == 1,
            name=f"assign_{i}",
        )

    # Constraint (2c): capacity
    for b in bins:
        for p in periods:
            items_in_period = [i for i in item_ids if p in feasible_periods[i]]
            if items_in_period:
                model.addConstr(
                    gp.quicksum(weights[i] * x[i, b, p] for i in items_in_period)
                    <= W * u[b, p],
                    name=f"cap_{b}_{p}",
                )

    # Constraint (2d): time lag constraints between real items
    # For arcs between two real items: l_{i,j} + p_i <= p_j
    # where p_i = sum_p p * sum_b x[i,b,p]
    real_id_set = set(item_ids)
    for arc in arcs:
        i_arc, j_arc, lag = arc["from"], arc["to"], arc["lag"]
        if i_arc in real_id_set and j_arc in real_id_set:
            # l + sum_p p * sum_b x[i,b,p] <= sum_p p * sum_b x[j,b,p]
            model.addConstr(
                lag
                + gp.quicksum(
                    p * x[i_arc, b, p]
                    for p in feasible_periods[i_arc]
                    for b in bins
                )
                <= gp.quicksum(
                    p * x[j_arc, b, p]
                    for p in feasible_periods[j_arc]
                    for b in bins
                ),
                name=f"lag_{i_arc}_{j_arc}",
            )

    # Symmetry breaking: order bins within each period
    # u[b,p] >= u[b+1,p] to break symmetry among identical bins
    for p in periods:
        for b in range(num_bins - 1):
            model.addConstr(
                u[b, p] >= u[b + 1, p], name=f"sym_{b}_{p}"
            )

    model.update()
    model.optimize()

    # ---------------------------------------------------------------
    # Extract solution
    # ---------------------------------------------------------------
    result = {
        "status": model.Status,
        "status_name": _status_name(model.Status),
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val

        assignments = []
        for i in item_ids:
            for b in bins:
                for p in feasible_periods[i]:
                    if x[i, b, p].X > 0.5:
                        assignments.append(
                            {"item_id": i, "bin": b, "period": p, "weight": weights[i]}
                        )
                        break
                else:
                    continue
                break

        # Build bin-period usage summary
        active_bins = []
        for b in bins:
            for p in periods:
                if u[b, p].X > 0.5:
                    items_here = [
                        a for a in assignments if a["bin"] == b and a["period"] == p
                    ]
                    active_bins.append(
                        {
                            "bin": b,
                            "period": p,
                            "items": [a["item_id"] for a in items_here],
                            "total_weight": sum(a["weight"] for a in items_here),
                        }
                    )

        result["assignments"] = assignments
        result["active_bin_periods"] = active_bins
        result["num_active_bin_periods"] = len(active_bins)

        if model.Status == GRB.OPTIMAL:
            result["optimal"] = True
            result["gap"] = 0.0
        else:
            result["optimal"] = False
            result["gap"] = model.MIPGap
    else:
        result["objective_value"] = None
        result["assignments"] = []
        result["active_bin_periods"] = []
        result["num_active_bin_periods"] = None
        result["optimal"] = False

    if hasattr(model, "Runtime"):
        result["runtime_seconds"] = model.Runtime

    return result


def _status_name(status):
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(status, f"UNKNOWN_{status}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve BPPTL (Model 2) using Gurobi"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True, help="Path to instance JSON file"
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path to output solution JSON file",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=None,
        help="Gurobi time limit in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_bpptl(instance, time_limit=args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status_name']}")
    else:
        print(f"No feasible solution found. Status: {result['status_name']}")


if __name__ == "__main__":
    main()
