"""
Gurobi implementation for RCPSP/max-π (Resource-Constrained Project Scheduling
Problem with Partially Renewable Resources and General Temporal Constraints).

From: Watermeyer & Zimmermann (2022), OR Spectrum 44:575-602

The paper does NOT provide an explicit time-indexed ILP formulation.
**INFERRED ASSUMPTION**: We use a standard time-indexed binary LP formulation
where binary variables x_{i,t} indicate whether activity i starts at time t.
This is the natural linearization of the compact model given in the paper.
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


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    n_total = data["num_activities_total"]  # |V| = n + 2
    d_bar = data["max_project_duration"]

    # Parse activities
    activities = {a["id"]: a for a in data["activities"]}
    activity_ids = sorted(activities.keys())
    p = {a_id: activities[a_id]["processing_time"] for a_id in activity_ids}

    # Source and sink
    source = 0
    sink = n_total - 1  # n+1

    # Temporal constraints
    temporal_constraints = data["temporal_constraints"]

    # Resources
    resources = data["resources"]

    # -------------------------------------------------------------------------
    # Precompute resource usage r^u_{ik}(t) for each activity i, resource k,
    # and possible start time t.
    # r^u_{ik}(t) = |{t' in Pi_k : t < t' <= t + p_i}|
    # i.e., the number of periods in Pi_k during which activity i is executing
    # if it starts at time t.
    # -------------------------------------------------------------------------
    Pi_k = {}  # resource_id -> set of periods
    R_k = {}   # resource_id -> capacity
    r_d = {}   # (activity_id, resource_id) -> demand per period
    for res in resources:
        kid = res["resource_id"]
        Pi_k[kid] = set(res["Pi_k"])
        R_k[kid] = res["capacity"]
        for aid_str, demand in res["demands"].items():
            r_d[(int(aid_str), kid)] = demand

    def resource_usage(i, k, t):
        """Number of periods in Pi_k during which activity i executes if started at t."""
        pi = p[i]
        if pi == 0:
            return 0
        count = 0
        for tp in Pi_k[k]:
            if t < tp <= t + pi:
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Compute earliest/latest start times via longest-path (Bellman-Ford)
    # on the temporal constraint graph to tighten variable domains.
    # -------------------------------------------------------------------------
    # Build adjacency for longest path (S_j - S_i >= delta_ij  =>  dist[j] >= dist[i] + delta_ij)
    adj = {a_id: [] for a_id in activity_ids}
    for tc in temporal_constraints:
        i, j, d = tc["from_activity"], tc["to_activity"], tc["time_lag"]
        adj[i].append((j, d))

    # Bellman-Ford for earliest start times (longest paths from source)
    ES = {a_id: 0 for a_id in activity_ids}
    ES[source] = 0
    for _ in range(n_total):
        updated = False
        for tc in temporal_constraints:
            i, j, d = tc["from_activity"], tc["to_activity"], tc["time_lag"]
            if ES[i] + d > ES[j]:
                ES[j] = ES[i] + d
                updated = True
        if not updated:
            break

    # Bellman-Ford for latest start times (reverse: longest paths to sink, then invert)
    # S_j - S_i >= delta_ij  =>  S_i <= S_j - delta_ij
    # We compute latest starts by setting LS[sink] = d_bar and propagating backwards.
    LS = {a_id: d_bar for a_id in activity_ids}
    LS[source] = 0
    # Reverse edges: from S_j - S_i >= delta_ij, we get S_i <= S_j - delta_ij
    for _ in range(n_total):
        updated = False
        for tc in temporal_constraints:
            i, j, d = tc["from_activity"], tc["to_activity"], tc["time_lag"]
            if LS[j] - d < LS[i]:
                LS[i] = LS[j] - d
                updated = True
        if not updated:
            break

    # Clamp to valid range
    for a_id in activity_ids:
        ES[a_id] = max(ES[a_id], 0)
        LS[a_id] = min(LS[a_id], d_bar - p[a_id])
        # Ensure LS >= ES
        if LS[a_id] < ES[a_id]:
            LS[a_id] = ES[a_id]

    # -------------------------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("RCPSP_max_pi")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Binary variables: x[i,t] = 1 if activity i starts at time t
    x = {}
    for i in activity_ids:
        for t in range(ES[i], LS[i] + 1):
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    model.update()

    # Constraint: each activity starts exactly once
    for i in activity_ids:
        model.addConstr(
            gp.quicksum(x[i, t] for t in range(ES[i], LS[i] + 1)) == 1,
            name=f"start_once_{i}"
        )

    # Start time expressions: S_i = sum_t t * x[i,t]
    S = {}
    for i in activity_ids:
        S[i] = gp.quicksum(t * x[i, t] for t in range(ES[i], LS[i] + 1))

    # Constraint: S_0 = 0
    model.addConstr(S[source] == 0, name="source_start")

    # Temporal constraints: S_j - S_i >= delta_ij
    for tc in temporal_constraints:
        i, j, d = tc["from_activity"], tc["to_activity"], tc["time_lag"]
        model.addConstr(S[j] - S[i] >= d, name=f"temp_{i}_{j}")

    # Resource constraints (partially renewable):
    # For each resource k: sum_i sum_t x[i,t] * r^u_{ik}(t) * r^d_{ik} <= R_k
    for res in resources:
        kid = res["resource_id"]
        expr = gp.LinExpr()
        for i in activity_ids:
            demand = r_d.get((i, kid), 0)
            if demand == 0:
                continue
            for t in range(ES[i], LS[i] + 1):
                ru = resource_usage(i, kid, t)
                if ru > 0:
                    expr.addTerms(ru * demand, x[i, t])
        model.addConstr(expr <= R_k[kid], name=f"res_{kid}")

    # Objective: minimize makespan (start time of sink activity)
    makespan = model.addVar(vtype=GRB.INTEGER, name="makespan", lb=ES[sink], ub=LS[sink])
    model.addConstr(makespan == S[sink], name="makespan_def")
    model.setObjective(makespan, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        schedule = {}
        for i in activity_ids:
            for t in range(ES[i], LS[i] + 1):
                if x[i, t].X > 0.5:
                    schedule[str(i)] = t
                    break
        result["schedule"] = schedule
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
    else:
        result["objective_value"] = None
        result["schedule"] = {}
        result["status"] = "infeasible"

    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status']}")
    else:
        print("No feasible solution found.")


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for RCPSP/max-pi"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
