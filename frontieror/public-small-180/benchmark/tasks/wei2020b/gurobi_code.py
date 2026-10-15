#!/usr/bin/env python3
"""
Gurobi solver for the Two-Dimensional Vector Packing Problem (2DVPP).

Paper: Wei, Lai, Lim, Hu (2019) — A branch-and-price algorithm for the
two-dimensional vector packing problem.

We replace the paper's set-partitioning formulation F (which has an
exponential number of pattern variables and therefore needs column
generation) with the textbook compact assignment MIP. The compact MIP
is a problem-equivalent reformulation: any feasible packing in F maps
1:1 to a feasible packing in the compact MIP with the same bin count.
Gurobi's built-in branch-and-bound therefore solves to optimality
within the time limit, without any custom column generation, branching,
or goal-cut machinery.

Compact assignment MIP:
  Variables:
    y[b, i] in {0,1}  : 1 iff item i is placed in bin b
    x[b]    in {0,1}  : 1 iff bin b is used
  Objective:
    min sum_{b} x[b]
  Constraints:
    sum_{b} y[b, i] = 1                            for all items i
    sum_{i} w_i * y[b, i] <= W * x[b]              weight capacity
    sum_{i} v_i * y[b, i] <= V * x[b]              volume capacity
    x[b] >= x[b+1]                                 symmetry breaking
"""

import argparse
import json
import time

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


def load_instance(path):
    with open(path, "r") as f:
        d = json.load(f)
    n = d["n"]
    W = d["W"]
    V = d["V"]
    items = d["items"]
    weights = [it["w"] for it in items]
    volumes = [it["v"] for it in items]
    return n, W, V, weights, volumes


def ffd_2d_upper_bound(n, W, V, weights, volumes):
    """First-Fit-Decreasing on the 2D capacity ratio. Returns an UB on
    the optimal bin count, used to size the compact MIP."""
    order = sorted(range(n),
                   key=lambda i: -max(weights[i] / W, volumes[i] / V))
    bins = []  # list of [w_used, v_used]
    for i in order:
        placed = False
        for b in bins:
            if b[0] + weights[i] <= W and b[1] + volumes[i] <= V:
                b[0] += weights[i]
                b[1] += volumes[i]
                placed = True
                break
        if not placed:
            bins.append([weights[i], volumes[i]])
    return max(1, len(bins))


def solve_2dvpp_compact(n, W, V, weights, volumes, time_limit):
    M_ub = ffd_2d_upper_bound(n, W, V, weights, volumes)

    model = gp.Model("2DVPP_compact")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)

    y = model.addVars(M_ub, n, vtype=GRB.BINARY, name="y")
    x = model.addVars(M_ub, vtype=GRB.BINARY, name="x")

    # Each item assigned to exactly one bin
    for i in range(n):
        model.addConstr(gp.quicksum(y[b, i] for b in range(M_ub)) == 1,
                         name=f"assign_{i}")

    # Capacity + linking
    for b in range(M_ub):
        model.addConstr(
            gp.quicksum(weights[i] * y[b, i] for i in range(n)) <= W * x[b],
            name=f"weight_{b}",
        )
        model.addConstr(
            gp.quicksum(volumes[i] * y[b, i] for i in range(n)) <= V * x[b],
            name=f"volume_{b}",
        )

    # Symmetry breaking: used bins come first.
    for b in range(M_ub - 1):
        model.addConstr(x[b] >= x[b + 1], name=f"sym_{b}")

    model.setObjective(gp.quicksum(x[b] for b in range(M_ub)), GRB.MINIMIZE)
    model.optimize()

    obj = None
    bins_out = []
    if model.SolCount > 0:
        obj = model.ObjVal
        for b in range(M_ub):
            if x[b].X > 0.5:
                items_in_b = sorted(i for i in range(n) if y[b, i].X > 0.5)
                if items_in_b:
                    tw = sum(weights[i] for i in items_in_b)
                    tv = sum(volumes[i] for i in items_in_b)
                    bins_out.append({
                        "items": items_in_b,
                        "total_weight": int(tw),
                        "total_volume": int(tv),
                    })
    return obj, bins_out, model.Status


def main():
    parser = argparse.ArgumentParser(
        description="Solve 2DVPP via compact assignment MIP (Wei 2019)."
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    n, W, V, weights, volumes = load_instance(args.instance_path)
    start = time.time()
    obj, bins_out, status = solve_2dvpp_compact(
        n, W, V, weights, volumes, args.time_limit
    )
    elapsed = time.time() - start

    solution = {
        "objective_value": int(round(obj)) if obj is not None else None,
        "bins": bins_out,
        "status": (
            "optimal" if status == GRB.OPTIMAL else
            "time_limit" if status == GRB.TIME_LIMIT else
            "feasible" if obj is not None else
            "infeasible"
        ),
        "time_seconds": elapsed,
    }

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Objective value: {solution['objective_value']}")
    print(f"Status: {solution['status']}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
