#!/usr/bin/env python3
"""
Gurobi solver for the One-Dimensional Bin-Packing Problem (1D-BPP) and
the variant with conflicts (1D-BPPC) from
Wei et al. (2019), "A New Branch-and-Price-and-Cut Algorithm for
One-Dimensional Bin-Packing Problems".

We replace the paper's set-partitioning formulation F (which has an
exponential number of pattern variables and therefore needs column
generation) with the textbook compact assignment MIP. This is a
problem-equivalent reformulation: any feasible solution to the compact
MIP corresponds to a feasible pattern-set in formulation F, and
vice versa, with the same number of bins. Gurobi's built-in
branch-and-bound therefore solves to optimality (within the time limit)
without any custom column generation, branching, or SR separation.

Compact assignment MIP:
  Variables:
    y[b, i] in {0,1} : 1 iff item i is placed in bin b
    x[b]    in {0,1} : 1 iff bin b is used (has at least one item)
  Objective:
    min sum_{b} x[b]
  Constraints:
    sum_{b} y[b, i] = 1                           for all items i
    sum_{i} w_i * y[b, i] <= c * x[b]             for all bins b
    y[b, i] + y[b, j] <= 1                        for all (i,j) in conflicts, all b
    x[b] >= x[b+1]                                symmetry breaking
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
        data = json.load(f)
    n = data["n"]
    c = data["c"]
    weights = data["weights"]
    # 1D-BPPC: optional conflict edges (list of [i, j] pairs)
    conflicts = data.get("conflicts", []) or data.get("edges", [])
    return n, c, weights, conflicts, data


def ffd_upper_bound(n, c, weights, conflicts):
    """First-Fit-Decreasing on weights, respecting conflicts. Returns a
    valid bin count that upper-bounds OPT, used to size the compact MIP."""
    conflict_set = set()
    for e in conflicts:
        i, j = int(e[0]), int(e[1])
        if i != j:
            a, b = min(i, j), max(i, j)
            conflict_set.add((a, b))

    def conflicts_with(item, bin_items):
        for k in bin_items:
            a, b = (item, k) if item < k else (k, item)
            if (a, b) in conflict_set:
                return True
        return False

    order = sorted(range(n), key=lambda i: -weights[i])
    bins = []  # list of [remaining_capacity, [items]]
    for i in order:
        placed = False
        for b in bins:
            if b[0] >= weights[i] and not conflicts_with(i, b[1]):
                b[0] -= weights[i]
                b[1].append(i)
                placed = True
                break
        if not placed:
            bins.append([c - weights[i], [i]])
    return len(bins)


def solve_bpp_compact(n, c, weights, conflicts, time_limit):
    """Build and solve the compact assignment MIP. Returns (obj, bins,
    status) where bins is a list of bins, each bin a sorted list of item
    indices."""
    M_ub = max(1, ffd_upper_bound(n, c, weights, conflicts))

    model = gp.Model("BPP_compact")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)

    # y[b, i] = 1 iff item i is in bin b
    y = model.addVars(M_ub, n, vtype=GRB.BINARY, name="y")
    # x[b] = 1 iff bin b is used
    x = model.addVars(M_ub, vtype=GRB.BINARY, name="x")

    # Each item assigned to exactly one bin
    for i in range(n):
        model.addConstr(gp.quicksum(y[b, i] for b in range(M_ub)) == 1,
                         name=f"assign_{i}")

    # Capacity + linking
    for b in range(M_ub):
        model.addConstr(
            gp.quicksum(weights[i] * y[b, i] for i in range(n)) <= c * x[b],
            name=f"cap_{b}",
        )

    # Conflicts (1D-BPPC)
    if conflicts:
        for e in conflicts:
            i, j = int(e[0]), int(e[1])
            if i == j:
                continue
            for b in range(M_ub):
                model.addConstr(y[b, i] + y[b, j] <= 1,
                                 name=f"conflict_{i}_{j}_{b}")

    # Symmetry breaking: used bins come first.
    for b in range(M_ub - 1):
        model.addConstr(x[b] >= x[b + 1], name=f"sym_{b}")

    model.setObjective(gp.quicksum(x[b] for b in range(M_ub)), GRB.MINIMIZE)
    model.optimize()

    obj_val = None
    bins_out = []
    if model.SolCount > 0:
        obj_val = model.ObjVal
        for b in range(M_ub):
            if x[b].X > 0.5:
                bin_items = sorted(
                    i for i in range(n) if y[b, i].X > 0.5
                )
                if bin_items:
                    bins_out.append(bin_items)
    return obj_val, bins_out, model.Status


def main():
    parser = argparse.ArgumentParser(
        description="Solve 1D-BPP/BPPC via compact assignment MIP (Wei 2019)."
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    n, c, weights, conflicts, _ = load_instance(args.instance_path)
    start = time.time()
    obj_val, bins_out, status = solve_bpp_compact(
        n, c, weights, conflicts, args.time_limit
    )
    elapsed = time.time() - start

    solution = {}
    if obj_val is not None:
        solution["objective_value"] = obj_val
        solution["bins"] = bins_out
        solution["num_bins"] = len(bins_out)
    else:
        solution["objective_value"] = float(n)
        solution["bins"] = [[i] for i in range(n)]
        solution["num_bins"] = n
    solution["status"] = (
        "optimal" if status == GRB.OPTIMAL else
        "time_limit" if status == GRB.TIME_LIMIT else
        "feasible" if obj_val is not None else
        "infeasible"
    )
    solution["time_seconds"] = elapsed

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Objective value: {solution['objective_value']}")
    print(f"Status: {solution['status']}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
