r"""
Gurobi implementation of the Set Covering ILP for Pm||sum w_j C_j
from Kowalczyk & Leus (2018), "A Branch-and-Price Algorithm for Parallel
Machine Scheduling Using ZDDs and Generic Branching",
INFORMS Journal on Computing 30(4):768-782.

The problem is modeled as a set covering formulation:
  min  sum_{s in F} c_s * lambda_s                          (6a)
  s.t. sum_{s in F: j in s} lambda_s >= 1   for all j in J  (6b)
       sum_{s in F} lambda_s <= m                            (6c)
       lambda_s in {0,1}                    for all s in F   (6d)

where F is the set of all feasible single-machine schedules, and c_s is
the total weighted completion time of the jobs in schedule s when sequenced
in SWPT (nonincreasing w_j/p_j) order.

A schedule s is feasible if:
  - Jobs are sequenced in SWPT order
  - r_j + p_j <= C_j(s) <= d_bar_j for all j in s
  - H_min <= makespan(s) <= H_max

Release dates r_j and deadlines d_bar_j are tightened using Properties 3-6
from the paper (Belouadah & Potts 1994; Azizoglu & Kirca 1999;
Elmaghraby & Park 1974).

For n=20, all 2^20 subsets are enumerated with feasibility pruning.
"""

import json
import argparse
import math
from itertools import combinations

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
        data = json.load(f)
    return data


def preprocess(jobs, m):
    """
    Sort jobs by SWPT rule and compute release dates r_j, deadlines d_bar_j,
    and global bounds H_min, H_max.

    Returns (sorted_jobs, r, d_bar, H_min, H_max) where sorted_jobs is a list
    of dicts with keys 'j_orig', 'p', 'w', and r/d_bar are lists indexed by
    position in sorted order.
    """
    n = len(jobs)

    # Sort in nonincreasing order of w_j / p_j (SWPT rule).
    # Ties broken by nondecreasing p_j (jobs with smaller p first).
    sorted_jobs = sorted(jobs, key=lambda j: (-j["w"] / j["p"], j["p"]))

    p = [j["p"] for j in sorted_jobs]
    w = [j["w"] for j in sorted_jobs]

    total_p = sum(p)
    p_max = max(p)

    # Sorted processing times (ascending) for selecting m-1 largest
    p_sorted_asc = sorted(p)

    # Property 3: H_max = sum(p)/m + (m-1)/m * p_max
    H_max_real = total_p / m + (m - 1) / m * p_max
    H_max = math.ceil(H_max_real)

    # Property 4: H_min = (1/m) * (sum(p) - sum of (m-1) largest p)
    # The (m-1) largest processing times are the last (m-1) in ascending sort
    sum_m1_largest = sum(p_sorted_asc[-(m - 1):]) if m > 1 else 0
    H_min_real = (total_p - sum_m1_largest) / m
    H_min = math.ceil(H_min_real)

    # Initialize release dates and deadlines
    r = [0] * n
    d_bar = [H_max] * n

    # Precompute prefix sums of processing times in SWPT order
    # prefix_p[k] = sum_{h=0}^{k-1} p[h]  (sum of first k jobs)
    prefix_p = [0] * (n + 1)
    for i in range(n):
        prefix_p[i + 1] = prefix_p[i] + p[i]

    # suffix_p[k] = sum_{h=k}^{n-1} p[h]
    suffix_p = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_p[i] = suffix_p[i + 1] + p[i]

    # Threshold for P^2 and E^2 conditions
    # (1/m) * (sum p + sum of m-1 largest p)
    threshold = (total_p + sum_m1_largest) / m

    # Tighten release dates (Properties 5, 6)
    for j in range(n):
        # P^1_j: {k | (w_k > w_j and p_k <= p_j) or (w_k >= w_j and p_k < p_j)}
        P_j = set()
        for k in range(n):
            if k == j:
                continue
            if (w[k] > w[j] and p[k] <= p[j]) or (w[k] >= w[j] and p[k] < p[j]):
                P_j.add(k)

        # P^2_j: {k | k < j and prefix_p[k] <= threshold - suffix_p[j]}
        # k < j means k has a smaller index in SWPT order (higher priority)
        # sum_{h=1}^{k-1} p_h in paper is prefix_p[k] (0-indexed: sum of first k items)
        # sum_{h=j}^{n} p_h in paper is suffix_p[j]
        rhs_p2 = threshold - suffix_p[j]
        for k in range(j):
            if prefix_p[k] <= rhs_p2:
                P_j.add(k)

        if len(P_j) >= m - 1:
            # rho_j = sum of (|P_j| - m + 1) smallest p_k in P_j
            p_in_P = sorted([p[k] for k in P_j])
            count = len(P_j) - m + 1
            rho_j = sum(p_in_P[:count])
            r[j] = math.ceil(rho_j / m)

    # Tighten deadlines
    for j in range(n):
        # E^1_j: {k | (w_k < w_j and p_k >= p_j) or (w_k <= w_j and p_k > p_j)}
        E_j = set()
        for k in range(n):
            if k == j:
                continue
            if (w[k] < w[j] and p[k] >= p[j]) or (w[k] <= w[j] and p[k] > p[j]):
                E_j.add(k)

        # E^2_j: {k | k > j and prefix_p[k] <= threshold - suffix_p[k]}
        # sum_{h=1}^{k-1} p_h = prefix_p[k], sum_{h=k}^{n} p_h = suffix_p[k]
        for k in range(j + 1, n):
            if prefix_p[k] <= threshold - suffix_p[k]:
                E_j.add(k)

        # delta_j = H_max - ceil((sum_{j' in E_j} p_{j'} + p_j) / m)
        sum_E_p = sum(p[k] for k in E_j)
        delta_j = H_max - math.ceil((sum_E_p + p[j]) / m)
        if delta_j + p[j] < d_bar[j]:
            d_bar[j] = delta_j + p[j]

    return sorted_jobs, p, w, r, d_bar, H_min, H_max


def enumerate_feasible_schedules(n, p, w, r, d_bar, H_min, H_max):
    """
    Enumerate all feasible single-machine schedules using recursive
    enumeration with pruning. Jobs are in SWPT order (indices 0..n-1).

    A schedule is a subset of jobs sequenced in SWPT order. For each
    feasible schedule, we return (job_set_as_frozenset, cost).

    Returns a list of (frozenset_of_job_indices, cost) tuples.
    """
    feasible = []

    # Recursive DFS: at position `pos`, accumulated processing time `t`,
    # current cost `cost_so_far`, current job set `current`.
    # We decide whether to include job `pos` or skip it.
    def dfs(pos, t, cost_so_far, current):
        # We can terminate the schedule here (if current is nonempty)
        if current and H_min <= t <= H_max:
            feasible.append((frozenset(current), cost_so_far))

        if pos >= n:
            return

        # Try including job `pos`
        new_t = t + p[pos]
        # Check feasibility: r[pos] + p[pos] <= C_pos(s) = new_t and new_t <= d_bar[pos]
        if r[pos] + p[pos] <= new_t <= d_bar[pos]:
            # Prune: new_t must not exceed H_max (can't finish schedule legally)
            if new_t <= H_max:
                new_cost = cost_so_far + w[pos] * new_t
                current.append(pos)
                dfs(pos + 1, new_t, new_cost, current)
                current.pop()

        # Try excluding job `pos`
        dfs(pos + 1, t, cost_so_far, current)

    dfs(0, 0, 0, [])
    return feasible


def solve_wct(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    n = data["n"]
    m = data["m"]
    jobs = data["jobs"]

    # Preprocess: sort by SWPT, compute bounds
    sorted_jobs, p, w, r, d_bar, H_min, H_max = preprocess(jobs, m)

    print(f"Instance: n={n}, m={m}")
    print(f"H_min={H_min}, H_max={H_max}")
    print(f"Release dates: {r}")
    print(f"Deadlines: {d_bar}")

    # Enumerate all feasible single-machine schedules
    print("Enumerating feasible schedules...")
    schedules = enumerate_feasible_schedules(n, p, w, r, d_bar, H_min, H_max)
    print(f"Found {len(schedules)} feasible schedules.")

    if len(schedules) == 0:
        solution = {
            "objective_value": None,
            "status_str": "NO_FEASIBLE_SCHEDULES"
        }
        with open(solution_path, "w") as f:
            solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(solution, f, indent=2)
        print("No feasible schedules found.")
        return

    # Build the set covering ILP
    model = gp.Model("WCT_SetCovering")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    # OOM mitigation: spill B&B node data to disk after 2GB RAM. Without
    # this the model OOMed @ 640G within ~26-40 min on large_21/_31 in the
    # 2026-05-03 rerun.
    model.setParam("NodefileStart", 2)
    import os as _o, tempfile as _t
    model.setParam("NodefileDir",
                   _o.environ.get("NODEFILE_DIR") or _t.gettempdir())

    # Decision variables: lambda_s in {0,1} for each schedule s
    lam = {}
    for s_idx, (job_set, cost) in enumerate(schedules):
        lam[s_idx] = model.addVar(
            vtype=GRB.BINARY,
            obj=cost,
            name=f"lam_{s_idx}"
        )

    model.update()

    # Objective: minimize sum c_s * lambda_s (already set via obj= in addVar)
    model.setAttr("ModelSense", GRB.MINIMIZE)

    # Covering constraints (6b): each job must be covered at least once
    for j in range(n):
        # Find all schedules containing job j
        covering_vars = []
        for s_idx, (job_set, cost) in enumerate(schedules):
            if j in job_set:
                covering_vars.append(lam[s_idx])
        if covering_vars:
            model.addConstr(
                gp.quicksum(covering_vars) >= 1,
                name=f"cover_{j}"
            )
        else:
            # Job j is not in any feasible schedule - problem is infeasible
            print(f"Warning: job {j} (orig={sorted_jobs[j]['j']}) not covered "
                  f"by any feasible schedule.")

    # Capacity constraint (6c): at most m machines
    model.addConstr(
        gp.quicksum(lam[s_idx] for s_idx in range(len(schedules))) <= m,
        name="capacity"
    )

    # Optimize
    print("Solving ILP...")
    model.optimize()

    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution = {
            "objective_value": obj_val,
            "status": model.Status,
            "status_str": {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
            }.get(model.Status, f"STATUS_{model.Status}"),
            "n": n,
            "m": m,
            "H_min": H_min,
            "H_max": H_max,
            "num_feasible_schedules": len(schedules),
            "machines": []
        }

        # Report which schedules are selected
        for s_idx in range(len(schedules)):
            if lam[s_idx].X > 0.5:
                job_set, cost = schedules[s_idx]
                # Reconstruct the schedule details
                job_indices = sorted(job_set)
                machine_info = {
                    "schedule_cost": cost,
                    "jobs": []
                }
                cumulative_p = 0
                for ji in job_indices:
                    cumulative_p += p[ji]
                    machine_info["jobs"].append({
                        "j_orig": sorted_jobs[ji]["j"],
                        "p": p[ji],
                        "w": w[ji],
                        "C_j": cumulative_p
                    })
                solution["machines"].append(machine_info)

        with open(solution_path, "w") as f:
            solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(solution, f, indent=2)
        print(f"Solution written to {solution_path}")
        print(f"Objective value: {obj_val}")
    else:
        solution = {
            "objective_value": None,
            "status": model.Status,
            "status_str": "NO_SOLUTION_FOUND"
        }
        with open(solution_path, "w") as f:
            solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(solution, f, indent=2)
        print("No feasible solution found.")


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for Pm||sum w_j C_j (Set Covering Formulation)"
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

    solve_wct(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
