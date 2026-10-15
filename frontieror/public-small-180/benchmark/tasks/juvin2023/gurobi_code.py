#!/usr/bin/env python3
"""
Gurobi implementation of the time-indexed MILP for the preemptive Flexible
Job-Shop Scheduling Problem (pFJSSP).

Based on: Juvin, Houssin, Lopez (2023) - "Logic-based Benders decomposition
for the preemptive Flexible Job-Shop Scheduling Problem",
Computers and Operations Research, 152, 106156.

Model from Section 2.1 (constraints 1-12).
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


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def solve_pfjssp(instance, time_limit):
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    jobs = instance["jobs"]

    # -------------------------------------------------------------------------
    # Parse instance data
    # -------------------------------------------------------------------------
    # operations[i] = list of operation dicts for job i
    # eligible[i][j] = list of eligible machine indices
    # proc_time[i][j][m] = processing time of O_{i,j} on machine m
    operations = []
    eligible = []
    proc_time = []
    for job in jobs:
        i = job["job_id"]
        ops = job["operations"]
        operations.append(ops)
        elig_i = []
        pt_i = []
        for op in ops:
            j = op["operation_id"]
            machines = op["eligible_machines"]
            pt_dict = {int(k): v for k, v in op["processing_times"].items()}
            elig_i.append(machines)
            pt_i.append(pt_dict)
        eligible.append(elig_i)
        proc_time.append(pt_i)

    n = [len(operations[i]) for i in range(num_jobs)]  # n_i

    # -------------------------------------------------------------------------
    # Derived parameters
    # -------------------------------------------------------------------------
    # p_min[i][j] = min processing time of O_{i,j}
    p_min = []
    for i in range(num_jobs):
        p_min_i = []
        for j in range(n[i]):
            p_min_i.append(min(proc_time[i][j].values()))
        p_min.append(p_min_i)

    # r_min[i][j] = shortest release date of O_{i,j}
    r_min = []
    for i in range(num_jobs):
        r_min_i = []
        cumsum = 0
        for j in range(n[i]):
            r_min_i.append(cumsum)
            cumsum += p_min[i][j]
        r_min.append(r_min_i)

    # q_min[i][j] = shortest delivery time of O_{i,j}
    q_min = []
    for i in range(num_jobs):
        q_min_i = [0] * n[i]
        cumsum = 0
        for j in range(n[i] - 1, -1, -1):
            q_min_i[j] = cumsum
            cumsum += p_min[i][j]
        q_min.append(q_min_i)

    # -------------------------------------------------------------------------
    # Time horizon h
    # **NOT SPECIFIED IN PAPER** - we use the sum of maximum processing times
    # of all operations as an upper bound on the makespan.
    # -------------------------------------------------------------------------
    h = 0
    for i in range(num_jobs):
        for j in range(n[i]):
            h += max(proc_time[i][j].values())
    # Ensure h >= 1
    h = max(h, 1)

    # Time set H = {1, 2, ..., h} (paper uses 1-indexed)
    H = range(1, h + 1)

    # I_m: set of (i, j) that can be processed on machine m
    I_m = {m: [] for m in range(num_machines)}
    for i in range(num_jobs):
        for j in range(n[i]):
            for m in eligible[i][j]:
                I_m[m].append((i, j))

    # -------------------------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("pFJSSP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, time_limit * 0.05))

    # Decision variables
    # x[i,j,m]: 1 if O_{i,j} assigned to machine m
    x = {}
    for i in range(num_jobs):
        for j in range(n[i]):
            for m in eligible[i][j]:
                x[i, j, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{m}")

    # y[i,j,t]: 1 if O_{i,j} is in process at time t
    y = {}
    for i in range(num_jobs):
        for j in range(n[i]):
            # Only create y variables for feasible time periods
            earliest = r_min[i][j] + 1  # 1-indexed, earliest possible start
            latest = h - q_min[i][j]  # latest possible time period
            for t in range(max(1, earliest), min(h, latest) + 1):
                y[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{t}")

    # z[i,j,m,t]: linearisation variable
    z = {}
    for i in range(num_jobs):
        for j in range(n[i]):
            for m in eligible[i][j]:
                earliest = r_min[i][j] + 1
                latest = h - q_min[i][j]
                for t in range(max(1, earliest), min(h, latest) + 1):
                    z[i, j, m, t] = model.addVar(vtype=GRB.BINARY,
                                                   name=f"z_{i}_{j}_{m}_{t}")

    # C_max: makespan
    C_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C_max")

    model.update()

    # Objective (1): min C_max
    model.setObjective(C_max, GRB.MINIMIZE)

    # Constraint (2): each operation assigned to exactly one eligible machine
    for i in range(num_jobs):
        for j in range(n[i]):
            model.addConstr(
                gp.quicksum(x[i, j, m] for m in eligible[i][j]) == 1,
                name=f"assign_{i}_{j}"
            )

    # Constraint (3): duration - operation processed for EXACTLY its required time
    # NOTE: changed from `>=` to `==` 2026-05-19 — `>=` allowed solver to pad
    # extra y slots beyond min processing time on the assigned machine, which
    # then triggered checker's precedence constraint (4) at early t where the
    # model itself skipped the constraint (because y[i,j+1,t] doesn't exist
    # for t < earliest_start_of_next_op). With `==`, total y slots match
    # exactly, so lhs ≤ max_p is always tight, no slack to overshoot.
    for i in range(num_jobs):
        for j in range(n[i]):
            y_sum = gp.quicksum(y[i, j, t] for t in H if (i, j, t) in y)
            model.addConstr(
                y_sum == gp.quicksum(
                    x[i, j, m] * proc_time[i][j][m] for m in eligible[i][j]
                ),
                name=f"duration_{i}_{j}"
            )

    # Constraint (4): precedence between consecutive operations
    for i in range(num_jobs):
        for j in range(n[i] - 1):  # O_{i,j} not the last operation
            max_p = max(proc_time[i][j].values())
            for t in H:
                if (i, j + 1, t) not in y:
                    continue
                lhs = gp.quicksum(
                    y[i, j, tp] for tp in range(t, h + 1) if (i, j, tp) in y
                )
                model.addConstr(
                    lhs <= max_p * (1 - y[i, j + 1, t]),
                    name=f"prec_{i}_{j}_{t}"
                )

    # Constraints (9)-(11): linearisation of z = x * y
    for i in range(num_jobs):
        for j in range(n[i]):
            for m in eligible[i][j]:
                earliest = r_min[i][j] + 1
                latest = h - q_min[i][j]
                for t in range(max(1, earliest), min(h, latest) + 1):
                    if (i, j, m, t) not in z:
                        continue
                    # (9): z >= x + y - 1
                    model.addConstr(
                        z[i, j, m, t] >= x[i, j, m] + y[i, j, t] - 1,
                        name=f"lin9_{i}_{j}_{m}_{t}"
                    )
                    # (10): z <= x
                    model.addConstr(
                        z[i, j, m, t] <= x[i, j, m],
                        name=f"lin10_{i}_{j}_{m}_{t}"
                    )
                    # (11): z <= y
                    model.addConstr(
                        z[i, j, m, t] <= y[i, j, t],
                        name=f"lin11_{i}_{j}_{m}_{t}"
                    )

    # Constraint (12): at most one operation per machine at each time
    for m_idx in range(num_machines):
        for t in H:
            z_terms = []
            for (i, j) in I_m[m_idx]:
                if (i, j, m_idx, t) in z:
                    z_terms.append(z[i, j, m_idx, t])
            if z_terms:
                model.addConstr(
                    gp.quicksum(z_terms) <= 1,
                    name=f"resource_{m_idx}_{t}"
                )

    # Constraint (6): makespan definition
    # C_max >= (t+1) * sum_i y_{i, n_i, t}
    # Note: the paper states the last operation of job i is O_{i, n_i}.
    # In 0-indexed terms, last operation of job i is index n[i]-1.
    for t in H:
        last_ops = []
        for i in range(num_jobs):
            last_j = n[i] - 1
            if (i, last_j, t) in y:
                last_ops.append(y[i, last_j, t])
        if last_ops:
            model.addConstr(
                C_max >= (t + 1) * gp.quicksum(last_ops),
                name=f"makespan_{t}"
            )

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        # Extract assignment
        assignment = {}
        for i in range(num_jobs):
            for j in range(n[i]):
                for m in eligible[i][j]:
                    if x[i, j, m].X > 0.5:
                        assignment[(i, j)] = m
                        break

        # Extract schedule (time periods each operation is active)
        schedule = {}
        for i in range(num_jobs):
            for j in range(n[i]):
                active_times = []
                for t in H:
                    if (i, j, t) in y and y[i, j, t].X > 0.5:
                        active_times.append(t)
                schedule[(i, j)] = active_times

        result["assignment"] = {f"({i},{j})": m for (i, j), m in assignment.items()}
        result["schedule"] = {
            f"({i},{j})": times for (i, j), times in schedule.items()
        }
        result["status"] = "Optimal" if model.Status == GRB.OPTIMAL else "Feasible"
    else:
        result["objective_value"] = None
        result["status"] = "Infeasible or no solution found"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve pFJSSP using Gurobi (time-indexed MILP)"
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

    instance = load_instance(args.instance_path)
    result = solve_pfjssp(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value (C_max): {result['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
