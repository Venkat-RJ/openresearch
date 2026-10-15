#!/usr/bin/env python3
"""
Gurobi implementation of the MNPFSP (Mixed No-Idle Permutation Flowshop Scheduling Problem).

Based on: Ruiz et al. (2007), "The Mixed No-Idle Permutation Flowshop Scheduling Problem".

This implements the MILP formulation for minimizing makespan in a permutation flowshop
where some machines are designated as "no-idle" (no idle time between consecutive jobs).

Decision variables:
  - X[j,k] binary: 1 if job j is assigned to position k in the sequence
  - C[i,k] >= 0: completion time of the job in position k on machine i
  - Cmax: makespan (= C[m-1, n-1])

Constraints:
  (1) Each job assigned to exactly one position
  (2) Each position assigned to exactly one job
  (3) Completion time base: C[0,k] >= sum_j X[j,k]*p[0,j]
  (4) Machine precedence: C[i,k] >= C[i-1,k] + sum_j X[j,k]*p[i,j]
  (5) Sequence precedence (no-idle machines use equality, regular use inequality):
      C[i,k] (>=|=) C[i,k-1] + sum_j X[j,k]*p[i,j]
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
        pass# =============================================================================
# DATA LOADING
# =============================================================================

def load_instance(path):
    """Load an MNPFSP instance from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# =============================================================================
# MAIN MODEL
# =============================================================================

def solve_mnpfsp(instance_path, solution_path, time_limit):
    """Build and solve the MNPFSP MILP model using Gurobi."""
    data = load_instance(instance_path)

    n = data['n_jobs']
    m = data['n_machines']
    no_idle = data['no_idle_flags']  # length m, True if machine i is no-idle
    pt = data['processing_times']    # pt[j][i] = processing time of job j on machine i

    # p[i][j] = processing time on machine i of job j
    p = [[pt[j][i] for j in range(n)] for i in range(m)]

    print(f"MNPFSP Instance: n={n} jobs, m={m} machines")
    print(f"No-idle machines: {[i for i in range(m) if no_idle[i]]}")
    print(f"Time limit: {time_limit} seconds")

    model = gp.Model("MNPFSP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # -------------------------------------------------------------------------
    # Decision variables
    # -------------------------------------------------------------------------

    # X[j,k]: binary, 1 if job j is assigned to position k
    X = {}
    for j in range(n):
        for k in range(n):
            X[j, k] = model.addVar(vtype=GRB.BINARY, name=f"X_{j}_{k}")

    # C[i,k]: completion time of job in position k on machine i
    C = {}
    for i in range(m):
        for k in range(n):
            C[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"C_{i}_{k}")

    model.update()

    # -------------------------------------------------------------------------
    # Objective: minimize makespan = C[m-1, n-1]
    # -------------------------------------------------------------------------
    model.setObjective(C[m - 1, n - 1], GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Constraint (1): Each job occupies exactly one position
    # -------------------------------------------------------------------------
    for j in range(n):
        model.addConstr(
            gp.quicksum(X[j, k] for k in range(n)) == 1,
            name=f"job_{j}_one_pos"
        )

    # -------------------------------------------------------------------------
    # Constraint (2): Each position has exactly one job
    # -------------------------------------------------------------------------
    for k in range(n):
        model.addConstr(
            gp.quicksum(X[j, k] for j in range(n)) == 1,
            name=f"pos_{k}_one_job"
        )

    # -------------------------------------------------------------------------
    # Constraint (3): Base completion time on first machine
    # C[0,k] >= sum_j X[j,k]*p[0][j]  for all k
    # -------------------------------------------------------------------------
    for k in range(n):
        model.addConstr(
            C[0, k] >= gp.quicksum(X[j, k] * p[0][j] for j in range(n)),
            name=f"base_m0_k{k}"
        )

    # -------------------------------------------------------------------------
    # Constraint (4): Machine precedence
    # C[i,k] >= C[i-1,k] + sum_j X[j,k]*p[i][j]  for i=1..m-1, k=0..n-1
    # -------------------------------------------------------------------------
    for i in range(1, m):
        for k in range(n):
            model.addConstr(
                C[i, k] >= C[i - 1, k] + gp.quicksum(X[j, k] * p[i][j] for j in range(n)),
                name=f"mach_prec_i{i}_k{k}"
            )

    # -------------------------------------------------------------------------
    # Constraint (5): Sequence precedence
    # For k=1..n-1, i=0..m-1:
    #   If no-idle: C[i,k] = C[i,k-1] + sum_j X[j,k]*p[i][j]  (equality)
    #   If regular: C[i,k] >= C[i,k-1] + sum_j X[j,k]*p[i][j]  (inequality)
    # -------------------------------------------------------------------------
    for i in range(m):
        for k in range(1, n):
            proc_sum = gp.quicksum(X[j, k] * p[i][j] for j in range(n))
            if no_idle[i]:
                model.addConstr(
                    C[i, k] == C[i, k - 1] + proc_sum,
                    name=f"seq_noidle_i{i}_k{k}"
                )
            else:
                model.addConstr(
                    C[i, k] >= C[i, k - 1] + proc_sum,
                    name=f"seq_regular_i{i}_k{k}"
                )

    model.update()

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    result = {"problem": "MNPFSP", "instance": instance_path}

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["gap"] = model.MIPGap if hasattr(model, 'MIPGap') else 0.0

        # Extract job sequence from X variables
        sequence = [None] * n
        for k in range(n):
            for j in range(n):
                if X[j, k].X > 0.5:
                    sequence[k] = j
                    break

        result["sequence"] = sequence

        # Extract completion times for verification
        completion_times = {}
        for i in range(m):
            completion_times[f"machine_{i}"] = [C[i, k].X for k in range(n)]
        result["completion_times"] = completion_times
    else:
        result["objective_value"] = None
        result["status"] = "infeasible_or_no_solution"

    result["solve_time"] = model.Runtime

    # Write solution
    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"\nSolution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status']}")
        print(f"Sequence: {result['sequence']}")

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve MNPFSP using Gurobi MILP")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json",
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_mnpfsp(args.instance_path, args.solution_path, args.time_limit)
