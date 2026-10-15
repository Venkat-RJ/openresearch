#!/usr/bin/env python3
"""
Gurobi implementation of Model I for F|nwt, d_j|C_max
(No-Wait Flow Shop Scheduling with Due Date Constraints)

Source: Samarghandi & Behroozi (2017), Computers and Operations Research 81, pp. 141-159

Model I is a Mixed Integer Programming formulation that works directly with
processing times (does not require pre-computation of the contribution matrix).

Decision variables:
  x_{jk} in {0,1}: 1 if job J_k is placed immediately after job J_j in the sequence
  S_{ij} >= 0    : starting time of i-th operation of job J_j (continuous)
  C_max >= 0     : makespan (continuous)

Objective: minimize C_max

Constraints (equations reference paper numbering):
  (5)  C_max >= S_{mj} + p_{mj}                     for all j
  (6)  S_{ik} + M*(1-x_{jk}) >= S_{ij} + p_{ij}    for all i,j,k  [big-M sequencing]
  (7)  S_{(i+1)j} = S_{ij} + p_{ij}                 for all i<m, j  [no-wait]
  (8)  S_{mj} + p_{mj} <= d_j                        for all j  [due date]
  (9)  sum_j x_{jk} <= 1                             for all k  [at most one predecessor]
  (10) sum_k x_{jk} <= 1                             for all j  [at most one successor]
  (11) x_{jk} + x_{kj} <= 1                          for all j<k  [no bidirectional]
  (12) sum_j sum_k x_{jk} = n-1                       [exactly n-1 links = Hamiltonian path]
  (13) x_{jj} = 0                                    for all j  [no self-loops]
  (14) S_{ij} >= 0
  (15) x_{jk} in {0,1}

NOTE on big-M (Eq. 6): The paper warns that choosing M too large causes CPLEX to
eliminate it in pre-solve. The paper recommends indicator constraints as an alternative
but notes the original big-M formulation performs better overall (Section 6.1).
INFERRED ASSUMPTION: M is set to the sum of all processing times, which is a safe
upper bound on any start time S_{ij} + p_{ij} in a feasible schedule.

NOTE on sub-tour elimination: The time-indexed variables S_{ij} with positive
processing times implicitly prevent sub-tours, so no MTZ-style constraints are needed.
"""

import argparse
import json
import sys
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

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy not found. Install Gurobi and its Python API.")
    sys.exit(1)


def load_instance(path):
    """Load problem instance from JSON file.

    Returns:
        n  : number of jobs
        m  : number of machines
        p  : processing_times[i][j] = p_{i+1, j+1}  (0-indexed; machine i+1, job j+1)
        d  : due_dates[j] = d_{j+1}  (0-indexed)
    """
    with open(path, "r") as f:
        data = json.load(f)
    n = data["n_jobs"]
    m = data["n_machines"]
    p = data["processing_times"]   # p[i][j]  0-indexed
    d = data["due_dates"]          # d[j]      0-indexed
    return n, m, p, d


def compute_makespan(seq_0indexed, n, m, p):
    """Recompute the no-wait flow shop makespan C_max from a job permutation.

    Mirrors the start-time recursion in feasibility_check.py so that the
    reported objective_value equals evaluate(sequence) exactly, eliminating
    MIPGap-induced drift between model.ObjVal (continuous) and the true
    integer-feasible makespan.
    """
    if not seq_0indexed:
        return None
    S = [[0.0] * n for _ in range(m)]
    first = seq_0indexed[0]
    S[0][first] = 0.0
    for i in range(1, m):
        S[i][first] = S[i - 1][first] + p[i - 1][first]
    for pos in range(1, len(seq_0indexed)):
        prev_j = seq_0indexed[pos - 1]
        curr_j = seq_0indexed[pos]
        max_s0 = 0.0
        for i in range(m):
            offset_curr = sum(p[l][curr_j] for l in range(i))
            needed = S[i][prev_j] + p[i][prev_j] - offset_curr
            if needed > max_s0:
                max_s0 = needed
        S[0][curr_j] = max_s0
        for i in range(1, m):
            S[i][curr_j] = S[i - 1][curr_j] + p[i - 1][curr_j]
    return max(S[m - 1][j] + p[m - 1][j] for j in seq_0indexed)


def extract_sequence(x_vals, n):
    """Reconstruct the job permutation from the x variable values.

    Builds a successor map then walks from the unique first job
    (the job with no predecessor) to produce an ordered sequence.
    Returns a 0-indexed list of jobs.
    """
    successor = {}
    for j in range(n):
        for k in range(n):
            if j != k and x_vals.get((j, k), 0) > 0.5:
                successor[j] = k

    # First job = job with no predecessor among those that appear as a successor
    all_jobs = set(range(n))
    jobs_with_predecessor = set(successor.values())
    first_candidates = all_jobs - jobs_with_predecessor

    if not first_candidates:
        # Fallback: arbitrary order (should not happen in a valid solution)
        return list(range(n))

    first = next(iter(first_candidates))
    seq = [first]
    visited = {first}
    cur = first
    while cur in successor:
        nxt = successor[cur]
        if nxt in visited:
            break
        seq.append(nxt)
        visited.add(nxt)
        cur = nxt

    # Append any jobs not reached (safety fallback)
    for j in range(n):
        if j not in visited:
            seq.append(j)

    return seq


def solve(n, m, p, d, time_limit):
    """Build and solve Model I using Gurobi.

    Args:
        n          : number of jobs
        m          : number of machines
        p[i][j]    : processing time (0-indexed: machine i+1, job j+1)
        d[j]       : due date (0-indexed: job j+1)
        time_limit : maximum solver runtime in seconds (int)

    Returns:
        dict with keys: status, objective_value, sequence
    """
    # ----------------------------------------------------------------
    # Big-M: upper bound on any possible S_{ij} + p_{ij}.
    # Sum of ALL processing times is a safe (if loose) upper bound.
    # INFERRED ASSUMPTION: paper only states "M is a sufficiently large
    # number" without giving an exact value (math_model.txt note 10).
    # ----------------------------------------------------------------
    M = sum(p[i][j] for i in range(m) for j in range(n))

    model = gp.Model("NoWaitFlowShop_ModelI")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # Binary assignment: x[j,k] = 1  <=>  J_{k+1} immediately follows J_{j+1}
    x = {}
    for j in range(n):
        for k in range(n):
            x[j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j+1}_{k+1}")

    # Starting times: S[i,j] = starting time of (i+1)-th operation of J_{j+1}
    S = {}
    for i in range(m):
        for j in range(n):
            S[i, j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                   name=f"S_{i+1}_{j+1}")

    # Makespan
    C_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

    model.update()

    # ------------------------------------------------------------------
    # Objective: minimize C_max  (Eq. 4)
    # ------------------------------------------------------------------
    model.setObjective(C_max, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Constraint (5): C_max >= S_{mj} + p_{mj}  for j = 1,...,n
    # ------------------------------------------------------------------
    for j in range(n):
        model.addConstr(
            C_max >= S[m - 1, j] + p[m - 1][j],
            name=f"Cmax_lb_{j+1}"
        )

    # ------------------------------------------------------------------
    # Constraint (6): S_{ik} + M*(1-x_{jk}) >= S_{ij} + p_{ij}
    # for i=1,...,m;  j,k=1,...,n  (j != k, since x_{jj}=0)
    # ------------------------------------------------------------------
    for i in range(m):
        for j in range(n):
            for k in range(n):
                if j != k:
                    model.addConstr(
                        S[i, k] + M * (1 - x[j, k]) >= S[i, j] + p[i][j],
                        name=f"Seq_{i+1}_{j+1}_{k+1}"
                    )

    # ------------------------------------------------------------------
    # Constraint (7): S_{(i+1)j} = S_{ij} + p_{ij}   (no-wait)
    # for i=1,...,m-1;  j=1,...,n
    # ------------------------------------------------------------------
    for i in range(m - 1):
        for j in range(n):
            model.addConstr(
                S[i + 1, j] == S[i, j] + p[i][j],
                name=f"NoWait_{i+1}_{j+1}"
            )

    # ------------------------------------------------------------------
    # Constraint (8): S_{mj} + p_{mj} <= d_j   (due date)
    # for j = 1,...,n
    # ------------------------------------------------------------------
    for j in range(n):
        model.addConstr(
            S[m - 1, j] + p[m - 1][j] <= d[j],
            name=f"DueDate_{j+1}"
        )

    # ------------------------------------------------------------------
    # Constraint (9): sum_j x_{jk} <= 1  for k = 1,...,n
    # ------------------------------------------------------------------
    for k in range(n):
        model.addConstr(
            gp.quicksum(x[j, k] for j in range(n)) <= 1,
            name=f"InDeg_{k+1}"
        )

    # ------------------------------------------------------------------
    # Constraint (10): sum_k x_{jk} <= 1  for j = 1,...,n
    # ------------------------------------------------------------------
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[j, k] for k in range(n)) <= 1,
            name=f"OutDeg_{j+1}"
        )

    # ------------------------------------------------------------------
    # Constraint (11): x_{jk} + x_{kj} <= 1  for j,k = 1,...,n
    # ------------------------------------------------------------------
    for j in range(n):
        for k in range(j + 1, n):
            model.addConstr(
                x[j, k] + x[k, j] <= 1,
                name=f"Antipar_{j+1}_{k+1}"
            )

    # ------------------------------------------------------------------
    # Constraint (12): sum_{j,k} x_{jk} = n-1
    # ------------------------------------------------------------------
    model.addConstr(
        gp.quicksum(x[j, k] for j in range(n) for k in range(n)) == n - 1,
        name="TotalLinks"
    )

    # ------------------------------------------------------------------
    # Constraint (13): x_{jj} = 0  for j = 1,...,n
    # ------------------------------------------------------------------
    for j in range(n):
        model.addConstr(x[j, j] == 0, name=f"NoDiag_{j+1}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # Extract and return result
    # ------------------------------------------------------------------
    result = {}
    status = model.Status

    if status == GRB.INFEASIBLE:
        result["status"] = "INFEASIBLE"
        result["objective_value"] = None
        result["sequence"] = None

    elif model.SolCount > 0:
        # Read x variable values and reconstruct the permutation first
        x_vals = {(j, k): x[j, k].X for j in range(n) for k in range(n) if j != k}
        seq_0indexed = extract_sequence(x_vals, n)
        result["sequence"] = [j + 1 for j in seq_0indexed]   # 1-indexed output

        # Report objective_value = evaluate(sequence). Saves Gurobi's bound-driven
        # value separately for diagnostics; it can drift below the true makespan
        # by up to MIPGap.
        result["objective_value"] = compute_makespan(seq_0indexed, n, m, p)
        result["solver_obj_val"] = float(model.ObjVal)
        try:
            result["mip_gap"] = float(model.MIPGap)
        except Exception:
            result["mip_gap"] = None

        if status == GRB.OPTIMAL:
            result["status"] = "OPTIMAL"
        else:
            result["status"] = "TIME_LIMIT_FEASIBLE"

    else:
        result["status"] = "NO_SOLUTION_FOUND"
        result["objective_value"] = None
        result["sequence"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi Model I for F|nwt, d_j|C_max (Samarghandi & Behroozi 2017)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON problem instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the solution JSON will be written.")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds (default: 3600).")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    n, m, p, d = load_instance(args.instance_path)
    print(f"Loaded instance: {n} jobs, {m} machines")
    print(f"Time limit: {args.time_limit} s")

    result = solve(n, m, p, d, args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Status          : {result['status']}")
    print(f"Objective value : {result['objective_value']}")
    if result.get("sequence"):
        print(f"Sequence        : {result['sequence']}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
