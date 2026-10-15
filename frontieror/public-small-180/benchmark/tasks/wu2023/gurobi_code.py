#!/usr/bin/env python3
"""
Gurobi implementation of the mathematical optimization models from:
"Optimal Scenario Reduction for One- and Two-Stage Robust Optimization"
by Marc Goerigk and Mohammad Khosravi (2022)

Implements:
- IP-mu formulation for one-stage scenario reduction (Eqs. 10, 5-8)
  Binary mu, continuous lambda, big-M linearization.
- Simplified IP for two-stage scenario reduction (Eqs. 33-36, 38-39)
  Binary lambda, continuous mu, using precomputed d_{i,ell}.
"""

import json
import argparse
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
def solve_one_stage_ip_mu(scenarios, K, time_limit):
    """
    One-stage scenario reduction using IP-mu formulation.

    max  t
    s.t. t * c^i_j <= sum_ell lambda_{k,ell} * c^ell_j + M*(1 - mu_{i,k})
         for all i in [N], j in [n], k in [K]                          (10)
         sum_{i in [N]} lambda_{k,i} = 1       for all k in [K]        (5)
         sum_{k in [K]} mu_{i,k} = 1           for all i in [N]        (6)
         t >= 0                                                         (7)
         lambda_{k,i} >= 0                     for all k,i             (8)
         mu_{i,k} in {0,1}                     for all i,k
    """
    N = len(scenarios)
    n = len(scenarios[0])

    # Big-M constant: M >= c^i_j for all i, j
    M = max(scenarios[i][j] for i in range(N) for j in range(n))

    model = gp.Model("IP_mu_one_stage")
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)

    # Variables
    t = model.addVar(lb=0, name="t")

    lam = {}
    for k in range(K):
        for i in range(N):
            lam[k, i] = model.addVar(lb=0, name=f"lam_{k}_{i}")

    mu = {}
    for i in range(N):
        for k in range(K):
            mu[i, k] = model.addVar(vtype=GRB.BINARY, name=f"mu_{i}_{k}")

    model.setObjective(t, GRB.MAXIMIZE)

    # Constraint (10): t * c^i_j <= sum_ell lam_{k,ell} * c^ell_j + M*(1 - mu_{i,k})
    for i in range(N):
        for j in range(n):
            for k in range(K):
                model.addConstr(
                    t * scenarios[i][j]
                    <= gp.quicksum(lam[k, ell] * scenarios[ell][j] for ell in range(N))
                    + M * (1 - mu[i, k]),
                    name=f"c10_{i}_{j}_{k}"
                )

    # Constraint (5): sum_i lam_{k,i} = 1 for all k
    for k in range(K):
        model.addConstr(
            gp.quicksum(lam[k, i] for i in range(N)) == 1,
            name=f"c5_{k}"
        )

    # Constraint (6): sum_k mu_{i,k} = 1 for all i
    for i in range(N):
        model.addConstr(
            gp.quicksum(mu[i, k] for k in range(K)) == 1,
            name=f"c6_{i}"
        )

    model.optimize()

    result = {"objective_value": None, "status": model.Status}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap
        # Extract assignment: which scenario is assigned to which cluster
        assignment = {}
        for i in range(N):
            for k in range(K):
                if mu[i, k].X > 0.5:
                    assignment[i] = k
        result["assignment"] = assignment
        # Extract aggregated scenarios (lambda weights)
        lambda_weights = {}
        for k in range(K):
            weights = {
                ell: lam[k, ell].X
                for ell in range(N)
                if lam[k, ell].X > 1e-10
            }
            lambda_weights[k] = weights
        result["lambda_weights"] = lambda_weights

    return result


def solve_two_stage_ip(scenarios, K, time_limit):
    """
    Two-stage scenario reduction using simplified IP with d_{i,ell}.

    max  t
    s.t. t <= sum_ell d_{i,ell} * mu_{i,ell}   for all i in [N]        (39)
         mu_{i,ell} <= lambda_ell               for all i,ell in [N]    (33)
         sum_{i in [N]} lambda_i = K                                    (34)
         sum_{ell in [N]} mu_{i,ell} = 1        for all i in [N]        (35)
         lambda_i in {0,1}                      for all i in [N]        (36)
         mu_{i,ell} >= 0                        for all i,ell
         t >= 0                                                         (38)

    where d_{i,ell} = min_{j in [n]} c^ell_j / c^i_j
    """
    N = len(scenarios)
    n = len(scenarios[0])

    # Precompute d_{i,ell} = min_j c^ell_j / c^i_j
    d = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for ell in range(N):
            min_ratio = float('inf')
            for j in range(n):
                if scenarios[i][j] > 0:
                    ratio = scenarios[ell][j] / scenarios[i][j]
                    min_ratio = min(min_ratio, ratio)
                else:
                    # NOT SPECIFIED IN PAPER: handling c^i_j = 0.
                    # Assumption: if c^i_j = 0 and c^ell_j = 0, ratio is 1.0 (neutral).
                    # If c^i_j = 0 and c^ell_j > 0, ratio is inf (skip, does not constrain min).
                    if scenarios[ell][j] == 0:
                        min_ratio = min(min_ratio, 1.0)
                    # else: ratio = +inf, does not affect min
            if min_ratio == float('inf'):
                # All components of c^i are 0; set d to 0 as a safe default
                min_ratio = 0.0
            d[i][ell] = min_ratio

    model = gp.Model("IP_two_stage")
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)

    t = model.addVar(lb=0, name="t")
    lam = {}
    mu = {}
    for i in range(N):
        lam[i] = model.addVar(vtype=GRB.BINARY, name=f"lam_{i}")
        for ell in range(N):
            mu[i, ell] = model.addVar(lb=0, name=f"mu_{i}_{ell}")

    model.setObjective(t, GRB.MAXIMIZE)

    # Constraint (39): t <= sum_ell d_{i,ell} * mu_{i,ell}
    for i in range(N):
        model.addConstr(
            t <= gp.quicksum(d[i][ell] * mu[i, ell] for ell in range(N)),
            name=f"c39_{i}"
        )

    # Constraint (33): mu_{i,ell} <= lambda_ell
    for i in range(N):
        for ell in range(N):
            model.addConstr(mu[i, ell] <= lam[ell], name=f"c33_{i}_{ell}")

    # Constraint (34): sum_i lambda_i = K
    model.addConstr(gp.quicksum(lam[i] for i in range(N)) == K, name="c34")

    # Constraint (35): sum_ell mu_{i,ell} = 1
    for i in range(N):
        model.addConstr(
            gp.quicksum(mu[i, ell] for ell in range(N)) == 1,
            name=f"c35_{i}"
        )

    model.optimize()

    result = {"objective_value": None, "status": model.Status}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        selected = [i for i in range(N) if lam[i].X > 0.5]
        result["selected_scenarios"] = selected
        assignment = {}
        for i in range(N):
            for ell in range(N):
                if mu[i, ell].X > 0.5:
                    assignment[i] = ell
                    break
        result["assignment"] = assignment

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for scenario reduction (one-stage IP-mu, two-stage IP)"
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

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    scenarios = instance["scenarios"]
    K_values = instance["K_values"]
    N = len(scenarios)

    start_time = time.time()

    # Allocate time budget across all sub-problems
    # For each K: one-stage IP-mu + two-stage IP = 2 problems
    num_problems = 2 * len(K_values)
    time_per_problem = max(1, args.time_limit // max(num_problems, 1))

    one_stage_results = {}
    two_stage_results = {}

    for K in K_values:
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining <= 1:
            break

        # Solve one-stage IP-mu
        tl = min(time_per_problem, remaining / 2)
        if tl > 0:
            res = solve_one_stage_ip_mu(scenarios, K, tl)
            if res["objective_value"] is not None:
                one_stage_results[str(K)] = {
                    "t": res["objective_value"],
                    "approximation_ratio": 1.0 / res["objective_value"] if res["objective_value"] > 0 else float('inf'),
                    # Keep the actual IP-mu witness.  The previous release
                    # discarded these variables and therefore left the
                    # checker with an unverifiable scalar t.
                    "assignment": res["assignment"],
                    "lambda_weights": res["lambda_weights"],
                    "solver_status": res["status"],
                    "best_bound": res["best_bound"],
                    "mip_gap": res["mip_gap"],
                }

        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining <= 1:
            break

        # Solve two-stage IP
        tl = min(time_per_problem, remaining)
        if tl > 0:
            res = solve_two_stage_ip(scenarios, K, tl)
            if res["objective_value"] is not None:
                two_stage_results[str(K)] = {
                    "t": res["objective_value"],
                    "approximation_ratio": 1.0 / res["objective_value"] if res["objective_value"] > 0 else float('inf'),
                    "selected_scenarios": res.get("selected_scenarios"),
                }

    # Compute single objective_value as the best (highest) t found across all problems
    all_t_values = []
    for v in one_stage_results.values():
        all_t_values.append(v["t"])
    for v in two_stage_results.values():
        all_t_values.append(v["t"])

    best_obj = max(all_t_values) if all_t_values else 0.0

    solution = {
        "objective_value": best_obj,
        "one_stage_IP_mu": one_stage_results,
        "two_stage_IP": two_stage_results,
    }

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Best objective value (max t): {best_obj}")
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
