#!/usr/bin/env python3
"""
Gurobi-based MIP solver for the three paper-supported CSCP variants.

Source: Pérez Martínez, Adulyasak, and Jans (2022),
"Logic-Based Benders Decomposition for Integrated Process Configuration
and Production Planning Problems," INFORMS Journal on Computing, 34(4), pp. 2177-2191.

The OP has a nonlinear constraint q_{ij} = a_{ij} * z_j (eq. 18).
This is linearized via:
  1. Binary substitution (eq. 24): a_{ij} = sum_{k in K_i} b_k * w_{ijk},
     where b_k = 2^{k-1}, K_i = {1, ..., n_i}, n_i = ceil(log2(N_i+1)).
  2. McCormick linearization: u_{ijk} = w_{ijk} * z_j (binary x integer):
       u_{ijk} >= 0
       u_{ijk} <= M * w_{ijk}
       u_{ijk} <= z_j
       u_{ijk} >= z_j - M * (1 - w_{ijk})
  Then q_{ij} = sum_{k in K_i} b_k * u_{ijk}  (exact, linear).
"""

import argparse
import json
import math

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


def main():
    parser = argparse.ArgumentParser(
        description="Solve Open-Ended CSCP using Gurobi (linearized MIP)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ------------------------------------------------------------------
    # Load instance
    # ------------------------------------------------------------------
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    products = instance["products"]
    num_products = instance["num_products"]
    num_patterns = instance["num_patterns"]
    L = instance["machine_length"]
    C_s = instance["cost_setup_pattern"]
    C_r = instance["cost_repetition"]
    M = instance["M"]  # M = max_{i in T} {d_i}  (large number, eq. defined in paper)
    problem_type = instance.get("problem_type", "open_ended_cscp")
    supported_types = {
        "open_ended_cscp",
        "closed_ended_cscp",
        "cscp_with_knife_dependent_setups",
    }
    if problem_type not in supported_types:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    is_closed_ended = problem_type == "closed_ended_cscp"
    is_knife_dependent = problem_type == "cscp_with_knife_dependent_setups"
    C_k = instance.get("cost_knife_setup", 0)
    if is_closed_ended and not instance.get("x_i_forced_zero", False):
        raise ValueError("closed_ended_cscp requires x_i_forced_zero=true")
    if is_knife_dependent and C_k < 0:
        raise ValueError("cost_knife_setup must be nonnegative")

    T = list(range(num_products))   # 0-indexed product indices
    P = list(range(num_patterns))   # 0-indexed pattern indices

    d = [products[i]["demand"] for i in T]
    l = [products[i]["length"] for i in T]
    N_i = [products[i]["N_i"] for i in T]  # floor(L / l_i)

    # ------------------------------------------------------------------
    # Binary substitution parameters (eq. 24)
    # n_i = minimum number of bits to represent N_i
    # K_i = {1, ..., n_i}  (1-indexed)
    # b_k = 2^{k-1}
    # ------------------------------------------------------------------
    def get_n_bits(Ni):
        if Ni == 0:
            return 1
        return max(1, math.ceil(math.log2(Ni + 1)))

    n_bits = [get_n_bits(N_i[i]) for i in T]
    K_i = [list(range(1, n_bits[i] + 1)) for i in T]  # 1-indexed

    def bk(k):
        return 2 ** (k - 1)  # b_k = 2^{k-1}

    # ------------------------------------------------------------------
    # Build Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model(f"CSCP_Linearized_{problem_type}")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("OutputFlag", 1)

    # --- Variables ---

    # s_j in {0,1}: 1 if pattern j is used (eq. 22)
    s = model.addVars(P, vtype=GRB.BINARY, name="s")

    # w_{ijk} in {0,1}: bit k of a_{ij} (binary substitution, eq. 24)
    w = {}
    for i in T:
        for j in P:
            for k in K_i[i]:
                w[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}_{k}")

    # z_j in Z_+: number of repetitions of pattern j (eq. 23)
    z = model.addVars(P, vtype=GRB.INTEGER, lb=0, ub=M, name="z")

    # x_i >= 0: extra pieces of product i.
    # INFERRED ASSUMPTION: declared continuous (x_i >= 0 per eq. 23, in R_+ component
    # of y-vector). Physically integer, but relaxed here for faithfulness to OP form.
    x = model.addVars(T, lb=0.0, ub=0.0 if is_closed_ended else GRB.INFINITY, name="x")

    # n_j: number of knives in pattern j for the knife-dependent variant.
    knife_count = None
    if is_knife_dependent:
        knife_count = model.addVars(P, lb=0.0, name="knife_count")

    # u_{ijk} = w_{ijk} * z_j  (McCormick linearization auxiliary)
    u = {}
    for i in T:
        for j in P:
            for k in K_i[i]:
                u[i, j, k] = model.addVar(lb=0.0, ub=float(M), name=f"u_{i}_{j}_{k}")

    # q_{ij} >= 0: production quantity (eq. 23), defined by linearization below
    q = {}
    for i in T:
        for j in P:
            q[i, j] = model.addVar(lb=0.0, name=f"q_{i}_{j}")

    # --- Objective (eq. 15) ---
    objective = (
        C_s * gp.quicksum(s[j] for j in P)
        + C_r * gp.quicksum(z[j] for j in P)
    )
    if is_knife_dependent:
        objective += C_k * gp.quicksum(knife_count[j] for j in P)
    model.setObjective(objective, GRB.MINIMIZE)

    # --- Constraints ---

    # (16)/(25) Configuration constraints:
    # sum_{i in T} l_i * sum_{k in K_i} b_k * w_{ijk} <= L * s_j  for all j in P
    for j in P:
        model.addConstr(
            gp.quicksum(
                l[i] * gp.quicksum(bk(k) * w[i, j, k] for k in K_i[i])
                for i in T
            ) <= L * s[j],
            name=f"config_{j}"
        )

    # (17) Linking constraints: z_j <= M * s_j  for all j in P
    for j in P:
        model.addConstr(z[j] <= M * s[j], name=f"link_{j}")

    # Online supplement, Section 1.1.2: the knife setup is proportional to
    # the number of pieces cut simultaneously by each used pattern.
    if is_knife_dependent:
        for j in P:
            model.addConstr(
                knife_count[j]
                >= gp.quicksum(
                    bk(k) * w[i, j, k] for i in T for k in K_i[i]
                ) + M * (s[j] - 1),
                name=f"knife_count_{j}",
            )

    # McCormick linearization of u_{ijk} = w_{ijk} * z_j
    # (w_{ijk} binary, z_j integer with upper bound M)
    for i in T:
        for j in P:
            for k in K_i[i]:
                model.addConstr(u[i, j, k] <= M * w[i, j, k],         name=f"mck_a_{i}_{j}_{k}")
                model.addConstr(u[i, j, k] <= z[j],                    name=f"mck_b_{i}_{j}_{k}")
                model.addConstr(u[i, j, k] >= z[j] - M * (1 - w[i, j, k]), name=f"mck_c_{i}_{j}_{k}")

    # q_{ij} = sum_{k in K_i} b_k * u_{ijk}  (exact linearization of a_{ij} * z_j)
    for i in T:
        for j in P:
            model.addConstr(
                q[i, j] == gp.quicksum(bk(k) * u[i, j, k] for k in K_i[i]),
                name=f"q_def_{i}_{j}"
            )

    # (19) Demand fulfillment: sum_j q_{ij} + x_i >= d_i  for all i in T
    for i in T:
        model.addConstr(
            gp.quicksum(q[i, j] for j in P) + x[i] >= d[i],
            name=f"demand_{i}"
        )

    # (20) Extra pieces bound: sum_j z_j >= sum_i x_i
    model.addConstr(
        gp.quicksum(z[j] for j in P) >= gp.quicksum(x[i] for i in T),
        name="extra_bound"
    )

    # (21) Symmetry-breaking: z_{j-1} >= z_j  for all j in P, j > 1
    for j in P:
        if j > 0:
            model.addConstr(z[j - 1] >= z[j], name=f"sym_{j}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    if model.SolCount > 0:
        obj_val = model.ObjVal
        s_sol = {j: int(round(s[j].X)) for j in P}
        z_sol = {j: int(round(z[j].X)) for j in P}
        x_sol = {i: x[i].X for i in T}
        a_sol = {
            (i, j): int(sum(bk(k) * round(w[i, j, k].X) for k in K_i[i]))
            for i in T for j in P
        }
        q_sol = {(i, j): q[i, j].X for i in T for j in P}

        solution = {
            "objective_value": obj_val,
            "status": "optimal" if model.status == GRB.OPTIMAL else "feasible_time_limit",
            "patterns": {
                str(j): {
                    "used": bool(s_sol[j]),
                    "repetitions": z_sol[j],
                    "pieces": {
                        str(products[i]["id"]): a_sol[i, j]
                        for i in T
                    },
                    **(
                        {"knife_count": int(round(knife_count[j].X))}
                        if is_knife_dependent
                        else {}
                    ),
                }
                for j in P
            },
            "extra_pieces": {str(products[i]["id"]): x_sol[i] for i in T},
            "production_quantities": {
                f"product_{products[i]['id']}_pattern_{j}": q_sol[i, j]
                for i in T for j in P
            }
        }
    else:
        solution = {
            "objective_value": None,
            "status": "no_feasible_solution_found"
        }

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution.get("objective_value") is not None:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Status: {solution['status']}")


if __name__ == "__main__":
    main()
