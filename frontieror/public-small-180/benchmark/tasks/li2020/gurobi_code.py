#!/usr/bin/env python3
"""
Gurobi-based solver for the bi-level partial interdiction problem on
capacitated nested hierarchical facilities.

Reference: Forghani, Dehghanian, Salari, Ghiami (2020)
"A bi-level model and solution methods for partial interdiction problem
on capacitated hierarchical facilities"
Computers and Operations Research 114 (2020) 104831

Approach: Single-level MILP reformulation via LP duality of the lower level.
The upper level (interdictor) maximizes H(z), the lower level (defender) minimizes
cost. Since the lower level is an LP, we replace it with its dual and jointly
optimize: max_{x, dual} dual_obj s.t. upper-level constraints + dual feasibility.
Bilinear terms (binary x * dual variable) in the dual objective's capacity terms
are linearized using standard big-M techniques.
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


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def compute_s2_distances(inst):
    """Compute Euclidean distances between level-2 facilities (not in instance data)."""
    s2x = inst["level2_facilities"]["x_coordinates"]
    s2y = inst["level2_facilities"]["y_coordinates"]
    nS2 = len(s2x)
    return [[math.sqrt((s2x[j] - s2x[f]) ** 2 + (s2y[j] - s2y[f]) ** 2)
             for f in range(nS2)] for j in range(nS2)]


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for bi-level partial interdiction on hierarchical facilities")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    inst = load_instance(args.instance_path)
    gp_ = inst["global_parameters"]
    ip = inst["instance_parameters"]
    inter = inst["interdiction_parameters"]

    # Global parameters
    theta = gp_["theta"]
    sigma = gp_["sigma"]
    alpha = gp_["alpha"]
    beta = gp_["beta"]
    gamma = gp_["gamma"]
    alpha_p = gp_["alpha_prime"]
    beta_p = gp_["beta_prime"]
    gamma_p = gp_["gamma_prime"]

    # Instance dimensions
    nI = ip["num_customers"]
    nS1 = ip["num_level1_facilities"]
    nS2 = ip["num_level2_facilities"]
    k_max = ip["k_max"]
    B = ip["total_interdiction_budget"]
    nFac = nS1 + nS2
    K = list(range(k_max + 1))  # {0, 1, ..., k_max}

    # Interdiction costs: h1[k], h2[k]; h[0]=0 by convention
    h1 = {0: 0.0}
    for ks, v in inter["interdiction_cost_level1_h1"].items():
        h1[int(ks)] = v
    h2 = {0: 0.0}
    for ks, v in inter["interdiction_cost_level2_h2"].items():
        h2[int(ks)] = v

    # Capacity reduction ratios: d1[k], d2[k]; d[0]=0
    d1 = {0: 0.0}
    for ks, v in inter["capacity_reduction_level1_d1"].items():
        d1[int(ks)] = v
    d2 = {0: 0.0}
    for ks, v in inter["capacity_reduction_level2_d2"].items():
        d2[int(ks)] = v

    # Data arrays
    z = inst["customers"]["demands"]
    c1_s1 = inst["level1_facilities"]["capacity_type1"]
    c1_s2 = inst["level2_facilities"]["capacity_type1"]
    c2_s2 = inst["level2_facilities"]["capacity_type2"]

    l_cs1 = inst["distances"]["customer_to_level1_facility"]
    l_cs2 = inst["distances"]["customer_to_level2_facility"]
    l_s1s2 = inst["distances"]["level1_to_level2_facility"]
    l_s2s2 = compute_s2_distances(inst)

    # Combined facility-to-S2 distances: S1 rows first, then S2 rows
    l_fac_s2 = l_s1s2 + l_s2s2

    # --- Compute big-M for linearization ---
    # Dual variable bounds analysis:
    # lambda_i <= alpha' + sigma*gamma' (from D5)
    # mu_i <= beta' (from D6)
    # nu_j <= gamma' (from D7)
    # pi_j, rho_j, tau_j <= 0
    # Lower bounds on pi/rho/tau: determined by dual constraints.
    # Conservative estimate: max outsourcing cost provides an upper bound on magnitude.
    BIG_M = max(alpha_p + sigma * gamma_p, beta_p, gamma_p) * 2.0

    # --- Build the single-level MILP reformulation ---
    m = gp.Model("bilevel_partial_interdiction")
    m.setParam("Threads", 1)
    m.setParam("TimeLimit", args.time_limit)
    m.setParam("OutputFlag", 1)

    # === Upper-level binary variables: x[j,k] ===
    # x[j,k] = 1 if facility j is interdicted at level k
    x = {}
    for j in range(nFac):
        for k in K:
            x[j, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{k}")

    # === Lower-level dual variables ===
    # lambda_i (dual of constraint 6 - type I demand balance, equality -> free)
    lam = [m.addVar(lb=-GRB.INFINITY, ub=alpha_p + sigma * gamma_p,
                     name=f"lam_{i}") for i in range(nI)]
    # mu_i (dual of constraint 7 - type II demand balance, equality -> free)
    mu = [m.addVar(lb=-GRB.INFINITY, ub=beta_p,
                    name=f"mu_{i}") for i in range(nI)]
    # nu_j (dual of constraint 8 - referral balance, equality -> free)
    nu = [m.addVar(lb=-GRB.INFINITY, ub=gamma_p,
                    name=f"nu_{j}") for j in range(nFac)]
    # pi_j <= 0 (dual of constraint 9 - type I capacity at S1, <= ineq)
    pi_ = [m.addVar(ub=0.0, lb=-BIG_M, name=f"pi_{j}") for j in range(nS1)]
    # rho_j <= 0 (dual of constraint 10 - type I capacity at S2, <= ineq)
    rho = [m.addVar(ub=0.0, lb=-BIG_M, name=f"rho_{j}") for j in range(nS2)]
    # tau_j <= 0 (dual of constraint 11 - type II capacity at S2, <= ineq)
    tau = [m.addVar(ub=0.0, lb=-BIG_M, name=f"tau_{j}") for j in range(nS2)]

    # === Linearization variables for bilinear terms in dual objective ===
    # The dual objective contains terms like:
    #   [c1_j - c1_j * sum_k d_k * x_{jk}] * pi_j
    # The bilinear product x_{jk} * pi_j is linearized as w1_{jk}.
    # w1[j,k] = x[j,k] * pi_[j]  for j in S1, k >= 1
    w1 = {}
    for j in range(nS1):
        for k in range(1, k_max + 1):
            w1[j, k] = m.addVar(ub=0.0, lb=-BIG_M, name=f"w1_{j}_{k}")
    # w2[j,k] = x[nS1+j,k] * rho[j]  for j in S2, k >= 1
    w2 = {}
    for j in range(nS2):
        for k in range(1, k_max + 1):
            w2[j, k] = m.addVar(ub=0.0, lb=-BIG_M, name=f"w2_{j}_{k}")
    # w3[j,k] = x[nS1+j,k] * tau[j]  for j in S2, k >= 1
    w3 = {}
    for j in range(nS2):
        for k in range(1, k_max + 1):
            w3[j, k] = m.addVar(ub=0.0, lb=-BIG_M, name=f"w3_{j}_{k}")

    # === Primal variables (express explicit strong duality condition) ===
    u1 = {}
    for i in range(nI):
        for j in range(nFac):
            u1[i, j] = m.addVar(lb=0.0, name=f"u1_{i}_{j}")
    u2 = {}
    for i in range(nI):
        for j in range(nS2):
            u2[i, j] = m.addVar(lb=0.0, name=f"u2_{i}_{j}")
    u3 = {}
    for j in range(nFac):
        for f in range(nS2):
            u3[j, f] = m.addVar(lb=0.0, name=f"u3_{j}_{f}")
    o1 = [m.addVar(lb=0.0, name=f"o1_{i}") for i in range(nI)]
    o2 = [m.addVar(lb=0.0, name=f"o2_{i}") for i in range(nI)]
    o3 = [m.addVar(lb=0.0, name=f"o3_{j}") for j in range(nFac)]

    m.update()

    # =============================================
    # Upper-level constraints
    # =============================================

    # (2): Exactly one interdiction level per facility
    for j in range(nFac):
        m.addConstr(gp.quicksum(x[j, k] for k in K) == 1, name=f"one_level_{j}")

    # (3): Interdiction budget constraint
    m.addConstr(
        gp.quicksum(h1[k] * x[j, k] for j in range(nS1) for k in K)
        + gp.quicksum(h2[k] * x[nS1 + j, k] for j in range(nS2) for k in K)
        <= B, name="budget"
    )

    # =============================================
    # Dual feasibility constraints
    # =============================================

    # Dual of u1[i,j] for j in S1:
    #   Primal cost coeff: l_cs1[i][j] * alpha
    #   Appears in (6) coeff +1, (8) coeff -sigma (rewritten), (9) coeff +1
    #   => lam[i] - sigma*nu[j] + pi_[j] <= l_cs1[i][j]*alpha
    for i in range(nI):
        for j in range(nS1):
            m.addConstr(
                lam[i] - sigma * nu[j] + pi_[j] <= l_cs1[i][j] * alpha,
                name=f"du1s1_{i}_{j}")

    # Dual of u1[i,j] for j in S2 (j indexes S2 as 0..nS2-1, facility index nS1+j):
    #   Primal cost coeff: l_cs2[i][j] * beta
    #   Appears in (6) coeff +1, (8) coeff -sigma, (10) coeff +1
    #   => lam[i] - sigma*nu[nS1+j] + rho[j] <= l_cs2[i][j]*beta
    for i in range(nI):
        for j in range(nS2):
            m.addConstr(
                lam[i] - sigma * nu[nS1 + j] + rho[j] <= l_cs2[i][j] * beta,
                name=f"du1s2_{i}_{j}")

    # Dual of u2[i,j] for j in S2:
    #   Primal cost coeff: l_cs2[i][j] * beta
    #   Appears in (7) coeff +1, (11) coeff +1
    #   => mu[i] + tau[j] <= l_cs2[i][j]*beta
    for i in range(nI):
        for j in range(nS2):
            m.addConstr(
                mu[i] + tau[j] <= l_cs2[i][j] * beta,
                name=f"du2_{i}_{j}")

    # Dual of u3[j,f] for j in S1 union S2, f in S2:
    #   Primal cost coeff: l_fac_s2[j][f] * gamma
    #   Appears in (8) coeff +1 for facility j, (11) coeff +1 for facility f
    #   => nu[j] + tau[f] <= l_fac_s2[j][f]*gamma
    for j in range(nFac):
        for f in range(nS2):
            m.addConstr(
                nu[j] + tau[f] <= l_fac_s2[j][f] * gamma,
                name=f"du3_{j}_{f}")

    # Dual of o1[i]: lam[i] <= alpha' + sigma*gamma'  (set as UB on lam)
    # Dual of o2[i]: mu[i] <= beta'  (set as UB on mu)
    # Dual of o3[j]: nu[j] <= gamma'  (set as UB on nu)
    # These are already enforced via variable upper bounds.

    # =============================================
    # Linearization constraints
    # =============================================
    # For w = x_bin * y_nonpos where x in {0,1}, y <= 0:
    #   x=0 => w=0;  x=1 => w=y
    # Enforced by:
    #   w >= y - M*(1-x)    [x=1: w>=y; x=0: w>=y-M (loose)]
    #   w <= y + M*(1-x)    [x=1: w<=y; x=0: w<=y+M (loose)]
    #   w >= -M*x           [x=0: w>=0; x=1: w>=-M (loose)]
    #   w <= 0              (variable upper bound)

    for j in range(nS1):
        for k in range(1, k_max + 1):
            xvar = x[j, k]
            m.addConstr(w1[j, k] >= pi_[j] - BIG_M * (1 - xvar),
                        name=f"lin1a_{j}_{k}")
            m.addConstr(w1[j, k] <= pi_[j] + BIG_M * (1 - xvar),
                        name=f"lin1b_{j}_{k}")
            m.addConstr(w1[j, k] >= -BIG_M * xvar,
                        name=f"lin1c_{j}_{k}")

    for j in range(nS2):
        for k in range(1, k_max + 1):
            xvar = x[nS1 + j, k]
            m.addConstr(w2[j, k] >= rho[j] - BIG_M * (1 - xvar),
                        name=f"lin2a_{j}_{k}")
            m.addConstr(w2[j, k] <= rho[j] + BIG_M * (1 - xvar),
                        name=f"lin2b_{j}_{k}")
            m.addConstr(w2[j, k] >= -BIG_M * xvar,
                        name=f"lin2c_{j}_{k}")

            m.addConstr(w3[j, k] >= tau[j] - BIG_M * (1 - xvar),
                        name=f"lin3a_{j}_{k}")
            m.addConstr(w3[j, k] <= tau[j] + BIG_M * (1 - xvar),
                        name=f"lin3b_{j}_{k}")
            m.addConstr(w3[j, k] >= -BIG_M * xvar,
                        name=f"lin3c_{j}_{k}")

    # =============================================
    # Primal feasibility (eqs 6-11) + strong duality
    # =============================================
    for i in range(nI):
        m.addConstr(
            gp.quicksum(u1[i, j] for j in range(nFac)) + o1[i] == theta * z[i],
            name=f"primal6_{i}")
        m.addConstr(
            gp.quicksum(u2[i, j] for j in range(nS2)) + o2[i] == (1 - theta) * z[i],
            name=f"primal7_{i}")

    for j in range(nFac):
        m.addConstr(
            sigma * gp.quicksum(u1[i, j] for i in range(nI))
            == gp.quicksum(u3[j, f] for f in range(nS2)) + o3[j],
            name=f"primal8_{j}")

    for j in range(nS1):
        m.addConstr(
            gp.quicksum(u1[i, j] for i in range(nI))
            <= c1_s1[j] - c1_s1[j] * gp.quicksum(d1[k] * x[j, k]
                                                  for k in range(1, k_max + 1)),
            name=f"primal9_{j}")

    for j in range(nS2):
        m.addConstr(
            gp.quicksum(u1[i, nS1 + j] for i in range(nI))
            <= c1_s2[j] - c1_s2[j] * gp.quicksum(d2[k] * x[nS1 + j, k]
                                                  for k in range(1, k_max + 1)),
            name=f"primal10_{j}")
        m.addConstr(
            gp.quicksum(u2[i, j] for i in range(nI))
            + gp.quicksum(u3[f, j] for f in range(nFac))
            <= c2_s2[j] - c2_s2[j] * gp.quicksum(d2[k] * x[nS1 + j, k]
                                                  for k in range(1, k_max + 1)),
            name=f"primal11_{j}")

    # Strong duality: primal_obj == dual_obj (dual side uses w1/w2/w3 to linearize x*pi/x*rho/x*tau)
    primal_obj_expr = gp.LinExpr()
    for i in range(nI):
        for j in range(nS1):
            primal_obj_expr += l_cs1[i][j] * alpha * u1[i, j]
        for j in range(nS2):
            primal_obj_expr += l_cs2[i][j] * beta * u1[i, nS1 + j]
            primal_obj_expr += l_cs2[i][j] * beta * u2[i, j]
    for j in range(nFac):
        for f in range(nS2):
            primal_obj_expr += l_fac_s2[j][f] * gamma * u3[j, f]
    for i in range(nI):
        primal_obj_expr += alpha_p * o1[i]
        primal_obj_expr += beta_p * o2[i]
        primal_obj_expr += sigma * gamma_p * o1[i]
    for j in range(nFac):
        primal_obj_expr += gamma_p * o3[j]

    dual_obj_expr = gp.LinExpr()
    for i in range(nI):
        dual_obj_expr += theta * z[i] * lam[i]
        dual_obj_expr += (1 - theta) * z[i] * mu[i]
    for j in range(nS1):
        dual_obj_expr += c1_s1[j] * pi_[j]
        for k in range(1, k_max + 1):
            dual_obj_expr -= c1_s1[j] * d1[k] * w1[j, k]
    for j in range(nS2):
        dual_obj_expr += c1_s2[j] * rho[j]
        for k in range(1, k_max + 1):
            dual_obj_expr -= c1_s2[j] * d2[k] * w2[j, k]
    for j in range(nS2):
        dual_obj_expr += c2_s2[j] * tau[j]
        for k in range(1, k_max + 1):
            dual_obj_expr -= c2_s2[j] * d2[k] * w3[j, k]

    m.addConstr(primal_obj_expr == dual_obj_expr, name="strong_duality")

    # =============================================
    # Objective: maximize dual objective of lower-level LP
    # =============================================
    # dual_obj = sum_i theta*z_i*lam_i + sum_i (1-theta)*z_i*mu_i
    #          + sum_{j in S1} [c1_j * pi_j - sum_{k>=1} c1_j * d1_k * w1_{jk}]
    #          + sum_{j in S2} [c1_j * rho_j - sum_{k>=1} c1_j * d2_k * w2_{jk}]
    #          + sum_{j in S2} [c2_j * tau_j - sum_{k>=1} c2_j * d2_k * w3_{jk}]
    #
    # The negative terms -c*d*w = -c*d*x*pi become positive when x=1 and pi<0,
    # reflecting that interdiction removes capacity and increases defender cost.

    obj = gp.LinExpr()

    # Demand-related dual terms
    for i in range(nI):
        obj += theta * z[i] * lam[i]
        obj += (1 - theta) * z[i] * mu[i]

    # S1 capacity terms (constraint 9 duals)
    for j in range(nS1):
        obj += c1_s1[j] * pi_[j]
        for k in range(1, k_max + 1):
            obj -= c1_s1[j] * d1[k] * w1[j, k]

    # S2 type-I capacity terms (constraint 10 duals)
    for j in range(nS2):
        obj += c1_s2[j] * rho[j]
        for k in range(1, k_max + 1):
            obj -= c1_s2[j] * d2[k] * w2[j, k]

    # S2 type-II capacity terms (constraint 11 duals)
    for j in range(nS2):
        obj += c2_s2[j] * tau[j]
        for k in range(1, k_max + 1):
            obj -= c2_s2[j] * d2[k] * w3[j, k]

    m.setObjective(obj, GRB.MAXIMIZE)

    # =============================================
    # Solve
    # =============================================
    m.optimize()

    # =============================================
    # Extract and output solution
    # =============================================
    result = {"objective_value": None, "status": "no_solution"}

    if m.SolCount > 0:
        result["objective_value"] = m.ObjVal

        # Extract interdiction pattern
        pattern = {}
        for j in range(nFac):
            for k in K:
                if x[j, k].X > 0.5:
                    fac_type = "S1" if j < nS1 else "S2"
                    fac_idx = j if j < nS1 else j - nS1
                    pattern[f"{fac_type}_{fac_idx}"] = k
                    break
        result["interdiction_pattern"] = pattern

        if m.Status == GRB.OPTIMAL:
            result["status"] = "optimal"
            result["mip_gap"] = m.MIPGap
        elif m.Status == GRB.TIME_LIMIT:
            result["status"] = "time_limit"
            result["mip_gap"] = m.MIPGap
        else:
            result["status"] = "feasible"
    else:
        if m.Status == GRB.INFEASIBLE:
            result["status"] = "infeasible"
        elif m.Status == GRB.TIME_LIMIT:
            result["status"] = "time_limit_no_solution"

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Objective value: {result['objective_value']}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
