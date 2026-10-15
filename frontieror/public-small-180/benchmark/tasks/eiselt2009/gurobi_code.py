#!/usr/bin/env python3
"""
Gurobi implementation of the Competitive p-Hub Location Problem with Attraction Functions.

Reference: Eiselt & Marianov, "A conditional p-hub location problem with attraction
functions," Computers & Operations Research 36 (2009) 3128-3135.

Mathematical Model P [Eqs. (3)-(7)]:
    Maximize z = sum_{i,j} h_ij * sum_{k,l in Own} w_kl * u_ijklo /
                 (sum_{m,n in Own} w_mn * u_ijmno + sum_{s!=o} sum_{m,n in Comp} u_ijmns)

    s.t.  w_kl <= y_k               for all k, l in Own   (4)
          w_kl <= y_l               for all k, l in Own   (5)
          sum_k y_k = p                                    (6)
          y_k, w_kl in {0,1}                               (7)

The objective is a sum-of-ratios nonlinear function.  Because the paper states it
"precludes using standard integer programming software," this implementation reformulates
it as a Mixed Integer Linear Program (MILP) using McCormick linearization.

Reformulation idea:
    Define r_ij = h_ij * N_ij / D_ij (captured demand for OD pair ij), where
        N_ij = sum_{active pairs (k,l)} u_ijklo   (own utility numerator)
        D_ij = N_ij + C_ij                          (full denominator)
        C_ij = sum_{competitor pairs} u_ijmns      (fixed competitor term)
    Then r_ij * D_ij = h_ij * N_ij.
    Introducing x_{ij,kl} = r_ij * w_kl (continuous × binary) and applying McCormick
    envelopes, this becomes a linear constraint, yielding an exact MILP.
"""

import argparse
import json
import math
import itertools

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_instance(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Utility computation
# ---------------------------------------------------------------------------

def compute_utilities(inst: dict):
    """
    Pre-compute:
      u_own[(i, j, k, l)] : best-direction utility for entering airline on unordered
                              hub pair (k, l), k <= l, for OD pair (i, j).
      C_ij[(i, j)]         : total competitor utility (fixed denominator term) for (i, j).

    Returns (u_own, C_ij, Own, own_pairs_1hub, own_pairs_2hub)
    """
    n = inst['n_nodes']
    nodes = inst['nodes']
    dist = inst['distance_matrix']
    params = inst['parameters']

    alpha = params['alpha']          # interhub fare discount factor
    gamma = params['gamma']          # time vs. cost trade-off weight
    beta  = params['beta']           # time decay exponent
    lam   = params['lambda']         # cost decay exponent
    c_sc  = params['cost_scalar']    # fare proportionality constant
    A_2   = params['attractiveness_2hub']
    A_1   = params['attractiveness_1hub']  # 25% higher than A_2 per paper

    Own  = list(range(n))
    Comp = inst['competitor_hub_locations']

    def utility(t_val: float, c_val: float, A: float) -> float:
        """Gravity utility: A / (gamma * t^beta + (1-gamma) * c^lambda)  [Eq. (1)]"""
        denom = gamma * (t_val ** beta) + (1.0 - gamma) * (c_val ** lam)
        if denom <= 0.0:
            return 0.0
        return A / denom

    def t_2hub(i: int, k: int, l: int, j: int) -> float:
        """
        Travel time for 2-hub route i -> k -> l -> j.
        Formula from paper Section 3:
            t_ik + t_kl + t_lj + wait_i + wait_k + wait_l
        """
        return (dist[i][k] + dist[k][l] + dist[l][j]
                + nodes[i]['waiting_time']
                + nodes[k]['waiting_time']
                + nodes[l]['waiting_time'])

    def c_2hub(i: int, k: int, l: int, j: int) -> float:
        """
        Travel cost for 2-hub route i -> k -> l -> j.
        Formula from paper: c * (t_ik + alpha * t_kl + t_lj)
        where costs are proportional to distances.
        """
        return c_sc * (dist[i][k] + alpha * dist[k][l] + dist[l][j])

    # INFERRED ASSUMPTION: For 1-hub routes (k = l), the travel time uses only one
    # hub waiting time (physically, the passenger stops at hub k once).  The paper
    # gives the general formula t = t_ik + t_kl + t_lj + wait_i + wait_k + wait_l;
    # for k = l, t_kk = 0 and mechanically this gives wait_i + 2*wait_k, but that
    # double-counts the single hub dwell.  We therefore use:
    #     t_1hub = t_ik + t_kj + wait_i + wait_k
    # and c_1hub = c * (t_ik + t_kj)  (no interhub leg => no alpha discount).
    def t_1hub(i: int, k: int, j: int) -> float:
        return (dist[i][k] + dist[k][j]
                + nodes[i]['waiting_time']
                + nodes[k]['waiting_time'])

    def c_1hub(i: int, k: int, j: int) -> float:
        return c_sc * (dist[i][k] + dist[k][j])

    # ------------------------------------------------------------------
    # Build own-airline utility table (unordered pairs k <= l)
    # ------------------------------------------------------------------
    own_pairs_1hub = [(k, k) for k in Own]
    own_pairs_2hub = [(k, l) for k in Own for l in Own if k < l]
    all_own_pairs  = own_pairs_1hub + own_pairs_2hub

    u_own = {}
    for i in range(n):
        for j in range(n):
            for (k, l) in all_own_pairs:
                if k == l:
                    u = utility(t_1hub(i, k, j), c_1hub(i, k, j), A_1)
                else:
                    # Take best direction between i-k-l-j and i-l-k-j  [paper Section 3]
                    u1 = utility(t_2hub(i, k, l, j), c_2hub(i, k, l, j), A_2)
                    u2 = utility(t_2hub(i, l, k, j), c_2hub(i, l, k, j), A_2)
                    u = max(u1, u2)
                u_own[(i, j, k, l)] = u

    # ------------------------------------------------------------------
    # Build competitor fixed utility C_ij
    # Competitor uses hubs in Comp; all unordered pairs (m,n) with m<=n
    # ------------------------------------------------------------------
    comp_pairs = [(m, q) for m in Comp for q in Comp if m <= q]

    C_ij = {}
    for i in range(n):
        for j in range(n):
            total = 0.0
            for (m, q) in comp_pairs:
                if m == q:
                    total += utility(t_1hub(i, m, j), c_1hub(i, m, j), A_1)
                else:
                    u1 = utility(t_2hub(i, m, q, j), c_2hub(i, m, q, j), A_2)
                    u2 = utility(t_2hub(i, q, m, j), c_2hub(i, q, m, j), A_2)
                    total += max(u1, u2)
            C_ij[(i, j)] = total

    return u_own, C_ij, Own, own_pairs_1hub, own_pairs_2hub


# ---------------------------------------------------------------------------
# Gurobi model
# ---------------------------------------------------------------------------

def build_and_solve(inst: dict, time_limit: int) -> dict:
    n      = inst['n_nodes']
    p      = inst['p']
    demand = inst['demand_matrix']

    u_own, C_ij, Own, pairs_1hub, pairs_2hub = compute_utilities(inst)

    # OD pairs with positive demand (others contribute 0 to objective)
    od_pairs = [(i, j) for i in range(n) for j in range(n) if demand[i][j] > 0]

    model = gp.Model("CompetitiveHubLocation")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # ------------------------------------------------------------------
    # Binary decision variables
    # y[k]    = 1 if hub opened at node k
    # w[k, l] = 1 if hubs at both k and l (k < l, 2-hub pairs only)
    # For 1-hub pairs: w_kk = y_k (handled directly)
    # ------------------------------------------------------------------
    y = model.addVars(Own, vtype=GRB.BINARY, name="y")
    w = {}
    for (k, l) in pairs_2hub:
        w[k, l] = model.addVar(vtype=GRB.BINARY, name=f"w_{k}_{l}")
    model.update()

    # Hub-pair activation constraints  [Eqs. (4) and (5)]
    for (k, l) in pairs_2hub:
        model.addConstr(w[k, l] <= y[k], name=f"wle_k_{k}_{l}")
        model.addConstr(w[k, l] <= y[l], name=f"wle_l_{k}_{l}")

    # Number of hubs constraint  [Eq. (6)]
    model.addConstr(gp.quicksum(y[k] for k in Own) == p, name="num_hubs")

    # ------------------------------------------------------------------
    # Continuous variables
    # r[i, j]      = captured demand for OD pair (i, j)
    #                By linearization: r_ij = h_ij * N_ij / D_ij
    # x1[i, j, k]  = McCormick product r[i,j] * y[k]   (1-hub routes)
    # x2[i, j, k, l] = McCormick product r[i,j] * w[k,l] (2-hub routes)
    # ------------------------------------------------------------------
    r  = {}
    x1 = {}
    x2 = {}
    for (i, j) in od_pairs:
        hij = float(demand[i][j])
        r[i, j] = model.addVar(lb=0.0, ub=hij, name=f"r_{i}_{j}")
        for k in Own:
            x1[i, j, k] = model.addVar(lb=0.0, ub=hij, name=f"x1_{i}_{j}_{k}")
        for (k, l) in pairs_2hub:
            x2[i, j, k, l] = model.addVar(lb=0.0, ub=hij, name=f"x2_{i}_{j}_{k}_{l}")
    model.update()

    # ------------------------------------------------------------------
    # McCormick envelopes for x1[i,j,k] = r[i,j] * y[k]
    # Valid because y[k] is binary and 0 <= r[i,j] <= h_ij:
    #   (a) x1 <= h_ij * y[k]           (x1=0 when y[k]=0)
    #   (b) x1 >= r - h_ij*(1 - y[k])   (x1=r when y[k]=1)
    #   (c) x1 <= r                      (x1 <= r always)
    # ------------------------------------------------------------------
    for (i, j) in od_pairs:
        hij = float(demand[i][j])
        for k in Own:
            model.addConstr(x1[i, j, k] <= hij * y[k])
            model.addConstr(x1[i, j, k] >= r[i, j] - hij * (1.0 - y[k]))
            model.addConstr(x1[i, j, k] <= r[i, j])

    # McCormick envelopes for x2[i,j,k,l] = r[i,j] * w[k,l]
    for (i, j) in od_pairs:
        hij = float(demand[i][j])
        for (k, l) in pairs_2hub:
            model.addConstr(x2[i, j, k, l] <= hij * w[k, l])
            model.addConstr(x2[i, j, k, l] >= r[i, j] - hij * (1.0 - w[k, l]))
            model.addConstr(x2[i, j, k, l] <= r[i, j])

    # ------------------------------------------------------------------
    # Market-share equality constraint for each OD pair:
    #
    #   r_ij * D_ij = h_ij * N_ij
    #
    # Linearized (using x1/x2 as McCormick products of r with hub indicators):
    #
    #   sum_{k in Own}    x1[i,j,k] * u_own[i,j,k,k]
    # + sum_{(k,l) 2hub}  x2[i,j,k,l] * u_own[i,j,k,l]
    # + r[i,j] * C_ij
    # =
    #   sum_{k in Own}    y[k] * h_ij * u_own[i,j,k,k]
    # + sum_{(k,l) 2hub}  w[k,l] * h_ij * u_own[i,j,k,l]
    #
    # Proof: when hub indicator = 1, x = r, so LHS becomes r*(N+C) = r*D;
    #        RHS = h*N; hence r = h*N/D.
    # ------------------------------------------------------------------
    for (i, j) in od_pairs:
        hij   = float(demand[i][j])
        c_ij  = C_ij[(i, j)]

        lhs = gp.LinExpr()
        rhs = gp.LinExpr()

        for k in Own:
            u = u_own[(i, j, k, k)]
            lhs += x1[i, j, k] * u
            rhs += y[k] * (hij * u)

        for (k, l) in pairs_2hub:
            u = u_own[(i, j, k, l)]
            lhs += x2[i, j, k, l] * u
            rhs += w[k, l] * (hij * u)

        lhs += r[i, j] * c_ij

        model.addConstr(lhs == rhs, name=f"mkt_{i}_{j}")

    # ------------------------------------------------------------------
    # Objective: maximize total captured passenger demand  [Eq. (3)]
    # ------------------------------------------------------------------
    model.setObjective(
        gp.quicksum(r[i, j] for (i, j) in od_pairs),
        GRB.MAXIMIZE
    )

    model.optimize()

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    result = {
        "objective_value": None,
        "hub_locations": [],
        "status": "no_solution"
    }

    if model.SolCount > 0:
        hub_locs = [k for k in Own if y[k].X > 0.5]
        result["objective_value"] = model.ObjVal
        result["hub_locations"]   = hub_locs
        if model.status == GRB.OPTIMAL:
            result["status"] = "optimal"
        elif model.status == GRB.TIME_LIMIT:
            result["status"] = "time_limit"
        else:
            result["status"] = "suboptimal"
    elif model.status == GRB.INFEASIBLE:
        result["status"] = "infeasible"

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Competitive Hub Location Problem – Gurobi MILP Solver"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to write the solution JSON (e.g. gurobi_solution_1.json).")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    inst   = load_instance(args.instance_path)
    result = build_and_solve(inst, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Status           : {result['status']}")
    print(f"Hub locations    : {result['hub_locations']}")
    print(f"Objective value  : {result['objective_value']}")


if __name__ == "__main__":
    main()
