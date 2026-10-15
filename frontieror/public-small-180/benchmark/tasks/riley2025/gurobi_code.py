#!/usr/bin/env python3
"""
Full-space deterministic equivalent (P) for the supply chain with
relocatable manufacturing facilities — solved directly with Gurobi.

Reference: Riley, Daoutidis, Zhang (arXiv:2511.06340), Case Study 1.
Model equations (11)-(24) from the paper.

Usage:
  python gurobi_code.py --instance_path instance_1.json \
                        --solution_path gurobi_solution_1.json \
                        --time_limit 3600
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_instance(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model builder & solver
# ---------------------------------------------------------------------------

def build_and_solve(data, time_limit, fixed_y=None, output_flag=1):
    I_n = data['num_demand_sites']
    J_n = data['num_production_sites']
    S_n = data['num_scenarios']
    T_n = data['num_time_periods']
    M_n = data['num_module_sizes']
    K_n = data['units_per_module_size']   # K_m (same for all m in this instance)
    u_max = data['u_max']

    scenarios = data['scenarios']
    p_s = [sc['probability'] for sc in scenarios]
    # d[s][i][t] : demand at site i in period t under scenario s
    d = [[[sc['demands'][i][t] for t in range(T_n)]
          for i in range(I_n)]
         for sc in scenarios]

    # u_m[m]: production capacity of a single unit of size m
    u_m = [ms['capacity'] for ms in data['module_sizes']]

    # g[m][k]: capital cost for selecting unit k of size m
    g = [[0.0] * K_n for _ in range(M_n)]
    for mu in data['module_units']:
        g[mu['module_size_id']][mu['unit_id']] = mu['capital_cost']

    # c_prod[i][j]: unit production/transport cost — time-invariant in this instance
    # (paper has c_{ijt}; instance provides one value per (i,j) used for all t)
    c_prod = data['production_costs']

    # h[j][jp][m]: relocation cost per unit of size m from production site j to jp
    # (paper has h_{jj'mt}; instance provides one value per (j,jp,m) for all t)
    # INFERRED ASSUMPTION: relocations between depot and production sites cost 0,
    # since the instance only provides a J×J×M matrix (production sites only).
    h = data['relocation_costs']

    # pen[i][t]: penalty for unmet demand at site i in period t
    pen = data['penalty_costs']

    I = range(I_n)
    J = range(J_n)
    S = range(S_n)
    T = range(T_n)
    M = range(M_n)
    K = range(K_n)

    model = gp.Model("supply_chain_full_space")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", output_flag)
    if fixed_y is not None:
        model.setParam("MIPGap", 1e-4)
        model.setParam("MIPGapAbs", 1e-6)

    # -----------------------------------------------------------------------
    # First-stage binary variables
    # y[m,k] ∈ {0,1}: select manufacturing unit k of size m   [Constraint (19)]
    # -----------------------------------------------------------------------
    y = model.addVars(M_n, K_n, vtype=GRB.BINARY, name="y")
    if fixed_y is not None:
        for m in range(M_n):
            for k in range(K_n):
                model.addConstr(y[m, k] == int(round(fixed_y[m][k])))

    # -----------------------------------------------------------------------
    # Second-stage variables (per scenario)
    # -----------------------------------------------------------------------
    # z[m,s] ∈ Z+: number of units of size m selected (auxiliary)  [Constraint (22)]
    z = model.addVars(M_n, S_n, vtype=GRB.INTEGER, lb=0, name="z")

    # x[i,j,t,s] ∈ [0,1]: fraction of demand at i served from j at t, s  [Constraint (21)]
    x = model.addVars(I_n, J_n, T_n, S_n, lb=0.0, ub=1.0, name="x")

    # q[i,t,s] ∈ [0,1]: fraction of unmet demand  [Constraint (20)]
    q = model.addVars(I_n, T_n, S_n, lb=0.0, ub=1.0, name="q")

    # v[j,m,t,s] ∈ Z+: units of size m at production site j at time t  [Constraint (23)]
    v = model.addVars(J_n, M_n, T_n, S_n, vtype=GRB.INTEGER, lb=0, name="v")

    # v_depot[m,t,s] ∈ Z+: units at depot at time t  (tracked via constraint 15)
    v_depot = model.addVars(M_n, T_n, S_n, vtype=GRB.INTEGER, lb=0, name="v_depot")

    # w[j,jp,m,t,s] ∈ Z+: relocations between production sites  [Constraint (24)]
    w = model.addVars(
        [(j, jp, m, t, s)
         for j in J for jp in J for m in M for t in T for s in S if j != jp],
        vtype=GRB.INTEGER, lb=0, name="w"
    )

    # Depot ↔ production site movements (cost assumed 0 — NOT SPECIFIED IN PAPER)
    # w_fd[j,m,t,s]: units from depot → production site j
    # w_td[j,m,t,s]: units from production site j → depot
    w_fd = model.addVars(J_n, M_n, T_n, S_n, vtype=GRB.INTEGER, lb=0, name="w_fd")
    w_td = model.addVars(J_n, M_n, T_n, S_n, vtype=GRB.INTEGER, lb=0, name="w_td")

    # -----------------------------------------------------------------------
    # Objective (11):
    #   min  Σ_s p_s [ Σ_j Σ_t ( Σ_i c_{ijt} d_{its} x_{ijts}
    #                             + Σ_{j',m} h_{jj'mt} w_{jj'mts} )
    #                 + Σ_{i,t} p_{it} d_{its} q_{its} ]
    #       + Σ_m Σ_k g_{mk} y_{mk}
    # -----------------------------------------------------------------------
    obj = gp.LinExpr()

    # Capital costs (first-stage)
    for m in M:
        for k in K:
            obj += g[m][k] * y[m, k]

    # Scenario-weighted operating costs
    for s in S:
        ps = p_s[s]
        for j in J:
            for t in T:
                for i in I:
                    obj += ps * c_prod[i][j] * d[s][i][t] * x[i, j, t, s]
                for jp in J:
                    if j != jp:
                        for m in M:
                            obj += ps * h[j][jp][m] * w[j, jp, m, t, s]
        for i in I:
            for t in T:
                obj += ps * pen[i][t] * d[s][i][t] * q[i, t, s]

    model.setObjective(obj, GRB.MINIMIZE)

    # -----------------------------------------------------------------------
    # Constraint (12): Symmetry breaking  y_{mk} <= y_{m,k-1}  for k >= 2
    # -----------------------------------------------------------------------
    for m in M:
        for k in range(1, K_n):
            model.addConstr(y[m, k] <= y[m, k - 1], name=f"sym_{m}_{k}")

    # -----------------------------------------------------------------------
    # Constraint (13): z_{ms} = Σ_k y_{mk}  for all m, s
    # -----------------------------------------------------------------------
    for m in M:
        for s in S:
            model.addConstr(
                z[m, s] == gp.quicksum(y[m, k] for k in K),
                name=f"z_def_{m}_{s}"
            )

    # -----------------------------------------------------------------------
    # Constraint (14): Σ_j x_{ijts} + q_{its} = 1  for all i, t, s
    # -----------------------------------------------------------------------
    for i in I:
        for t in T:
            for s in S:
                model.addConstr(
                    gp.quicksum(x[i, j, t, s] for j in J) + q[i, t, s] == 1,
                    name=f"demand_{i}_{t}_{s}"
                )

    # -----------------------------------------------------------------------
    # Constraint (15): Depot balance (recursive form)
    #   v_depot[m,0,s] = z[m,s] + Σ_j w_td - Σ_j w_fd
    #   v_depot[m,t,s] = v_depot[m,t-1,s] + Σ_j w_td - Σ_j w_fd   for t > 0
    # -----------------------------------------------------------------------
    for m in M:
        for s in S:
            for t in T:
                net = (gp.quicksum(w_td[j, m, t, s] for j in J)
                       - gp.quicksum(w_fd[j, m, t, s] for j in J))
                if t == 0:
                    model.addConstr(
                        v_depot[m, t, s] == z[m, s] + net,
                        name=f"depot_bal_{m}_{t}_{s}"
                    )
                else:
                    model.addConstr(
                        v_depot[m, t, s] == v_depot[m, t - 1, s] + net,
                        name=f"depot_bal_{m}_{t}_{s}"
                    )

    # -----------------------------------------------------------------------
    # Constraint (16): Production site balance (recursive form)
    #   v[j,m,0,s] = in_flow - out_flow
    #   v[j,m,t,s] = v[j,m,t-1,s] + in_flow - out_flow  for t > 0
    # -----------------------------------------------------------------------
    for j in J:
        for m in M:
            for s in S:
                for t in T:
                    in_f = (w_fd[j, m, t, s]
                            + gp.quicksum(w[jp, j, m, t, s] for jp in J if jp != j))
                    out_f = (w_td[j, m, t, s]
                             + gp.quicksum(w[j, jp, m, t, s] for jp in J if jp != j))
                    if t == 0:
                        model.addConstr(
                            v[j, m, t, s] == in_f - out_f,
                            name=f"site_bal_{j}_{m}_{t}_{s}"
                        )
                    else:
                        model.addConstr(
                            v[j, m, t, s] == v[j, m, t - 1, s] + in_f - out_f,
                            name=f"site_bal_{j}_{m}_{t}_{s}"
                        )

    # -----------------------------------------------------------------------
    # Constraint (17): Production capacity per site
    #   Σ_i d_{its} x_{ijts} <= Σ_m v_{jmts} u_m
    # -----------------------------------------------------------------------
    for j in J:
        for t in T:
            for s in S:
                model.addConstr(
                    gp.quicksum(d[s][i][t] * x[i, j, t, s] for i in I)
                    <= gp.quicksum(v[j, m, t, s] * u_m[m] for m in M),
                    name=f"cap_{j}_{t}_{s}"
                )

    # -----------------------------------------------------------------------
    # Constraint (18): Global production capacity
    #   Σ_i d_{its} x_{ijts} <= u^max
    # -----------------------------------------------------------------------
    for j in J:
        for t in T:
            for s in S:
                model.addConstr(
                    gp.quicksum(d[s][i][t] * x[i, j, t, s] for i in I) <= u_max,
                    name=f"maxcap_{j}_{t}_{s}"
                )

    model.optimize()
    return model, y, M_n, K_n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Solve supply chain MILP (full-space) with Gurobi."
    )
    parser.add_argument('--instance_path', required=True,
                        help="Path to instance JSON file.")
    parser.add_argument('--solution_path', required=True,
                        help="Path for output solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True,
                        help="Solver time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    model, y, M_n, K_n = build_and_solve(data, args.time_limit)

    result = {}
    if model.SolCount > 0:
        result['objective_value'] = model.ObjVal
        result['status'] = 'Optimal' if model.Status == GRB.OPTIMAL else 'Feasible'
        result['optimality_gap'] = model.MIPGap
        result['best_bound'] = model.ObjBound
        result['y'] = {
            f"y_{m}_{k}": round(y[m, k].X)
            for m in range(M_n)
            for k in range(K_n)
        }
    else:
        result['objective_value'] = None
        result['status'] = 'NoSolution'
        result['optimality_gap'] = None
        result['best_bound'] = model.ObjBound

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Status: {result['status']}")
    if result['objective_value'] is not None:
        print(f"Objective value: {result['objective_value']:.4f}")
        print(f"Optimality gap: {result['optimality_gap']:.4%}")


if __name__ == '__main__':
    main()
