"""
Gurobi implementation of the linearized PP-DESUP model.
Reference: Hewitt & Pantuso (2024/2025), "Production planning under demand and
endogenous supply uncertainty", Appendix A (Eqs. 22a–22s).

The bilinear terms delta_pd * w_pds and delta_pd * o_pds are linearized via
McCormick inequalities using auxiliary variables mu_pds and rho_pds.
"""

import argparse
import json
import sys

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


def build_and_solve(instance, time_limit):
    # ------------------------------------------------------------------ #
    # Parse instance data                                                  #
    # ------------------------------------------------------------------ #
    P = [p["id"] for p in instance["products"]]
    F = [f["id"] for f in instance["facilities"]]

    B = {f["id"]: f["capacity"] for f in instance["facilities"]}
    sale_price = {p["id"]: p["sale_price"] for p in instance["products"]}        # P_p
    disc_price  = {p["id"]: p["discounted_price"] for p in instance["products"]} # O_p

    # C_pf: manufacturing cost
    C = {}
    # L_pfl, U_pfl: production level bounds
    L_bound = {}  # L_bound[(p, f, l)] = lower bound
    U_bound = {}  # U_bound[(p, f, l)] = upper bound
    # Levels per (p,f)
    levels = {}   # levels[(p, f)] = list of level ids

    for pfd in instance["product_facility_data"]:
        p = pfd["product_id"]
        f = pfd["facility_id"]
        C[(p, f)] = pfd["manufacturing_cost"]
        lv_list = []
        for lv in pfd["production_levels"]:
            l = lv["level_id"]
            lv_list.append(l)
            L_bound[(p, f, l)] = lv["lower_bound"]
            U_bound[(p, f, l)] = lv["upper_bound"]
        levels[(p, f)] = lv_list

    # Distributions and scenarios
    # distributions_data[(p)] = list of dist dicts
    distributions_data = {}
    for pd_entry in instance["products_distributions"]:
        p = pd_entry["product_id"]
        distributions_data[p] = pd_entry["distributions"]

    # Build index sets
    # D_p: distribution ids per product
    D_p = {}
    for p in P:
        D_p[p] = [d["distribution_id"] for d in distributions_data[p]]

    # S_pd: scenario ids per (product, distribution)
    S_pd = {}
    for p in P:
        for d_dict in distributions_data[p]:
            d = d_dict["distribution_id"]
            S_pd[(p, d)] = [s["scenario_id"] for s in d_dict["scenarios"]]

    # pi_sd: probability of scenario s under distribution d (same for each p)
    # NOTE: probabilities are indexed by (d, s) but in the JSON they are per product-dist-scenario
    pi = {}   # pi[(p, d, s)]
    for p in P:
        for d_dict in distributions_data[p]:
            d = d_dict["distribution_id"]
            for s_dict in d_dict["scenarios"]:
                s = s_dict["scenario_id"]
                pi[(p, d, s)] = s_dict["probability"]

    # Y_pfds: yield realization; for demand_only distributions, yield = 0
    Y = {}  # Y[(p, f, d, s)]
    for p in P:
        for d_dict in distributions_data[p]:
            d = d_dict["distribution_id"]
            for s_dict in d_dict["scenarios"]:
                s = s_dict["scenario_id"]
                for f in F:
                    if "yields" in s_dict:
                        Y[(p, f, d, s)] = s_dict["yields"].get(str(f), 0.0)
                    else:
                        Y[(p, f, d, s)] = 0.0  # demand_only distribution

    # D_pds: demand realization
    D_dem = {}  # D_dem[(p, d, s)]
    for p in P:
        for d_dict in distributions_data[p]:
            d = d_dict["distribution_id"]
            for s_dict in d_dict["scenarios"]:
                s = s_dict["scenario_id"]
                D_dem[(p, d, s)] = s_dict["demand"]

    # level_combo[(p, d)]: list of level ids per facility (indexed by facility position)
    # giving l(p, f, d) for each f; None for demand_only
    level_combo = {}
    for p in P:
        for d_dict in distributions_data[p]:
            d = d_dict["distribution_id"]
            level_combo[(p, d)] = d_dict["level_combination"]  # list or None

    # l_map[(p, f, d)]: production level required at facility f for distribution d of product p
    # For demand_only (level_combo is None), no production level constraint applies.
    l_map = {}
    for p in P:
        for d_dict in distributions_data[p]:
            d = d_dict["distribution_id"]
            combo = level_combo[(p, d)]
            if combo is not None:
                for fi, f in enumerate(F):
                    l_map[(p, f, d)] = combo[fi]

    # N_pds: McCormick upper bound on o_pds
    # N_pds = sum_f Y_pfds * min(B_f, max_l U_pfl)
    N = {}  # N[(p, d, s)]
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                val = 0.0
                for f in F:
                    max_U = max(U_bound[(p, f, l)] for l in levels[(p, f)])
                    val += Y[(p, f, d, s)] * min(B[f], max_U)
                N[(p, d, s)] = val

    # ------------------------------------------------------------------ #
    # Build Gurobi model                                                   #
    # ------------------------------------------------------------------ #
    model = gp.Model("PP-DESUP-Linearized")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OptimalityTol", 1e-4)

    # First-stage variables
    x = {}  # x[(p, f)]
    for p in P:
        for f in F:
            x[(p, f)] = model.addVar(lb=0.0, name=f"x_{p}_{f}")

    y = {}  # y[(p, f, l)]
    for p in P:
        for f in F:
            for l in levels[(p, f)]:
                y[(p, f, l)] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{f}_{l}")

    delta = {}  # delta[(p, d)]
    for p in P:
        for d in D_p[p]:
            delta[(p, d)] = model.addVar(vtype=GRB.BINARY, name=f"delta_{p}_{d}")

    # Second-stage variables
    z = {}   # z[(p, d, s)]
    w = {}   # w[(p, d, s)]
    o = {}   # o[(p, d, s)]
    mu = {}  # mu[(p, d, s)] = delta_pd * w_pds
    rho = {} # rho[(p, d, s)] = delta_pd * o_pds

    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                z[(p, d, s)]   = model.addVar(lb=0.0, name=f"z_{p}_{d}_{s}")
                w[(p, d, s)]   = model.addVar(lb=0.0, name=f"w_{p}_{d}_{s}")
                o[(p, d, s)]   = model.addVar(lb=0.0, name=f"o_{p}_{d}_{s}")
                mu[(p, d, s)]  = model.addVar(lb=0.0, name=f"mu_{p}_{d}_{s}")
                rho[(p, d, s)] = model.addVar(lb=0.0, name=f"rho_{p}_{d}_{s}")

    model.update()

    # ------------------------------------------------------------------ #
    # Objective (22a)                                                      #
    # ------------------------------------------------------------------ #
    obj = gp.LinExpr()
    for p in P:
        for f in F:
            obj -= C[(p, f)] * x[(p, f)]
    for p in P:
        Pp = sale_price[p]
        Op = disc_price[p]
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                obj += pi[(p, d, s)] * (Pp * mu[(p, d, s)] + Op * rho[(p, d, s)])
    model.setObjective(obj, GRB.MAXIMIZE)

    # ------------------------------------------------------------------ #
    # Constraints                                                          #
    # ------------------------------------------------------------------ #
    # (22b) Facility capacity
    for f in F:
        model.addConstr(
            gp.quicksum(x[(p, f)] for p in P) <= B[f],
            name=f"cap_{f}"
        )

    # (22c) Exactly one production level per (p, f)
    for p in P:
        for f in F:
            model.addConstr(
                gp.quicksum(y[(p, f, l)] for l in levels[(p, f)]) == 1,
                name=f"one_level_{p}_{f}"
            )

    # (22d) Production bounds tightened with min(B_f, U_pfl)
    for p in P:
        for f in F:
            model.addConstr(
                gp.quicksum(L_bound[(p, f, l)] * y[(p, f, l)] for l in levels[(p, f)]) <= x[(p, f)],
                name=f"lb_{p}_{f}"
            )
            model.addConstr(
                x[(p, f)] <= gp.quicksum(
                    min(B[f], U_bound[(p, f, l)]) * y[(p, f, l)]
                    for l in levels[(p, f)]
                ),
                name=f"ub_{p}_{f}"
            )

    # (22e) Exactly one distribution per product
    for p in P:
        model.addConstr(
            gp.quicksum(delta[(p, d)] for d in D_p[p]) == 1,
            name=f"one_dist_{p}"
        )

    # (22f) Endogeneity link: if delta_pd=1, the required production levels must be active
    # Only for distributions with a level_combination (not demand_only)
    for p in P:
        for d in D_p[p]:
            combo = level_combo[(p, d)]
            if combo is not None:
                model.addConstr(
                    gp.quicksum(y[(p, f, l_map[(p, f, d)])] for f in F) >= len(F) * delta[(p, d)],
                    name=f"endo_{p}_{d}"
                )

    # (22g) Inventory level
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                model.addConstr(
                    z[(p, d, s)] == gp.quicksum(Y[(p, f, d, s)] * x[(p, f)] for f in F),
                    name=f"inv_{p}_{d}_{s}"
                )

    # (22h) Sales at full price limited by demand
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                model.addConstr(
                    w[(p, d, s)] <= D_dem[(p, d, s)],
                    name=f"dem_{p}_{d}_{s}"
                )

    # (22i) Inventory balance: w + o = z
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                model.addConstr(
                    w[(p, d, s)] + o[(p, d, s)] == z[(p, d, s)],
                    name=f"bal_{p}_{d}_{s}"
                )

    # McCormick constraints for mu_pds = delta_pd * w_pds
    # (22n): mu <= w
    # (22o): mu <= D_pds * delta_pd
    # (22p): mu >= w - D_pds * (1 - delta_pd)
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                Dpds = D_dem[(p, d, s)]
                model.addConstr(mu[(p, d, s)] <= w[(p, d, s)],                         name=f"mc_n_{p}_{d}_{s}")
                model.addConstr(mu[(p, d, s)] <= Dpds * delta[(p, d)],                 name=f"mc_o_{p}_{d}_{s}")
                model.addConstr(mu[(p, d, s)] >= w[(p, d, s)] - Dpds * (1 - delta[(p, d)]), name=f"mc_p_{p}_{d}_{s}")

    # McCormick constraints for rho_pds = delta_pd * o_pds
    # (22q): rho <= o
    # (22r): rho <= N_pds * delta_pd
    # (22s): rho >= o - N_pds * (1 - delta_pd)
    for p in P:
        for d in D_p[p]:
            for s in S_pd[(p, d)]:
                Npds = N[(p, d, s)]
                model.addConstr(rho[(p, d, s)] <= o[(p, d, s)],                          name=f"mc_q_{p}_{d}_{s}")
                model.addConstr(rho[(p, d, s)] <= Npds * delta[(p, d)],                  name=f"mc_r_{p}_{d}_{s}")
                model.addConstr(rho[(p, d, s)] >= o[(p, d, s)] - Npds * (1 - delta[(p, d)]), name=f"mc_s_{p}_{d}_{s}")

    # ------------------------------------------------------------------ #
    # Solve                                                                #
    # ------------------------------------------------------------------ #
    model.optimize()

    # ------------------------------------------------------------------ #
    # Extract solution                                                     #
    # ------------------------------------------------------------------ #
    status = model.Status
    obj_val = None
    solution = {}

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        if model.SolCount > 0:
            obj_val = model.ObjVal
            solution["x"] = {f"{p}_{f}": x[(p, f)].X for p in P for f in F}
            solution["y"] = {f"{p}_{f}_{l}": y[(p, f, l)].X
                             for p in P for f in F for l in levels[(p, f)]}
            solution["delta"] = {f"{p}_{d}": delta[(p, d)].X for p in P for d in D_p[p]}
            solution["z"] = {f"{p}_{d}_{s}": z[(p, d, s)].X
                             for p in P for d in D_p[p] for s in S_pd[(p, d)]}
            solution["w"] = {f"{p}_{d}_{s}": w[(p, d, s)].X
                             for p in P for d in D_p[p] for s in S_pd[(p, d)]}
            solution["o"] = {f"{p}_{d}_{s}": o[(p, d, s)].X
                             for p in P for d in D_p[p] for s in S_pd[(p, d)]}
    else:
        print(f"Solver status: {status}. No feasible solution found.", file=sys.stderr)
        obj_val = None

    return obj_val, solution


def main():
    parser = argparse.ArgumentParser(description="Solve PP-DESUP using Gurobi (linearized formulation).")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", required=True, help="Path for output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Solver time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    obj_val, solution = build_and_solve(instance, args.time_limit)

    output = {
        "instance_id": instance.get("instance_id"),
        "objective_value": obj_val,
        "solution": solution,
    }

    with open(args.solution_path, "w") as f:
        output["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(output, f, indent=2)

    print(f"Objective value: {obj_val}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
