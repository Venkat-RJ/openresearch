"""
Stochastic Unit Commitment Problem (UCP) from Salemi and Davarnia (2022).

Two-stage stochastic program:
  First stage:  commitment (x, y, y_bar binary) + startup cost (q continuous)
  Second stage: production (p, p_bar continuous) per demand scenario

Minimises first-stage costs plus expected second-stage production costs.
"""

import argparse
import json
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


def build_model(instance):
    n_gen = instance["n_generators"]
    n_scen = instance["n_scenarios"]
    T = instance["T"]
    gens = instance["generators"]
    demand = instance["demand_scenarios"]   # [s][j]
    reserve = instance["reserve_scenarios"]  # [s][j]

    model = gp.Model("StochasticUCP")
    model.setParam("Threads", 1)

    # Index ranges
    periods = range(T)          # 0-indexed: period j corresponds to j+1 in paper
    generators = range(n_gen)
    scenarios = range(n_scen)

    # ---- First-stage variables ----
    # x[i,j]: commitment status of generator i in period j
    x = model.addVars(generators, periods, vtype=GRB.BINARY, name="x")
    # y[i,j]: start-up indicator
    y = model.addVars(generators, periods, vtype=GRB.BINARY, name="y")
    # y_bar[i,j]: shut-down indicator
    y_bar = model.addVars(generators, periods, vtype=GRB.BINARY, name="ybar")
    # q[i,j]: start-up cost
    q = model.addVars(generators, periods, lb=0.0, name="q")

    # ---- Second-stage variables ----
    # p[i,j,s]: production of generator i in period j under scenario s
    p = model.addVars(generators, periods, scenarios, lb=0.0, name="p")
    # p_bar[i,j,s]: available capacity
    p_bar = model.addVars(generators, periods, scenarios, lb=0.0, name="pbar")

    # Initial conditions: x_0^i = 0 for all i (all generators start OFF)
    # We handle this inline when j-1 or j-h references period 0 or earlier.

    # ---- Objective (1a) ----
    first_stage = gp.quicksum(
        gens[i]["c_f"] * x[i, j] + q[i, j]
        for i in generators
        for j in periods
    )
    second_stage = (1.0 / n_scen) * gp.quicksum(
        gens[i]["c_g"] * p[i, j, s]
        for i in generators
        for j in periods
        for s in scenarios
    )
    model.setObjective(first_stage + second_stage, GRB.MINIMIZE)

    # ---- Constraint (1b): Start-up cost linearization ----
    # q_j^i >= K_k^i * (x_j^i - sum_{h=1}^{k} x_{j-h}^i)
    # for k in {1, ..., j} (paper says k in {1,...,j-1} but j is 1-indexed;
    # here j is 0-indexed so k in {1, ..., j+1-1} = {1,...,j} when j>=1,
    # and for j=0 (first period), k can be 1 only with x_{-1}=0).
    # Actually: paper j in {1,...,T}, k in {1,...,j-1}. With 0-indexing:
    #   j in {0,...,T-1}, k in {1,...,j} (since paper's j-1 maps to our j).
    # For j=0 in paper (j=1), k in {1,...,0} is empty. But we also need k
    # up to T for the full startup cost schedule.
    # Corrected: paper says k in {1,...,j-1} for period j (1-indexed).
    # 0-indexed: for period j, k in {1,...,j}.  (paper j -> our j+1, so k <= j+1-1 = j)
    # Also k is bounded by len(startup_costs_K) = T.
    for i in generators:
        K = gens[i]["startup_costs_K"]  # K[k-1] = cost after k inactive periods
        for j in periods:
            # k ranges from 1 to j (0-indexed j corresponds to 1-indexed j+1)
            # In paper: k in {1,...,j-1} for 1-indexed j, i.e. k in {1,...,j} for 0-indexed j
            max_k = j  # since paper's j-1 = our j when converting
            # Also k <= T (length of K array)
            max_k = min(max_k, len(K))
            for k in range(1, max_k + 1):
                # sum_{h=1}^{k} x_{j-h}^i, where x for negative index = 0
                past_sum = gp.quicksum(
                    x[i, j - h] for h in range(1, k + 1) if j - h >= 0
                )
                model.addConstr(
                    q[i, j] >= K[k - 1] * (x[i, j] - past_sum),
                    name=f"startup_{i}_{j}_{k}"
                )

    # ---- Constraint (1c): Logical relation ----
    # y_j^i - y_bar_j^i = x_j^i - x_{j-1}^i,  x_0^i = 0
    for i in generators:
        for j in periods:
            x_prev = x[i, j - 1] if j >= 1 else 0
            model.addConstr(
                y[i, j] - y_bar[i, j] == x[i, j] - x_prev,
                name=f"logical_{i}_{j}"
            )

    # ---- Constraint (1d): Minimum up-time ----
    # sum_{j'=j-L+1}^{j} y_{j'}^i <= x_j^i,  for j >= L-1 (0-indexed)
    for i in generators:
        L = gens[i]["L"]
        for j in range(L - 1, T):
            model.addConstr(
                gp.quicksum(y[i, jp] for jp in range(j - L + 1, j + 1)) <= x[i, j],
                name=f"minup_{i}_{j}"
            )

    # ---- Constraint (1e): Minimum down-time ----
    # sum_{j'=j-ell+1}^{j} y_bar_{j'}^i <= 1 - x_j^i,  for j >= ell-1 (0-indexed)
    for i in generators:
        ell = gens[i]["ell"]
        for j in range(ell - 1, T):
            model.addConstr(
                gp.quicksum(y_bar[i, jp] for jp in range(j - ell + 1, j + 1))
                <= 1 - x[i, j],
                name=f"mindown_{i}_{j}"
            )

    # ---- Per-scenario constraints ----
    for s in scenarios:
        for i in generators:
            RU = gens[i]["RU"]
            SU = gens[i]["SU"]
            RD = gens[i]["RD"]
            SD = gens[i]["SD"]
            m_cap = gens[i]["m"]
            M_cap = gens[i]["M"]

            for j in periods:
                x_prev = x[i, j - 1] if j >= 1 else 0
                p_prev = p[i, j - 1, s] if j >= 1 else 0

                # (1f) Ramp-up
                model.addConstr(
                    p[i, j, s] - p_prev <= RU * x_prev + SU * y[i, j],
                    name=f"rampup_{i}_{j}_{s}"
                )

                # (1g) Ramp-down
                model.addConstr(
                    p_prev - p[i, j, s] <= RD * x[i, j] + SD * y_bar[i, j],
                    name=f"rampdown_{i}_{j}_{s}"
                )

                # (1h) Generation capacity bounds
                model.addConstr(
                    p[i, j, s] >= m_cap * x[i, j],
                    name=f"pmin_{i}_{j}_{s}"
                )
                model.addConstr(
                    p[i, j, s] <= p_bar[i, j, s],
                    name=f"p_le_pbar_{i}_{j}_{s}"
                )
                model.addConstr(
                    p_bar[i, j, s] <= M_cap * x[i, j],
                    name=f"pbarmax_{i}_{j}_{s}"
                )

        # (1i) Demand satisfaction
        for j in periods:
            model.addConstr(
                gp.quicksum(p[i, j, s] for i in generators) >= demand[s][j],
                name=f"demand_{j}_{s}"
            )

        # (1j) Spinning reserve
        for j in periods:
            model.addConstr(
                gp.quicksum(p_bar[i, j, s] for i in generators)
                >= demand[s][j] + reserve[s][j],
                name=f"reserve_{j}_{s}"
            )

    model.update()
    return model, x, y, y_bar, q, p, p_bar


def solve_and_save(model, instance, x, y, y_bar, q, p, p_bar, solution_path):
    n_gen = instance["n_generators"]
    n_scen = instance["n_scenarios"]
    T = instance["T"]

    model.optimize()

    result = {
        "status": model.Status,
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
        }.get(model.Status, str(model.Status)),
        "objective_value": None,
        "best_bound": None,
        "mip_gap": None,
        "solve_time": model.Runtime,
        "x": None,
        "y": None,
        "y_bar": None,
        "q": None,
        "p": None,
        "p_bar": None,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap

        # Extract first-stage solution
        result["x"] = [
            [int(round(x[i, j].X)) for j in range(T)]
            for i in range(n_gen)
        ]
        result["y"] = [
            [int(round(y[i, j].X)) for j in range(T)]
            for i in range(n_gen)
        ]
        result["y_bar"] = [
            [int(round(y_bar[i, j].X)) for j in range(T)]
            for i in range(n_gen)
        ]
        result["q"] = [
            [q[i, j].X for j in range(T)]
            for i in range(n_gen)
        ]

        # Extract second-stage solution
        result["p"] = [
            [
                [p[i, j, s].X for j in range(T)]
                for s in range(n_scen)
            ]
            for i in range(n_gen)
        ]
        result["p_bar"] = [
            [
                [p_bar[i, j, s].X for j in range(T)]
                for s in range(n_scen)
            ]
            for i in range(n_gen)
        ]

    with open(solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve the stochastic UCP (Salemi & Davarnia 2022) with Gurobi."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=float, default=3600,
                        help="Gurobi time limit in seconds (default: 3600).")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    model, x, y, y_bar, q, p, p_bar = build_model(instance)
    model.setParam("TimeLimit", args.time_limit)

    result = solve_and_save(model, instance, x, y, y_bar, q, p, p_bar,
                            args.solution_path)

    print(f"Status: {result['status_name']}")
    if result["objective_value"] is not None:
        print(f"Objective: {result['objective_value']:.4f}")
        print(f"Best bound: {result['best_bound']:.4f}")
        print(f"MIP gap: {result['mip_gap']:.6f}")
    print(f"Solve time: {result['solve_time']:.2f}s")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
