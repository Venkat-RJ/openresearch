#!/usr/bin/env python3
"""
Gurobi implementation of the Two-Stage Stochastic Linear Program (2-SLP)
for the Multilocation Transshipment Problem with Random Cost Coefficients.

Based on: Gangammanavar, Liu, and Sen (2020) - "Stochastic Decomposition for
Two-Stage Stochastic Linear Programs with Random Cost Coefficients"

This implements the SAA (Sample Average Approximation) formulation using
the regularized L-shaped method (RLS) as described in Section 5 of the paper.
The transshipment model is from Appendix C (Eq. C.1 and C.2).
"""

import argparse
import json
import numpy as np
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


def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)


def build_and_solve(instance, time_limit):
    """
    Build the extensive form (DEP) of the two-stage stochastic program
    for the transshipment problem using SAA with all provided scenarios.

    Model from Appendix C:
      First stage:  min_{s >= 0} E[h(s, omega)]
      Second stage: h(s, omega) = min sum_i(h_i*e_i + p_i*r_i) + sum_{i!=j}(c_ij(omega)*t_ij)
        s.t. f_i + sum_{j!=i} t_ij + e_i = s_i       (stock allocation)
             f_i + sum_{j!=i} t_ji + r_i = d_i(omega)  (demand satisfaction)
             e_i + q_i = s_i                            (inventory-replenishment balance)
             sum_i(r_i + q_i) = sum_i d_i(omega)        (total system balance)
             e_i, f_i, q_i, r_i, t_ij >= 0
    """
    N = instance["N"]
    second = instance["second_stage"]
    h_costs = second["holding_costs"]
    p_costs = second["penalty_costs"]
    c_mean = second["transshipment_costs_mean"]
    c_std = second["transshipment_costs_std"]

    demand_info = second["demand_distribution"]
    scenarios = demand_info["scenarios"]
    probabilities = demand_info["probabilities"]
    num_scenarios = demand_info["num_scenarios"]

    # Random cost links: pairs [i, j] (0-indexed) with random transshipment cost
    random_links = second.get("random_cost_links", [])

    # Build scenario-specific transshipment costs
    # For deterministic links: c_ij(omega) = c_mean[i][j]
    # For random links: c_ij(omega) is sampled from Normal(mean, std)
    # In the SAA formulation with discrete scenarios, we use the mean for
    # transshipment costs since the scenarios only specify demand realizations.
    #
    # **INFERRED ASSUMPTION**: The instance provides discrete demand scenarios
    # with probabilities. For random cost coefficients, the paper states they
    # follow Normal(mean, 0.2*mean). Since the instance only provides demand
    # scenarios (not joint demand-cost scenarios), we generate random cost
    # realizations for each demand scenario using the instance seed.
    # This is consistent with the paper's statement that demand and cost
    # coefficients are independent (Section C.3).

    rng = np.random.RandomState(instance.get("seed", 42))

    # Generate transshipment cost realizations for each scenario
    scenario_costs = []
    for s in range(num_scenarios):
        costs_s = [row[:] for row in c_mean]  # copy mean costs
        for link in random_links:
            i, j = link
            std = c_std[i][j]
            if std > 0:
                costs_s[i][j] = rng.normal(c_mean[i][j], std)
                # Ensure non-negative transshipment cost
                costs_s[i][j] = max(costs_s[i][j], 0.0)
        scenario_costs.append(costs_s)

    # Build Gurobi model (extensive form / DEP)
    model = gp.Model("transship_2SLP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # First-stage variables: s_i >= 0 (order-up-to quantities)
    # Note: The paper states no first-stage cost (c^T x = 0 for transship).
    # The first-stage objective is purely E[h(s, omega)].
    s_vars = model.addVars(N, lb=0.0, name="s")

    # Second-stage variables for each scenario
    e_vars = {}  # ending inventory
    f_vars = {}  # stock used to satisfy demand
    q_vars = {}  # inventory through replenishment
    r_vars = {}  # backlog
    t_vars = {}  # transshipment

    obj_expr = gp.LinExpr()

    for sc in range(num_scenarios):
        prob = probabilities[sc]
        d = scenarios[sc]
        c_sc = scenario_costs[sc]

        # Variables for this scenario
        for i in range(N):
            e_vars[sc, i] = model.addVar(lb=0.0, name=f"e_{sc}_{i}")
            f_vars[sc, i] = model.addVar(lb=0.0, name=f"f_{sc}_{i}")
            q_vars[sc, i] = model.addVar(lb=0.0, name=f"q_{sc}_{i}")
            r_vars[sc, i] = model.addVar(lb=0.0, name=f"r_{sc}_{i}")
            for j in range(N):
                if i != j:
                    t_vars[sc, i, j] = model.addVar(lb=0.0, name=f"t_{sc}_{i}_{j}")

        # Objective contribution for this scenario
        for i in range(N):
            obj_expr += prob * h_costs[i] * e_vars[sc, i]
            obj_expr += prob * p_costs[i] * r_vars[sc, i]
            for j in range(N):
                if i != j:
                    obj_expr += prob * c_sc[i][j] * t_vars[sc, i, j]

        # Constraints for this scenario
        for i in range(N):
            # Stock allocation: f_i + sum_{j!=i} t_ij + e_i = s_i
            model.addConstr(
                f_vars[sc, i]
                + gp.quicksum(t_vars[sc, i, j] for j in range(N) if j != i)
                + e_vars[sc, i]
                == s_vars[i],
                name=f"stock_alloc_{sc}_{i}"
            )

            # Demand satisfaction: f_i + sum_{j!=i} t_ji + r_i = d_i
            model.addConstr(
                f_vars[sc, i]
                + gp.quicksum(t_vars[sc, j, i] for j in range(N) if j != i)
                + r_vars[sc, i]
                == d[i],
                name=f"demand_sat_{sc}_{i}"
            )

            # Inventory-replenishment balance: e_i + q_i = s_i
            model.addConstr(
                e_vars[sc, i] + q_vars[sc, i] == s_vars[i],
                name=f"inv_repl_{sc}_{i}"
            )

        # Total system balance: sum_i(r_i + q_i) = sum_i d_i
        model.addConstr(
            gp.quicksum(r_vars[sc, i] + q_vars[sc, i] for i in range(N))
            == sum(d),
            name=f"sys_balance_{sc}"
        )

    model.setObjective(obj_expr, GRB.MINIMIZE)
    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["s"] = [s_vars[i].X for i in range(N)]
        result["status"] = model.Status
        result["mip_gap"] = model.MIPGap if hasattr(model, 'MIPGap') else None

        # Per-scenario second-stage details
        result["scenarios"] = []
        for sc in range(num_scenarios):
            sc_detail = {
                "e": [e_vars[sc, i].X for i in range(N)],
                "f": [f_vars[sc, i].X for i in range(N)],
                "q": [q_vars[sc, i].X for i in range(N)],
                "r": [r_vars[sc, i].X for i in range(N)],
                "t": [[t_vars[sc, i, j].X if i != j else 0.0
                        for j in range(N)] for i in range(N)]
            }
            result["scenarios"].append(sc_detail)
    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["s"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for 2-SLP Transshipment Problem "
                    "(Gangammanavar, Liu, Sen 2020)"
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
    result = build_and_solve(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']:.4f}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == "__main__":
    main()
