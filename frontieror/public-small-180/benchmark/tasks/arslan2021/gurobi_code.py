#!/usr/bin/env python3
"""
Gurobi MILP implementation of the benchmark's LP-relaxed recourse variant of
the Two-Stage Robust Knapsack Problem.

Based on: Arslan & Detienne (2022), "Decomposition-based approaches for a class of
two-stage robust binary optimization problems", INFORMS Journal on Computing 34(2).

Implements formulation (42) / deterministic equivalent (10)-(14) applied to the
knapsack application (Section 4.1, Equation 43).

Problem (43):
  min_{x in {0,1}^I} sum_i (f_i - p_bar_i) x_i
    + max_{xi in Xi} min_{(y,r) in Y(x)} sum_i [(p_hat_i xi_i - f_i) y_i - p_hat_i xi_i r_i]

where:
  Y_LP(x) = {(y,r) in [0,1]^{2I} | sum c_i y_i + t_i r_i <= C, y_i <= x_i, r_i <= y_i}
  Xi = {xi in R^I_+ | sum xi_i <= Gamma, 0 <= xi_i <= 1}

Deterministic equivalent approach:
  By Proposition 2.1, the inner min over Y(x) = min over conv(Y(x)).
  By Proposition 2.4, since linking is y_i <= x_i (H=I, T=-I, d=0),
  conv(Y(x)) = Y_bar(x) = conv(Y) intersect {y <= x}.
  By minimax theorem: max_xi min_y = min_y max_xi (both sets compact convex, bilinear).
  Dualizing max_xi gives the MILP below.

Benchmark convention: the second-stage binary set is replaced by its elementary
LP relaxation. This is an approximation of the paper's Eq. (43), not a claim
that the elementary relaxation equals the binary knapsack polytope's convex hull.
"""

import json
import argparse
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
    """Load the knapsack instance from JSON."""
    with open(path) as f:
        return json.load(f)


def build_and_solve(data, time_limit):
    """
    Build and solve the deterministic equivalent MILP.

    The formulation dualizes the inner max over the uncertainty set Xi:
      max_{xi >= 0, sum xi <= Gamma, xi <= 1} sum_i p_hat_i (y_i - r_i) xi_i
    Dual: min u0 * Gamma + sum v_i  s.t. u0 + v_i >= p_hat_i (y_i - r_i), u0, v_i >= 0

    Full MILP:
      min sum_i (f_i - p_bar_i) x_i - sum_i f_i y_i + Gamma u0 + sum_i v_i
      s.t. u0 + v_i >= p_hat_i (y_i - r_i)    for all i   [dual feasibility]
           sum_i c_i y_i + t_i r_i <= C                    [knapsack capacity]
           r_i <= y_i                          for all i   [repair requires production]
           y_i <= x_i                          for all i   [linking first/second stage]
           x_i in {0,1}                        for all i
           0 <= y_i <= 1, 0 <= r_i <= 1        for all i   [LP relaxation of conv(Y)]
           u0 >= 0, v_i >= 0                   for all i
    """
    items = data['items']
    I = len(items)
    C = data['problem_parameters']['knapsack_capacity']
    Gamma = data['problem_parameters']['uncertainty_budget']

    # Extract parameters
    weight = [item['weight'] for item in items]
    p_bar = [item['nominal_profit'] for item in items]
    p_hat = [item['max_degradation'] for item in items]
    f_out = [item['outsource_penalty'] for item in items]
    t_rep = [item['repair_capacity'] for item in items]

    model = gp.Model("TwoStageRobustKnapsack")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # --- Decision variables ---
    # First-stage: x_i = 1 if item i is selected for production
    x = model.addVars(I, vtype=GRB.BINARY, name="x")

    # Second-stage (LP relaxation of conv(Y)):
    # y_i: fraction of item i produced as-is
    # r_i: fraction of item i repaired
    y = model.addVars(I, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    r = model.addVars(I, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="r")

    # Dual variables for uncertainty set
    # u0: dual for sum_i xi_i <= Gamma
    # v_i: dual for xi_i <= 1
    u0 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="u0")
    v = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="v")

    # --- Objective ---
    # min sum_i (f_i - p_bar_i) x_i  [first-stage cost]
    #   + sum_i (-f_i) y_i            [nominal second-stage cost]
    #   + Gamma * u0 + sum_i v_i      [worst-case uncertainty premium via LP duality]
    model.setObjective(
        gp.quicksum((f_out[i] - p_bar[i]) * x[i] for i in range(I))
        + gp.quicksum(-f_out[i] * y[i] for i in range(I))
        + Gamma * u0
        + gp.quicksum(v[i] for i in range(I)),
        GRB.MINIMIZE
    )

    # --- Constraints ---

    # Dual feasibility: u0 + v_i >= p_hat_i (y_i - r_i) for all i
    # From LP duality of: max_{xi in Xi} sum_i p_hat_i xi_i (y_i - r_i)
    for i in range(I):
        model.addConstr(u0 + v[i] >= p_hat[i] * (y[i] - r[i]),
                        name=f"dual_feas_{i}")

    # Knapsack capacity constraint (from Y):
    # sum_i c_i y_i + t_i r_i <= C
    model.addConstr(
        gp.quicksum(weight[i] * y[i] + t_rep[i] * r[i] for i in range(I)) <= C,
        name="knapsack_cap"
    )

    # Repair requires production: r_i <= y_i for all i (from Y)
    for i in range(I):
        model.addConstr(r[i] <= y[i], name=f"repair_req_{i}")

    # Linking constraint: y_i <= x_i for all i (from Y(x))
    # This couples first-stage selection with second-stage production
    for i in range(I):
        model.addConstr(y[i] <= x[i], name=f"linking_{i}")

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    solution = {"solver_status": model.Status}

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        # Original formulation (Eq. 43) has first-stage binary x only; the
        # second-stage (y, r) and dual variables (u0, v) are artifacts of the
        # deterministic-equivalent single-level reformulation with LP duality.
        # They are NOT part of the original problem's decision space, so they
        # are not exported.
        solution["x"] = {str(i): int(round(x[i].X)) for i in range(I)}

        if model.Status == GRB.OPTIMAL:
            solution["optimality_gap"] = 0.0
        else:
            solution["optimality_gap"] = model.MIPGap

        # Interpret solution
        selected_items = [i for i in range(I) if round(x[i].X) == 1]
        solution["selected_items"] = selected_items
    else:
        solution["objective_value"] = None

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Robust Knapsack - Gurobi deterministic equivalent MILP"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    solution = build_and_solve(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.6f}")
        print(f"Status: {solution['solver_status']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
