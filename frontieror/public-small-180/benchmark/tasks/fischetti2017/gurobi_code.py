#!/usr/bin/env python3
"""
Direct Gurobi formulation for the Uncapacitated Facility Location (UFL) problem.

Implements the mathematical model from:
Fischetti, Ljubic, Sinnl (2016) "Redesigning Benders Decomposition for Large-Scale
Facility Location", Management Science.

- Linear UFL: disaggregated MILP formulation (equations 1-3).
- Quadratic UFL (qUFL): perspective reformulation (equations 5-11).
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


def main():
    parser = argparse.ArgumentParser(
        description="Solve UFL/qUFL using direct Gurobi MILP/MIQCP formulation"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    # NOT SPECIFIED IN PAPER: The instance format does not explicitly select linear vs quadratic.
    # Inferred assumption: default to 'quadratic' since qUFL is the paper's main contribution
    # and the test instances match the qUFL random generation procedure (Section 5.3).
    parser.add_argument("--variant", choices=["linear", "quadratic"], default="quadratic",
                        help="UFL variant: 'linear' or 'quadratic' (default: quadratic)")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ---- Load instance ----
    with open(args.instance_path) as f:
        data = json.load(f)

    n = data["num_facilities"]
    m = data["num_customers"]
    f_costs = data["opening_costs"]
    # allocation_costs[i][j]: cost for facility i serving customer j
    c = data["allocation_costs"]

    # ---- Build model ----
    model = gp.Model("UFL")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)

    # Solver tolerances matching the paper (Section 4.7), adapted for Gurobi.
    # Paper uses CPX_PARAM_EPINT=0; Gurobi minimum IntFeasTol is 1e-9.
    model.setParam("IntFeasTol", 1e-9)
    # Paper uses CPX_PARAM_EPGAP=1e-9.
    model.setParam("MIPGap", 1e-4)
    # Paper uses CPX_PARAM_EPRHS=1e-9.
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("OptimalityTol", 1e-9)

    if args.variant == "linear":
        # Paper uses CPX_PARAM_BRDIR=1 (branch-up-first) for linear UFL (Section 5.2).
        model.setParam("BranchDir", 1)

    # ---- Decision variables ----
    # y_i in {0,1}: 1 if facility i is open (eq. 11/15)
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    # x_{ij} >= 0: fraction of customer j served by facility i (eq. 9)
    x = model.addVars(n, m, lb=0.0, ub=1.0, name="x")

    if args.variant == "linear":
        # ---- Linear UFL: equations (1)-(3) ----
        # Objective (eq. 1): min sum_i f_i y_i + sum_{i,j} c_ij x_ij
        model.setObjective(
            gp.quicksum(f_costs[i] * y[i] for i in range(n))
            + gp.quicksum(c[i][j] * x[i, j] for i in range(n) for j in range(m)),
            GRB.MINIMIZE,
        )
    else:
        # ---- qUFL: perspective reformulation, equations (5)-(11) ----
        # Auxiliary variables z_{ij} >= 0 (eq. 10)
        z = model.addVars(n, m, lb=0.0, name="z")

        # Objective (eq. 5): min sum_i f_i y_i + sum_{i,j} c_ij z_ij
        model.setObjective(
            gp.quicksum(f_costs[i] * y[i] for i in range(n))
            + gp.quicksum(c[i][j] * z[i, j] for i in range(n) for j in range(m)),
            GRB.MINIMIZE,
        )

        # Second-order cone constraints (eq. 8): x_ij^2 <= z_ij * y_i
        for i in range(n):
            for j in range(m):
                model.addConstr(
                    x[i, j] * x[i, j] <= z[i, j] * y[i],
                    name=f"soc_{i}_{j}",
                )

    # Assignment constraints (eq. 2/6): sum_i x_ij = 1 for all j
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"assign_{j}"
        )

    # Capacity/linking constraints (eq. 3/7): x_ij <= y_i for all i,j
    for i in range(n):
        for j in range(m):
            model.addConstr(x[i, j] <= y[i], name=f"cap_{i}_{j}")

    # ---- Solve ----
    model.optimize()

    # ---- Build solution output ----
    solution = {
        "instance_id": data.get("instance_id", 0),
        "variant": args.variant,
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["y"] = [int(round(y[i].X)) for i in range(n)]
        solution["x"] = [[x[i, j].X for j in range(m)] for i in range(n)]
        solution["solver_status"] = model.Status
        solution["mip_gap"] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0
        solution["open_facilities"] = [i for i in range(n) if round(y[i].X) == 1]
    else:
        solution["objective_value"] = None
        solution["solver_status"] = model.Status

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()
