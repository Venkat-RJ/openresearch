#!/usr/bin/env python3
"""
Gurobi implementation of the QSAP/AQSAP compact BQP formulation.

Paper: "A dual bounding framework for binary quadratic combinatorial optimization"
Authors: M. Bayani, B. Rostami, Y. Adulyasak, L.-M. Rousseau (2021)

This implements the compact QSAP formulation (Equations 12-14) with
Standard Linearization Technique (SLT, Appendix A, Equations A.1-A.4)
to linearize quadratic terms, solved directly by Gurobi (GUROBI+SLT method).

For AQSAP instances (out_of_star_density == 0), quadratic costs only exist
between adjacent edges (edges sharing a server endpoint), and q_{ef} = c_e + c_f.
For general QSAP instances, out-of-star quadratic interactions also exist.
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
def load_instance(instance_path):
    """Load a QSAP/AQSAP instance from JSON."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """
    Build the linearized QSAP model (GUROBI+SLT) and solve it.

    The compact formulation is:
      min  sum_{e in A} c_e x_e + sum_{(e,f) in E} q_{ef} x_e x_f
      s.t. sum_{e in delta(i)} x_e = 1   for all i in N   (Eq. 13)
           x_e in {0,1}                   for all e in A   (Eq. 14)

    We linearize the quadratic terms using SLT:
      y_{ef} >= x_e + x_f - 1   (A.1)
      y_{ef} <= x_e              (A.2)
      y_{ef} <= x_f              (A.3)
      y_{ef} >= 0                (A.4)

    So the linearized model becomes:
      min  sum_{e in A} c_e x_e + sum_{(e,f)} q_{ef} y_{ef}
      s.t. assignment constraints (13)
           SLT constraints (A.1)-(A.4)
           x_e in {0,1}, y_{ef} >= 0
    """
    n = data["n_clients"]
    h = data["n_servers"]
    linear_costs = data["linear_costs"]  # n x h matrix
    quadratic_costs = data["quadratic_costs"]  # list of [i, j, k, l, q_val]

    model = gp.Model("QSAP_SLT")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x[i][j] = 1 if client i assigned to server j
    x = {}
    for i in range(n):
        for j in range(h):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}",
                                   obj=linear_costs[i][j])

    # Parse quadratic costs into a dictionary keyed by a canonical unordered pair.
    # Per Eq. (12) the objective sums q_{ef} x_e x_f over ordered pairs in
    # E = E x E, so every listed entry must contribute its q_val; we accumulate
    # on the canonical key so any redundant (e,f)/(f,e) listings are summed
    # rather than silently dropped.
    quad_dict = {}
    for entry in quadratic_costs:
        i, j, k, l = int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3])
        q_val = entry[4]
        e = (i, j)
        f = (k, l)
        key = (e, f) if e <= f else (f, e)
        quad_dict[key] = quad_dict.get(key, 0.0) + q_val

    # Linearization variables y_{ef} for each quadratic pair
    y = {}
    for (e, f), q_val in quad_dict.items():
        y[(e, f)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                 name=f"y_{e[0]}{e[1]}_{f[0]}{f[1]}",
                                 obj=q_val)

    model.update()

    # Constraint (13): each client assigned to exactly one server
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(h)) == 1,
            name=f"assign_{i}"
        )

    # SLT constraints for each quadratic pair
    for (e, f), q_val in quad_dict.items():
        xe = x[e[0], e[1]]
        xf = x[f[0], f[1]]
        yef = y[(e, f)]

        # (A.1) y_{ef} >= x_e + x_f - 1
        model.addConstr(yef >= xe + xf - 1,
                        name=f"slt_a1_{e}_{f}")
        # (A.2) y_{ef} <= x_e
        model.addConstr(yef <= xe,
                        name=f"slt_a2_{e}_{f}")
        # (A.3) y_{ef} <= x_f
        model.addConstr(yef <= xf,
                        name=f"slt_a3_{e}_{f}")
        # (A.4) y_{ef} >= 0  (already enforced by lb=0.0)

    # Optimize
    model.optimize()

    # Extract solution
    objective_value = None
    assignment = {}

    if model.SolCount > 0:
        objective_value = model.ObjVal
        for i in range(n):
            for j in range(h):
                if x[i, j].X > 0.5:
                    assignment[i] = j
                    break
    else:
        # No feasible solution found
        objective_value = None

    return objective_value, assignment, model


def main():
    parser = argparse.ArgumentParser(
        description="Solve QSAP/AQSAP using Gurobi with SLT linearization"
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

    # Load instance
    data = load_instance(args.instance_path)

    # Solve
    objective_value, assignment, model = build_and_solve(data, args.time_limit)

    # Build solution output
    solution = {
        "objective_value": objective_value,
        "assignment": {str(k): v for k, v in assignment.items()},
        "solver_status": model.Status,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "solve_time": model.Runtime,
    }

    # Write solution
    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if objective_value is not None:
        print(f"Objective value: {objective_value}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == "__main__":
    main()
