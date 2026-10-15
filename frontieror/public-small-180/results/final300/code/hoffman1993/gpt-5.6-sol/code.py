"""
Gurobi implementation of the Set Partitioning Problem with Base Constraints (SPB)
from Hoffman & Padberg (1993) "Solving Airline Crew Scheduling Problems by Branch-and-Cut"

Formulation (SPB):
    min  sum_{j=1}^{n} c_j x_j
    s.t. A x = e_m           (each flight leg covered exactly once)
         d_1 <= D x <= d_2   (base constraints, if present)
         x in {0,1}^n
"""

import argparse
import json
import sys

import os
_license=os.environ.get('GRB_LICENSE_FILE')
if _license and not os.path.exists(_license):os.environ.pop('GRB_LICENSE_FILE')
from solution_logger import SolutionLogger
import gurobipy as gp
_original_optimize=gp.Model.optimize
def _logged_optimize(self,callback=None):
    self.Params.Seed=0;self.Params.MIPGap=1e-4;self.Params.NumericFocus=0;self.Params.Threads=1;self.Params.TimeLimit=max(.1,self.Params.TimeLimit-4)
    logger=SolutionLogger(sys.argv[sys.argv.index('--log_path')+1],sense='minimize') if '--log_path' in sys.argv else None
    if logger:logger.min_interval=0;logger.start_time=_GUROBI_CODE_START_TIME
    def cb(model,where):
        if logger and where==gp.GRB.Callback.MIPSOL:logger.log(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ))
        if callback:callback(model,where)
    return _original_optimize(self,cb)
gp.Model.optimize=_logged_optimize
from gurobipy import GRB
import os as _os, sys as _sys
import time
_GUROBI_CODE_START_TIME = time.time()
# Walk up from this file's directory to find repo root (containing scripts/).
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
    """Load a problem instance from JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_model(data, time_limit):
    """Build the Gurobi model for the SPB problem."""
    num_rows = data["dimensions"]["num_rows"]  # m: number of flight legs
    num_cols = data["dimensions"]["num_cols"]  # n: number of rotations
    cost_vector = data["cost_vector"]           # c_j for j = 0,...,n-1
    columns = data["constraint_matrix_A"]["columns"]  # column-wise storage

    model = gp.Model("SPB")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    # Suppress solver output for cleaner results
    model.setParam("OutputFlag", 1)

    # Decision variables: x_j in {0,1}
    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x")

    # Objective: minimize sum c_j * x_j
    model.setObjective(
        gp.quicksum(cost_vector[j] * x[j] for j in range(num_cols)),
        GRB.MINIMIZE,
    )

    # Set partitioning constraints: Ax = e_m
    # Each flight leg i must be covered by exactly one rotation
    for i in range(num_rows):
        # Find all columns j that cover row i
        covering_cols = []
        for j in range(num_cols):
            if i in columns[j]:
                covering_cols.append(j)
        model.addConstr(
            gp.quicksum(x[j] for j in covering_cols) == 1,
            name=f"partition_{i}",
        )

    # Base constraints: d_1 <= D x <= d_2 (if present)
    if data.get("has_base_constraints", False) and data.get("base_constraints") is not None:
        base_constraints = data["base_constraints"]
        # Instance format stores base constraints as:
        #   D_matrix.rows[b][j] = coefficient for base b, column j
        #   lower_bounds_d1[b] = lower bound for base b
        #   upper_bounds_d2[b] = upper bound for base b
        D_rows = base_constraints["D_matrix"]["rows"]
        d1 = base_constraints["lower_bounds_d1"]
        d2 = base_constraints["upper_bounds_d2"]
        num_base = base_constraints["num_constraints"]

        for k in range(num_base):
            expr = gp.LinExpr()
            for j in range(num_cols):
                if D_rows[k][j] != 0.0:
                    expr.addTerms(D_rows[k][j], x[j])

            lb = d1[k]
            ub = d2[k]

            # Add range constraint: lb <= expr <= ub
            if lb > -GRB.INFINITY:
                model.addConstr(expr >= lb, name=f"base_lb_{k}")
            if ub < GRB.INFINITY:
                model.addConstr(expr <= ub, name=f"base_ub_{k}")

    model.update()
    return model, x


def solve_and_output(model, x, solution_path, num_cols):
    """Solve the model and write the solution to a JSON file."""
    model.optimize()

    solution = {}

    if model.SolCount > 0:
        # We have at least one feasible solution
        solution["objective_value"] = model.ObjVal
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        solution["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
        solution["best_bound"] = model.ObjBound

        # Extract variable values
        selected_rotations = []
        variable_values = {}
        for j in range(num_cols):
            val = x[j].X
            variable_values[str(j)] = val
            if val > 0.5:
                selected_rotations.append(j)

        solution["selected_rotations"] = selected_rotations
        solution["variable_values"] = variable_values
        solution["num_selected"] = len(selected_rotations)
    else:
        # No feasible solution found
        solution["objective_value"] = None
        solution["status"] = "infeasible" if model.Status == GRB.INFEASIBLE else "no_solution"
        solution["selected_rotations"] = []
        solution["variable_values"] = {}
        solution["num_selected"] = 0

    solution["solver_status_code"] = model.Status
    solution["runtime_seconds"] = model.Runtime

    with open(solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Status: {solution['status']}")
    print(f"Objective value: {solution['objective_value']}")
    if solution.get("selected_rotations"):
        print(f"Selected rotations: {solution['selected_rotations']}")
    print(f"Runtime: {solution['runtime_seconds']:.2f}s")
    print(f"Solution written to: {solution_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve Set Partitioning Problem with Base Constraints (SPB) using Gurobi"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the problem instance",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path where the final solution JSON file will be written",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="Maximum solver runtime in seconds",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)
    print(f"Problem type: {data.get('problem_type', 'unknown')}")
    print(f"Dimensions: {data['dimensions']['num_rows']} rows x {data['dimensions']['num_cols']} cols")
    print(f"Density: {data['dimensions']['density_percent']}%")
    print(f"Has base constraints: {data.get('has_base_constraints', False)}")
    print(f"Time limit: {args.time_limit}s")

    # Build and solve model
    model, x = build_model(data, args.time_limit)
    solve_and_output(model, x, args.solution_path, data["dimensions"]["num_cols"])


if __name__ == "__main__":
    main()
