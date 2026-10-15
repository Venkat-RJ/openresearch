"""
Gurobi ILP implementation of the Maximum Edge Weight Clique (MEWC) problem.

Based on: Hosseinian, Fontes, and Butenko (2020)
"A Lagrangian Bound on the Clique Number and an Exact Algorithm for the
 Maximum Edge Weight Clique Problem", INFORMS Journal on Computing.

Formulation (Equation 1 from Park et al. 1996):
  max  sum_{(i,j) in E} w_ij * y_ij
  s.t. y_ij <= x_i                   for all (i,j) in E
       y_ij <= x_j                   for all (i,j) in E
       x_i + x_j <= 1                for all (i,j) not in E
       x_i in {0,1}                  for all i in V
       y_ij in {0,1}                 for all (i,j) in E
"""

import argparse
import json
import os,sys
_license=os.environ.get('GRB_LICENSE_FILE')
if _license and not os.path.exists(_license):os.environ.pop('GRB_LICENSE_FILE')
from solution_logger import SolutionLogger
import gurobipy as gp
_original_optimize=gp.Model.optimize
def _logged_optimize(self,callback=None):
    self.Params.Seed=0;self.Params.MIPGap=1e-4;self.Params.NumericFocus=0;self.Params.Threads=1;self.Params.TimeLimit=max(.1,self.Params.TimeLimit-4)
    logger=SolutionLogger(sys.argv[sys.argv.index('--log_path')+1],sense='maximize') if '--log_path' in sys.argv else None
    if logger:logger.min_interval=0;logger.start_time=_GUROBI_CODE_START_TIME
    def cb(model,where):
        if logger and where==gp.GRB.Callback.MIPSOL:logger.log(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ))
        if callback:callback(model,where)
    return _original_optimize(self,cb)
gp.Model.optimize=_logged_optimize
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


def solve_mewc(instance_path, solution_path, time_limit):
    # Load instance
    with open(instance_path, 'r') as f:
        data = json.load(f)

    vertices = data["vertices"]
    edges = data["edges"]
    n = data["num_vertices"]

    # Build edge set and complement edge set
    edge_set = set()
    edge_weights = {}
    for e in edges:
        i, j = e["i"], e["j"]
        edge_set.add((i, j))
        edge_set.add((j, i))
        edge_weights[(i, j)] = e["weight"]
        edge_weights[(j, i)] = e["weight"]

    # Create model
    model = gp.Model("MEWC")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = {}
    for v in vertices:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    y = {}
    for e in edges:
        i, j = e["i"], e["j"]
        key = (i, j)
        y[key] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    model.update()

    # Objective: maximize sum of w_ij * y_ij
    model.setObjective(
        gp.quicksum(e["weight"] * y[(e["i"], e["j"])] for e in edges),
        GRB.MAXIMIZE
    )

    # Constraints: y_ij <= x_i and y_ij <= x_j for all edges
    for e in edges:
        i, j = e["i"], e["j"]
        key = (i, j)
        model.addConstr(y[key] <= x[i], name=f"y_le_xi_{i}_{j}")
        model.addConstr(y[key] <= x[j], name=f"y_le_xj_{i}_{j}")

    # Constraints: x_i + x_j <= 1 for all non-edges
    for idx_a in range(len(vertices)):
        for idx_b in range(idx_a + 1, len(vertices)):
            i = vertices[idx_a]
            j = vertices[idx_b]
            if (i, j) not in edge_set:
                model.addConstr(x[i] + x[j] <= 1, name=f"nonadj_{i}_{j}")

    # Solve
    model.optimize()

    # Extract solution
    result = {"objective_value": None, "clique": [], "status": ""}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["clique"] = [v for v in vertices if x[v].X > 0.5]
        if model.Status == GRB.OPTIMAL:
            result["status"] = "optimal"
        else:
            result["status"] = "feasible (time limit)"
    else:
        result["objective_value"] = 0
        result["status"] = "no feasible solution found"

    # Write solution
    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Objective value: {result['objective_value']}")
    print(f"Clique: {result['clique']}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gurobi solver for Maximum Edge Weight Clique (MEWC)"
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

    solve_mewc(args.instance_path, args.solution_path, args.time_limit)
