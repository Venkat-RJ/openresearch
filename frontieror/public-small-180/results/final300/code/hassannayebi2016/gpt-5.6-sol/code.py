"""
Gurobi implementation of the MINLP model for rail rapid transit timetabling.

Paper: "Variable and adaptive neighbourhood search algorithms for rail rapid
       transit timetabling problem"
Authors: Erfan Hassannayebi, Seyed Hessameddin Zegordi
Published in: Computers and Operations Research (2016)

Model: MINLP (Section 4.1, equations 1-18)
  Objective: min z = sum_{i in I} (w_i + 0.5*delta_i) * H_i   (eq. 1)

Usage:
  python gurobi_code.py --instance_path instance_1.json \
                        --solution_path gurobi_solution_1.json \
                        --time_limit 300
"""

import argparse
import json
import math

import os,sys
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
        description="Gurobi MINLP solver for train timetabling problem."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to write the JSON solution.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # -----------------------------------------------------------------------
    # Load instance
    # -----------------------------------------------------------------------
    with open(args.instance_path) as fh:
        inst = json.load(fh)

    T     = inst["parameters"]["T_planning_horizon_minutes"]
    D     = inst["parameters"]["D_num_train_services"]
    NT    = inst["parameters"]["NT_num_demand_periods"]
    theta = inst["parameters"]["theta_period_length_minutes"]
    h_min = inst["parameters"]["h_min_minutes"]
    h_max = inst["parameters"]["h_max_minutes"]
    C     = inst["parameters"]["C_train_capacity_passengers"]

    demand_periods = sorted(inst["demand_periods"], key=lambda x: x["period"])
    lambda_per = [dp["lambda_pax_per_min"] for dp in demand_periods]

    # -----------------------------------------------------------------------
    # Piecewise-linear cumulative demand function (used for Delta_i).
    # Delta(t) grows linearly at rate lambda^(p) over period p = [t_{p-1}, t_p].
    # This is an exact encoding of eqs. (11)-(12) that avoids Big-M.
    # -----------------------------------------------------------------------
    t_bp = [0.0]
    demand_bp = [0.0]
    cumul = 0.0
    for p in range(NT):
        t_end = float((p + 1) * theta)
        cumul += theta * lambda_per[p]
        t_bp.append(t_end)
        demand_bp.append(float(cumul))
    total_demand = demand_bp[-1]

    # -----------------------------------------------------------------------
    # Build Gurobi MINLP model (Section 4.1, equations 1-18)
    # -----------------------------------------------------------------------
    model = gp.Model("TTP_MINLP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("NonConvex", 2)   # required: bilinear objective (eq. 1)
    model.setParam("OutputFlag", 1)

    trains    = range(1, D + 1)     # train services i = 1..D
    intervals = range(0, D + 1)     # I = {0, 1, ..., D}; H_0, H_D are boundary headways

    # --- Decision variables (domains per eq. 18) ---
    d     = model.addVars(trains,    lb=0.0, ub=float(T), name="d")
    H     = model.addVars(intervals, lb=0.0, ub=float(T), name="H")
    Delta = model.addVars(trains,    lb=0.0, name="Delta")
    delta = model.addVars(intervals, lb=0.0, name="delta")
    w     = model.addVars(intervals, lb=0.0, name="w")
    b     = model.addVars(trains,    lb=0.0, name="b")

    # --- Eq (2): sum_i H_i = T ---
    model.addConstr(gp.quicksum(H[i] for i in intervals) == T, name="totalH")

    # --- Eq (3): H_i = d_{i+1} - d_i, i in I\{D} ---
    # Boundary convention (paper, Fig. 1):
    #   H_0 = dispatch time of first train  = d_1 - 0
    #   H_D = distance between end of horizon and last departure = T - d_D
    model.addConstr(H[0] == d[1],     name="H_0_def")
    model.addConstr(H[D] == T - d[D], name="H_D_def")
    for i in range(1, D):
        model.addConstr(H[i] == d[i + 1] - d[i], name=f"H_{i}_def")

    # --- Eq (4): h_min <= H_i <= h_max, i in I\{0, D} ---
    for i in range(1, D):
        model.addConstr(H[i] >= h_min, name=f"hmin_{i}")
        model.addConstr(H[i] <= h_max, name=f"hmax_{i}")

    # --- Eqs (11)-(12): Delta_i = cumulative demand at d_i ---
    # The paper's Big-M linearisation (11)-(12) with y_i^{(p)} (constraints (5)-(7))
    # encodes exactly the piecewise-linear cumulative demand evaluated at d_i.
    # We use Gurobi's native PWL constraint, which is equivalent and removes
    # both the binaries y_i^{(p)} and the unspecified Big-M.
    for i in trains:
        model.addGenConstrPWL(d[i], Delta[i], t_bp, demand_bp, name=f"Delta_{i}")

    # --- Eq (8): delta_i = Delta_{i+1} - Delta_i, i in I\{0, D} ---
    for i in range(1, D):
        model.addConstr(delta[i] == Delta[i + 1] - Delta[i], name=f"delta_{i}")

    # --- Eqs (13)-(14): delta_0 = arrivals in [0, d_1] = Delta_1 ---
    model.addConstr(delta[0] == Delta[1], name="delta_0")

    # --- Eqs (15)-(16): delta_D = arrivals in [d_D, T] = total_demand - Delta_D ---
    model.addConstr(delta[D] == total_demand - Delta[D], name="delta_D")

    # --- Eq (17): w_0 = 0 (b_0 is not instantiated -- no 0-th train service) ---
    model.addConstr(w[0] == 0, name="w0_init")

    # --- Eq (9): w_i = w_{i-1} + delta_{i-1} - b_i, i in I ---
    for i in trains:
        model.addConstr(w[i] == w[i - 1] + delta[i - 1] - b[i], name=f"w_flow_{i}")

    # --- Eq (10): b_i <= C ---
    for i in trains:
        model.addConstr(b[i] <= C, name=f"b_cap_{i}")

    # --- Objective (eq. 1): min sum_{i in I} (w_i + 0.5*delta_i) * H_i  (bilinear) ---
    obj = gp.QuadExpr()
    for i in intervals:
        obj += w[i] * H[i] + 0.5 * delta[i] * H[i]
    model.setObjective(obj, GRB.MINIMIZE)

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    model.optimize()

    # -----------------------------------------------------------------------
    # Extract and write solution
    # -----------------------------------------------------------------------
    solution = {}
    if model.SolCount > 0:
        solution["objective_value"] = float(model.ObjVal)
        solution["optimality_gap"]  = float(model.MIPGap)
        solution["status"]          = int(model.Status)

        # Departure schedule: train i -> departure time (minutes)
        schedule = {str(i): float(d[i].X) for i in trains}
        solution["departure_schedule_minutes"] = schedule

        # AWT = total waiting time (passenger-minutes) / total demand (passengers)
        total_waiting_passenger_minutes = float(model.ObjVal)
        solution["AWT_minutes"] = (total_waiting_passenger_minutes / total_demand
                                   if total_demand > 0 else None)
    else:
        solution["objective_value"] = None
        solution["status"]          = int(model.Status)
        solution["departure_schedule_minutes"] = {}
        solution["AWT_minutes"] = None

    with open(args.solution_path, "w") as fh:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, fh, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"  Objective (sum_i (w_i + 0.5*delta_i)*H_i) = {solution['objective_value']:.4f}")
        print(f"  AWT                                        = {solution['AWT_minutes']:.4f} min")


if __name__ == "__main__":
    main()
