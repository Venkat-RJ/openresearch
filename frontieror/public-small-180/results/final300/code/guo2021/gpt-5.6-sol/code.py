"""
Gurobi implementation of the Stochastic Distributed Operating Room Scheduling (SDORS) problem.
Based on: Guo et al. (2021), "Logic-based Benders Decomposition and Binary Decision Diagram
Based Approaches for Stochastic Distributed Operating Room Scheduling"

Implements the SAA Deterministic Equivalent (DE) formulation (Equation 3) directly in Gurobi.
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


def solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    instance_id = data["instance_id"]
    num_patients = data["num_patients"]
    num_hospitals = data["num_hospitals"]
    num_days = data["num_days"]
    num_ors_per_hospital = data["num_ors_per_hospital"]
    num_scenarios = data["num_scenarios"]

    # Sets (0-indexed)
    P = list(range(num_patients))
    H = list(range(num_hospitals))
    D = list(range(num_days))
    # R_h: ORs per hospital (same for all hospitals)
    R = list(range(num_ors_per_hospital))
    S = list(range(num_scenarios))

    # Patient data
    patients = data["patients"]
    # mandatory patients: those with is_mandatory = True
    P_mandatory = [p["patient_id"] for p in patients if p["is_mandatory"]]
    P_nonmandatory = [p["patient_id"] for p in patients if not p["is_mandatory"]]

    # c_sched_{dp}: scheduling benefit for patient p on day d (day_label is 1-indexed in JSON)
    c_sched = {}
    for p_data in patients:
        pid = p_data["patient_id"]
        for day_label_str, val in p_data["c_sched_per_day"].items():
            d_idx = int(day_label_str) - 1  # convert to 0-indexed
            c_sched[(d_idx, pid)] = val

    c_unsched = {p_data["patient_id"]: p_data["c_unsched"] for p_data in patients}
    c_cancel = {p_data["patient_id"]: p_data["c_cancel"] for p_data in patients}

    # Hospital data: B_{hd}, F_{hd}, G_{hd}
    B = {}  # B[h][d] = OR time limit
    F = {}  # F[h][d] = OR opening cost
    G = {}  # G[h][d] = suite opening cost
    for h_data in data["hospitals"]:
        hid = h_data["hospital_id"]
        B[hid] = {}
        F[hid] = {}
        G[hid] = {}
        for day_data in h_data["days"]:
            did = day_data["day_id"]  # 0-indexed
            B[hid][did] = day_data["B_hd"]
            F[hid][did] = day_data["F_hd"]
            G[hid][did] = day_data["G_hd"]

    # Scenario data: T^s_p = surgery duration of patient p under scenario s
    T_s = {}  # T_s[s][p]
    for sc in data["scenarios"]:
        sid = sc["scenario_id"]
        T_s[sid] = {p: sc["surgery_durations_minutes"][p] for p in P}

    # Build Gurobi model
    model = gp.Model("SDORS_DE")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 1e-4)
    model.setParam("Threads", 1)    # 1 thread as in paper

    # ============================================================
    # DECISION VARIABLES
    # ============================================================

    # u_{hd}: 1 if surgical suite of hospital h is open on day d
    u = {}
    for h in H:
        for d in D:
            u[h, d] = model.addVar(vtype=GRB.BINARY, name=f"u_{h}_{d}")

    # y_{hdr}: 1 if OR r of hospital h is open on day d
    y = {}
    for h in H:
        for d in D:
            for r in R:
                y[h, d, r] = model.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{r}")

    # x_{hdpr}: 1 if patient p is assigned to OR r of hospital h on day d
    x = {}
    for h in H:
        for d in D:
            for p in P:
                for r in R:
                    x[h, d, p, r] = model.addVar(vtype=GRB.BINARY, name=f"x_{h}_{d}_{p}_{r}")

    # w_p: 1 if patient p is postponed
    w = {}
    for p in P_nonmandatory:
        w[p] = model.addVar(vtype=GRB.BINARY, name=f"w_{p}")

    # z^s_{hdpr}: 1 if patient p's surgery in OR r of hospital h on day d is operated under scenario s
    z = {}
    for s in S:
        for h in H:
            for d in D:
                for p in P:
                    for r in R:
                        z[s, h, d, p, r] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{h}_{d}_{p}_{r}")

    model.update()

    # ============================================================
    # OBJECTIVE (3a)
    # ============================================================
    # min: suite opening + OR opening + scheduling benefit + postponement penalty
    #      + (1/|S|) * sum_s cancellation costs

    obj = gp.LinExpr()

    # Suite opening costs
    for h in H:
        for d in D:
            obj += G[h][d] * u[h, d]

    # OR opening costs
    for h in H:
        for d in D:
            for r in R:
                obj += F[h][d] * y[h, d, r]

    # Scheduling benefit/cost (c_sched <= 0, so this subtracts from cost)
    for h in H:
        for d in D:
            for p in P:
                for r in R:
                    obj += c_sched[d, p] * x[h, d, p, r]

    # Postponement penalty
    for p in P_nonmandatory:
        obj += c_unsched[p] * w[p]

    # Expected cancellation cost: (1/|S|) * sum_s sum_h sum_d sum_p sum_r c^cancel_p * (x - z^s)
    inv_S = 1.0 / num_scenarios
    for s in S:
        for h in H:
            for d in D:
                for p in P:
                    for r in R:
                        obj += inv_S * c_cancel[p] * (x[h, d, p, r] - z[s, h, d, p, r])

    model.setObjective(obj, GRB.MINIMIZE)

    # ============================================================
    # CONSTRAINTS
    # ============================================================

    # (1b) Mandatory patients must be scheduled exactly once
    for p in P_mandatory:
        model.addConstr(
            gp.quicksum(x[h, d, p, r] for h in H for d in D for r in R) == 1,
            name=f"mandatory_{p}"
        )

    # (1c) Non-mandatory patients are either scheduled or postponed
    for p in P_nonmandatory:
        model.addConstr(
            gp.quicksum(x[h, d, p, r] for h in H for d in D for r in R) + w[p] == 1,
            name=f"nonmandatory_{p}"
        )

    # (1d) Symmetry-breaking: ORs opened in index order
    for h in H:
        for d in D:
            for r in R:
                if r > 0:
                    model.addConstr(
                        y[h, d, r] <= y[h, d, r - 1],
                        name=f"sym_y_{h}_{d}_{r}"
                    )

    # (1e) Symmetry-breaking: OR with lower index has >= total cancellation cost
    for h in H:
        for d in D:
            for r in R:
                if r > 0:
                    lhs = gp.quicksum(c_cancel[p] * x[h, d, p, r] for p in P)
                    rhs = gp.quicksum(c_cancel[p] * x[h, d, p, r - 1] for p in P)
                    model.addConstr(lhs <= rhs, name=f"sym_x_{h}_{d}_{r}")

    # (1f) OR can only be open if surgical suite is open
    for h in H:
        for d in D:
            for r in R:
                model.addConstr(
                    y[h, d, r] <= u[h, d],
                    name=f"suite_{h}_{d}_{r}"
                )

    # (1g) Patient can only be assigned to an open OR
    for h in H:
        for d in D:
            for p in P:
                for r in R:
                    model.addConstr(
                        x[h, d, p, r] <= y[h, d, r],
                        name=f"assign_{h}_{d}_{p}_{r}"
                    )

    # (3c) Scenario-specific time limit constraints
    for s in S:
        for h in H:
            for d in D:
                for r in R:
                    model.addConstr(
                        gp.quicksum(T_s[s][p] * z[s, h, d, p, r] for p in P) <= B[h][d] * y[h, d, r],
                        name=f"time_{s}_{h}_{d}_{r}"
                    )

    # (3d) Scenario-specific linking: can only operate if scheduled
    for s in S:
        for h in H:
            for d in D:
                for p in P:
                    for r in R:
                        model.addConstr(
                            z[s, h, d, p, r] <= x[h, d, p, r],
                            name=f"link_{s}_{h}_{d}_{p}_{r}"
                        )

    # ============================================================
    # SOLVE
    # ============================================================
    model.optimize()

    # ============================================================
    # EXTRACT SOLUTION
    # ============================================================
    solution = {"instance_id": instance_id}

    status = model.Status
    if status == GRB.INFEASIBLE:
        solution["objective_value"] = None
        solution["status"] = "infeasible"
    elif model.SolCount == 0:
        solution["objective_value"] = None
        solution["status"] = "no_solution_found"
    else:
        solution["objective_value"] = model.ObjVal
        solution["status"] = "optimal" if status == GRB.OPTIMAL else "time_limit_best"
        solution["mip_gap"] = model.MIPGap

        # Extract first-stage decisions
        solution["u"] = {
            f"{h},{d}": round(u[h, d].X) for h in H for d in D
        }
        solution["y"] = {
            f"{h},{d},{r}": round(y[h, d, r].X) for h in H for d in D for r in R
        }
        solution["x"] = {
            f"{h},{d},{p},{r}": round(x[h, d, p, r].X)
            for h in H for d in D for p in P for r in R
            if round(x[h, d, p, r].X) > 0
        }
        solution["w"] = {
            f"{p}": round(w[p].X) for p in P_nonmandatory
        }
        solution['z']={f'{s},{h},{d},{p},{r}':round(z[s,h,d,p,r].X) for s in S for h in H for d in D for p in P for r in R if z[s,h,d,p,r].X>.5}

    with open(solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    if solution.get("objective_value") is not None:
        print(f"Objective value: {solution['objective_value']}")
    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve SDORS using Gurobi (DE formulation)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the solution JSON will be written")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
