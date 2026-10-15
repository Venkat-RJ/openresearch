#!/usr/bin/env python3
"""
Gurobi implementation of the MILP formulation for RCPSP/max-pi.

Based on: Watermeyer & Zimmermann (2020), "A branch-and-bound procedure for the
Resource-Constrained Project Scheduling Problem with partially renewable resources
and general temporal constraints."

This implements the step-variable MILP formulation (constraints (3)-(4)) from the
paper. Step variables z_{it} in {0,1} indicate whether activity i starts at time t
or earlier. The pulse variable zeta_{it} = z_{it} - z_{i,t-1} equals 1 exactly when
activity i starts at time t.

ES/LS bounds are computed via Floyd-Warshall on the temporal constraint network.
"""

import argparse
import json

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
        pass# =============================================================================
# DATA LOADING
# =============================================================================

def load_instance(path):
    """Load an RCPSP/max-pi instance from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# =============================================================================
# FLOYD-WARSHALL FOR ES/LS COMPUTATION
# =============================================================================

def compute_es_ls(data):
    """
    Compute earliest start (ES) and latest start (LS) times for all activities
    using Floyd-Warshall on the temporal constraint network.

    The distance matrix d[i][j] stores the longest path from i to j.
    - ES_i = d[0][i] (longest path from source 0 to activity i)
    - LS_i = -d[i][0] (derived from longest path from i back to source 0)
    """
    num_activities = data["num_activities"]
    NEG_INF = float("-inf")

    # Initialize distance matrix
    dist = [[NEG_INF] * num_activities for _ in range(num_activities)]
    for i in range(num_activities):
        dist[i][i] = 0

    # Populate with temporal constraints: d[i][j] = max(d[i][j], delta_{ij})
    for tc in data["temporal_constraints"]:
        i = tc["i"]
        j = tc["j"]
        delta = tc["delta"]
        if delta > dist[i][j]:
            dist[i][j] = delta

    # Floyd-Warshall: compute all-pairs longest paths
    for k in range(num_activities):
        for i in range(num_activities):
            if dist[i][k] == NEG_INF:
                continue
            for j in range(num_activities):
                if dist[k][j] == NEG_INF:
                    continue
                candidate = dist[i][k] + dist[k][j]
                if candidate > dist[i][j]:
                    dist[i][j] = candidate

    # Extract ES and LS
    ES = {}
    LS = {}
    for i in range(num_activities):
        ES[i] = dist[0][i]
        LS[i] = -dist[i][0]

    return ES, LS


# =============================================================================
# MILP MODEL
# =============================================================================

def build_and_solve(data, time_limit):
    """
    Build and solve the MILP formulation for RCPSP/max-pi using step variables.

    Returns a dict with solution information.
    """
    num_activities = data["num_activities"]
    n = data["n"]  # number of real activities; activities 0..n+1
    sink = n + 1  # index of the sink (fictitious end activity)

    activities = data["activities"]
    resources = data["resources"]
    temporal_constraints = data["temporal_constraints"]

    # Processing times
    p = {}
    for i_str, act in activities.items():
        p[int(i_str)] = act["processing_time"]

    # Resource demands: r[i][k]
    r = {}
    for i_str, act in activities.items():
        i = int(i_str)
        r[i] = {}
        for k_str, dem in act["resource_demands"].items():
            r[i][int(k_str)] = dem

    # Resource capacities and period sets
    R_cap = {}
    Pi = {}  # Pi[k] = set of periods for resource k
    for k_str, res in resources.items():
        k = int(k_str)
        R_cap[k] = res["capacity"]
        Pi[k] = set(res["periods"])

    # Compute ES and LS
    ES, LS = compute_es_ls(data)

    # Time-feasible start windows
    # T_i = {ES_i, ..., LS_i}
    T = {}
    for i in range(num_activities):
        T[i] = list(range(ES[i], LS[i] + 1))

    # T_i^+ = {ES_i, ..., LS_i + p_i - 1}
    T_plus = {}
    for i in range(num_activities):
        T_plus[i] = list(range(ES[i], LS[i] + p[i]))

    # -------------------------------------------------------------------------
    # Create Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("RCPSP_max_pi")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # Step variables: z[i,t] in {0,1}, defined for t in T_i
    # z[i,t] = 1 iff activity i starts at time t or earlier
    z = {}
    for i in range(num_activities):
        for t in T[i]:
            z[i, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}")

    model.update()

    # -------------------------------------------------------------------------
    # Objective: Minimize S_{n+1} = sum_{t in T_{n+1}} t * zeta_{n+1,t}
    # where zeta_{it} = z_{it} - z_{i,t-1}
    # Expanding: sum_t t*(z_{it} - z_{i,t-1})
    # For the first time step ES_i: zeta_{i,ES_i} = z_{i,ES_i} (since z_{i,ES_i-1}=0)
    # This simplifies to: sum_t t*z_{it} - sum_t t*z_{i,t-1}
    #   = sum_t t*z_{it} - sum_{t'} (t'+1)*z_{it'}  (shifting index: t-1->t', so t=t'+1)
    #   = ES_i * z_{i,ES_i} + sum_{t=ES_i+1}^{LS_i} t*z_{it} - sum_{t'=ES_i}^{LS_i-1} (t'+1)*z_{it'}
    #   = ES_i * z_{i,ES_i} + sum_{t=ES_i+1}^{LS_i-1} (t - t - 1)*z_{it} + LS_i*z_{i,LS_i}
    #       - (LS_i-1+1)*z_{i,LS_i-1}   ... this is getting complex.
    # Simpler: S_i = sum_{t in T_i} t * zeta_{it}
    #        = sum_{t in T_i} t * (z_{it} - z_{i,t-1})
    # where z_{i,t-1} = 0 for t-1 < ES_i.
    # Equivalently: S_i = sum_{t in T_i} t * z_{it} - sum_{t in T_i} t * z_{i,t-1}
    # Second sum: for t in {ES_i, ..., LS_i}, z_{i,t-1} where t-1 in {ES_i-1,...,LS_i-1}
    # z_{i,ES_i-1} = 0, so that term vanishes.
    # = sum_{t=ES_i}^{LS_i} t*z_{it} - sum_{t=ES_i+1}^{LS_i} t*z_{i,t-1}
    # = ES_i*z_{i,ES_i} + sum_{t=ES_i+1}^{LS_i} t*z_{it} - sum_{t=ES_i+1}^{LS_i} t*z_{i,t-1}
    # = ES_i*z_{i,ES_i} + sum_{t=ES_i+1}^{LS_i} t*(z_{it} - z_{i,t-1})
    # Another approach: S_i = sum_{t in T_i} t * zeta_{it}
    # Since sum zeta_{it} = 1, and z_{i,LS_i} = 1 (forced), we can write:
    # S_i = LS_i - sum_{t=ES_i}^{LS_i-1} z_{it}  (telescoping sum identity)
    # This is a well-known identity for step variables.
    # -------------------------------------------------------------------------

    # Use the identity: S_i = LS_i - sum_{t=ES_i}^{LS_i-1} z_{it}
    # Objective: minimize S_{sink}
    if ES[sink] == LS[sink]:
        # Trivial case: makespan is fixed
        obj = ES[sink]
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        obj = LS[sink] - gp.quicksum(
            z[sink, t] for t in range(ES[sink], LS[sink])
        )
        model.setObjective(obj, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Monotonicity constraints: z_{i,t-1} <= z_{i,t} for consecutive t in T_i
    # -------------------------------------------------------------------------
    for i in range(num_activities):
        ti = T[i]
        for idx in range(1, len(ti)):
            t = ti[idx]
            t_prev = ti[idx - 1]
            if t_prev == t - 1:
                model.addConstr(z[i, t_prev] <= z[i, t],
                                name=f"mono_{i}_{t}")

    # -------------------------------------------------------------------------
    # Boundary constraints:
    # z_{i,t} = 0 for t < ES_i  (handled by not creating variables)
    # z_{i,t} = 1 for t >= LS_i  (only z_{i,LS_i} exists and is fixed to 1)
    # -------------------------------------------------------------------------
    for i in range(num_activities):
        if LS[i] in [t for t in T[i]]:
            model.addConstr(z[i, LS[i]] == 1, name=f"fix_ls_{i}")

    # -------------------------------------------------------------------------
    # Temporal constraints (3):
    # S_j >= S_i + delta_{ij} for all (i,j) in E
    #
    # Using: S_i = LS_i - sum_{t=ES_i}^{LS_i-1} z_{it}
    # => LS_j - sum z_{jt} >= LS_i - sum z_{it} + delta_{ij}
    # => sum z_{it} - sum z_{jt} >= LS_i - LS_j + delta_{ij}
    # i.e. sum_{t in [ES_i, LS_i-1]} z_{it} - sum_{t in [ES_j, LS_j-1]} z_{jt}
    #       >= LS_i - LS_j + delta_{ij}
    # -------------------------------------------------------------------------
    for tc in temporal_constraints:
        i = tc["i"]
        j = tc["j"]
        delta = tc["delta"]

        lhs = gp.LinExpr()
        # sum z_{it} for t in [ES_i, LS_i - 1]
        for t in range(ES[i], LS[i]):
            lhs.add(z[i, t], 1.0)
        # - sum z_{jt} for t in [ES_j, LS_j - 1]
        for t in range(ES[j], LS[j]):
            lhs.add(z[j, t], -1.0)

        rhs = LS[i] - LS[j] + delta
        model.addConstr(lhs >= rhs, name=f"temp_{i}_{j}")

    # -------------------------------------------------------------------------
    # Resource constraints (4):
    # For each resource k:
    #   sum_{i in V} r^d_{ik} * sum_{tau in Q_{ik}} (z_{i,tau} - z_{i,tau-p_i}) <= R_k
    #
    # Q_{ik} = {tau in T_i^+ | tau+1 in Pi_k}
    # T_i^+ = {ES_i, ..., LS_i + p_i - 1}
    #
    # z_{i,tau} for tau in T_i^+ means tau can be >= LS_i, but z is only defined
    # up to LS_i. For tau >= LS_i, z_{i,tau} = 1.
    # Similarly z_{i,tau-p_i}: if tau-p_i < ES_i, then z_{i,tau-p_i} = 0.
    # If tau-p_i >= LS_i, then z_{i,tau-p_i} = 1.
    # -------------------------------------------------------------------------
    def get_z_value(i, t):
        """
        Return the variable z[i,t] or a constant if t is outside T_i.
        - t < ES_i => 0
        - t >= LS_i => 1  (forced by constraint)
        - ES_i <= t < LS_i => z[i,t] variable
        """
        if t < ES[i]:
            return 0
        elif t >= LS[i]:
            return 1
        else:
            return z[i, t]

    for k_str, res in resources.items():
        k = int(k_str)
        capacity = R_cap[k]
        pi_k = Pi[k]

        lhs = gp.LinExpr()
        constant = 0.0

        for i in range(num_activities):
            demand = r[i][k]
            if demand == 0:
                continue

            # Q_{ik} = {tau in T_i^+ | tau+1 in Pi_k}
            for tau in T_plus[i]:
                if (tau + 1) not in pi_k:
                    continue

                z_tau = get_z_value(i, tau)
                z_tau_minus_p = get_z_value(i, tau - p[i])

                # Add demand * (z_{i,tau} - z_{i,tau-p_i}) to the LHS
                if isinstance(z_tau, (int, float)):
                    if isinstance(z_tau_minus_p, (int, float)):
                        # Both constants
                        constant += demand * (z_tau - z_tau_minus_p)
                    else:
                        # z_tau is constant, z_tau_minus_p is variable
                        constant += demand * z_tau
                        lhs.add(z_tau_minus_p, -demand)
                else:
                    if isinstance(z_tau_minus_p, (int, float)):
                        # z_tau is variable, z_tau_minus_p is constant
                        lhs.add(z_tau, demand)
                        constant -= demand * z_tau_minus_p
                    else:
                        # Both variables
                        lhs.add(z_tau, demand)
                        lhs.add(z_tau_minus_p, -demand)

        model.addConstr(lhs + constant <= capacity, name=f"res_{k}")

    # -------------------------------------------------------------------------
    # Optimize
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    result = {
        "status": model.Status,
        "status_name": None,
        "objective_value": None,
        "start_times": None,
        "solve_time": model.Runtime,
    }

    if model.Status == GRB.OPTIMAL:
        result["status_name"] = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT:
        result["status_name"] = "TIME_LIMIT"
    elif model.Status == GRB.INFEASIBLE:
        result["status_name"] = "INFEASIBLE"
    elif model.Status == GRB.UNBOUNDED:
        result["status_name"] = "UNBOUNDED"
    else:
        result["status_name"] = str(model.Status)

    if model.SolCount > 0:
        # Compute start times: S_i = LS_i - sum_{t=ES_i}^{LS_i-1} z_{it}
        start_times = {}
        for i in range(num_activities):
            s_i = LS[i]
            for t in range(ES[i], LS[i]):
                s_i -= round(z[i, t].X)
            start_times[i] = int(s_i)

        result["objective_value"] = int(round(model.ObjVal))
        result["start_times"] = start_times

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Solve RCPSP/max-pi using the step-variable MILP from "
                    "Watermeyer & Zimmermann (2020)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=float, default=3600.0,
                        help="Gurobi time limit in seconds (default: 3600).")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)

    # Solve
    result = build_and_solve(data, args.time_limit)

    # Write solution
    solution = {
        "objective_value": result["objective_value"],
        "status": result["status_name"],
        "solve_time": result["solve_time"],
        "start_times": result["start_times"],
    }

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Status: {result['status_name']}")
    if result["objective_value"] is not None:
        print(f"Objective (makespan): {result['objective_value']}")
    if result["start_times"] is not None:
        print(f"Start times: {result['start_times']}")


if __name__ == "__main__":
    main()
