"""
Linearized MILP Model [M2] for the Integrated Berth Allocation,
Quay Crane Assignment, and Yard Assignment Problem.

Reference:
    Wang, Zhen, Wang, Laporte (2018).
    "Column Generation for the Integrated Berth Allocation,
     Quay Crane Assignment, and Yard Assignment Problem."
    Transportation Science 52(4):812-834.

This script reads a JSON instance file, builds the [M2] MILP model
using Gurobi, solves it, and writes the solution to a JSON file.

Usage:
    python gurobi_code.py --instance_path instance_1.json \
                          --solution_path solution_1.json \
                          --time_limit 3600
"""

import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

_GUROBI_CODE_START_TIME = time.time()

import os as _os, sys as _sys
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
    """Load and parse a problem instance from a JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """
    Build the linearized MILP model [M2] and solve it with Gurobi.

    Returns a dict with solution information.
    """
    # -------------------------------------------------------------------------
    # Extract instance data
    # -------------------------------------------------------------------------
    num_vessels = data["num_vessels"]
    num_berths = data["num_berths"]
    num_subblocks = data["num_subblocks"]
    H = data["H"]
    E = data["E"]
    co = data["penalty_cost_coefficient_co"]  # c^o, operation cost coefficient
    Q_cap = data["QC_capacity_per_timestep"]  # length H+E, 0-indexed in list
    D_L = data["loading_route_lengths_D_L"]   # K x B matrix (list of lists)
    D_U = data["average_unloading_route_lengths_D_U"]  # length B
    vessels = data["vessels"]

    # Index sets (0-indexed internally)
    V = list(range(num_vessels))       # vessels
    K = list(range(num_subblocks))     # subblocks
    B = list(range(num_berths))        # berths
    T_all = list(range(1, H + E + 1)) # time steps 1..H+E (1-indexed)
    T_H = list(range(1, H + 1))       # time steps 1..H (for mu)

    # Per-vessel data
    l = []          # loading containers
    u = []          # unloading containers
    r = []          # reserved subblocks
    cp = []         # penalty cost coefficient c^p_i
    af = []         # a_i^f (feasible start)
    bf = []         # b_i^f (feasible end)
    ae = []         # a_i^e (expected start)
    be = []         # b_i^e (expected end)
    profiles = []   # list of lists of profiles per vessel

    for v in vessels:
        l.append(v["loading_containers"])
        u.append(v["unloading_containers"])
        r.append(v["reserved_subblocks"])
        cp.append(v["penalty_cost_coefficient"])
        af.append(v["feasible_time_window"][0])
        bf.append(v["feasible_time_window"][1])
        ae.append(v["expected_time_window"][0])
        be.append(v["expected_time_window"][1])
        profiles.append(v["qc_profiles"])

    # P_i: set of profile indices for vessel i
    P = []
    for i in V:
        P.append(list(range(len(profiles[i]))))

    # h_ip: handling time of vessel i, profile p
    h = []
    for i in V:
        h_i = []
        for p_idx in P[i]:
            h_i.append(profiles[i][p_idx]["handling_time"])
        h.append(h_i)

    # q_ipm: number of QCs used by profile p for vessel i at the m-th step
    # q[i][p][m] where m is 0-indexed (m=0 corresponds to the 1st step)
    q = []
    for i in V:
        q_i = []
        for p_idx in P[i]:
            q_i.append(profiles[i][p_idx]["qc_per_timestep"])
        q.append(q_i)

    # Big-M constant
    M_big = 10000

    # -------------------------------------------------------------------------
    # Create Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("CGFP_M2")
    model.setParam("Seed", 0)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 1e-4

    # -------------------------------------------------------------------------
    # Decision variables
    # -------------------------------------------------------------------------

    # omega[i,b]: berth b allocated to vessel i
    omega = {}
    for i in V:
        for b in B:
            omega[i, b] = model.addVar(vtype=GRB.BINARY, name=f"omega_{i}_{b}")

    # delta[i,j,b]: vessel i before vessel j at berth b
    delta = {}
    for i in V:
        for j in V:
            if i != j:
                for b in B:
                    delta[i, j, b] = model.addVar(
                        vtype=GRB.BINARY, name=f"delta_{i}_{j}_{b}"
                    )

    # phi[i,k]: subblock k reserved for vessel i
    phi = {}
    for i in V:
        for k in K:
            phi[i, k] = model.addVar(vtype=GRB.BINARY, name=f"phi_{i}_{k}")

    # gamma[i,p]: vessel i served by QC profile p
    gamma = {}
    for i in V:
        for p_idx in P[i]:
            gamma[i, p_idx] = model.addVar(
                vtype=GRB.BINARY, name=f"gamma_{i}_{p_idx}"
            )

    # mu[i,t]: vessel i begins handling at time t (only for t in 1..H)
    mu = {}
    for i in V:
        for t in T_H:
            mu[i, t] = model.addVar(vtype=GRB.BINARY, name=f"mu_{i}_{t}")

    # eta[i,p,t]: vessel i served by profile p starting at time t
    eta = {}
    for i in V:
        for p_idx in P[i]:
            for t in T_all:
                eta[i, p_idx, t] = model.addVar(
                    vtype=GRB.BINARY, name=f"eta_{i}_{p_idx}_{t}"
                )

    # alpha[i]: start time of handling for vessel i (integer)
    alpha = {}
    for i in V:
        alpha[i] = model.addVar(
            vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"alpha_{i}"
        )

    # beta[i]: end time of handling for vessel i (integer)
    beta = {}
    for i in V:
        beta[i] = model.addVar(
            vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"beta_{i}"
        )

    # rho[b]: start time for berth b (integer)
    rho = {}
    for b in B:
        rho[b] = model.addVar(
            vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"rho_{b}"
        )

    # varsigma[b]: end time for berth b (integer)
    varsigma = {}
    for b in B:
        varsigma[b] = model.addVar(
            vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"varsigma_{b}"
        )

    # sigma[t]: number of QCs used at time step t
    sigma = {}
    for t in T_all:
        sigma[t] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0, name=f"sigma_{t}"
        )

    # Linearization variables for penalty cost
    # tau_a_plus[i], tau_a_minus[i], tau_b_plus[i], tau_b_minus[i]
    tau_a_plus = {}
    tau_a_minus = {}
    tau_b_plus = {}
    tau_b_minus = {}
    for i in V:
        tau_a_plus[i] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0, name=f"tau_a_plus_{i}"
        )
        tau_a_minus[i] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0, name=f"tau_a_minus_{i}"
        )
        tau_b_plus[i] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0, name=f"tau_b_plus_{i}"
        )
        tau_b_minus[i] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0, name=f"tau_b_minus_{i}"
        )

    # theta[i,k,b]: linearization of omega[i,b] * phi[i,k]
    theta = {}
    for i in V:
        for k in K:
            for b in B:
                theta[i, k, b] = model.addVar(
                    vtype=GRB.BINARY, name=f"theta_{i}_{k}_{b}"
                )

    model.update()

    # -------------------------------------------------------------------------
    # Objective (34): Linearized objective
    # -------------------------------------------------------------------------
    # min sum_i c^p_i * (tau_a_plus_i + tau_b_plus_i)
    #   + c^o * sum_i sum_b sum_k theta_ikb * D^L_kb * (l_i / r_i)
    #   + c^o * sum_i sum_b omega_ib * D^U_b * u_i

    obj = gp.LinExpr()

    # Penalty cost term
    for i in V:
        obj += cp[i] * (tau_a_plus[i] + tau_b_plus[i])

    # Loading route cost term
    for i in V:
        li_over_ri = l[i] / r[i]
        for b in B:
            for k in K:
                obj += co * theta[i, k, b] * D_L[k][b] * li_over_ri

    # Unloading route cost term
    for i in V:
        for b in B:
            obj += co * omega[i, b] * D_U[b] * u[i]

    model.setObjective(obj, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------

    # (2) Each subblock reserved for at most one vessel:
    #     sum_i phi[i,k] <= 1, for all k
    for k in K:
        model.addConstr(
            gp.quicksum(phi[i, k] for i in V) <= 1,
            name=f"c2_subblock_{k}"
        )

    # (3) Exactly r_i subblocks reserved for vessel i:
    #     sum_k phi[i,k] = r_i, for all i
    for i in V:
        model.addConstr(
            gp.quicksum(phi[i, k] for k in K) == r[i],
            name=f"c3_reserved_{i}"
        )

    # (4) Exactly one QC profile assigned to each vessel:
    #     sum_p gamma[i,p] = 1, for all i
    for i in V:
        model.addConstr(
            gp.quicksum(gamma[i, p_idx] for p_idx in P[i]) == 1,
            name=f"c4_profile_{i}"
        )

    # (5) Each vessel allocated to exactly one berth:
    #     sum_b omega[i,b] = 1, for all i
    for i in V:
        model.addConstr(
            gp.quicksum(omega[i, b] for b in B) == 1,
            name=f"c5_berth_{i}"
        )

    # (6) Each vessel starts handling in exactly one time step (within horizon):
    #     sum_{t=1..H} mu[i,t] = 1, for all i
    for i in V:
        model.addConstr(
            gp.quicksum(mu[i, t] for t in T_H) == 1,
            name=f"c6_start_{i}"
        )

    # (7) Link mu and alpha:
    #     sum_t mu[i,t] * t = alpha[i], for all i
    #     Note: sum over T (1..H+E) in the math model, but mu[i,t] is only
    #     defined for t in 1..H (constraint 6 ensures sum over 1..H = 1).
    #     We sum over T_H since mu is only created for t in 1..H.
    for i in V:
        model.addConstr(
            gp.quicksum(mu[i, t] * t for t in T_H) == alpha[i],
            name=f"c7_link_alpha_{i}"
        )

    # (8) Link start and end time steps:
    #     alpha[i] + sum_p gamma[i,p] * h[i][p] - 1 = beta[i], for all i
    for i in V:
        model.addConstr(
            alpha[i]
            + gp.quicksum(gamma[i, p_idx] * h[i][p_idx] for p_idx in P[i])
            - 1
            == beta[i],
            name=f"c8_end_{i}"
        )

    # (9) Sequencing constraint:
    #     alpha[i] + sum_p gamma[i,p]*h[i][p] <= alpha[j] + (1-delta[i,j,b])*M
    #     for all i != j, b
    for i in V:
        for j in V:
            if i != j:
                for b in B:
                    model.addConstr(
                        alpha[i]
                        + gp.quicksum(
                            gamma[i, p_idx] * h[i][p_idx] for p_idx in P[i]
                        )
                        <= alpha[j] + (1 - delta[i, j, b]) * M_big,
                        name=f"c9_seq_{i}_{j}_{b}"
                    )

    # (10) delta[i,j,b] + delta[j,i,b] <= omega[i,b], for all i!=j, b
    for i in V:
        for j in V:
            if i != j:
                for b in B:
                    model.addConstr(
                        delta[i, j, b] + delta[j, i, b] <= omega[i, b],
                        name=f"c10_seq_{i}_{j}_{b}"
                    )

    # (11) delta[i,j,b] + delta[j,i,b] >= omega[i,b] + omega[j,b] - 1,
    #      for all i!=j, b
    for i in V:
        for j in V:
            if i != j:
                for b in B:
                    model.addConstr(
                        delta[i, j, b] + delta[j, i, b]
                        >= omega[i, b] + omega[j, b] - 1,
                        name=f"c11_seq_{i}_{j}_{b}"
                    )

    # (12) alpha[i] >= a_i^f, for all i
    for i in V:
        model.addConstr(alpha[i] >= af[i], name=f"c12_feas_start_{i}")

    # (13) beta[i] <= b_i^f, for all i
    for i in V:
        model.addConstr(beta[i] <= bf[i], name=f"c13_feas_end_{i}")

    # (14) eta[i,p,t] >= gamma[i,p] + mu[i,t] - 1,
    #      for all i, p in P_i, t in T
    #      Note: mu[i,t] is only defined for t in 1..H.
    #      For t > H, mu[i,t] = 0, so eta[i,p,t] should be 0
    #      (vessels can only start within the planning horizon).
    for i in V:
        for p_idx in P[i]:
            for t in T_H:
                model.addConstr(
                    eta[i, p_idx, t] >= gamma[i, p_idx] + mu[i, t] - 1,
                    name=f"c14_eta_{i}_{p_idx}_{t}"
                )
            # For t > H, eta must be 0 since vessel cannot start after horizon
            for t in range(H + 1, H + E + 1):
                model.addConstr(
                    eta[i, p_idx, t] == 0,
                    name=f"c14_eta_zero_{i}_{p_idx}_{t}"
                )

    # (15) sigma[t] = sum_i sum_p sum_{m=max(1,t-h_ip+1)}^{t}
    #                 eta[i,p,m] * q[i][p][t-m]
    #      for all t in T
    #
    #      Explanation: if vessel i starts at time m using profile p, then at
    #      time t the vessel is at its (t - m + 1)-th handling step (1-indexed).
    #      The number of QCs used at that step is q[i][p][t-m] (0-indexed in
    #      our list, since step 1 = index 0, step (t-m+1) = index (t-m)).
    #      The summation range ensures 1 <= t - m + 1 <= h_ip, i.e.,
    #      m ranges from max(1, t - h_ip + 1) to t.
    #      Also, m must be in T_all (1..H+E), but since m <= t and t <= H+E,
    #      m is always valid on the upper end. We also need m >= 1.
    #      Additionally, eta[i,p,m] is only meaningful when m is a valid start
    #      time; however, the variable exists for all t in T_all.
    for t in T_all:
        expr = gp.LinExpr()
        for i in V:
            for p_idx in P[i]:
                h_ip = h[i][p_idx]
                m_start = max(1, t - h_ip + 1)
                m_end = t  # inclusive
                for m in range(m_start, m_end + 1):
                    # step index (0-based): t - m
                    step_0 = t - m
                    if step_0 < h_ip:
                        expr += eta[i, p_idx, m] * q[i][p_idx][step_0]
        model.addConstr(sigma[t] == expr, name=f"c15_sigma_{t}")

    # (16) sigma[t] <= Q_t, for t in {E+1, ..., H}
    #      Q_cap is 0-indexed: Q_cap[t-1] = Q_t
    for t in range(E + 1, H + 1):
        model.addConstr(
            sigma[t] <= Q_cap[t - 1],
            name=f"c16_qc_cap_{t}"
        )

    # (17) sigma[t] + sigma[t+H] <= Q_t, for t in {1, ..., E}
    for t in range(1, E + 1):
        model.addConstr(
            sigma[t] + sigma[t + H] <= Q_cap[t - 1],
            name=f"c17_qc_period_{t}"
        )

    # (18) rho[b] <= alpha[i] + (1 - omega[i,b]) * M, for all i, b
    for i in V:
        for b in B:
            model.addConstr(
                rho[b] <= alpha[i] + (1 - omega[i, b]) * M_big,
                name=f"c18_rho_{i}_{b}"
            )

    # (19) varsigma[b] >= beta[i] + (omega[i,b] - 1) * M, for all i, b
    for i in V:
        for b in B:
            model.addConstr(
                varsigma[b] >= beta[i] + (omega[i, b] - 1) * M_big,
                name=f"c19_varsigma_{i}_{b}"
            )

    # (20) varsigma[b] - rho[b] <= H - 1, for all b
    for b in B:
        model.addConstr(
            varsigma[b] - rho[b] <= H - 1,
            name=f"c20_period_{b}"
        )

    # (29) a_i^e - alpha[i] = tau_a_plus[i] - tau_a_minus[i], for all i
    for i in V:
        model.addConstr(
            ae[i] - alpha[i] == tau_a_plus[i] - tau_a_minus[i],
            name=f"c29_tau_a_{i}"
        )

    # (30) beta[i] - b_i^e = tau_b_plus[i] - tau_b_minus[i], for all i
    for i in V:
        model.addConstr(
            beta[i] - be[i] == tau_b_plus[i] - tau_b_minus[i],
            name=f"c30_tau_b_{i}"
        )

    # (31) tau variables >= 0 (already set as lb=0 during variable creation)

    # (32) theta[i,k,b] >= omega[i,b] + phi[i,k] - 1,
    #      for all i, k, b
    for i in V:
        for k in K:
            for b in B:
                model.addConstr(
                    theta[i, k, b] >= omega[i, b] + phi[i, k] - 1,
                    name=f"c32_theta_{i}_{k}_{b}"
                )

    # (33) theta[i,k,b] in {0,1} (already set as BINARY during creation)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.update()
    print(f"Model has {model.NumVars} variables and {model.NumConstrs} constraints.")

    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    result = {
        "instance_path": None,  # will be set by caller
        "solver": "Gurobi",
        "model": "M2_linearized_MILP",
        "time_limit": time_limit,
        "solve_time_seconds": model.Runtime,
        "status": None,
        "objective_value": None,
        "mip_gap": None,
        "vessels": []
    }

    if model.Status == GRB.OPTIMAL:
        result["status"] = "optimal"
        result["objective_value"] = model.ObjVal
        result["mip_gap"] = 0.0
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        result["status"] = "time_limit_feasible"
        result["objective_value"] = model.ObjVal
        result["mip_gap"] = model.MIPGap
    elif model.Status == GRB.TIME_LIMIT and model.SolCount == 0:
        result["status"] = "time_limit_no_solution"
        return result
    elif model.Status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
        return result
    else:
        result["status"] = f"gurobi_status_{model.Status}"
        if model.SolCount > 0:
            result["objective_value"] = model.ObjVal
            result["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
        return result

    # Extract detailed vessel assignments
    for i in V:
        vessel_info = {
            "vessel_id": vessels[i]["vessel_id"],
            "alpha": round(alpha[i].X),
            "beta": round(beta[i].X),
            "berth": None,
            "profile": None,
            "subblocks": [],
        }

        # Find assigned berth
        for b in B:
            if omega[i, b].X > 0.5:
                vessel_info["berth"] = b
                break

        # Find assigned QC profile
        for p_idx in P[i]:
            if gamma[i, p_idx].X > 0.5:
                vessel_info["profile"] = p_idx
                vessel_info["handling_time"] = h[i][p_idx]
                vessel_info["qc_per_timestep"] = q[i][p_idx]
                break

        # Find reserved subblocks
        for k in K:
            if phi[i, k].X > 0.5:
                vessel_info["subblocks"].append(k)

        result["vessels"].append(vessel_info)

    # Derive berth windows from the selected vessel assignments. The rho and
    # varsigma model variables are valid bounds but are not optimized to equal
    # the actual extrema, so exporting their raw values is ambiguous.
    result["berths"] = []
    for b in B:
        assigned = [v for v in result["vessels"] if v["berth"] == b]
        if assigned:
            rho_value = min(v["alpha"] for v in assigned)
            varsigma_value = max(v["beta"] for v in assigned)
        else:
            rho_value = None
            varsigma_value = None
        result["berths"].append({
            "berth": b,
            "rho": rho_value,
            "varsigma": varsigma_value,
        })

    # Derive QC usage from selected profiles. The eta variables only need the
    # lower linking inequality in M1 and may take extra nonzero values, so raw
    # sigma can overstate the usage of the exported assignment.
    result["qc_usage"] = {}
    derived_qc = {t: 0 for t in T_all}
    for vessel in result["vessels"]:
        for offset, qc in enumerate(vessel["qc_per_timestep"]):
            derived_qc[vessel["alpha"] + offset] += qc
    for t, val in derived_qc.items():
        if val > 0.01:
            result["qc_usage"][str(t)] = float(round(val, 2))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve the integrated berth allocation, QC assignment, "
                    "and yard assignment problem using linearized MILP [M2]."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path to write the JSON solution file."
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Time limit in seconds for the Gurobi solver (default: 3600)."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)
    print(f"Loaded instance: {args.instance_path}")
    print(f"  Vessels: {data['num_vessels']}, Berths: {data['num_berths']}, "
          f"Subblocks: {data['num_subblocks']}, H: {data['H']}, E: {data['E']}")

    # Build and solve
    result = build_and_solve(data, args.time_limit)
    result["instance_path"] = args.instance_path

    # Write solution
    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"\nSolution written to: {args.solution_path}")
    print(f"  Status: {result['status']}")
    print(f"  Objective value: {result['objective_value']}")
    if result["mip_gap"] is not None:
        print(f"  MIP gap: {result['mip_gap']:.6f}")
    print(f"  Runtime: {result['runtime']:.2f} seconds")


if __name__ == "__main__":
    main()
