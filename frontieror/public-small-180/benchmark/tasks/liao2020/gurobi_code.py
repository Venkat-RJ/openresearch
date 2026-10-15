#!/usr/bin/env python3
"""
Gurobi implementation of the DR-bimodal MILP (Equation 15) for the
Distributionally Robust Outpatient Colonoscopy Scheduling (DROCS) problem.

Reference: Shehadeh, Cohn, Jiang (2019), "A Distributionally Robust Optimization
Approach for Outpatient Colonoscopy Scheduling", European Journal of Operational
Research, 283(2), 549-561.

This implements Proposition 4 (Equation 15) -- the MILP reformulation that
jointly optimizes patient sequencing and appointment scheduling under
distributional ambiguity with bimodal procedure durations.
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


def solve_drocs(instance_path, solution_path, time_limit,
                fixed_assignment=None, fixed_schedule=None, quiet=False):
    # ----------------------------------------------------------------
    # Load instance data
    # ----------------------------------------------------------------
    with open(instance_path) as f:
        instance = json.load(f)

    P = instance["num_patients_P"]
    L = instance["provider_service_hours_L_minutes"]

    # Use the first cost structure (Cost1)
    cost_key = list(instance["cost_structures"].keys())[0]
    cost = instance["cost_structures"][cost_key]
    c_w = cost["c_w"]
    c_g = cost["c_g"]
    c_o = cost["c_o"]

    patients = instance["patients"]

    # ----------------------------------------------------------------
    # Patient parameters (1-indexed: p = 1,...,P)
    # Convention: index P+1 has zero values for durations and arrivals.
    # ----------------------------------------------------------------
    mu_A = {}
    mu_I = {}
    mu_u = {}
    mu_q = {}
    d_AL = {}
    d_AU = {}
    d_IL = {}
    d_IU = {}
    u_L = {}
    u_U = {}

    for p in range(1, P + 1):
        pat = patients[p - 1]
        mu_A[p] = pat["mean_duration_adequate_prep"]
        mu_I[p] = pat["mean_duration_inadequate_prep"]
        mu_u[p] = pat["mean_arrival_time_deviation"]
        mu_q[p] = pat["mean_prep_adequacy"]
        d_AL[p] = pat["lower_bound_duration_adequate_prep"]
        d_AU[p] = pat["upper_bound_duration_adequate_prep"]
        d_IL[p] = pat["lower_bound_duration_inadequate_prep"]
        d_IU[p] = pat["upper_bound_duration_inadequate_prep"]
        u_L[p] = pat["lower_bound_arrival_time_deviation"]
        u_U[p] = pat["upper_bound_arrival_time_deviation"]

    # Convention: P+1 values are zero
    d_AL[P + 1] = 0.0
    d_AU[P + 1] = 0.0
    d_IL[P + 1] = 0.0
    d_IU[P + 1] = 0.0
    u_L[P + 1] = 0.0
    u_U[P + 1] = 0.0

    # ----------------------------------------------------------------
    # Cost arrays indexed by position (paper's 1-indexed)
    # c_w_pos[i]: c_w for i=1,...,P; c_o for i=P+1
    # c_g_pos[j]: c_g for j=1,...,P; 0 for j=P+1; 0 for j=P+2
    # ----------------------------------------------------------------
    c_w_pos = {}
    for i in range(1, P + 1):
        c_w_pos[i] = c_w
    c_w_pos[P + 1] = c_o

    c_g_pos = {}
    for j in range(1, P + 1):
        c_g_pos[j] = c_g
    c_g_pos[P + 1] = 0.0
    c_g_pos[P + 2] = 0.0

    # ----------------------------------------------------------------
    # Precompute pi[i][j] for 1 <= i <= j <= P+2
    # pi[i,j] = -c_g_pos[j] + sum_{ell=i}^{j-1} c_w_pos[ell]
    # Special case: pi[P+2, P+2] = 0
    # ----------------------------------------------------------------
    pi = {}
    for i in range(1, P + 3):
        pi[i] = {}
        for j in range(i, P + 3):
            if i == P + 2 and j == P + 2:
                pi[i][j] = 0.0
            else:
                pi[i][j] = -c_g_pos[j] + sum(
                    c_w_pos[ell] for ell in range(i, j)
                )

    # ----------------------------------------------------------------
    # Precomputed patient-level quantities
    # ----------------------------------------------------------------
    # min(d_AL[p], d_IL[p]) and max(d_AU[p], d_IU[p]) for each patient
    min_d_pat = {p: min(d_AL[p], d_IL[p]) for p in range(1, P + 2)}
    max_d_pat = {p: max(d_AU[p], d_IU[p]) for p in range(1, P + 2)}

    # ----------------------------------------------------------------
    # Bounds for McCormick linearization (from Appendix G structure)
    # ----------------------------------------------------------------
    # For eta[p,i] = x[p,i]*rho[i]: rho bounded by [d_AL[p], d_AU[p]]
    # For tau[p,i] = x[p,i]*alpha[i]: alpha bounded by [d_IL[p], d_IU[p]]
    # For nu[p,i] = x[p,i]*lambda[i]: lambda bounded by [u_L[p], u_U[p]]
    # For Gamma[p,i] = x[p,i]*gamma[i]: gamma bounded by [-M_gam, M_gam]
    # For zeta[p,i] = x[p,i]*z[i]: z bounded by [0, d_AU[p]]
    # For psi[p,i] = x[p,i]*v[i]: v bounded by [0, d_IU[p]]
    # For phi[p,i] = x[p,i]*r[i]: r bounded by [0, u_U[p]]

    # Global bounds on dual variables (for variable declaration)
    max_d_AU = max(d_AU[p] for p in range(1, P + 1))
    min_d_AL = min(d_AL[p] for p in range(1, P + 1))
    max_d_IU = max(d_IU[p] for p in range(1, P + 1))
    min_d_IL = min(d_IL[p] for p in range(1, P + 1))
    max_u_U = max(u_U[p] for p in range(1, P + 1))
    min_u_L = min(u_L[p] for p in range(1, P + 1))

    # Bound for gamma: use a big-M derived from pi values
    max_pi_abs = max(
        abs(pi[i][j]) for i in range(1, P + 3) for j in range(i, P + 3)
    )
    M_gam = max_pi_abs + L

    # ----------------------------------------------------------------
    # Build Gurobi model
    # ----------------------------------------------------------------
    model = gp.Model("DR_bimodal_DROCS")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0 if quiet else 1)

    # ================================================================
    # DECISION VARIABLES
    # ================================================================

    # x[p,i]: binary assignment of patient p to position i
    x = model.addVars(
        range(1, P + 1), range(1, P + 1), vtype=GRB.BINARY, name="x"
    )

    # t[i]: scheduled start time of appointment position i
    t = model.addVars(range(1, P + 1), lb=0.0, ub=L, name="t")

    # Trusted-checker mode: fix the public first-stage witness and optimize
    # only the omitted DRO dual/linearization variables.
    if fixed_assignment is not None:
        normalized_assignment = {
            int(patient): int(position)
            for patient, position in fixed_assignment.items()
        }
        for p in range(1, P + 1):
            for i in range(1, P + 1):
                fixed_value = 1.0 if normalized_assignment[p] == i else 0.0
                x[p, i].lb = fixed_value
                x[p, i].ub = fixed_value
    if fixed_schedule is not None:
        normalized_schedule = {
            int(position): float(start)
            for position, start in fixed_schedule.items()
        }
        for i in range(1, P + 1):
            t[i].lb = normalized_schedule[i]
            t[i].ub = normalized_schedule[i]

    # Dual variables (free, but with global bounds for McCormick validity)
    rho = model.addVars(
        range(1, P + 1), lb=min_d_AL, ub=max_d_AU, name="rho"
    )
    alpha = model.addVars(
        range(1, P + 1), lb=min_d_IL, ub=max_d_IU, name="alpha"
    )
    lam = model.addVars(
        range(1, P + 1), lb=min_u_L, ub=max_u_U, name="lambda"
    )
    gam = model.addVars(
        range(1, P + 1), lb=-M_gam, ub=M_gam, name="gamma"
    )

    # beta[i] for i=1,...,P+2; beta[P+2] >= 0
    beta = model.addVars(
        range(1, P + 3), lb=-GRB.INFINITY, name="beta"
    )
    beta[P + 2].lb = 0.0

    # z[i] >= 0, z[i] >= rho[i]
    z = model.addVars(range(1, P + 1), lb=0.0, ub=max_d_AU, name="z")
    # v[i] >= 0, v[i] >= alpha[i]
    v = model.addVars(range(1, P + 1), lb=0.0, ub=max_d_IU, name="v")
    # r[i] >= 0, r[i] >= lambda[i]
    r = model.addVars(range(1, P + 1), lb=0.0, ub=max_u_U, name="r")
    # e[i] >= 0, e[i] >= -gamma[i]
    e = model.addVars(range(1, P + 1), lb=0.0, ub=M_gam, name="e")

    # ================================================================
    # McCormick linearization variables
    # ================================================================
    # eta[p,i] = x[p,i] * rho[i]
    eta = model.addVars(
        range(1, P + 1), range(1, P + 1),
        lb=-GRB.INFINITY, ub=GRB.INFINITY, name="eta"
    )
    # tau[p,i] = x[p,i] * alpha[i]
    tau = model.addVars(
        range(1, P + 1), range(1, P + 1),
        lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tau"
    )
    # nu[p,i] = x[p,i] * lambda[i]
    nu = model.addVars(
        range(1, P + 1), range(1, P + 1),
        lb=-GRB.INFINITY, ub=GRB.INFINITY, name="nu"
    )
    # Gam_mc[p,i] = x[p,i] * gamma[i]
    Gam_mc = model.addVars(
        range(1, P + 1), range(1, P + 1),
        lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Gam"
    )
    # zeta[p,i] = x[p,i] * z[i]
    zeta = model.addVars(
        range(1, P + 1), range(1, P + 1), lb=0.0, ub=GRB.INFINITY, name="zeta"
    )
    # psi[p,i] = x[p,i] * v[i]
    psi = model.addVars(
        range(1, P + 1), range(1, P + 1), lb=0.0, ub=GRB.INFINITY, name="psi"
    )
    # phi[p,i] = x[p,i] * r[i]
    phi = model.addVars(
        range(1, P + 1), range(1, P + 1), lb=0.0, ub=GRB.INFINITY, name="phi"
    )

    # ================================================================
    # OBJECTIVE FUNCTION (Equation 15a)
    # ================================================================
    # min sum_{i=1}^{P} sum_{p=1}^{P} (mu_A[p]*eta[p,i] + mu_I[p]*tau[p,i]
    #     + mu_u[p]*nu[p,i] + mu_q[p]*Gam_mc[p,i]) + sum_{i=1}^{P+2} beta[i]
    obj = gp.quicksum(
        mu_A[p] * eta[p, i]
        + mu_I[p] * tau[p, i]
        + mu_u[p] * nu[p, i]
        + mu_q[p] * Gam_mc[p, i]
        for i in range(1, P + 1)
        for p in range(1, P + 1)
    ) + gp.quicksum(beta[i] for i in range(1, P + 3))

    model.setObjective(obj, GRB.MINIMIZE)

    # ================================================================
    # CONSTRAINTS
    # ================================================================

    # ----------------------------------------------------------------
    # (15b / Eq 1) Assignment constraints
    # ----------------------------------------------------------------
    for i in range(1, P + 1):
        model.addConstr(
            gp.quicksum(x[p, i] for p in range(1, P + 1)) == 1,
            name=f"assign_pos_{i}",
        )
    for p in range(1, P + 1):
        model.addConstr(
            gp.quicksum(x[p, i] for i in range(1, P + 1)) == 1,
            name=f"assign_pat_{p}",
        )

    # ----------------------------------------------------------------
    # (15b / Eq 2) Scheduling constraints
    # ----------------------------------------------------------------
    model.addConstr(t[1] == 0.0, name="t1_zero")
    for i in range(2, P + 1):
        model.addConstr(t[i] >= t[i - 1], name=f"t_order_{i}")

    # ----------------------------------------------------------------
    # (13d) Sign constraints on auxiliary variables
    # ----------------------------------------------------------------
    for i in range(1, P + 1):
        model.addConstr(z[i] >= rho[i], name=f"z_ge_rho_{i}")
        model.addConstr(v[i] >= alpha[i], name=f"v_ge_alpha_{i}")
        model.addConstr(r[i] >= lam[i], name=f"r_ge_lam_{i}")
        model.addConstr(e[i] >= -gam[i], name=f"e_ge_neg_gam_{i}")

    # ----------------------------------------------------------------
    # McCormick linearization constraints (Appendix F)
    # ----------------------------------------------------------------
    for p in range(1, P + 1):
        for i in range(1, P + 1):
            # ----------------------------------------------------------
            # eta[p,i] = x[p,i] * rho[i]; rho bounds: [d_AL[p], d_AU[p]]
            # ----------------------------------------------------------
            LB_r = d_AL[p]
            UB_r = d_AU[p]
            model.addConstr(
                eta[p, i] <= UB_r * x[p, i], name=f"mc_eta_ub1_{p}_{i}"
            )
            model.addConstr(
                eta[p, i] >= LB_r * x[p, i], name=f"mc_eta_lb1_{p}_{i}"
            )
            model.addConstr(
                eta[p, i] <= rho[i] - LB_r * (1 - x[p, i]),
                name=f"mc_eta_ub2_{p}_{i}",
            )
            model.addConstr(
                eta[p, i] >= rho[i] - UB_r * (1 - x[p, i]),
                name=f"mc_eta_lb2_{p}_{i}",
            )

            # ----------------------------------------------------------
            # tau[p,i] = x[p,i] * alpha[i]; alpha bounds: [d_IL[p], d_IU[p]]
            # ----------------------------------------------------------
            LB_a = d_IL[p]
            UB_a = d_IU[p]
            model.addConstr(
                tau[p, i] <= UB_a * x[p, i], name=f"mc_tau_ub1_{p}_{i}"
            )
            model.addConstr(
                tau[p, i] >= LB_a * x[p, i], name=f"mc_tau_lb1_{p}_{i}"
            )
            model.addConstr(
                tau[p, i] <= alpha[i] - LB_a * (1 - x[p, i]),
                name=f"mc_tau_ub2_{p}_{i}",
            )
            model.addConstr(
                tau[p, i] >= alpha[i] - UB_a * (1 - x[p, i]),
                name=f"mc_tau_lb2_{p}_{i}",
            )

            # ----------------------------------------------------------
            # nu[p,i] = x[p,i] * lambda[i]; lambda bounds: [u_L[p], u_U[p]]
            # ----------------------------------------------------------
            LB_l = u_L[p]
            UB_l = u_U[p]
            model.addConstr(
                nu[p, i] <= UB_l * x[p, i], name=f"mc_nu_ub1_{p}_{i}"
            )
            model.addConstr(
                nu[p, i] >= LB_l * x[p, i], name=f"mc_nu_lb1_{p}_{i}"
            )
            model.addConstr(
                nu[p, i] <= lam[i] - LB_l * (1 - x[p, i]),
                name=f"mc_nu_ub2_{p}_{i}",
            )
            model.addConstr(
                nu[p, i] >= lam[i] - UB_l * (1 - x[p, i]),
                name=f"mc_nu_lb2_{p}_{i}",
            )

            # ----------------------------------------------------------
            # Gam_mc[p,i] = x[p,i] * gamma[i]; gamma bounds: [-M_gam, M_gam]
            # ----------------------------------------------------------
            model.addConstr(
                Gam_mc[p, i] <= M_gam * x[p, i],
                name=f"mc_Gam_ub1_{p}_{i}",
            )
            model.addConstr(
                Gam_mc[p, i] >= -M_gam * x[p, i],
                name=f"mc_Gam_lb1_{p}_{i}",
            )
            model.addConstr(
                Gam_mc[p, i] <= gam[i] - (-M_gam) * (1 - x[p, i]),
                name=f"mc_Gam_ub2_{p}_{i}",
            )
            model.addConstr(
                Gam_mc[p, i] >= gam[i] - M_gam * (1 - x[p, i]),
                name=f"mc_Gam_lb2_{p}_{i}",
            )

            # ----------------------------------------------------------
            # zeta[p,i] = x[p,i] * z[i]; z bounds: [0, d_AU[p]]
            # ----------------------------------------------------------
            UB_z = d_AU[p]
            model.addConstr(
                zeta[p, i] <= UB_z * x[p, i], name=f"mc_zeta_ub1_{p}_{i}"
            )
            # zeta >= 0 already from lb
            model.addConstr(
                zeta[p, i] <= z[i], name=f"mc_zeta_ub2_{p}_{i}"
            )
            model.addConstr(
                zeta[p, i] >= z[i] - UB_z * (1 - x[p, i]),
                name=f"mc_zeta_lb2_{p}_{i}",
            )

            # ----------------------------------------------------------
            # psi[p,i] = x[p,i] * v[i]; v bounds: [0, d_IU[p]]
            # ----------------------------------------------------------
            UB_v = d_IU[p]
            model.addConstr(
                psi[p, i] <= UB_v * x[p, i], name=f"mc_psi_ub1_{p}_{i}"
            )
            # psi >= 0 already from lb
            model.addConstr(
                psi[p, i] <= v[i], name=f"mc_psi_ub2_{p}_{i}"
            )
            model.addConstr(
                psi[p, i] >= v[i] - UB_v * (1 - x[p, i]),
                name=f"mc_psi_lb2_{p}_{i}",
            )

            # ----------------------------------------------------------
            # phi[p,i] = x[p,i] * r[i]; r bounds: [0, u_U[p]]
            # ----------------------------------------------------------
            UB_rr = u_U[p]
            model.addConstr(
                phi[p, i] <= UB_rr * x[p, i], name=f"mc_phi_ub1_{p}_{i}"
            )
            # phi >= 0 already from lb
            model.addConstr(
                phi[p, i] <= r[i], name=f"mc_phi_ub2_{p}_{i}"
            )
            model.addConstr(
                phi[p, i] >= r[i] - UB_rr * (1 - x[p, i]),
                name=f"mc_phi_lb2_{p}_{i}",
            )

    # ----------------------------------------------------------------
    # Helper: position-dependent parameter expressions
    # When patient p is assigned to position pos, the position inherits
    # that patient's parameters. These are linear in x.
    # ----------------------------------------------------------------
    def u_L_at(pos):
        """u_L at position pos (1-indexed). Returns 0 for pos > P."""
        if 1 <= pos <= P:
            return gp.quicksum(u_L[p] * x[p, pos] for p in range(1, P + 1))
        return 0.0

    def u_U_at(pos):
        """u_U at position pos (1-indexed). Returns 0 for pos > P."""
        if 1 <= pos <= P:
            return gp.quicksum(u_U[p] * x[p, pos] for p in range(1, P + 1))
        return 0.0

    def min_d_at(pos):
        """min(d_AL, d_IL) at position pos. Returns 0 for pos > P."""
        if 1 <= pos <= P:
            return gp.quicksum(
                min_d_pat[p] * x[p, pos] for p in range(1, P + 1)
            )
        return 0.0

    def max_d_at(pos):
        """max(d_AU, d_IU) at position pos. Returns 0 for pos > P."""
        if 1 <= pos <= P:
            return gp.quicksum(
                max_d_pat[p] * x[p, pos] for p in range(1, P + 1)
            )
        return 0.0

    # ----------------------------------------------------------------
    # K'_i (linearized via McCormick variables), for i = 1,...,P
    #
    # K'_i = -(sum_p d_AU[p]*eta[p,i] + sum_p (d_AL[p]-d_AU[p])*zeta[p,i])
    #        -(sum_p d_IU[p]*tau[p,i] + sum_p (d_IL[p]-d_IU[p])*psi[p,i])
    #        -(sum_p u_U[p]*nu[p,i]   + sum_p (u_L[p]-u_U[p])*phi[p,i])
    #        + e[i]
    # ----------------------------------------------------------------
    def K_prime(i):
        return (
            -gp.quicksum(d_AU[p] * eta[p, i] for p in range(1, P + 1))
            - gp.quicksum(
                (d_AL[p] - d_AU[p]) * zeta[p, i] for p in range(1, P + 1)
            )
            - gp.quicksum(d_IU[p] * tau[p, i] for p in range(1, P + 1))
            - gp.quicksum(
                (d_IL[p] - d_IU[p]) * psi[p, i] for p in range(1, P + 1)
            )
            - gp.quicksum(u_U[p] * nu[p, i] for p in range(1, P + 1))
            - gp.quicksum(
                (u_L[p] - u_U[p]) * phi[p, i] for p in range(1, P + 1)
            )
            + e[i]
        )

    # ----------------------------------------------------------------
    # Constraint (13b): for j = 1, ..., P+2
    #
    # sum_{i=1}^{j} beta[i] >=
    #   (-t[1] - u_L_at(1)) * pi[1,j]
    #   + sum_{i=2}^{min(j,P)} (-t[i] - u_L_at(i) + t[i-1]
    #       + u_L_at(i-1) + min_d_at(i-1)) * pi[i,j]
    #   + sum_{i=2}^{min(j,P+1)} (max_d_at(i-1)
    #       - min_d_at(i-1)) * max(pi[i,j], 0)
    #   + sum_{i=1}^{min(j,P)} K'_i
    #   + [if j >= P+1]: (t[P] + u_L_at(P) - L
    #       + min_d_at(P)) * pi[P+1,j]
    # ----------------------------------------------------------------
    for j in range(1, P + 3):
        lhs = gp.quicksum(beta[i] for i in range(1, j + 1))
        rhs = 0

        # First term: i=1 contributes if j >= 1
        rhs += (-t[1] - u_L_at(1)) * pi[1][j]

        # Second term: i=2,...,min(j,P)
        for i in range(2, min(j, P) + 1):
            rhs += (
                -t[i] - u_L_at(i) + t[i - 1] + u_L_at(i - 1) + min_d_at(i - 1)
            ) * pi[i][j]

        # Third term: i=2,...,min(j,P+1)
        for i in range(2, min(j, P + 1) + 1):
            pi_plus = max(pi[i][j], 0.0)
            if pi_plus > 0:
                rhs += (max_d_at(i - 1) - min_d_at(i - 1)) * pi_plus

        # Fourth term: sum of K'_i for i=1,...,min(j,P)
        for i in range(1, min(j, P) + 1):
            rhs += K_prime(i)

        # Fifth term: overtime (i=P+1), only if j >= P+1
        if j >= P + 1:
            rhs += (
                t[P] + u_L_at(P) - L + min_d_at(P)
            ) * pi[P + 1][j]

        model.addConstr(lhs >= rhs, name=f"c13b_j{j}")

    # ----------------------------------------------------------------
    # Constraint (13c): for k = 2,...,P+1; j = k,...,P+2
    #
    # sum_{i=k}^{j} beta[i] >=
    #   (u_U_at(k-1) - u_L_at(k-1)) * max(pi[k,j]+c_g_pos[k-1], 0)
    #   + sum_{i=min(k,P+1)}^{min(j,P)} (-t[i] - u_L_at(i) + t[i-1]
    #       + u_L_at(i-1) + min_d_at(i-1)) * pi[i,j]
    #   + sum_{i=min(k,P+1)}^{min(j,P+1)} (max_d_at(i-1)
    #       - min_d_at(i-1)) * max(pi[i,j], 0)
    #   + sum_{i=min(k,P+1)}^{min(j,P)} K'_i
    #   + [if j >= P+1 and P+1 >= k]: (t[P] + u_L_at(P) - L
    #       + min_d_at(P)) * pi[P+1,j]
    # ----------------------------------------------------------------
    for k in range(2, P + 2):
        for j in range(k, P + 3):
            lhs = gp.quicksum(beta[i] for i in range(k, j + 1))
            rhs = 0

            # First term: (u_U_at(k-1) - u_L_at(k-1)) * max(pi[k,j] + c_g_pos[k-1], 0)
            pi_plus_cg = max(pi[k][j] + c_g_pos[k - 1], 0.0)
            if pi_plus_cg > 0:
                rhs += (u_U_at(k - 1) - u_L_at(k - 1)) * pi_plus_cg

            # Starting index for the sums
            i_start = min(k, P + 1)

            # Second term
            for i in range(i_start, min(j, P) + 1):
                rhs += (
                    -t[i]
                    - u_L_at(i)
                    + t[i - 1]
                    + u_L_at(i - 1)
                    + min_d_at(i - 1)
                ) * pi[i][j]

            # Third term
            for i in range(i_start, min(j, P + 1) + 1):
                pi_plus = max(pi[i][j], 0.0)
                if pi_plus > 0:
                    rhs += (max_d_at(i - 1) - min_d_at(i - 1)) * pi_plus

            # Fourth term
            for i in range(i_start, min(j, P) + 1):
                rhs += K_prime(i)

            # Fifth term: overtime, only if P+1 is in range [k, j]
            if j >= P + 1 and k <= P + 1:
                rhs += (
                    t[P] + u_L_at(P) - L + min_d_at(P)
                ) * pi[P + 1][j]

            model.addConstr(lhs >= rhs, name=f"c13c_k{k}_j{j}")

    # ================================================================
    # SOLVE
    # ================================================================
    model.optimize()

    # ================================================================
    # EXTRACT SOLUTION
    # ================================================================
    result = {
        "instance_path": instance_path,
        "cost_structure": cost_key,
        "c_w": c_w,
        "c_g": c_g,
        "c_o": c_o,
        "num_patients": P,
        "session_length": L,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound
        result["mip_gap"] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0
        result["status"] = (
            "optimal" if model.Status == GRB.OPTIMAL else f"status_{model.Status}"
        )
        result["solve_time_seconds"] = model.Runtime

        # Patient-to-position assignment
        assignment = {}
        for p in range(1, P + 1):
            for i in range(1, P + 1):
                if x[p, i].X > 0.5:
                    assignment[str(p)] = i
                    break
        result["assignment"] = assignment

        # Appointment schedule (start times indexed by position)
        schedule = {}
        for i in range(1, P + 1):
            schedule[str(i)] = round(t[i].X, 6)
        result["schedule"] = schedule

        # Appointment schedule indexed by patient (for convenience)
        patient_schedule = {}
        for p_str, pos in assignment.items():
            patient_schedule[p_str] = round(t[pos].X, 6)
        result["patient_start_times"] = patient_schedule

    else:
        result["objective_value"] = None
        result["status"] = f"status_{model.Status}"
        result["solve_time_seconds"] = model.Runtime

    if solution_path is not None:
        with open(solution_path, "w") as f:
            result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(result, f, indent=2)

    if not quiet:
        print(f"\nSolution written to: {solution_path}")
        if model.SolCount > 0:
            print(f"Objective value: {model.ObjVal:.6f}")
            print(f"Status: {result['status']}")
            print(f"Solve time: {model.Runtime:.2f}s")
            print(f"Assignment (patient -> position): {assignment}")
            print(
                f"Schedule (position -> start time): "
                + ", ".join(f"{k}: {v}" for k, v in sorted(schedule.items()))
            )
        else:
            print(f"No feasible solution found. Status: {model.Status}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Solve the DR-bimodal MILP (Eq. 15) for the DROCS problem "
            "using Gurobi."
        )
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the instance JSON file.",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path for the output solution JSON file.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Solver time limit in seconds (default: 3600).",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve_drocs(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
