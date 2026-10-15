"""
Gurobi MILP implementation for the Lock and Quay Co-scheduling Problem (LQCP).
Based on: Ji et al. (2019), "Exact and heuristic methods for optimizing
         lock-quay system in inland waterway", EJOR 277, 740-755.

Implements the full MILP model (Eqs. 1-57) from the paper.
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
        pass


def load_instance(instance_path):
    """Load instance data from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def build_and_solve_model(data, time_limit):
    """Build and solve the LQCP MILP model."""

    # ----- Extract data -----
    n = data["num_ships"]
    m = data["num_berths"]
    W = data["chamber_width_W"]
    L = data["chamber_length_L"]
    Du = data["lockage_duration_Du"]
    MT = data["min_time_between_lockages_MT"]
    sc = data["lock_start_availability_sc"]
    C_max = data["C_max"]

    ships = data["ships"]
    berths = data["berths"]
    handling_times = data["handling_times"]  # handling_times[i][k]

    # Ship properties
    w = {i: ships[i]["width"] for i in range(n)}
    l = {i: ships[i]["length"] for i in range(n)}
    a = {i: ships[i]["arrival_time"] for i in range(n)}
    Pe = {i: ships[i]["transshipment_penalty"] for i in range(n)}
    MR = {i: ships[i]["mooring_set_MR"] for i in range(n)}
    B_eligible = {i: ships[i]["eligible_berths_B"] for i in range(n)}

    # Berth properties
    s_k = {k: berths[k]["start_availability_sk"] for k in range(m)}
    # Berth end availability
    e_k_val = data.get("berth_end_availability_ek", 1000.0)
    e_k = {k: e_k_val for k in range(m)}

    # Handling times t[i][k]
    t = {}
    for i in range(n):
        for k in range(m):
            t[i, k] = handling_times[i][k]

    # Sets
    N = list(range(n))
    M = list(range(m))
    Q = list(range(n))  # |Q| = n (upper bound on lockages)
    P = list(range(n))  # |P| = n (service orders)

    # N(k) = {i in N : a_i >= s_k} — ships arriving after berth k becomes available
    N_k = {}
    for k in M:
        N_k[k] = [i for i in N if a[i] >= s_k[k]]

    # P(p) = {l in P : l < p} — service orders before p
    # Used inline below

    # **NOT SPECIFIED IN PAPER**: Sets SL and SB are not explicitly defined in the
    # instance data. The paper states "all ships can choose both modes" in experiments.
    # Inferred assumption: SL = empty, SB = empty (all ships can choose either mode).
    SL = []
    SB = []

    # ----- Create model -----
    model = gp.Model("LQCP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # ----- Decision variables -----
    # u_i: mode chosen by ship i (1=transshipment, 0=lock)
    u = model.addVars(N, vtype=GRB.BINARY, name="u")

    # x_i, y_i: position of ship i in lock chamber
    x = model.addVars(N, vtype=GRB.INTEGER, lb=0, name="x")
    y = model.addVars(N, vtype=GRB.INTEGER, lb=0, name="y")

    # lt_{ij}: ship i completely to the left of ship j
    lt = model.addVars(N, N, vtype=GRB.BINARY, name="lt")

    # b_{ij}: ship i completely behind ship j
    b = model.addVars(N, N, vtype=GRB.BINARY, name="b")

    # ml_{ij}, mr_{ij}: mooring variables
    ml = {}
    mr = {}
    for i in N:
        for j in MR[i]:
            ml[i, j] = model.addVar(vtype=GRB.BINARY, name=f"ml_{i}_{j}")
            mr[i, j] = model.addVar(vtype=GRB.BINARY, name=f"mr_{i}_{j}")

    # ml_{i,n+1}: ship i moors to chamber's right side
    ml_right = model.addVars(N, vtype=GRB.BINARY, name="ml_right")
    # mr_{i,0}: ship i moors to chamber's left side
    mr_left = model.addVars(N, vtype=GRB.BINARY, name="mr_left")

    # z_q: whether lockage q is used
    z = model.addVars(Q, vtype=GRB.BINARY, name="z")

    # f_{iq}: ship i transferred by lockage q
    f = model.addVars(N, Q, vtype=GRB.BINARY, name="f")

    # v_{ij}: ships i and j in the same lockage (i < j)
    v = {}
    for i in N:
        for j in N:
            if i < j:
                v[i, j] = model.addVar(vtype=GRB.BINARY, name=f"v_{i}_{j}")

    # c_i: departure/completion time of ship i
    c = model.addVars(N, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="c")

    # C_q: completion time of lockage q
    C = model.addVars(Q, lb=0, ub=C_max, vtype=GRB.CONTINUOUS, name="C_lock")

    # delta_{ipk}: ship i assigned to p-th order of berth k
    delta = model.addVars(N, P, M, vtype=GRB.BINARY, name="delta")

    # omega_{ipk}: idle time of berth k before arrival of ship i at p-th order
    omega = model.addVars(N, P, M, lb=0, vtype=GRB.CONTINUOUS, name="omega")

    model.update()

    # ----- Objective function (Eq. 1, 2, 4) -----
    # T_1 = sum_{i in N} (c_i - a_i)
    T_1 = gp.quicksum(c[i] - a[i] for i in N)

    # T_2 (linearized, Eq. 4):
    # T_2 = sum_{k in M} sum_{i in N} sum_{p in P}
    #   { sum_{j in N} sum_{l in P(p)} (t_{jk} * delta_{jlk} + omega_{jlk})
    #     + omega_{ipk} + t_{ik} + s_k - a_i } * delta_{ipk}
    #
    # This is nonlinear due to delta_{ipk} multiplying sum terms that also contain delta.
    # The paper says Eq. (4) is the linearized version. Looking more carefully at Eq. (4),
    # the entire expression inside { } is multiplied by delta_{ipk}, making it quadratic.
    #
    # **INFERRED ASSUMPTION**: We use Eq. (3) which is the equivalent nonlinear form:
    # T_2 = sum_{k in M} sum_{i in N} sum_{p in P} {(n - p + 1) * t_{ik} + s_k - a_i} * delta_{ipk}
    #       + sum_{k in M} sum_{i in N} sum_{p in P} (n - p + 1) * omega_{ipk}
    # This is actually linear since delta_{ipk} multiplies constants and omega_{ipk} is
    # multiplied by a constant.

    T_2 = gp.quicksum(
        ((n - p + 1) * t[i, k] + s_k[k] - a[i]) * delta[i, p, k]
        for k in M for i in N for p in P
    ) + gp.quicksum(
        (n - p + 1) * omega[i, p, k]
        for k in M for i in N for p in P
    )

    # Penalty term
    penalty = gp.quicksum(Pe[i] * u[i] for i in N)

    model.setObjective(T_1 + T_2 + penalty, GRB.MINIMIZE)

    # ----- Constraints -----

    # --- Ship Placement Sub-problem (Constraints 5-30) ---

    # (5) Non-overlap: lt_{ij} + lt_{ji} + b_{ij} + b_{ji} + (1-f_{iq}) + (1-f_{jq}) >= 1
    #     for i < j, q in Q
    for i in N:
        for j in N:
            if i < j:
                for q in Q:
                    model.addConstr(
                        lt[i, j] + lt[j, i] + b[i, j] + b[j, i]
                        + (1 - f[i, q]) + (1 - f[j, q]) >= 1,
                        name=f"nonoverlap_{i}_{j}_{q}"
                    )

    # (6) x_i + w_i <= x_j + W*(1 - lt_{ij}), for i != j
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(
                    x[i] + w[i] <= x[j] + W * (1 - lt[i, j]),
                    name=f"lt_x_{i}_{j}"
                )

    # (7) y_i + l_i <= y_j + L*(1 - b_{ij}), for i != j
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(
                    y[i] + l[i] <= y[j] + L * (1 - b[i, j]),
                    name=f"b_y_{i}_{j}"
                )

    # (8) x_i + w_i <= W
    for i in N:
        model.addConstr(x[i] + w[i] <= W, name=f"dim_x_{i}")

    # (9) y_i + l_i <= L
    for i in N:
        model.addConstr(y[i] + l[i] <= L, name=f"dim_y_{i}")

    # (10) sum_{q in Q} f_{iq} = 1 - u_i
    for i in N:
        model.addConstr(
            gp.quicksum(f[i, q] for q in Q) == 1 - u[i],
            name=f"lockage_assign_{i}"
        )

    # (11) f_{iq} <= z_q
    for i in N:
        for q in Q:
            model.addConstr(f[i, q] <= z[q], name=f"lockage_used_{i}_{q}")

    # Mooring constraints (12)-(15): ship i moors to LEFT of ship j
    for i in N:
        for j in MR[i]:
            # (12) y_j - y_i <= (1 - ml_{ij}) * L
            model.addConstr(
                y[j] - y[i] <= (1 - ml[i, j]) * L,
                name=f"ml_y1_{i}_{j}"
            )
            # (13) y_i - y_j <= l_j - l_i + (1 - ml_{ij}) * L
            model.addConstr(
                y[i] - y[j] <= l[j] - l[i] + (1 - ml[i, j]) * L,
                name=f"ml_y2_{i}_{j}"
            )
            # (14) x_j - x_i <= w_i + (1 - ml_{ij}) * W
            model.addConstr(
                x[j] - x[i] <= w[i] + (1 - ml[i, j]) * W,
                name=f"ml_x1_{i}_{j}"
            )
            # (15) x_j - x_i >= w_i - (1 - ml_{ij}) * W
            model.addConstr(
                x[j] - x[i] >= w[i] - (1 - ml[i, j]) * W,
                name=f"ml_x2_{i}_{j}"
            )

    # (16)-(17): Ship i adjacent to chamber's right side (sp_{n+1}: x_{n+1}=W)
    for i in N:
        # (16) x_{n+1} - x_i <= w_i + (1 - ml_{i,n+1}) * W
        # x_{n+1} = W, so: W - x_i <= w_i + (1 - ml_right[i]) * W
        model.addConstr(
            W - x[i] <= w[i] + (1 - ml_right[i]) * W,
            name=f"mlright_x1_{i}"
        )
        # (17) x_{n+1} - x_i >= w_i - (1 - ml_{i,n+1}) * W
        # W - x_i >= w_i - (1 - ml_right[i]) * W
        model.addConstr(
            W - x[i] >= w[i] - (1 - ml_right[i]) * W,
            name=f"mlright_x2_{i}"
        )

    # Mooring constraints (18)-(21): ship i moors to RIGHT of ship j
    for i in N:
        for j in MR[i]:
            # (18) y_j - y_i <= (1 - mr_{ij}) * L
            model.addConstr(
                y[j] - y[i] <= (1 - mr[i, j]) * L,
                name=f"mr_y1_{i}_{j}"
            )
            # (19) y_i - y_j <= l_j - l_i + (1 - mr_{ij}) * L
            model.addConstr(
                y[i] - y[j] <= l[j] - l[i] + (1 - mr[i, j]) * L,
                name=f"mr_y2_{i}_{j}"
            )
            # (20) x_j - x_i <= -w_j + (1 - mr_{ij}) * W
            model.addConstr(
                x[j] - x[i] <= -w[j] + (1 - mr[i, j]) * W,
                name=f"mr_x1_{i}_{j}"
            )
            # (21) x_j - x_i >= -w_j - (1 - mr_{ij}) * W
            model.addConstr(
                x[j] - x[i] >= -w[j] - (1 - mr[i, j]) * W,
                name=f"mr_x2_{i}_{j}"
            )

    # (22)-(23): Ship i adjacent to chamber's left side (sp_0: x_0=0)
    for i in N:
        # (22) x_0 - x_i <= -w_i + (1 - mr_{i,0}) * W
        # x_0 = 0: -x_i <= -w_i + (1 - mr_left[i]) * W
        model.addConstr(
            -x[i] <= -w[i] + (1 - mr_left[i]) * W,
            name=f"mrleft_x1_{i}"
        )
        # (23) x_0 - x_i >= -w_i - (1 - mr_{i,0}) * W
        # -x_i >= -w_i - (1 - mr_left[i]) * W
        model.addConstr(
            -x[i] >= -w[i] - (1 - mr_left[i]) * W,
            name=f"mrleft_x2_{i}"
        )

    # (24) Each lock-mode ship must moor to another ship or chamber side
    # sum_{j in MR_i} (ml_{ij} + mr_{ij}) + mr_{i,0} + ml_{i,n+1} >= 1 - u_i
    # Note: The paper has "sum_{t in TP} ml_{i,n+1}" but this is just ml_{i,n+1}.
    for i in N:
        model.addConstr(
            gp.quicksum(ml[i, j] + mr[i, j] for j in MR[i])
            + mr_left[i] + ml_right[i] >= 1 - u[i],
            name=f"moor_required_{i}"
        )

    # (25) Prevent two ships of same length both unattached
    # ml_{ij} + mr_{ji} <= 1, for i in N, j in MR_i
    for i in N:
        for j in MR[i]:
            if (j, i) in mr:  # mr_{ji} exists if i in MR[j]
                model.addConstr(
                    ml[i, j] + mr[j, i] <= 1,
                    name=f"no_both_unattached_{i}_{j}"
                )

    # (26) Two ships not in same lockage cannot moor: ml_{ij} + mr_{ij} <= v_{ij}
    for i in N:
        for j in MR[i]:
            if i < j and (i, j) in v:
                model.addConstr(
                    ml[i, j] + mr[i, j] <= v[i, j],
                    name=f"moor_same_lockage_a_{i}_{j}"
                )
            elif j < i and (j, i) in v:
                model.addConstr(
                    ml[i, j] + mr[i, j] <= v[j, i],
                    name=f"moor_same_lockage_b_{i}_{j}"
                )

    # (27) f_{iq} - f_{jq} <= 1 - v_{ij}, for i < j, q in Q
    for i in N:
        for j in N:
            if i < j:
                for q in Q:
                    model.addConstr(
                        f[i, q] - f[j, q] <= 1 - v[i, j],
                        name=f"same_lockage1_{i}_{j}_{q}"
                    )

    # (28) f_{jq} - f_{iq} <= 1 - v_{ij}, for i < j, q in Q
    for i in N:
        for j in N:
            if i < j:
                for q in Q:
                    model.addConstr(
                        f[j, q] - f[i, q] <= 1 - v[i, j],
                        name=f"same_lockage2_{i}_{j}_{q}"
                    )

    # (29) v_{ij} < 1 - u_i  =>  v_{ij} <= 1 - u_i - epsilon
    # Since v is binary and u is binary, v_{ij} < 1 - u_i means v_{ij} + u_i <= 0 when u_i=1
    # Equivalent to: v_{ij} <= 1 - u_i (since strict < with integers means <=)
    # Actually with binary: v_{ij} < 1 - u_i. If u_i = 0, v_{ij} < 1 so v_{ij} = 0 doesn't help.
    # If u_i = 0, v_{ij} can be 0. If u_i = 1, v_{ij} < 0 impossible.
    # The paper uses strict inequality. For binary variables, v_{ij} < 1-u_i is equivalent
    # to v_{ij} <= -u_i when u_i=1 (impossible for v>=0), so v_{ij} + u_i <= 0.
    # But if u_i = 0, v_{ij} < 1 which is v_{ij} = 0 for binary.
    # Actually re-reading: this should just be v_{ij} <= 1 - u_i for practical purposes.
    for i in N:
        for j in N:
            if i < j:
                model.addConstr(
                    v[i, j] <= 1 - u[i],
                    name=f"v_mode1_{i}_{j}"
                )

    # (30) v_{ij} <= 1 - u_j
    for i in N:
        for j in N:
            if i < j:
                model.addConstr(
                    v[i, j] <= 1 - u[j],
                    name=f"v_mode2_{i}_{j}"
                )

    # --- Lockage Scheduling Sub-problem (Constraints 31-42) ---

    # (31) c_i >= C_max*(f_{iq} - 1) + C_q
    for i in N:
        for q in Q:
            model.addConstr(
                c[i] >= C_max * (f[i, q] - 1) + C[q],
                name=f"comp_time_lb_{i}_{q}"
            )

    # (32) c_i <= C_max*(1 - f_{iq}) + C_q
    for i in N:
        for q in Q:
            model.addConstr(
                c[i] <= C_max * (1 - f[i, q]) + C[q],
                name=f"comp_time_ub_{i}_{q}"
            )

    # (33) c_i >= C_max*(u_i - 1) + a_i
    for i in N:
        model.addConstr(
            c[i] >= C_max * (u[i] - 1) + a[i],
            name=f"trans_comp_lb_{i}"
        )

    # (34) c_i <= C_max*(1 - u_i) + a_i
    for i in N:
        model.addConstr(
            c[i] <= C_max * (1 - u[i]) + a[i],
            name=f"trans_comp_ub_{i}"
        )

    # (35) C_{q+1} - C_q >= MT, for q < |Q|
    for q in Q:
        if q < n - 1:
            model.addConstr(
                C[q + 1] - C[q] >= MT,
                name=f"min_interval_{q}"
            )

    # (36) C_q - Du >= sc
    for q in Q:
        model.addConstr(C[q] - Du >= sc, name=f"chamber_avail_{q}")

    # (37) C_q - Du >= f_{iq} * a_i
    for i in N:
        for q in Q:
            model.addConstr(
                C[q] - Du >= f[i, q] * a[i],
                name=f"ship_arrival_{i}_{q}"
            )

    # (38) z_q <= sum_{i in N} f_{iq}
    for q in Q:
        model.addConstr(
            z[q] <= gp.quicksum(f[i, q] for i in N),
            name=f"lockage_scheduled_{q}"
        )

    # (39) FCFS: c_i - c_j <= C_max*(u_i + u_j), for i < j
    for i in N:
        for j in N:
            if i < j:
                model.addConstr(
                    c[i] - c[j] <= C_max * (u[i] + u[j]),
                    name=f"fcfs_{i}_{j}"
                )

    # (40) z_{q+1} <= z_q
    for q in Q:
        if q < n - 1:
            model.addConstr(z[q + 1] <= z[q], name=f"lock_order_z_{q}")

    # (41) C_q <= C_{q+1}
    for q in Q:
        if q < n - 1:
            model.addConstr(C[q] <= C[q + 1], name=f"lock_order_C_{q}")

    # (42) sum_{q < l, q in Q} (f_{iq} - f_{jq}) >= -(u_i + u_j), for i < j, l in Q
    for i in N:
        for j in N:
            if i < j:
                for l_idx in Q:
                    model.addConstr(
                        gp.quicksum(f[i, q] - f[j, q] for q in Q if q < l_idx)
                        >= -(u[i] + u[j]),
                        name=f"lock_order_f_{i}_{j}_{l_idx}"
                    )

    # --- Mode Fixing Constraints (43, 48) ---
    # (43) u_i = 0 for i in SL
    for i in SL:
        model.addConstr(u[i] == 0, name=f"lock_only_{i}")

    # (48) u_i = 1 for i in SB
    for i in SB:
        model.addConstr(u[i] == 1, name=f"trans_only_{i}")

    # --- BAPD Constraints (44-47) ---

    # (44) sum_{k in B(i)} sum_{p in P} delta_{ipk} = u_i
    for i in N:
        model.addConstr(
            gp.quicksum(delta[i, p, k] for k in B_eligible[i] for p in P) == u[i],
            name=f"berth_assign_{i}"
        )

    # (45) sum_{i in N} delta_{ipk} <= 1, for k in M, p in P
    for k in M:
        for p in P:
            model.addConstr(
                gp.quicksum(delta[i, p, k] for i in N) <= 1,
                name=f"berth_capacity_{k}_{p}"
            )

    # (46) Idle time constraint at berth
    # sum_{j in N} sum_{l in P(p)} (t_{jk} * delta_{jlk} + omega_{jlk})
    #   + omega_{ipk} - (a_i - s_k) * delta_{ipk} >= 0
    # for k in M, i in N(k), p in P
    for k in M:
        for i in N_k[k]:
            for p in P:
                model.addConstr(
                    gp.quicksum(
                        t[j, k] * delta[j, l_idx, k] + omega[j, l_idx, k]
                        for j in N for l_idx in P if l_idx < p
                    )
                    + omega[i, p, k]
                    - (a[i] - s_k[k]) * delta[i, p, k] >= 0,
                    name=f"idle_time_{k}_{i}_{p}"
                )

    # (47) Berth availability: sum_{i in N} sum_{p in P} (t_{ik}*delta_{ipk} + omega_{ipk}) <= e_k - s_k
    for k in M:
        model.addConstr(
            gp.quicksum(
                t[i, k] * delta[i, p, k] + omega[i, p, k]
                for i in N for p in P
            ) <= e_k[k] - s_k[k],
            name=f"berth_avail_{k}"
        )

    # Force delta_{ipk} = 0 for berths not eligible for ship i
    for i in N:
        for k in M:
            if k not in B_eligible[i]:
                for p in P:
                    model.addConstr(delta[i, p, k] == 0,
                                    name=f"ineligible_{i}_{p}_{k}")

    # ----- Solve -----
    model.optimize()

    # ----- Extract solution -----
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal

        # Extract mode decisions
        result["modes"] = {str(i): int(round(u[i].X)) for i in N}

        # Extract completion times
        result["completion_times"] = {str(i): c[i].X for i in N}

        # Extract lockage assignments for lock-mode ships
        lockage_assignments = {}
        for i in N:
            if round(u[i].X) == 0:  # lock mode
                for q in Q:
                    if round(f[i, q].X) == 1:
                        lockage_assignments[str(i)] = q
                        break
        result["lockage_assignments"] = lockage_assignments

        # Extract lockage completion times
        lockage_times = {}
        for q in Q:
            if round(z[q].X) == 1:
                lockage_times[str(q)] = C[q].X
        result["lockage_completion_times"] = lockage_times

        # Extract berth assignments for transshipment-mode ships
        berth_assignments = {}
        for i in N:
            if round(u[i].X) == 1:  # transshipment mode
                for k in B_eligible[i]:
                    for p in P:
                        if round(delta[i, p, k].X) == 1:
                            berth_assignments[str(i)] = {"berth": k, "order": p}
                            break
        result["berth_assignments"] = berth_assignments

        result["solver_status"] = model.Status
        result["mip_gap"] = model.MIPGap if hasattr(model, 'MIPGap') else None
    else:
        result["objective_value"] = None
        result["solver_status"] = model.Status
        result["error"] = "No feasible solution found within time limit."

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MILP solver for LQCP (Ji et al., 2019)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = build_and_solve_model(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
