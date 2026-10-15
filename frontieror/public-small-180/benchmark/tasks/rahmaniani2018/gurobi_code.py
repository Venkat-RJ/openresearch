"""
Monolithic MILP energy systems capacity expansion planning model using Gurobi.

Implements the model from Jacobson et al. (2023) as described in
Rahmaniani et al. (2018). Minimizes total system cost subject to
investment, operational, and policy constraints.
"""

import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
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


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_and_solve(instance, time_limit, fixed_solution=None, quiet=False):
    # ----------------------------------------------------------------
    # Unpack instance data
    # ----------------------------------------------------------------
    dims = instance["problem_dimensions"]
    algo = instance["algorithm_parameters"]
    policy = instance["policy"]
    subperiods = instance["subperiods"]
    generators = instance["generators"]
    trans_lines = instance["transmission_lines"]
    demand_data = instance["demand"]
    nse_segments = instance["nse_segments"]
    avail_data = instance["availability_profiles"]

    num_zones = dims["num_zones"]
    total_T = dims["total_timesteps"]
    formulation = algo["formulation"]  # "MILP" or "LP"

    num_subperiods = subperiods["num_subperiods"]
    hours_per_subperiod = subperiods["hours_per_subperiod"]
    subperiod_weights = subperiods["subperiod_weights"]  # beta_w

    # Build subperiod structure: list of (w, [t indices within subperiod])
    # Subperiods are sequential blocks of hours_per_subperiod timesteps.
    subperiod_timesteps = []  # subperiod_timesteps[w] = list of global t indices
    for w in range(num_subperiods):
        start = w * hours_per_subperiod
        end = start + hours_per_subperiod
        subperiod_timesteps.append(list(range(start, end)))

    # alpha_t = beta_w / delta_w
    alpha = {}
    for w in range(num_subperiods):
        beta_w = subperiod_weights[w]
        delta_w = hours_per_subperiod
        a = beta_w / delta_w
        for t in subperiod_timesteps[w]:
            alpha[t] = a

    # Generator sets
    G_all = list(range(len(generators)))
    G_UC = [i for i in G_all if generators[i]["is_UC"]]
    G_nonUC = [i for i in G_all if not generators[i]["is_UC"]]
    G_STOR = [i for i in G_all if generators[i]["is_storage"]]
    G_HYDRO = [i for i in G_all if generators[i]["is_hydro"]]
    G_RPS = [i for i in G_all if generators[i]["is_RPS_qualifying"]]

    # Zone sets
    zones = list(range(1, num_zones + 1))

    # Generators by zone
    gens_in_zone = {z: [] for z in zones}
    for i in G_all:
        gens_in_zone[generators[i]["zone"]].append(i)

    # Availability profiles: build sigma[i][t]
    sigma = {}
    for i in G_all:
        gid = str(generators[i]["id"])
        if gid in avail_data:
            prof = avail_data[gid]
            if prof["type"] == "timeseries":
                sigma[i] = prof["values"]
            else:
                sigma[i] = [prof["value"]] * total_T
        else:
            sigma[i] = [1.0] * total_T

    # Demand: demand[z][t]
    demand = {}
    for z in zones:
        demand[z] = demand_data[str(z)]

    # Transmission lines indexed 0..L-1
    L_all = list(range(len(trans_lines)))

    # NSE segments indexed 0..S-1
    S_all = list(range(len(nse_segments)))

    # ----------------------------------------------------------------
    # Build Gurobi model
    # ----------------------------------------------------------------
    model = gp.Model("energy_planning")
    model.setParam("Threads", 1)

    # Solver parameters
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPFocus", 1)
        model.setParam("NoRelHeurTime", min(60.0, time_limit * 0.05))
    model.setParam("MIPGap", 1e-4)
    if formulation == "LP":
        model.setParam("Method", 2)      # barrier
        model.setParam("Crossover", 0)   # crossover off

    # Variable type for investment
    inv_vtype = GRB.INTEGER if formulation == "MILP" else GRB.CONTINUOUS

    # ----------------------------------------------------------------
    # Decision Variables
    # ----------------------------------------------------------------

    # --- Investment variables (per generator) ---
    y_P_new = {}   # new power capacity (units)
    y_P_ret = {}   # retired power capacity (units)
    y_P = {}       # total installed power capacity (MW)
    for i in G_all:
        g = generators[i]
        max_new_units = g["max_capacity_MW"] / g["capacity_size_MW"] if g["capacity_size_MW"] > 0 else 0
        max_ret_units = g["existing_capacity_MW"] / g["capacity_size_MW"] if g["capacity_size_MW"] > 0 else 0
        y_P_new[i] = model.addVar(lb=0, ub=max_new_units, vtype=inv_vtype, name=f"yPnew_{i}")
        y_P_ret[i] = model.addVar(lb=0, ub=max_ret_units, vtype=inv_vtype, name=f"yPret_{i}")
        y_P[i] = model.addVar(lb=0, name=f"yP_{i}")

    # --- Storage energy investment variables ---
    y_E_new = {}
    y_E_ret = {}
    y_E = {}
    for i in G_STOR:
        g = generators[i]
        max_new_e = g["max_storage_capacity_MWh"] / g["storage_capacity_size_MWh"] if g["storage_capacity_size_MWh"] > 0 else 0
        max_ret_e = g["existing_storage_capacity_MWh"] / g["storage_capacity_size_MWh"] if g["storage_capacity_size_MWh"] > 0 else 0
        y_E_new[i] = model.addVar(lb=0, ub=max_new_e, vtype=inv_vtype, name=f"yEnew_{i}")
        y_E_ret[i] = model.addVar(lb=0, ub=max_ret_e, vtype=inv_vtype, name=f"yEret_{i}")
        y_E[i] = model.addVar(lb=0, name=f"yE_{i}")

    # --- Transmission investment ---
    y_F_new = {}
    for l in L_all:
        tl = trans_lines[l]
        y_F_new[l] = model.addVar(lb=0, ub=tl["max_new_capacity_MW"], vtype=inv_vtype, name=f"yFnew_{l}")

    # --- Operational variables ---
    # Power injection (generation) for all generators
    x_inj = {}
    for i in G_all:
        for t in range(total_T):
            x_inj[i, t] = model.addVar(lb=0, name=f"xinj_{i}_{t}")

    # Storage charge (withdrawal from grid)
    x_wdw = {}
    for i in G_STOR:
        for t in range(total_T):
            x_wdw[i, t] = model.addVar(lb=0, name=f"xwdw_{i}_{t}")

    # Storage state of charge
    x_soc = {}
    for i in G_STOR:
        for t in range(total_T):
            x_soc[i, t] = model.addVar(lb=0, name=f"xsoc_{i}_{t}")

    # Hydro reservoir level
    x_hydro_level = {}
    for i in G_HYDRO:
        for t in range(total_T):
            x_hydro_level[i, t] = model.addVar(lb=0, name=f"xhydro_{i}_{t}")

    # Transmission flow (bidirectional: positive = from_zone -> to_zone)
    x_flow = {}
    for l in L_all:
        for t in range(total_T):
            x_flow[l, t] = model.addVar(lb=-GRB.INFINITY, name=f"xflow_{l}_{t}")

    # NSE (non-served energy) per segment, zone, timestep
    x_nse = {}
    for s in S_all:
        for z in zones:
            for t in range(total_T):
                x_nse[s, z, t] = model.addVar(lb=0, name=f"xnse_{s}_{z}_{t}")

    # UC variables (commit, start, shut) - continuous relaxation
    x_commit = {}
    x_start = {}
    x_shut = {}
    for i in G_UC:
        for t in range(total_T):
            x_commit[i, t] = model.addVar(lb=0, name=f"xcommit_{i}_{t}")
            x_start[i, t] = model.addVar(lb=0, name=f"xstart_{i}_{t}")
            x_shut[i, t] = model.addVar(lb=0, name=f"xshut_{i}_{t}")

    # Policy slack variables
    x_RPS = {}
    x_CO2 = {}
    for w in range(num_subperiods):
        x_RPS[w] = model.addVar(lb=0, name=f"xRPS_{w}")
        x_CO2[w] = model.addVar(lb=0, name=f"xCO2_{w}")

    model.update()

    # Checker mode: investment decisions are the submitted first-stage witness.
    # Fix them and optimize only the omitted operational recourse variables.
    if fixed_solution is not None:
        # The original global-MIP run reserves up to 60 seconds for a no-relaxation
        # heuristic.  All integer first-stage variables are fixed here, so that
        # phase is pure overhead for checker recourse.
        model.setParam("NoRelHeurTime", 0)
        fixed_generators = {
            entry["generator_id"]: entry
            for entry in fixed_solution.get("investment_decisions", [])
        }
        fixed_lines = {
            entry["line_id"]: entry
            for entry in fixed_solution.get("transmission_decisions", [])
        }
        expected_generator_ids = {g["id"] for g in generators}
        expected_line_ids = {line["id"] for line in trans_lines}
        if set(fixed_generators) != expected_generator_ids:
            raise ValueError("fixed solution must cover every generator exactly once")
        if set(fixed_lines) != expected_line_ids:
            raise ValueError("fixed solution must cover every transmission line exactly once")

        for i, generator in enumerate(generators):
            entry = fixed_generators[generator["id"]]
            y_P_new[i].LB = y_P_new[i].UB = float(entry["y_P_new"])
            y_P_ret[i].LB = y_P_ret[i].UB = float(entry["y_P_ret"])
            if i in y_E_new:
                y_E_new[i].LB = y_E_new[i].UB = float(entry["y_E_new"])
                y_E_ret[i].LB = y_E_ret[i].UB = float(entry["y_E_ret"])
        for l, line in enumerate(trans_lines):
            value = float(fixed_lines[line["id"]]["y_F_new_MW"])
            y_F_new[l].LB = y_F_new[l].UB = value

    if quiet:
        model.setParam("OutputFlag", 0)

    # ----------------------------------------------------------------
    # Objective Function
    # ----------------------------------------------------------------

    # c^FIXED
    c_fixed = gp.LinExpr()
    for i in G_all:
        g = generators[i]
        gamma_P = g["capacity_size_MW"]
        # Investment cost for new power capacity
        c_fixed += g["inv_cost_per_MW_yr"] * gamma_P * y_P_new[i]
        # FOM cost on total installed capacity
        c_fixed += g["fom_cost_per_MW_yr"] * y_P[i]

    for i in G_STOR:
        g = generators[i]
        gamma_E = g["storage_capacity_size_MWh"]
        c_fixed += g["inv_cost_energy_per_MWh_yr"] * gamma_E * y_E_new[i]
        c_fixed += g.get("fom_cost_energy_per_MWh_yr", 0) * y_E[i]

    for i in G_HYDRO:
        g = generators[i]
        rho = g["duration_MWh_per_MW"]
        # Hydro energy investment and FOM (assumed 0 per notes, but include if present)
        hydro_e_inv = g.get("inv_cost_energy_per_MWh_yr", 0)
        hydro_e_fom = g.get("fom_cost_energy_per_MWh_yr", 0)
        c_fixed += hydro_e_inv * rho * g["capacity_size_MW"] * y_P_new[i]
        c_fixed += hydro_e_fom * rho * y_P[i]

    for l in L_all:
        tl = trans_lines[l]
        c_fixed += tl["inv_cost_per_MW_yr"] * y_F_new[l]

    # c^VAR
    c_var = gp.LinExpr()
    for i in G_all:
        g = generators[i]
        vc = g["var_cost_per_MWh"]
        if vc > 0:
            for t in range(total_T):
                c_var += vc * alpha[t] * x_inj[i, t]
    for i in G_STOR:
        g = generators[i]
        vc = g["var_cost_per_MWh"]
        if vc > 0:
            for t in range(total_T):
                c_var += vc * alpha[t] * x_wdw[i, t]

    # c^NSE
    c_nse = gp.LinExpr()
    for s in S_all:
        cost_s = nse_segments[s]["cost_per_MWh"]
        for z in zones:
            for t in range(total_T):
                c_nse += cost_s * alpha[t] * x_nse[s, z, t]

    # c^START
    c_start = gp.LinExpr()
    for i in G_UC:
        g = generators[i]
        sc = g["start_cost_per_unit"]
        if sc > 0:
            for t in range(total_T):
                c_start += sc * alpha[t] * x_start[i, t]

    # c^RPS + c^CO2
    c_policy = gp.LinExpr()
    rps_cost = policy["RPS_noncompliance_cost_per_MWh"]
    co2_cost = policy["CO2_noncompliance_cost_per_ton"]
    for w in range(num_subperiods):
        c_policy += rps_cost * x_RPS[w]
        c_policy += co2_cost * x_CO2[w]

    model.setObjective(c_fixed + c_var + c_nse + c_start + c_policy, GRB.MINIMIZE)

    # ----------------------------------------------------------------
    # Investment Constraints
    # ----------------------------------------------------------------

    # (12a) New capacity limit
    for i in G_all:
        g = generators[i]
        model.addConstr(
            g["capacity_size_MW"] * y_P_new[i] <= g["max_capacity_MW"],
            name=f"inv_new_cap_{i}"
        )

    # (12b) Retirement limit
    for i in G_all:
        g = generators[i]
        model.addConstr(
            g["capacity_size_MW"] * y_P_ret[i] <= g["existing_capacity_MW"],
            name=f"inv_ret_cap_{i}"
        )

    # (12c) Cannot retire if can_retire is False
    for i in G_all:
        if not generators[i]["can_retire"]:
            model.addConstr(y_P_ret[i] == 0, name=f"inv_no_ret_{i}")

    # (12d) Total installed capacity
    for i in G_all:
        g = generators[i]
        model.addConstr(
            y_P[i] == g["existing_capacity_MW"] + g["capacity_size_MW"] * (y_P_new[i] - y_P_ret[i]),
            name=f"inv_total_cap_{i}"
        )

    # (13a-13f) Storage energy capacity
    for i in G_STOR:
        g = generators[i]
        gamma_E = g["storage_capacity_size_MWh"]
        # (13a) New energy limit
        model.addConstr(
            gamma_E * y_E_new[i] <= g["max_storage_capacity_MWh"],
            name=f"stor_new_ecap_{i}"
        )
        # (13b) Retirement energy limit
        model.addConstr(
            gamma_E * y_E_ret[i] <= g["existing_storage_capacity_MWh"],
            name=f"stor_ret_ecap_{i}"
        )
        # (13c) Cannot retire energy if can_retire is False
        if not g["can_retire"]:
            model.addConstr(y_E_ret[i] == 0, name=f"stor_no_eret_{i}")
        # (13d) Total energy capacity
        model.addConstr(
            y_E[i] == g["existing_storage_capacity_MWh"] + gamma_E * (y_E_new[i] - y_E_ret[i]),
            name=f"stor_total_ecap_{i}"
        )
        # (13e) Min duration: y_E >= min_duration * y_P
        model.addConstr(
            y_E[i] >= g["min_duration_MWh_per_MW"] * y_P[i],
            name=f"stor_min_dur_{i}"
        )
        # (13f) Max duration: y_E <= max_duration * y_P
        model.addConstr(
            y_E[i] <= g["max_duration_MWh_per_MW"] * y_P[i],
            name=f"stor_max_dur_{i}"
        )

    # (14a-14b) Transmission capacity (total = existing + new)
    # We don't need an explicit y_F total variable; we use it inline.
    # The flow constraints will use existing + y_F_new.

    # ----------------------------------------------------------------
    # Operational Constraints
    # ----------------------------------------------------------------

    # Build net flow into each zone for each timestep
    # flow_into_zone[z][t] = sum of flows into zone z at time t
    # For line l: from_zone -> to_zone with flow x_flow[l,t]
    # Positive flow: from_zone -> to_zone (adds to to_zone, subtracts from from_zone)

    # (15) Power balance: for each zone z, timestep t
    for z in zones:
        for t in range(total_T):
            lhs = gp.LinExpr()
            # Generation injection from generators in zone z
            for i in gens_in_zone[z]:
                lhs += x_inj[i, t]
            # Subtract storage charging in zone z
            for i in gens_in_zone[z]:
                if generators[i]["is_storage"]:
                    lhs -= x_wdw[i, t]
            # Transmission: net flow into zone z
            for l in L_all:
                tl = trans_lines[l]
                if tl["to_zone"] == z:
                    lhs += x_flow[l, t]
                if tl["from_zone"] == z:
                    lhs -= x_flow[l, t]
            # NSE
            for s in S_all:
                lhs += x_nse[s, z, t]
            # Must equal demand
            model.addConstr(lhs == demand[z][t], name=f"balance_{z}_{t}")

    # (16a-16g) Capacity limits
    for i in G_all:
        g = generators[i]
        is_uc = g["is_UC"]
        is_stor = g["is_storage"]
        is_hydro = g["is_hydro"]

        for t in range(total_T):
            if is_uc:
                # (16b) UC: x_inj <= sigma * capacity_size * x_commit
                model.addConstr(
                    x_inj[i, t] <= sigma[i][t] * g["capacity_size_MW"] * x_commit[i, t],
                    name=f"cap_uc_{i}_{t}"
                )
            elif is_stor:
                # (16c) Storage discharge: x_inj <= sigma * y_P
                model.addConstr(
                    x_inj[i, t] <= sigma[i][t] * y_P[i],
                    name=f"cap_stor_dis_{i}_{t}"
                )
                # (16d) Storage charge: x_wdw <= sigma * y_P
                model.addConstr(
                    x_wdw[i, t] <= sigma[i][t] * y_P[i],
                    name=f"cap_stor_chg_{i}_{t}"
                )
            elif is_hydro:
                # (16e) Hydro: x_inj <= sigma * y_P
                model.addConstr(
                    x_inj[i, t] <= sigma[i][t] * y_P[i],
                    name=f"cap_hydro_{i}_{t}"
                )
            else:
                # (16a) Non-UC, non-storage, non-hydro (VRE, etc.): x_inj <= sigma * y_P
                model.addConstr(
                    x_inj[i, t] <= sigma[i][t] * y_P[i],
                    name=f"cap_gen_{i}_{t}"
                )

    # (16f) Storage SOC limit: x_soc <= y_E
    for i in G_STOR:
        for t in range(total_T):
            model.addConstr(x_soc[i, t] <= y_E[i], name=f"soc_limit_{i}_{t}")

    # (16g) Hydro level limit: x_hydro_level <= rho * y_P
    for i in G_HYDRO:
        g = generators[i]
        rho = g["duration_MWh_per_MW"]
        for t in range(total_T):
            model.addConstr(
                x_hydro_level[i, t] <= rho * y_P[i],
                name=f"hydro_level_limit_{i}_{t}"
            )

    # (17a-17b) Minimum output
    for i in G_all:
        g = generators[i]
        min_frac = g.get("min_output_frac", 0.0)
        if min_frac <= 0:
            continue
        is_uc = g["is_UC"]
        for t in range(total_T):
            if is_uc:
                # (17b) UC: x_inj >= min_frac * capacity_size * x_commit
                model.addConstr(
                    x_inj[i, t] >= min_frac * g["capacity_size_MW"] * x_commit[i, t],
                    name=f"minout_uc_{i}_{t}"
                )
            else:
                # (17a) Non-UC: x_inj >= min_frac * sigma * y_P
                model.addConstr(
                    x_inj[i, t] >= min_frac * sigma[i][t] * y_P[i],
                    name=f"minout_{i}_{t}"
                )

    # (18a-18b) Transmission flow limits
    for l in L_all:
        tl = trans_lines[l]
        for t in range(total_T):
            # (18a) x_flow <= existing + y_F_new
            model.addConstr(
                x_flow[l, t] <= tl["existing_capacity_MW"] + y_F_new[l],
                name=f"flow_ub_{l}_{t}"
            )
            # (18b) x_flow >= -(existing + y_F_new)
            model.addConstr(
                x_flow[l, t] >= -(tl["existing_capacity_MW"] + y_F_new[l]),
                name=f"flow_lb_{l}_{t}"
            )

    # (19) NSE limits: x_nse[s,z,t] <= max_frac_s * demand[z][t]
    for s in S_all:
        seg = nse_segments[s]
        for z in zones:
            for t in range(total_T):
                model.addConstr(
                    x_nse[s, z, t] <= seg["max_frac"] * demand[z][t],
                    name=f"nse_limit_{s}_{z}_{t}"
                )

    # (20a-20d) Storage SOC dynamics (circular within subperiod)
    for i in G_STOR:
        g = generators[i]
        eta_c = g["charge_efficiency"]
        eta_d = g["discharge_efficiency"]
        self_dis = g["self_discharge_rate"]
        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            T_w = len(ts)
            for idx in range(T_w):
                t = ts[idx]
                t_prev = ts[(idx - 1) % T_w]  # circular: wraps to last hour
                # (20a) SOC[t] = (1-self_discharge)*SOC[t-1] + eta_c*charge[t] - (1/eta_d)*discharge[t]
                model.addConstr(
                    x_soc[i, t] == (1 - self_dis) * x_soc[i, t_prev]
                    + eta_c * x_wdw[i, t]
                    - (1.0 / eta_d) * x_inj[i, t],
                    name=f"soc_dyn_{i}_{t}"
                )

    # (20c-20d) Hydro level dynamics (circular within subperiod)
    # Inferred: sigma[i][t] serves as normalized inflow
    for i in G_HYDRO:
        g = generators[i]
        rho = g["duration_MWh_per_MW"]
        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            T_w = len(ts)
            for idx in range(T_w):
                t = ts[idx]
                t_prev = ts[(idx - 1) % T_w]
                # level[t] = level[t-1] + inflow[t] - generation[t]
                # inflow = sigma[i][t] * rho * y_P[i] (normalized inflow * reservoir capacity)
                # Actually, inflow should be scaled: sigma as fraction of capacity per hour
                # From paper: inflow_{g,t} = sigma^{inflow}_{g,t} * rho_g * y^P_g
                # But that would mean inflow ~ capacity which is huge. More likely:
                # inflow per hour is sigma[i][t] * y_P[i] (matching generation scale)
                # and reservoir = rho * y_P, so sigma ~ inflow/capacity.
                # Let's use: level[t] = level[t-1] + sigma[i][t] * y_P[i] - x_inj[i,t]
                model.addConstr(
                    x_hydro_level[i, t] == x_hydro_level[i, t_prev]
                    + sigma[i][t] * y_P[i]
                    - x_inj[i, t],
                    name=f"hydro_dyn_{i}_{t}"
                )

    # (21a-21d) Ramping for non-UC generators (circular within subperiod)
    for i in G_nonUC:
        g = generators[i]
        if g["is_storage"]:
            continue  # storage handled by SOC dynamics, not ramping
        ramp_up = g.get("ramp_up_frac_per_hr", 1.0)
        ramp_dn = g.get("ramp_dn_frac_per_hr", 1.0)
        if ramp_up >= 1.0 and ramp_dn >= 1.0:
            continue  # no binding ramp constraints
        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            T_w = len(ts)
            for idx in range(T_w):
                t = ts[idx]
                t_prev = ts[(idx - 1) % T_w]
                # (21a) Ramp up: x_inj[t] - x_inj[t-1] <= ramp_up * y_P
                if ramp_up < 1.0:
                    model.addConstr(
                        x_inj[i, t] - x_inj[i, t_prev] <= ramp_up * y_P[i],
                        name=f"ramp_up_{i}_{t}"
                    )
                # (21b) Ramp down: x_inj[t-1] - x_inj[t] <= ramp_dn * y_P
                if ramp_dn < 1.0:
                    model.addConstr(
                        x_inj[i, t_prev] - x_inj[i, t] <= ramp_dn * y_P[i],
                        name=f"ramp_dn_{i}_{t}"
                    )

    # (22a-22g) Unit commitment constraints
    for i in G_UC:
        g = generators[i]
        cap_size = g["capacity_size_MW"]

        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            T_w = len(ts)
            for idx in range(T_w):
                t = ts[idx]
                t_prev = ts[(idx - 1) % T_w]

                # (22a) Commitment state transition: commit[t] = commit[t-1] + start[t] - shut[t]
                model.addConstr(
                    x_commit[i, t] == x_commit[i, t_prev] + x_start[i, t] - x_shut[i, t],
                    name=f"uc_trans_{i}_{t}"
                )

        # (22b) commit <= max_units = max_capacity / capacity_size
        max_units = g["max_capacity_MW"] / cap_size if cap_size > 0 else 0
        for t in range(total_T):
            # Actually commit is bounded by installed capacity / cap_size
            # commit[t] <= y_P / cap_size
            model.addConstr(
                x_commit[i, t] * cap_size <= y_P[i],
                name=f"uc_max_commit_{i}_{t}"
            )

    # (23a-23d) Ramping for UC generators (circular within subperiod)
    for i in G_UC:
        g = generators[i]
        cap_size = g["capacity_size_MW"]
        ramp_up = g.get("ramp_up_frac_per_hr", 1.0)
        ramp_dn = g.get("ramp_dn_frac_per_hr", 1.0)
        min_frac = g.get("min_output_frac", 0.0)

        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            T_w = len(ts)
            for idx in range(T_w):
                t = ts[idx]
                t_prev = ts[(idx - 1) % T_w]

                # (23a) Ramp up: x_inj[t] - x_inj[t-1] <= ramp_up * cap_size * commit[t-1]
                #        + min(sigma*cap_size, ramp_up*cap_size + min_frac*cap_size) * start[t]
                # Simplified: ramp up limited by committed units
                # x_inj[t] - x_inj[t-1] <= ramp_up * cap_size * commit[t-1]
                #                          + max(min_frac, ramp_up) * cap_size * start[t]
                model.addConstr(
                    x_inj[i, t] - x_inj[i, t_prev] <=
                    ramp_up * cap_size * x_commit[i, t_prev]
                    + max(min_frac, ramp_up) * cap_size * x_start[i, t],
                    name=f"uc_ramp_up_{i}_{t}"
                )

                # (23b) Ramp down: x_inj[t-1] - x_inj[t] <= ramp_dn * cap_size * commit[t]
                #                   + max(min_frac, ramp_dn) * cap_size * shut[t]
                model.addConstr(
                    x_inj[i, t_prev] - x_inj[i, t] <=
                    ramp_dn * cap_size * x_commit[i, t]
                    + max(min_frac, ramp_dn) * cap_size * x_shut[i, t],
                    name=f"uc_ramp_dn_{i}_{t}"
                )

    # (24a-24b) Minimum up/down time (circular within subperiod)
    for i in G_UC:
        g = generators[i]
        min_up = g.get("min_up_time_hr", 0)
        min_dn = g.get("min_down_time_hr", 0)

        if min_up <= 1 and min_dn <= 1:
            continue

        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            T_w = len(ts)

            for idx in range(T_w):
                t = ts[idx]

                # (24a) Min up time: commit[t] >= sum of start[t'] for t' in [t-min_up+1, t]
                if min_up > 1:
                    start_sum = gp.LinExpr()
                    for k in range(min_up):
                        idx_k = (idx - k) % T_w
                        t_k = ts[idx_k]
                        start_sum += x_start[i, t_k]
                    model.addConstr(
                        x_commit[i, t] >= start_sum,
                        name=f"min_up_{i}_{t}"
                    )

                # (24b) Min down time: (y_P/cap_size) - commit[t] >= sum of shut[t'] for t' in [t-min_dn+1, t]
                if min_dn > 1:
                    shut_sum = gp.LinExpr()
                    for k in range(min_dn):
                        idx_k = (idx - k) % T_w
                        t_k = ts[idx_k]
                        shut_sum += x_shut[i, t_k]
                    cap_size = g["capacity_size_MW"]
                    if cap_size > 0:
                        model.addConstr(
                            y_P[i] / cap_size - x_commit[i, t] >= shut_sum,
                            name=f"min_dn_{i}_{t}"
                        )

    # ----------------------------------------------------------------
    # Policy Constraints
    # ----------------------------------------------------------------

    # Compute total weighted demand for CO2 cap
    total_weighted_demand = 0.0
    for z in zones:
        for t in range(total_T):
            total_weighted_demand += alpha[t] * demand[z][t]

    # (26) RPS constraint (if enabled)
    if policy["RPS_enabled"]:
        rps_share = policy["RPS_share"]
        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            # Sum of RPS-qualifying generation in subperiod w
            rps_gen = gp.LinExpr()
            for i in G_RPS:
                for t in ts:
                    rps_gen += alpha[t] * x_inj[i, t]
            # Total demand in subperiod w
            total_demand_w = gp.LinExpr()
            for z in zones:
                for t in ts:
                    total_demand_w += alpha[t] * demand[z][t]
            # rps_gen + x_RPS[w] >= rps_share * total_demand_w
            model.addConstr(
                rps_gen + x_RPS[w] >= rps_share * total_demand_w,
                name=f"rps_{w}"
            )

    # (27-28) CO2 constraint (if enabled)
    if policy["CO2_cap_enabled"]:
        co2_cap_rate = policy["CO2_cap_tons_per_MWh"]
        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            # Total CO2 emissions in subperiod w
            co2_emit = gp.LinExpr()
            for i in G_all:
                g = generators[i]
                co2_rate = g["co2_tons_per_MWh"]
                if co2_rate > 0:
                    for t in ts:
                        co2_emit += co2_rate * alpha[t] * x_inj[i, t]
            # Total demand in subperiod w
            demand_w_val = sum(alpha[t] * demand[z][t] for z in zones for t in ts)
            # CO2 cap = co2_cap_rate * demand_w (eq 28 uses total demand but applied per subperiod)
            # Actually eq 28: cap = 0.05 * total_weighted_demand (global cap)
            # But constraint 27 is per subperiod with slack.
            # Let's apply proportionally:
            co2_cap_w = co2_cap_rate * demand_w_val
            model.addConstr(
                co2_emit - x_CO2[w] <= co2_cap_w,
                name=f"co2_{w}"
            )

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------
    if not quiet:
        print("Starting optimization...")
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    result = {
        "status": model.Status,
        "status_name": {
            1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
            5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT",
            8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT",
            11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
            14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT",
            17: "MEM_LIMIT"
        }.get(model.Status, "UNKNOWN"),
        "solve_time_seconds": solve_time,
        "num_variables": model.NumVars,
        "num_constraints": model.NumConstrs,
        "num_nonzeros": model.NumNZs,
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["best_bound"] = model.ObjBound if hasattr(model, "ObjBound") else model.ObjVal
        result["mip_gap"] = model.MIPGap if formulation == "MILP" else 0.0

        # Investment decisions
        inv_decisions = []
        for i in G_all:
            g = generators[i]
            entry = {
                "generator_id": g["id"],
                "resource_type": g["resource_type"],
                "zone": g["zone"],
                "y_P_new": y_P_new[i].X,
                "y_P_ret": y_P_ret[i].X,
                "y_P_total_MW": y_P[i].X,
            }
            if g["is_storage"]:
                entry["y_E_new"] = y_E_new[i].X
                entry["y_E_ret"] = y_E_ret[i].X
                entry["y_E_total_MWh"] = y_E[i].X
            inv_decisions.append(entry)
        result["investment_decisions"] = inv_decisions

        # Transmission decisions
        trans_decisions = []
        for l in L_all:
            tl = trans_lines[l]
            trans_decisions.append({
                "line_id": tl["id"],
                "y_F_new_MW": y_F_new[l].X,
                "total_capacity_MW": tl["existing_capacity_MW"] + y_F_new[l].X,
            })
        result["transmission_decisions"] = trans_decisions

        # Cost breakdown
        result["cost_breakdown"] = {
            "fixed": c_fixed.getValue(),
            "variable": c_var.getValue(),
            "nse": c_nse.getValue(),
            "startup": c_start.getValue(),
            "policy": c_policy.getValue(),
        }

        # Generation summary per subperiod
        gen_summary = []
        for w in range(num_subperiods):
            ts = subperiod_timesteps[w]
            total_gen = sum(alpha[t] * x_inj[i, t].X for i in G_all for t in ts)
            total_dem = sum(alpha[t] * demand[z][t] for z in zones for t in ts)
            total_nse_val = sum(
                alpha[t] * x_nse[s, z, t].X
                for s in S_all for z in zones for t in ts
            )
            gen_summary.append({
                "subperiod": w,
                "weighted_generation_MWh": total_gen,
                "weighted_demand_MWh": total_dem,
                "weighted_nse_MWh": total_nse_val,
            })
        result["generation_summary"] = gen_summary
    else:
        result["objective_value"] = None
        result["best_bound"] = model.ObjBound if hasattr(model, "ObjBound") else None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Monolithic MILP energy systems planning model using Gurobi"
    )
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=172800, help="Solver time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = build_and_solve(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"\nSolution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']:.2f}")
    print(f"Status: {result['status_name']}")
    print(f"Solve time: {result['solve_time_seconds']:.2f}s")


if __name__ == "__main__":
    main()
