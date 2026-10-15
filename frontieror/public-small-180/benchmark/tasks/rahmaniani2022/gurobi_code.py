"""
Monolithic energy systems planning MILP model from Jacobson et al. (2023),
"A computationally efficient Benders decomposition for energy systems planning
problems with detailed operations and time-coupling constraints."

Implements the full formulation (Appendix B) using Gurobi.
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
    with open(path) as f:
        return json.load(f)


def build_model(inst, formulation="MILP", quiet=False):
    """Build the monolithic Gurobi model from the instance data."""

    m = gp.Model("energy_systems_planning")
    m.setParam("Threads", 1)
    if quiet:
        m.setParam("OutputFlag", 0)

    # =========================================================================
    # Unpack instance data
    # =========================================================================
    params = inst["problem_parameters"]
    policy = inst["policy"]
    generators = inst["generators"]
    gen_params = inst["generator_parameters"]
    existing_cap = inst["existing_capacity"]
    demand = inst["demand"]
    consumer_segs = inst["consumer_segments"]
    trans_lines = inst["transmission_lines"]
    availability = inst["availability"]
    hydro_inflow = inst["hydro_inflow"]
    subperiod_weights = inst["subperiod_weights"]

    num_zones = params["num_zones"]
    num_weeks = params["num_weeks"]
    delta_w = params["hours_per_subperiod"]  # 168
    total_hours = params["total_hours"]

    # =========================================================================
    # Build index sets
    # =========================================================================
    G = list(range(len(generators)))
    G_UC = [g for g in G if generators[g]["is_UC"]]
    G_nonUC = [g for g in G if not generators[g]["is_UC"]]
    G_STOR = [g for g in G if generators[g]["is_storage"]]
    G_HYDRO = [g for g in G if generators[g]["is_hydro"]]
    G_RPS = [g for g in G if generators[g]["is_RPS_qualifying"]]
    G_NONRET = [g for g in G if not generators[g]["can_retire"]]
    G_NONRET_set = set(G_NONRET)
    G_UC_set = set(G_UC)
    G_STOR_set = set(G_STOR)
    G_HYDRO_set = set(G_HYDRO)

    # Resources that have min output but are NOT UC, storage, or hydro
    G_minout = [g for g in G if not generators[g]["is_UC"]
                and not generators[g]["is_storage"]
                and not generators[g]["is_hydro"]]

    Z = list(range(num_zones))
    W = list(range(num_weeks))
    L = list(range(len(trans_lines)))

    # G_z: generators in zone z
    G_z = {z: [] for z in Z}
    for g in G:
        G_z[generators[g]["zone"]].append(g)

    G_z_STOR = {z: [g for g in G_z[z] if g in G_STOR_set] for z in Z}

    # Lines into / out of zone z
    I_z_in = {z: [] for z in Z}
    I_z_out = {z: [] for z in Z}
    for l_idx, line in enumerate(trans_lines):
        I_z_out[line["from_zone"]].append(l_idx)
        I_z_in[line["to_zone"]].append(l_idx)

    # Subperiod structure: H_w, t_w_0 (first), t_w (last)
    # Hours are 0-indexed internally
    H_w = {}
    t_w_0 = {}
    t_w_last = {}
    for w in W:
        start = w * delta_w
        end = (w + 1) * delta_w
        H_w[w] = list(range(start, end))
        t_w_0[w] = start
        t_w_last[w] = end - 1

    T = list(range(total_hours))

    # Map each timestep to its subperiod
    t_to_w = {}
    for w in W:
        for t in H_w[w]:
            t_to_w[t] = w

    # =========================================================================
    # Parameters
    # =========================================================================

    release_convention = inst["released_investment_convention"]

    # gamma_P: released investment increment.  UC resources use their physical
    # cluster size; non-UC resources use explicit 1-MW integer increments.
    gamma_P = {}
    for g in G:
        if g in G_UC_set:
            gamma_P[g] = generators[g]["capacity_size_MW"]
        else:
            gamma_P[g] = float(release_convention["non_uc_power_increment_MW"])

    # gamma_E: energy capacity size for storage
    gamma_E = {}
    for g in G_STOR:
        # Per math_model line 38: gamma_E_SIZE set to 1 for g not in G^UC
        # Battery is non-UC, so gamma_E = 1.0
        if g in G_UC_set:
            gamma_E[g] = gen_params["battery"]["energy_capacity_size_MWh"]
        else:
            gamma_E[g] = float(
                release_convention["non_uc_storage_energy_increment_MWh"]
            )

    # Existing capacities
    y_P_0 = {}
    for g in G:
        y_P_0[g] = existing_cap.get(str(g), 0.0)

    y_E_0 = {}
    for g in G_STOR:
        bp = gen_params["battery"]
        # existing energy = (existing_MW / capacity_size_MW) * energy_capacity_size_MWh
        if bp["capacity_size_MW"] > 0:
            y_E_0[g] = (y_P_0[g] / bp["capacity_size_MW"]) * bp["energy_capacity_size_MWh"]
        else:
            y_E_0[g] = 0.0

    # alpha_t = beta_w / delta_w
    alpha = {}
    for w in W:
        a = subperiod_weights[w] / delta_w
        for t in H_w[w]:
            alpha[t] = a

    # Generator type parameters
    def gp_val(g, key, default=0.0):
        rtype = generators[g]["resource_type"]
        p = gen_params[rtype]
        return p.get(key, default)

    # Cost parameters
    c_P_INV = {g: gp_val(g, "investment_cost_per_MW_yr") for g in G}
    c_P_FOM = {g: gp_val(g, "fixed_OM_cost_per_MW_yr") for g in G}
    c_VAR = {g: gp_val(g, "variable_cost_per_MWh") for g in G}
    c_START = {g: gp_val(g, "start_up_cost") for g in G_UC}

    c_E_INV_stor = {g: gen_params["battery"]["energy_investment_cost_per_MWh_yr"] for g in G_STOR}
    c_E_FOM_stor = {
        g: gen_params["battery"]["energy_fixed_OM_cost_per_MWh_yr"]
        for g in G_STOR
    }

    # These released zero values are now explicit instance parameters.
    c_E_INV_hydro = {
        g: gen_params["hydro"]["energy_investment_cost_per_MWh_yr"]
        for g in G_HYDRO
    }
    c_E_FOM_hydro = {
        g: gen_params["hydro"]["energy_fixed_OM_cost_per_MWh_yr"]
        for g in G_HYDRO
    }

    # Hydro duration rho_g
    rho_hydro = {g: gen_params["hydro"]["duration_MWh_per_MW"] for g in G_HYDRO}

    # Max capacities
    y_P_max = {g: gp_val(g, "max_capacity_MW") for g in G}
    y_E_max = {g: gen_params["battery"]["max_energy_capacity_MWh"] for g in G_STOR}

    # Storage parameters
    eta_charge = {g: gen_params["battery"]["charging_efficiency"] for g in G_STOR}
    eta_discharge = {g: gen_params["battery"]["discharging_efficiency"] for g in G_STOR}
    eta_self_discharge = {g: gen_params["battery"]["self_discharge_rate_pct"] for g in G_STOR}
    rho_min_stor = {g: gen_params["battery"]["min_duration_MWh_per_MW"] for g in G_STOR}
    rho_max_stor = {g: gen_params["battery"]["max_duration_MWh_per_MW"] for g in G_STOR}

    # Operational parameters
    min_output = {g: gp_val(g, "min_output_frac") for g in G}
    ramp_up = {g: gp_val(g, "ramp_up_rate_pct_per_hr") for g in G}
    ramp_down = {g: gp_val(g, "ramp_down_rate_pct_per_hr") for g in G}
    min_up_time = {g: int(gp_val(g, "min_up_time_hr")) for g in G_UC}
    min_down_time = {g: int(gp_val(g, "min_down_time_hr")) for g in G_UC}

    # Availability
    avail = {}
    for g in G:
        key = str(g)
        if key in availability:
            avail[g] = availability[key]
        else:
            avail[g] = [1.0] * total_hours

    # Hydro inflow
    h_inflow = {}
    for g in G_HYDRO:
        key = str(g)
        if key in hydro_inflow:
            h_inflow[g] = hydro_inflow[key]
        else:
            h_inflow[g] = [0.0] * total_hours

    # Demand
    d_zt = {}
    for z in Z:
        dem = demand[str(z)]
        for t in T:
            d_zt[z, t] = dem[t]

    # Emissions
    co2_rate = {g: gp_val(g, "CO2_emission_tons_per_MWh") for g in G}

    # Policy
    RPS_share = policy["RPS_share"]
    CO2_cap_rate = policy["CO2_cap_tons_per_MWh"]
    c_RPS = policy["RPS_noncompliance_cost_per_MWh"]
    c_CO2 = policy["CO2_noncompliance_cost_per_ton"]

    # NSE segments: list of (segment_id, zone, max_nse_fraction, cost_per_MWh)
    # Build unique (s, z) pairs
    nse_segments = []
    for seg in consumer_segs:
        nse_segments.append((seg["segment_id"], seg["zone"],
                             seg["max_nse_fraction"], seg["cost_per_MWh"]))

    # Transmission
    y_F_0 = {l: trans_lines[l]["existing_capacity_MW"] for l in L}
    y_F_max = {l: trans_lines[l]["max_capacity_MW"] for l in L}
    c_F_INV = {l: trans_lines[l]["investment_cost_per_MW_yr"] for l in L}

    # Determine variable type for investment
    is_milp = (formulation == "MILP")
    inv_vtype = GRB.INTEGER if is_milp else GRB.CONTINUOUS

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================
    print("Creating variables...")

    # --- Investment variables ---
    y_P_new = m.addVars(G, vtype=inv_vtype, lb=0, name="y_P_new")
    y_P_ret = m.addVars(G, vtype=inv_vtype, lb=0, name="y_P_ret")
    y_P = m.addVars(G, vtype=GRB.CONTINUOUS, lb=0, name="y_P")

    y_E_new = m.addVars(G_STOR, vtype=inv_vtype, lb=0, name="y_E_new")
    y_E_ret = m.addVars(G_STOR, vtype=inv_vtype, lb=0, name="y_E_ret")
    y_E = m.addVars(G_STOR, vtype=GRB.CONTINUOUS, lb=0, name="y_E")

    # Transmission expansion follows the same integer-MW MILP convention.
    y_F_new = m.addVars(L, vtype=inv_vtype, lb=0, name="y_F_new")
    y_F = m.addVars(L, vtype=GRB.CONTINUOUS, lb=0, name="y_F")

    # --- Operational variables ---
    # x_inj[g,t]: generation injection
    x_inj = m.addVars(G, T, vtype=GRB.CONTINUOUS, lb=0, name="x_inj")

    # x_wdw[g,t]: storage withdrawal (charging)
    x_wdw = m.addVars(G_STOR, T, vtype=GRB.CONTINUOUS, lb=0, name="x_wdw")

    # x_nse[s_idx, t]: NSE for segment index s_idx
    nse_indices = [(i, t) for i in range(len(nse_segments)) for t in T]
    x_nse = m.addVars(nse_indices, vtype=GRB.CONTINUOUS, lb=0, name="x_nse")

    # x_soc[g,t]: state of charge (storage)
    x_soc = m.addVars(G_STOR, T, vtype=GRB.CONTINUOUS, lb=0, name="x_soc")

    # x_lvl[g,t]: reservoir level (hydro)
    x_lvl = m.addVars(G_HYDRO, T, vtype=GRB.CONTINUOUS, lb=0, name="x_lvl")

    # x_spill[g,t]: hydro spillage
    x_spill = m.addVars(G_HYDRO, T, vtype=GRB.CONTINUOUS, lb=0, name="x_spill")

    # x_flow[l,t]: transmission flow (free variable)
    x_flow = m.addVars(L, T, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="x_flow")

    # UC variables (always continuous/relaxed)
    x_commit = m.addVars(G_UC, T, vtype=GRB.CONTINUOUS, lb=0, name="x_commit")
    x_start = m.addVars(G_UC, T, vtype=GRB.CONTINUOUS, lb=0, name="x_start")
    x_shut = m.addVars(G_UC, T, vtype=GRB.CONTINUOUS, lb=0, name="x_shut")

    # Policy slack variables
    x_RPS = m.addVars(W, vtype=GRB.CONTINUOUS, lb=0, name="x_RPS")
    x_CO2 = m.addVars(W, vtype=GRB.CONTINUOUS, lb=0, name="x_CO2")

    m.update()

    # =========================================================================
    # OBJECTIVE FUNCTION (Eq. 29-34)
    # =========================================================================
    print("Building objective...")

    # c^FIXED (Eq. 29)
    c_fixed = gp.LinExpr()

    # Investment costs: sum_g c_P_INV * gamma_P * y_P_new
    for g in G:
        c_fixed += c_P_INV[g] * gamma_P[g] * y_P_new[g]

    # Storage energy investment: sum_{g in STOR} c_E_INV * gamma_E * y_E_new
    for g in G_STOR:
        c_fixed += c_E_INV_stor[g] * gamma_E[g] * y_E_new[g]

    # Hydro energy investment: sum_{g in HYDRO} c_E_INV_hydro * rho_g * y_P_new
    for g in G_HYDRO:
        c_fixed += c_E_INV_hydro[g] * rho_hydro[g] * y_P_new[g]

    # FOM: sum_g c_P_FOM * y_P
    for g in G:
        c_fixed += c_P_FOM[g] * y_P[g]

    # Storage energy FOM: sum_{g in STOR} c_E_FOM * y_E
    for g in G_STOR:
        c_fixed += c_E_FOM_stor[g] * y_E[g]

    # Hydro energy FOM: sum_{g in HYDRO} c_E_FOM_hydro * rho_g * y_P
    for g in G_HYDRO:
        c_fixed += c_E_FOM_hydro[g] * rho_hydro[g] * y_P[g]

    # Transmission investment: sum_l c_F_INV * y_F_new
    for l in L:
        c_fixed += c_F_INV[l] * y_F_new[l]

    # c^VAR (Eq. 30)
    c_var = gp.LinExpr()
    for g in G:
        if c_VAR[g] == 0.0:
            continue
        for t in T:
            c_var += c_VAR[g] * alpha[t] * x_inj[g, t]
    for g in G_STOR:
        if c_VAR[g] == 0.0:
            continue
        for t in T:
            c_var += c_VAR[g] * alpha[t] * x_wdw[g, t]

    # c^NSE (Eq. 31)
    c_nse = gp.LinExpr()
    for i, (sid, z, frac, cost) in enumerate(nse_segments):
        if cost == 0.0:
            continue
        for t in T:
            c_nse += cost * alpha[t] * x_nse[i, t]

    # c^START (Eq. 32)
    c_start = gp.LinExpr()
    for g in G_UC:
        if c_START[g] == 0.0:
            continue
        for t in T:
            c_start += c_START[g] * alpha[t] * x_start[g, t]

    # c^RPS + c^CO2 (Eq. 33)
    c_policy = gp.LinExpr()
    for w in W:
        c_policy += c_RPS * x_RPS[w]
        c_policy += c_CO2 * x_CO2[w]

    m.setObjective(c_fixed + c_var + c_nse + c_start + c_policy, GRB.MINIMIZE)

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # ---- B.1 Investment Constraints ----
    print("Adding investment constraints...")

    # (12a) gamma_P * y_P_new <= y_P_max
    for g in G:
        m.addConstr(gamma_P[g] * y_P_new[g] <= y_P_max[g], name=f"inv_cap_{g}")

    # (12b) gamma_P * y_P_ret <= y_P_0
    for g in G:
        m.addConstr(gamma_P[g] * y_P_ret[g] <= y_P_0[g], name=f"ret_cap_{g}")

    # (12c) y_P_ret = 0 for non-retirable
    for g in G_NONRET:
        m.addConstr(y_P_ret[g] == 0, name=f"noret_{g}")

    # (12d) y_P = y_P_0 + gamma_P * (y_P_new - y_P_ret)
    for g in G:
        m.addConstr(y_P[g] == y_P_0[g] + gamma_P[g] * (y_P_new[g] - y_P_ret[g]),
                     name=f"cap_bal_{g}")

    # (13a) gamma_E * y_E_new <= y_E_max
    for g in G_STOR:
        m.addConstr(gamma_E[g] * y_E_new[g] <= y_E_max[g], name=f"stor_inv_{g}")

    # (13b) gamma_E * y_E_ret <= y_E_0
    for g in G_STOR:
        m.addConstr(gamma_E[g] * y_E_ret[g] <= y_E_0[g], name=f"stor_ret_{g}")

    # (13c) y_E_ret = 0 for non-retirable storage
    for g in G_STOR:
        if g in G_NONRET_set:
            m.addConstr(y_E_ret[g] == 0, name=f"stor_noret_{g}")

    # (13d) y_E = y_E_0 + gamma_E * (y_E_new - y_E_ret)
    for g in G_STOR:
        m.addConstr(y_E[g] == y_E_0[g] + gamma_E[g] * (y_E_new[g] - y_E_ret[g]),
                     name=f"stor_bal_{g}")

    # (13e) rho_min * y_P <= y_E
    for g in G_STOR:
        m.addConstr(rho_min_stor[g] * y_P[g] <= y_E[g], name=f"stor_minrho_{g}")

    # (13f) rho_max * y_P >= y_E
    for g in G_STOR:
        m.addConstr(rho_max_stor[g] * y_P[g] >= y_E[g], name=f"stor_maxrho_{g}")

    # (14a) max_capacity_MW is total capacity after expansion.
    for l in L:
        m.addConstr(
            y_F_new[l] <= max(0.0, y_F_max[l] - y_F_0[l]),
            name=f"trans_inv_{l}",
        )

    # (14b) y_F = y_F_0 + y_F_new
    for l in L:
        m.addConstr(y_F[l] == y_F_0[l] + y_F_new[l], name=f"trans_bal_{l}")

    # ---- B.2 Operational Constraints ----
    print("Adding operational constraints...")

    # (15) Power balance: for each zone z, timestep t
    for z in Z:
        for t in T:
            lhs = gp.LinExpr()
            # + sum_{g in G_z} x_inj
            for g in G_z[z]:
                lhs += x_inj[g, t]
            # - sum_{g in G_z^STOR} x_wdw
            for g in G_z_STOR[z]:
                lhs -= x_wdw[g, t]
            # - sum_{l in I_z_out} x_flow + sum_{l in I_z_in} x_flow
            for l in I_z_out[z]:
                lhs -= x_flow[l, t]
            for l in I_z_in[z]:
                lhs += x_flow[l, t]
            # + sum_s x_nse
            for i, (sid, sz, frac, cost) in enumerate(nse_segments):
                if sz == z:
                    lhs += x_nse[i, t]
            m.addConstr(lhs == d_zt[z, t], name=f"pbal_{z}_{t}")

    # (16a) x_inj <= sigma * y_P for non-UC
    print("  Output limits...")
    for g in G_nonUC:
        for t in T:
            m.addConstr(x_inj[g, t] <= avail[g][t] * y_P[g],
                         name=f"out_lim_{g}_{t}")

    # (16b) x_wdw <= sigma * y_P for storage
    for g in G_STOR:
        for t in T:
            m.addConstr(x_wdw[g, t] <= avail[g][t] * y_P[g],
                         name=f"wdw_lim_{g}_{t}")

    # (16c) x_inj + x_wdw <= y_P for storage
    for g in G_STOR:
        for t in T:
            m.addConstr(x_inj[g, t] + x_wdw[g, t] <= y_P[g],
                         name=f"stor_pw_{g}_{t}")

    # (16d) x_soc <= y_E
    for g in G_STOR:
        for t in T:
            m.addConstr(x_soc[g, t] <= y_E[g], name=f"soc_lim_{g}_{t}")

    # (16e) x_lvl <= rho * y_P for hydro
    for g in G_HYDRO:
        for t in T:
            m.addConstr(x_lvl[g, t] <= rho_hydro[g] * y_P[g],
                         name=f"lvl_lim_{g}_{t}")

    # (16f) eta_charge * x_wdw <= y_E - x_soc
    for g in G_STOR:
        for t in T:
            m.addConstr(eta_charge[g] * x_wdw[g, t] <= y_E[g] - x_soc[g, t],
                         name=f"chrg_room_{g}_{t}")

    # (16g) x_inj / eta_discharge <= x_soc
    for g in G_STOR:
        for t in T:
            m.addConstr(x_inj[g, t] <= eta_discharge[g] * x_soc[g, t],
                         name=f"dchrg_avail_{g}_{t}")

    # (17a) x_inj >= theta * y_P for non-UC, non-storage, non-hydro
    print("  Minimum output...")
    for g in G_minout:
        theta = min_output[g]
        if theta > 0:
            for t in T:
                m.addConstr(x_inj[g, t] >= theta * y_P[g],
                             name=f"min_out_{g}_{t}")

    # (17b) x_inj + x_spill >= theta * y_P for hydro
    for g in G_HYDRO:
        theta = min_output[g]
        if theta > 0:
            for t in T:
                m.addConstr(x_inj[g, t] + x_spill[g, t] >= theta * y_P[g],
                             name=f"hydro_min_{g}_{t}")

    # (18a, 18b) -y_F <= x_flow <= y_F
    print("  Transmission limits...")
    for l in L:
        for t in T:
            m.addConstr(x_flow[l, t] <= y_F[l], name=f"flow_ub_{l}_{t}")
            m.addConstr(-x_flow[l, t] <= y_F[l], name=f"flow_lb_{l}_{t}")

    # (19) x_nse <= nu_s * d_{z,t}
    print("  NSE limits...")
    for i, (sid, z, frac, cost) in enumerate(nse_segments):
        for t in T:
            m.addConstr(x_nse[i, t] <= frac * d_zt[z, t],
                         name=f"nse_lim_{i}_{t}")

    # (20a, 20b) Storage dynamics
    print("  Storage dynamics...")
    for g in G_STOR:
        ec = eta_charge[g]
        ed = eta_discharge[g]
        sd = eta_self_discharge[g]
        for w in W:
            hours = H_w[w]
            first = hours[0]
            last = hours[-1]

            # (20b) Wrap-around: first timestep links to last
            m.addConstr(
                x_soc[g, first] - x_soc[g, last] ==
                ec * x_wdw[g, first] - x_inj[g, first] / ed - sd * x_soc[g, last],
                name=f"soc_wrap_{g}_{w}")

            # (20a) Interior timesteps
            for idx in range(1, len(hours)):
                t = hours[idx]
                t_prev = hours[idx - 1]
                m.addConstr(
                    x_soc[g, t] - x_soc[g, t_prev] ==
                    ec * x_wdw[g, t] - x_inj[g, t] / ed - sd * x_soc[g, t_prev],
                    name=f"soc_dyn_{g}_{t}")

    # (20c, 20d) Hydro dynamics
    print("  Hydro dynamics...")
    for g in G_HYDRO:
        for w in W:
            hours = H_w[w]
            first = hours[0]
            last = hours[-1]

            # (20d) Wrap-around
            m.addConstr(
                x_lvl[g, first] - x_lvl[g, last] ==
                h_inflow[g][first] * y_P[g] - x_inj[g, first] - x_spill[g, first],
                name=f"hydro_wrap_{g}_{w}")

            # (20c) Interior timesteps
            for idx in range(1, len(hours)):
                t = hours[idx]
                t_prev = hours[idx - 1]
                m.addConstr(
                    x_lvl[g, t] - x_lvl[g, t_prev] ==
                    h_inflow[g][t] * y_P[g] - x_inj[g, t] - x_spill[g, t],
                    name=f"hydro_dyn_{g}_{t}")

    # (21a-21d) Non-UC ramping
    print("  Non-UC ramping...")
    for g in G_nonUC:
        mu_up = ramp_up[g]
        mu_dn = ramp_down[g]
        if mu_up == 0 and mu_dn == 0:
            continue
        for w in W:
            hours = H_w[w]
            first = hours[0]
            last = hours[-1]

            # (21c) wrap: x_inj[first] - x_inj[last] <= mu_up * y_P
            if mu_up > 0:
                m.addConstr(x_inj[g, first] - x_inj[g, last] <= mu_up * y_P[g],
                             name=f"ramp_up_wrap_{g}_{w}")
            # (21d) wrap: x_inj[last] - x_inj[first] <= mu_dn * y_P
            if mu_dn > 0:
                m.addConstr(x_inj[g, last] - x_inj[g, first] <= mu_dn * y_P[g],
                             name=f"ramp_dn_wrap_{g}_{w}")

            # (21a, 21b) Interior
            for idx in range(1, len(hours)):
                t = hours[idx]
                t_prev = hours[idx - 1]
                if mu_up > 0:
                    m.addConstr(x_inj[g, t] - x_inj[g, t_prev] <= mu_up * y_P[g],
                                 name=f"ramp_up_{g}_{t}")
                if mu_dn > 0:
                    m.addConstr(x_inj[g, t_prev] - x_inj[g, t] <= mu_dn * y_P[g],
                                 name=f"ramp_dn_{g}_{t}")

    # (22a-22g) UC constraints
    print("  UC constraints...")
    for g in G_UC:
        gp_size = gamma_P[g]
        theta = min_output[g]
        for t in T:
            # (22a) gamma * commit <= y_P
            m.addConstr(gp_size * x_commit[g, t] <= y_P[g],
                         name=f"uc_cap_{g}_{t}")
            # (22b) gamma * start <= y_P
            m.addConstr(gp_size * x_start[g, t] <= y_P[g],
                         name=f"uc_start_cap_{g}_{t}")
            # (22c) gamma * shut <= y_P
            m.addConstr(gp_size * x_shut[g, t] <= y_P[g],
                         name=f"uc_shut_cap_{g}_{t}")
            # (22d) x_inj >= commit * theta * gamma
            m.addConstr(x_inj[g, t] >= x_commit[g, t] * theta * gp_size,
                         name=f"uc_min_{g}_{t}")
            # (22e) x_inj <= commit * sigma * gamma
            m.addConstr(x_inj[g, t] <= x_commit[g, t] * avail[g][t] * gp_size,
                         name=f"uc_max_{g}_{t}")

        # (22f, 22g) Commitment state transitions
        for w in W:
            hours = H_w[w]
            first = hours[0]
            last = hours[-1]

            # (22g) Wrap-around
            m.addConstr(
                x_commit[g, first] - x_commit[g, last] ==
                x_start[g, first] - x_shut[g, first],
                name=f"uc_trans_wrap_{g}_{w}")

            # (22f) Interior
            for idx in range(1, len(hours)):
                t = hours[idx]
                t_prev = hours[idx - 1]
                m.addConstr(
                    x_commit[g, t] - x_commit[g, t_prev] ==
                    x_start[g, t] - x_shut[g, t],
                    name=f"uc_trans_{g}_{t}")

    # (23a-23d) UC ramping
    print("  UC ramping...")
    for g in G_UC:
        gp_size = gamma_P[g]
        theta = min_output[g]
        mu_up = ramp_up[g]
        mu_dn = ramp_down[g]

        for w in W:
            hours = H_w[w]
            first = hours[0]
            last = hours[-1]

            # Helper for ramp coefficients
            def start_coeff_up(t_idx):
                sig = avail[g][t_idx]
                return min(sig, max(theta, mu_up))

            def start_coeff_dn(t_idx):
                sig = avail[g][t_idx]
                return min(sig, max(theta, mu_dn))

            # (23c) Wrap-around up: x_inj[first] - x_inj[last] <= ...
            m.addConstr(
                x_inj[g, first] - x_inj[g, last] <=
                gp_size * mu_up * (x_commit[g, first] - x_start[g, first])
                + gp_size * start_coeff_up(first) * x_start[g, first]
                - gp_size * theta * x_shut[g, first],
                name=f"uc_ramp_up_wrap_{g}_{w}")

            # (23d) Wrap-around down: x_inj[last] - x_inj[first] <= ...
            m.addConstr(
                x_inj[g, last] - x_inj[g, first] <=
                gp_size * mu_dn * (x_commit[g, first] - x_start[g, first])
                + gp_size * start_coeff_dn(first) * x_start[g, first]
                - gp_size * theta * x_shut[g, first],
                name=f"uc_ramp_dn_wrap_{g}_{w}")

            # (23a, 23b) Interior
            for idx in range(1, len(hours)):
                t = hours[idx]
                t_prev = hours[idx - 1]

                # (23a) Up
                m.addConstr(
                    x_inj[g, t] - x_inj[g, t_prev] <=
                    gp_size * mu_up * (x_commit[g, t] - x_start[g, t])
                    + gp_size * start_coeff_up(t) * x_start[g, t]
                    - gp_size * theta * x_shut[g, t],
                    name=f"uc_ramp_up_{g}_{t}")

                # (23b) Down
                m.addConstr(
                    x_inj[g, t_prev] - x_inj[g, t] <=
                    gp_size * mu_dn * (x_commit[g, t] - x_start[g, t])
                    + gp_size * start_coeff_dn(t) * x_start[g, t]
                    - gp_size * theta * x_shut[g, t],
                    name=f"uc_ramp_dn_{g}_{t}")

    # (24a, 24b, 25) Minimum up/down time
    print("  Min up/down time...")
    for g in G_UC:
        gp_size = gamma_P[g]
        tau_up = min_up_time[g]
        tau_dn = min_down_time[g]

        for w in W:
            hours = H_w[w]
            n_hours = len(hours)

            for local_idx in range(n_hours):
                t = hours[local_idx]

                # (24a) x_commit[g,t] >= sum_{k in Omega_up(t)} x_start[g,k]
                if tau_up > 0:
                    lhs = gp.LinExpr()
                    for step in range(tau_up + 1):  # 0..tau_up
                        # phi_{w,step}(t): step steps before t (circular)
                        k_local = (local_idx - step) % n_hours
                        k = hours[k_local]
                        lhs += x_start[g, k]
                    m.addConstr(x_commit[g, t] >= lhs,
                                 name=f"min_up_{g}_{t}")

                # (24b) x_commit[g,t] + sum_{k in Omega_dn(t)} x_shut[g,k] <= y_P / gamma
                if tau_dn > 0:
                    lhs = x_commit[g, t]
                    for step in range(tau_dn + 1):  # 0..tau_dn
                        k_local = (local_idx - step) % n_hours
                        k = hours[k_local]
                        lhs += x_shut[g, k]
                    m.addConstr(lhs <= y_P[g] / gp_size,
                                 name=f"min_dn_{g}_{t}")

    # ---- B.3 Policy Constraints ----
    print("Adding policy constraints...")

    # Compute total weighted demand
    total_weighted_demand = sum(alpha[t] * d_zt[z, t] for z in Z for t in T)

    # (26) RPS constraint
    if RPS_share > 0 or c_RPS > 0:
        rps_lhs = gp.LinExpr()
        for w in W:
            for t in H_w[w]:
                for g in G_RPS:
                    rps_lhs += alpha[t] * x_inj[g, t]
            rps_lhs += x_RPS[w]
        m.addConstr(rps_lhs >= RPS_share * total_weighted_demand, name="rps")

    # (27a, 28) CO2 cap
    # overline{c}^{CO2} = CO2_cap_tons_per_MWh * total_weighted_demand
    # But for CO2 scenario, eq 28 says cap = 0.05 * total_weighted_demand
    # The instance provides CO2_cap_tons_per_MWh which encodes this.
    co2_cap = CO2_cap_rate * total_weighted_demand

    co2_lhs = gp.LinExpr()
    for w in W:
        for t in H_w[w]:
            for g in G:
                if co2_rate[g] != 0:
                    co2_lhs += alpha[t] * co2_rate[g] * x_inj[g, t]
            for g in G_STOR:
                if co2_rate[g] != 0:
                    co2_lhs += alpha[t] * co2_rate[g] * x_wdw[g, t]
        co2_lhs -= x_CO2[w]
    m.addConstr(co2_lhs <= co2_cap, name="co2_cap")

    if not quiet:
        print(f"Model has {m.NumVars} variables and {m.NumConstrs} constraints.")
    return m, {
        "generators": generators,
        "y_P_new": y_P_new,
        "y_P_ret": y_P_ret,
        "y_P": y_P,
        "y_E_new": y_E_new,
        "y_E_ret": y_E_ret,
        "y_E": y_E,
        "y_F_new": y_F_new,
        "y_F": y_F,
        "G": G,
        "G_STOR": G_STOR,
        "G_HYDRO": G_HYDRO,
        "L": L,
        "gamma_P": gamma_P,
        "gamma_E": gamma_E,
    }


def solve_model(m, time_limit=None, formulation="MILP"):
    """Configure solver parameters and solve."""

    tol = 1e-3

    # Barrier method with crossover off (for monolithic)
    m.Params.Method = 2  # Barrier
    m.Params.Crossover = 0  # Crossover off
    m.Params.BarConvTol = tol
    m.Params.OptimalityTol = tol
    m.Params.FeasibilityTol = tol

    if formulation == "MILP":
        m.Params.MIPGap = 1e-4

    if time_limit is not None:
        m.Params.TimeLimit = time_limit
        m.Params.MIPFocus = 1
        m.Params.NoRelHeurTime = min(60.0, time_limit * 0.05)

    print("Solving...")
    start_time = time.time()
    m.optimize()
    elapsed = time.time() - start_time
    print(f"Solve time: {elapsed:.2f} seconds")

    return elapsed


def extract_solution(m, model_data):
    """Extract solution values and build output dict."""

    sol = {}

    status = m.Status
    sol["status"] = status
    sol["status_name"] = {
        1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
        5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT",
        8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT",
        11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
        14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT",
        17: "MEM_LIMIT"
    }.get(status, "UNKNOWN")

    if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT,
                  GRB.ITERATION_LIMIT, GRB.NODE_LIMIT, GRB.INTERRUPTED):
        try:
            sol["objective_value"] = m.ObjVal
        except Exception:
            sol["objective_value"] = None
    else:
        sol["objective_value"] = None

    if m.SolCount > 0:
        try:
            sol["objective_bound"] = m.ObjBound
        except Exception:
            sol["objective_bound"] = None

        # Extract investment decisions
        investments = {}
        generators = model_data["generators"]
        y_P_new = model_data["y_P_new"]
        y_P_ret = model_data["y_P_ret"]
        y_P = model_data["y_P"]
        gamma_P = model_data["gamma_P"]

        for g in model_data["G"]:
            gen = generators[g]
            pnew = y_P_new[g].X
            pret = y_P_ret[g].X
            pcap = y_P[g].X
            if abs(pnew) > 1e-6 or abs(pret) > 1e-6:
                investments[str(g)] = {
                    "resource_type": gen["resource_type"],
                    "zone": gen["zone"],
                    "new_units": pnew,
                    "new_MW": pnew * gamma_P[g],
                    "retired_units": pret,
                    "retired_MW": pret * gamma_P[g],
                    "total_capacity_MW": pcap
                }

        sol["investments"] = investments

        # Storage energy investments
        stor_investments = {}
        y_E_new = model_data["y_E_new"]
        y_E = model_data["y_E"]
        gamma_E = model_data["gamma_E"]
        for g in model_data["G_STOR"]:
            enew = y_E_new[g].X
            ecap = y_E[g].X
            if abs(enew) > 1e-6:
                stor_investments[str(g)] = {
                    "new_units": enew,
                    "new_MWh": enew * gamma_E[g],
                    "total_energy_MWh": ecap
                }
        sol["storage_energy_investments"] = stor_investments

        # Transmission investments
        trans_investments = {}
        y_F_new = model_data["y_F_new"]
        y_F = model_data["y_F"]
        for l in model_data["L"]:
            fnew = y_F_new[l].X
            if abs(fnew) > 1e-6:
                trans_investments[str(l)] = {
                    "new_MW": fnew,
                    "total_capacity_MW": y_F[l].X
                }
        sol["transmission_investments"] = trans_investments
    else:
        sol["investments"] = {}
        sol["storage_energy_investments"] = {}
        sol["transmission_investments"] = {}

    return sol


def main():
    parser = argparse.ArgumentParser(
        description="Monolithic energy systems planning model (Jacobson et al. 2023)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json",
                        help="Path for output solution JSON")
    parser.add_argument("--time_limit", type=int, default=None,
                        help="Solver time limit in seconds")
    parser.add_argument("--formulation", type=str, default="MILP",
                        choices=["LP", "MILP"],
                        help="Formulation type (LP or MILP)")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from {args.instance_path}")
    inst = load_instance(args.instance_path)

    formulation = args.formulation
    # If instance specifies only one formulation type, use that
    form_types = inst["problem_parameters"].get("formulation_types", ["LP", "MILP"])
    if formulation not in form_types:
        print(f"Warning: requested formulation {formulation} not in instance types {form_types}.")

    print(f"Building {formulation} model...")
    model, model_data = build_model(inst, formulation=formulation)

    elapsed = solve_model(model, time_limit=args.time_limit, formulation=formulation)

    solution = extract_solution(model, model_data)
    solution["solve_time_seconds"] = elapsed
    solution["formulation"] = formulation
    solution["instance_path"] = args.instance_path

    print(f"\nStatus: {solution['status_name']}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.6f}")

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
