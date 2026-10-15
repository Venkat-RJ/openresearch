"""
Gurobi implementation of the Time-Indexed Formulation (TIF) for the
Outbound Baggage Handling Problem (OBHP).

Reference: Frey, Kolisch, and Artigues (2017)
"Column Generation for Outbound Baggage Handling at Airports"
Transportation Science, Articles in Advance, pp. 1-16.

The TIF assigns flights to carousels and schedules baggage handling
to minimize utilization peaks across peak intervals. The formulation
consists of:
  - Objective (1): minimize penalty-weighted threshold violations
  - Constraint (2): each flight assigned exactly once
  - Constraint (3): working station capacity per carousel per period
  - Constraint (4): parking position capacity per carousel per period
  - Constraint (5): central storage capacity per period (preprocessed)
  - Constraint (6): utilization threshold linking constraints
  - Constraint (7): at most one threshold level per carousel per peak interval
  - Constraints (8)-(9): binary domains

ASSUMPTIONS:
  - The arrival_profile for flight i gives arrivals starting at earliest_start_period.
    i.e., A_{i, earliest_start_period + j} = arrival_profile[j].
  - Peak interval ranges are inclusive: [start_period, end_period].
  - The depletion offset Delta is taken from the instance JSON field
    central_storage.depletion_offset_periods.
  - Constraint (5) uses the baseline constant storage capacity K^s
    (Section 3.1, eq. (5) of the paper).
  - For constraints (3) and (4), a flight occupies a carousel during
    periods [s^h_tau, S^e_i - 1] (i.e., from handling start to one period
    before the end time).
  - S_i(w) = S_i for all w, meaning the set of feasible start time tuples
    does not depend on w (consistent with the paper's description where
    the tuple tau = <s^h, s^d> depends only on timing, not workers).
  - For constraint (5), Phi^i_{t,tau} does not depend on carousel c or
    working stations w. We use auxiliary variables z_{i,tau} = sum_c sum_w
    x_{i,c,w,tau} to avoid expanding over carousels in this constraint,
    which dramatically reduces the number of nonzeros.
"""

import json
import argparse
import sys
import time
from collections import defaultdict

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
# =============================================================================
# DATA LOADING AND PARSING
# =============================================================================

def load_instance(path):
    """Load an OBHP instance from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_instance(inst):
    """
    Parse the instance JSON into structured data.

    Returns a dict with all sets, parameters, and derived data needed
    for the TIF formulation.
    """
    data = {}

    # Planning horizon
    T = inst["planning_horizon"]["num_periods"]
    data["T"] = T

    # Objective parameters
    obj = inst["objective"]
    data["u_ta"] = obj["target_utilization"]
    data["threshold_values"] = obj["threshold_values"]  # u_1, ..., u_K
    data["penalties"] = obj["penalties"]                 # p_1, ..., p_K
    K = len(data["threshold_values"])
    data["K"] = K

    # Peak intervals (M disjoint peak intervals)
    peak_intervals = inst["peak_intervals"]
    data["peak_intervals"] = peak_intervals
    data["M"] = len(peak_intervals)

    # Central storage
    cs = inst["central_storage"]
    data["Ks"] = cs["capacity_bags"]
    data["depletion_rate"] = cs["depletion_rate_bags_per_period"]
    data["depletion_offset"] = cs["depletion_offset_periods"]

    # Working rate
    data["working_rate"] = inst["working_rate_bags_per_period"]

    # Carousels
    carousels = inst["carousels"]
    data["carousels"] = carousels
    data["num_carousels"] = len(carousels)
    data["carousel_types"] = inst["carousel_types"]

    # Build carousel type -> carousel list mapping
    Cr = defaultdict(list)  # type_r -> list of carousel indices (0-based)
    for idx, car in enumerate(carousels):
        Cr[str(car["type_r"])].append(idx)
    data["Cr"] = Cr

    # Build set of all carousel types present
    data["R"] = sorted(Cr.keys())

    # Flights
    flights = inst["flights"]
    data["flights"] = flights
    data["num_flights"] = len(flights)

    return data


# =============================================================================
# PREPROCESSING (Section 3.3)
# =============================================================================

def preprocess_storage_capacity(data):
    """
    Compute time-dependent storage capacity K^s_t for constraint (5).

    K^s_t = K^s - sum_{i in F : t < S^es_i} (arrivals of i up to time t)

    This tightens the storage capacity by accounting for bags that are
    guaranteed to be in storage at time t because their flights' handling
    has not yet started and bags have already arrived.
    """
    T = data["T"]
    flights = data["flights"]
    Ks_base = data["Ks"]

    # Precompute cumulative arrivals for each flight up to each period
    # Only relevant for flights whose earliest start is in the future at time t
    Ks_t = [Ks_base] * T

    for flight in flights:
        S_es = flight["earliest_start_period"]
        profile = flight["arrival_profile"]
        earliest = flight["earliest_start_period"]

        # Compute cumulative arrivals of this flight at each period
        cum_arr = 0
        for j in range(len(profile)):
            arr_period = earliest + j
            if arr_period >= T:
                break
            cum_arr += profile[j]
            # For any t in [arr_period, S_es - 1], this flight has
            # cum_arr bags guaranteed in storage
            if arr_period < S_es:
                # These bags are guaranteed to be in storage from arr_period
                # until at least S_es - 1
                for t in range(arr_period, S_es):
                    if t < T:
                        Ks_t[t] -= profile[j]

    # The above double-counts. Let me redo this more carefully.
    # Reset and recompute properly
    Ks_t = [Ks_base] * T
    for t in range(T):
        reserved = 0
        for flight in flights:
            S_es = flight["earliest_start_period"]
            if t < S_es:
                # This flight's handling hasn't started at time t.
                # Count all arrivals of this flight up to and including time t.
                profile = flight["arrival_profile"]
                earliest = flight["earliest_start_period"]
                cum = 0
                for j in range(len(profile)):
                    arr_period = earliest + j
                    if arr_period > t:
                        break
                    cum += profile[j]
                reserved += cum
        Ks_t[t] = max(Ks_base - reserved, 0)

    return Ks_t


def preprocess_tighten_start_times(data):
    """
    Preprocessing from Section 3.3:
    1. Tighten earliest start time: if no bags have arrived by S^es_i,
       shift it forward to the first period with arrivals.

    Modifies flights in-place and returns the updated data.
    """
    flights = data["flights"]

    for flight in flights:
        profile = flight["arrival_profile"]
        earliest = flight["earliest_start_period"]

        # Tighten earliest start: shift forward if no bags have arrived
        cumulative = 0
        new_earliest = earliest
        for j in range(len(profile)):
            arr_period = earliest + j
            cumulative += profile[j]
            if cumulative > 0:
                new_earliest = min(arr_period, flight["latest_start_period"])
                break

        if new_earliest > flight["earliest_start_period"]:
            flight["earliest_start_period"] = new_earliest

    return data


# =============================================================================
# FEASIBLE START TIME TUPLES
# =============================================================================

def generate_feasible_tuples(flight, data):
    """
    Generate all feasible start time tuples tau = (s_h, s_d) for a flight.

    A tuple is feasible iff:
      - s_h in [S^es_i, S^ls_i]
      - s_h <= s_d
      - s_d < S^e_i - Delta (depletion must finish before end)

    Returns:
        list of (s_h, s_d) tuples
    """
    S_es = flight["earliest_start_period"]
    S_ls = flight["latest_start_period"]
    S_e = flight["end_period"]
    Delta = data["depletion_offset"]

    tuples = []
    s_d_max = S_e - Delta - 1
    for s_h in range(S_es, S_ls + 1):
        for s_d in range(s_h, s_d_max + 1):
            tuples.append((s_h, s_d))

    return tuples


# =============================================================================
# DERIVED PARAMETERS: Gamma and Phi (Online Appendix B)
# =============================================================================

def compute_gamma(flight, w, tau, data):
    """
    Compute Gamma^{i,w}_{t,tau}: workload on carousel during each period.

    The workload is the number of bags on the conveyor belt at the end of
    each period t, given:
      - w working stations (loading rate = w * working_rate bags/period)
      - start time tuple tau = (s_h, s_d)

    Returns:
        dict mapping period t -> workload value (only non-zero entries)
    """
    s_h, s_d = tau
    S_e = flight["end_period"]
    profile = flight["arrival_profile"]
    earliest = flight["earliest_start_period"]
    working_rate = data["working_rate"]
    depletion_rate = data["depletion_rate"]

    gamma = {}

    # Compute stored bags accumulated before handling starts
    stored_bags = 0
    for j in range(len(profile)):
        arr_period = earliest + j
        if arr_period < s_h:
            stored_bags += profile[j]
        else:
            break  # profile starts at earliest, periods are contiguous

    # Simulate period by period from s_h to S_e - 1
    bags_on_belt = 0
    for t in range(s_h, S_e):
        # New arrivals directly to carousel
        arr_idx = t - earliest
        if 0 <= arr_idx < len(profile):
            bags_on_belt += profile[arr_idx]

        # Storage depletion
        if t >= s_d and stored_bags > 0:
            depleted = min(stored_bags, depletion_rate)
            bags_on_belt += depleted
            stored_bags -= depleted

        # Workers load bags into containers
        loaded = min(bags_on_belt, w * working_rate)
        bags_on_belt -= loaded

        if bags_on_belt > 0:
            gamma[t] = bags_on_belt

    return gamma


def compute_phi(flight, tau, data):
    """
    Compute Phi^{i}_{t,tau}: amount of flight i's baggage in central
    storage during each period t.

    Returns:
        dict mapping period t -> storage occupancy value (only non-zero entries)
    """
    s_h, s_d = tau
    profile = flight["arrival_profile"]
    earliest = flight["earliest_start_period"]
    S_e = flight["end_period"]
    depletion_rate = data["depletion_rate"]

    phi = {}
    stored_bags = 0

    # Only iterate over the relevant range of periods
    # Bags accumulate from earliest to s_h-1, then deplete from s_d onward
    first_period = earliest
    # Storage is empty once fully depleted; worst case is a bit after S_e
    last_period = S_e + 20  # generous upper bound
    last_period = min(last_period, data["T"])

    for t in range(first_period, last_period):
        # Arrivals that go to storage (only before handling starts)
        arr_idx = t - earliest
        if 0 <= arr_idx < len(profile) and t < s_h:
            stored_bags += profile[arr_idx]

        # Depletion from storage to carousel
        if t >= s_d and stored_bags > 0:
            depleted = min(stored_bags, depletion_rate)
            stored_bags -= depleted

        if stored_bags > 0:
            phi[t] = stored_bags
        elif t > s_d and stored_bags == 0:
            break  # Once depleted, storage stays at 0

    return phi


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_model(data, time_limit=3600):
    """
    Build the Gurobi MIP model for the TIF formulation.

    Uses index structures to efficiently generate constraints, and introduces
    auxiliary z_{i,tau} variables to reduce the size of constraint (5).

    Returns:
        model: gurobipy.Model object
        x_vars: dict mapping (i, c, w, tau_idx) -> gurobi variable
        y_vars: dict mapping (c, k, m) -> gurobi variable
        flight_tuples: dict mapping flight_index -> list of (s_h, s_d)
        flight_w_range: dict mapping (flight_index, type_str) -> (Wmin, Wmax)
    """
    T = data["T"]
    flights = data["flights"]
    carousels = data["carousels"]
    num_flights = data["num_flights"]
    num_carousels = data["num_carousels"]
    K = data["K"]
    M = data["M"]
    u_ta = data["u_ta"]
    threshold_values = data["threshold_values"]
    penalties = data["penalties"]
    peak_intervals = data["peak_intervals"]
    Cr = data["Cr"]
    R = data["R"]
    carousel_types = data["carousel_types"]

    # =========================================================================
    # STEP 1: Precompute feasible tuples for each flight
    # =========================================================================
    print("Precomputing feasible start time tuples...")
    flight_tuples = {}
    for i, flight in enumerate(flights):
        flight_tuples[i] = generate_feasible_tuples(flight, data)
        if len(flight_tuples[i]) == 0:
            print(f"  WARNING: Flight {flight['flight_id']} has no feasible "
                  f"start time tuples!")

    total_tuples = sum(len(v) for v in flight_tuples.values())
    print(f"  Total tuples across all flights: {total_tuples}")

    # =========================================================================
    # STEP 2: Precompute working station ranges per flight per carousel type
    # =========================================================================
    flight_w_range = {}  # (flight_idx, type_str) -> (Wmin, Wmax)
    for i, flight in enumerate(flights):
        ws_bounds = flight["working_station_bounds_by_carousel_type"]
        for type_str in R:
            if type_str in ws_bounds:
                wmin = ws_bounds[type_str]["Wmin"]
                wmax = ws_bounds[type_str]["Wmax"]
                flight_w_range[(i, type_str)] = (wmin, wmax)

    # =========================================================================
    # STEP 3: Precompute Gamma and Phi
    # =========================================================================
    print("Precomputing Gamma (workload) and Phi (storage) parameters...")
    gamma_vals = {}   # (i, w, tau_idx) -> {t: workload}
    phi_vals = {}     # (i, tau_idx) -> {t: storage}
    total_combos = 0

    for i, flight in enumerate(flights):
        tuples = flight_tuples[i]

        # Determine all w values needed for this flight
        w_values = set()
        for type_str in R:
            if (i, type_str) in flight_w_range:
                wmin, wmax = flight_w_range[(i, type_str)]
                for w in range(wmin, wmax + 1):
                    w_values.add(w)

        for tau_idx, tau in enumerate(tuples):
            phi_vals[(i, tau_idx)] = compute_phi(flight, tau, data)
            for w in w_values:
                gamma_vals[(i, w, tau_idx)] = compute_gamma(
                    flight, w, tau, data
                )
                total_combos += 1

    print(f"  Total (flight, w, tau) combinations: {total_combos}")

    # =========================================================================
    # STEP 5: Build index structures for efficient constraint generation
    # =========================================================================
    print("Building index structures...")

    # For each period t, which flights could be active (S_es_i <= t < S_e_i)?
    flights_active_at = defaultdict(list)  # t -> list of flight indices
    for i, flight in enumerate(flights):
        S_es = flight["earliest_start_period"]
        S_e = flight["end_period"]
        for t in range(S_es, S_e):
            flights_active_at[t].append(i)

    # For each (flight, tau_idx), which periods have s_h_tau <= t?
    # We store tuples sorted by s_h so we can binary-search later,
    # but for simplicity we just iterate.

    # =========================================================================
    # STEP 6: Create Gurobi model
    # =========================================================================
    print("Building Gurobi model...")
    model = gp.Model("OBHP_TIF")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    # Setting NumericFocus due to large penalty coefficients
    model.setParam("NumericFocus", 1)

    # ---- Decision variables x_{i,c,w,tau_idx} ----
    x_vars = {}
    num_x = 0
    for i in range(num_flights):
        tuples = flight_tuples[i]
        num_tau = len(tuples)
        for c_idx in range(num_carousels):
            type_str = str(carousels[c_idx]["type_r"])
            if (i, type_str) not in flight_w_range:
                continue
            wmin, wmax = flight_w_range[(i, type_str)]
            for w in range(wmin, wmax + 1):
                for tau_idx in range(num_tau):
                    x_vars[(i, c_idx, w, tau_idx)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{i}_{c_idx}_{w}_{tau_idx}"
                    )
                    num_x += 1

    # ---- Decision variables y_{c,k,m} ----
    y_vars = {}
    num_y = num_carousels * K * M
    for c_idx in range(num_carousels):
        for k in range(K):
            for m_idx in range(M):
                y_vars[(c_idx, k, m_idx)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_{c_idx}_{k}_{m_idx}"
                )

    # ---- Auxiliary variables z_{i,tau_idx} for constraint (5) ----
    # z_{i,tau_idx} = sum_c sum_w x_{i,c,w,tau_idx}
    # This aggregates over carousels and working stations, since Phi
    # does not depend on c or w. This reduces constraint (5) from
    # O(F * C * W * tau) nonzeros per period to O(F * tau) nonzeros.
    z_vars = {}
    for i in range(num_flights):
        for tau_idx in range(len(flight_tuples[i])):
            z_vars[(i, tau_idx)] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0,
                name=f"z_{i}_{tau_idx}"
            )

    model.update()
    print(f"  Variables: {num_x} x-vars, {num_y} y-vars, "
          f"{len(z_vars)} z-vars (auxiliary)")

    # =========================================================================
    # OBJECTIVE (1)
    # =========================================================================
    obj_expr = gp.LinExpr()
    for c_idx in range(num_carousels):
        for k in range(K):
            for m_idx in range(M):
                obj_expr.addTerms(penalties[k], y_vars[(c_idx, k, m_idx)])
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # =========================================================================
    # CONSTRAINT (2): Each flight assigned exactly once
    # sum_c sum_w sum_tau x_{i,c,w,tau} = 1 for all i
    # =========================================================================
    print("  Adding constraint (2): flight assignment...")
    for i in range(num_flights):
        tuples = flight_tuples[i]
        lhs = gp.LinExpr()
        for c_idx in range(num_carousels):
            type_str = str(carousels[c_idx]["type_r"])
            if (i, type_str) not in flight_w_range:
                continue
            wmin, wmax = flight_w_range[(i, type_str)]
            for w in range(wmin, wmax + 1):
                for tau_idx in range(len(tuples)):
                    lhs.addTerms(1.0, x_vars[(i, c_idx, w, tau_idx)])
        model.addConstr(lhs == 1, name=f"assign_{i}")

    # =========================================================================
    # LINKING CONSTRAINTS for z variables
    # z_{i,tau_idx} = sum_c sum_w x_{i,c,w,tau_idx}
    # =========================================================================
    print("  Adding z-variable linking constraints...")
    for i in range(num_flights):
        tuples = flight_tuples[i]
        for tau_idx in range(len(tuples)):
            lhs = gp.LinExpr()
            lhs.addTerms(-1.0, z_vars[(i, tau_idx)])
            for c_idx in range(num_carousels):
                type_str = str(carousels[c_idx]["type_r"])
                if (i, type_str) not in flight_w_range:
                    continue
                wmin, wmax = flight_w_range[(i, type_str)]
                for w in range(wmin, wmax + 1):
                    lhs.addTerms(1.0, x_vars[(i, c_idx, w, tau_idx)])
            model.addConstr(lhs == 0, name=f"link_z_{i}_{tau_idx}")

    # =========================================================================
    # CONSTRAINT (3): Working station capacity
    # sum_{applicable i,w,tau: s_h<=t} w * x_{i,c,w,tau} <= K^ws_r
    # for all r, c in C_r, t
    # =========================================================================
    print("  Adding constraint (3): working station capacity...")
    num_ws_constrs = 0
    for type_str in R:
        Kws_r = carousel_types[type_str]["Kws"]
        c_indices = Cr[type_str]

        for t in range(T):
            # Collect the terms that apply for this type and period
            # (shared across all carousels of this type)
            terms = []  # list of (i, w, tau_idx, coeff=w)
            for i in flights_active_at.get(t, []):
                if (i, type_str) not in flight_w_range:
                    continue
                wmin, wmax = flight_w_range[(i, type_str)]
                for w in range(wmin, wmax + 1):
                    for tau_idx, (s_h, _) in enumerate(flight_tuples[i]):
                        if s_h <= t:
                            terms.append((i, w, tau_idx))

            if not terms:
                continue

            # Add one constraint per carousel of this type
            for c_idx in c_indices:
                lhs = gp.LinExpr()
                for (i, w, tau_idx) in terms:
                    lhs.addTerms(float(w),
                                 x_vars[(i, c_idx, w, tau_idx)])
                model.addConstr(lhs <= Kws_r,
                                name=f"ws_{type_str}_{c_idx}_{t}")
                num_ws_constrs += 1
    print(f"    {num_ws_constrs} constraints added")

    # =========================================================================
    # CONSTRAINT (4): Parking position capacity
    # sum_{applicable i,w,tau: s_h<=t} P_i * x_{i,c,w,tau} <= K^pp_r
    # for all r, c in C_r, t
    # =========================================================================
    print("  Adding constraint (4): parking position capacity...")
    num_pp_constrs = 0
    for type_str in R:
        Kpp_r = carousel_types[type_str]["Kpp"]
        c_indices = Cr[type_str]

        for t in range(T):
            terms = []  # list of (i, w, tau_idx, P_i)
            for i in flights_active_at.get(t, []):
                if (i, type_str) not in flight_w_range:
                    continue
                P_i = flights[i]["num_containers"]
                wmin, wmax = flight_w_range[(i, type_str)]
                for w in range(wmin, wmax + 1):
                    for tau_idx, (s_h, _) in enumerate(flight_tuples[i]):
                        if s_h <= t:
                            terms.append((i, w, tau_idx, P_i))

            if not terms:
                continue

            for c_idx in c_indices:
                lhs = gp.LinExpr()
                for (i, w, tau_idx, P_i) in terms:
                    lhs.addTerms(float(P_i),
                                 x_vars[(i, c_idx, w, tau_idx)])
                model.addConstr(lhs <= Kpp_r,
                                name=f"pp_{type_str}_{c_idx}_{t}")
                num_pp_constrs += 1
    print(f"    {num_pp_constrs} constraints added")

    # =========================================================================
    # CONSTRAINT (5): Central storage capacity (using z-variables)
    # sum_i sum_tau Phi^i_{t,tau} * z_{i,tau} <= K^s_t  for all t
    # =========================================================================
    print("  Adding constraint (5): central storage capacity...")
    num_stor_constrs = 0
    for t in range(T):
        lhs = gp.LinExpr()
        has_terms = False
        for i in range(num_flights):
            for tau_idx in range(len(flight_tuples[i])):
                phi_t = phi_vals[(i, tau_idx)].get(t, 0)
                if phi_t > 0:
                    lhs.addTerms(float(phi_t), z_vars[(i, tau_idx)])
                    has_terms = True
        if has_terms:
            model.addConstr(lhs <= data["Ks"], name=f"stor_{t}")
            num_stor_constrs += 1
    print(f"    {num_stor_constrs} constraints added")

    # =========================================================================
    # CONSTRAINT (6): Utilization threshold
    # (1/K^cb_r) * sum_{applicable i,w,tau} Gamma * x_{i,c,w,tau}
    #   - sum_k u_k * y_{c,k,m} <= u^ta
    # for all r, c in C_r, m, t in T_m
    # =========================================================================
    print("  Adding constraint (6): utilization threshold...")
    num_util_constrs = 0
    for type_str in R:
        Kcb_r = carousel_types[type_str]["Kcb"]
        inv_Kcb = 1.0 / Kcb_r
        c_indices = Cr[type_str]

        for m_idx, peak in enumerate(peak_intervals):
            t_start = peak["start_period"]
            t_end = peak["end_period"]

            for t in range(t_start, t_end + 1):
                # Collect gamma terms for this type and period
                terms = []  # list of (i, w, tau_idx, gamma_coeff)
                for i in flights_active_at.get(t, []):
                    if (i, type_str) not in flight_w_range:
                        continue
                    wmin, wmax = flight_w_range[(i, type_str)]
                    for w in range(wmin, wmax + 1):
                        for tau_idx in range(len(flight_tuples[i])):
                            gamma_t = gamma_vals[(i, w, tau_idx)].get(t, 0)
                            if gamma_t > 0:
                                terms.append(
                                    (i, w, tau_idx, gamma_t * inv_Kcb)
                                )

                # Build y-term (same for all carousels of this type)
                # This is subtracted: - sum_k u_k * y_{c,k,m}

                for c_idx in c_indices:
                    lhs = gp.LinExpr()
                    for (i, w, tau_idx, coeff) in terms:
                        lhs.addTerms(coeff,
                                     x_vars[(i, c_idx, w, tau_idx)])
                    for k in range(K):
                        lhs.addTerms(-threshold_values[k],
                                     y_vars[(c_idx, k, m_idx)])
                    model.addConstr(lhs <= u_ta,
                                    name=f"util_{type_str}_{c_idx}_{m_idx}_{t}")
                    num_util_constrs += 1
    print(f"    {num_util_constrs} constraints added")

    # =========================================================================
    # CONSTRAINT (7): At most one threshold level per carousel per peak interval
    # sum_k y_{c,k,m} <= 1  for all c, m
    # =========================================================================
    print("  Adding constraint (7): threshold uniqueness...")
    for c_idx in range(num_carousels):
        for m_idx in range(M):
            lhs = gp.LinExpr()
            for k in range(K):
                lhs.addTerms(1.0, y_vars[(c_idx, k, m_idx)])
            model.addConstr(lhs <= 1, name=f"thresh_{c_idx}_{m_idx}")

    model.update()
    print(f"\nModel summary:")
    print(f"  Total variables: {model.NumVars}")
    print(f"  Total constraints: {model.NumConstrs}")
    print(f"  Total nonzeros: {model.NumNZs}")

    return model, x_vars, y_vars, flight_tuples, flight_w_range


# =============================================================================
# SOLUTION EXTRACTION
# =============================================================================

def extract_solution(model, x_vars, y_vars, data, flight_tuples, flight_w_range):
    """
    Extract the solution from a solved Gurobi model.

    Returns a dict with:
      - objective_value
      - status / status_name
      - flight assignments (flight -> carousel, w, start tuple)
      - utilization violations
      - solve statistics
    """
    flights = data["flights"]
    carousels = data["carousels"]
    R = data["R"]

    solution = {
        "solver": "Gurobi",
        "status": model.Status,
        "status_name": _status_name(model.Status),
    }

    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        if model.SolCount > 0:
            solution["objective_value"] = model.ObjVal
            solution["best_bound"] = model.ObjBound
            solution["mip_gap"] = model.MIPGap
            solution["solve_time_seconds"] = model.Runtime
            solution["num_solutions_found"] = model.SolCount

            # Extract flight assignments
            assignments = []
            for i, flight in enumerate(flights):
                tuples = flight_tuples[i]
                assigned = False
                for c_idx in range(len(carousels)):
                    type_str = str(carousels[c_idx]["type_r"])
                    if (i, type_str) not in flight_w_range:
                        continue
                    wmin, wmax = flight_w_range[(i, type_str)]
                    for w in range(wmin, wmax + 1):
                        for tau_idx in range(len(tuples)):
                            key = (i, c_idx, w, tau_idx)
                            if key in x_vars and x_vars[key].X > 0.5:
                                s_h, s_d = tuples[tau_idx]
                                assignments.append({
                                    "flight_id": flight["flight_id"],
                                    "carousel_id": carousels[c_idx][
                                        "carousel_id"],
                                    "carousel_type": carousels[c_idx][
                                        "type_r"],
                                    "working_stations": w,
                                    "handling_start_period": s_h,
                                    "depletion_start_period": s_d,
                                    "end_period": flight["end_period"],
                                })
                                assigned = True
                                break
                        if assigned:
                            break
                    if assigned:
                        break
                if not assigned:
                    assignments.append({
                        "flight_id": flight["flight_id"],
                        "carousel_id": None,
                        "note": "NOT ASSIGNED"
                    })

            solution["assignments"] = assignments

            # Extract utilization violations
            violations = []
            K = data["K"]
            M_count = data["M"]
            tv = data["threshold_values"]
            for c_idx in range(data["num_carousels"]):
                for m_idx in range(M_count):
                    for k in range(K):
                        key = (c_idx, k, m_idx)
                        if key in y_vars and y_vars[key].X > 0.5:
                            violations.append({
                                "carousel_id": carousels[c_idx][
                                    "carousel_id"],
                                "peak_interval": m_idx + 1,
                                "threshold_level": k + 1,
                                "threshold_value": tv[k],
                            })
            solution["utilization_violations"] = violations

        else:
            solution["objective_value"] = None
            solution["note"] = ("No feasible solution found within "
                                "time limit")
    elif model.Status == GRB.INFEASIBLE:
        solution["objective_value"] = None
        solution["note"] = "Model is infeasible"
        try:
            model.computeIIS()
            iis_constrs = [
                c.ConstrName for c in model.getConstrs() if c.IISConstr
            ]
            solution["iis_constraints"] = iis_constrs[:20]
        except Exception:
            pass
    else:
        solution["objective_value"] = None
        solution["note"] = f"Solver ended with status {model.Status}"

    solution["solve_time_seconds"] = model.Runtime
    return solution


def _status_name(status):
    """Map Gurobi status code to human-readable name."""
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.LOADED: "LOADED",
    }
    return names.get(status, f"UNKNOWN_{status}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=("Solve OBHP using Time-Indexed Formulation (TIF) "
                     "with Gurobi")
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the instance JSON file"
    )
    parser.add_argument(
        "--solution_path", type=str, default=None,
        help="Path to write the solution JSON file"
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600,
        help="Gurobi time limit in seconds (default: 3600)"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ---- Load instance ----
    print("=" * 70)
    print("OBHP - Time-Indexed Formulation (TIF) Solver")
    print("Frey, Kolisch, and Artigues (2017)")
    print("=" * 70)

    print(f"\nLoading instance from: {args.instance_path}")
    inst = load_instance(args.instance_path)
    data = parse_instance(inst)

    print(f"  Flights: {data['num_flights']}")
    print(f"  Carousels: {data['num_carousels']}")
    print(f"  Carousel types: {data['R']}")
    print(f"  Planning horizon: {data['T']} periods")
    print(f"  Peak intervals: {data['M']}")
    print(f"  Threshold levels (K): {data['K']}")
    print(f"  Target utilization: {data['u_ta']}")

    # ---- Preprocessing ----
    print("\nPreprocessing...")
    data = preprocess_tighten_start_times(data)

    # ---- Build and solve model ----
    print(f"\nTime limit: {args.time_limit} seconds")
    wall_start = time.time()

    model, x_vars, y_vars, flight_tuples, flight_w_range = build_model(
        data, time_limit=args.time_limit
    )

    print("\nSolving...")
    model.optimize()

    wall_elapsed = time.time() - wall_start
    print(f"\nTotal wall time: {wall_elapsed:.2f} seconds")
    print(f"  Gurobi solve time: {model.Runtime:.2f} seconds")
    print(f"  Status: {_status_name(model.Status)}")

    # ---- Extract solution ----
    solution = extract_solution(
        model, x_vars, y_vars, data, flight_tuples, flight_w_range
    )

    if solution["objective_value"] is not None:
        print(f"  Objective value: {solution['objective_value']}")
        if "mip_gap" in solution:
            print(f"  MIP gap: {solution['mip_gap']:.6f}")
        if "assignments" in solution:
            print(f"  Flights assigned: {len(solution['assignments'])}")
        if "utilization_violations" in solution:
            n_viol = len(solution["utilization_violations"])
            print(f"  Utilization violations: {n_viol}")
    else:
        print(f"  No solution found. Note: {solution.get('note', 'N/A')}")

    # ---- Save solution ----
    if args.solution_path:
        print(f"\nSaving solution to: {args.solution_path}")
        with open(args.solution_path, "w", encoding="utf-8") as f:
            solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
            json.dump(solution, f, indent=2, default=str)
    else:
        print("\n" + json.dumps(solution, indent=2, default=str))

    return solution


if __name__ == "__main__":
    main()
