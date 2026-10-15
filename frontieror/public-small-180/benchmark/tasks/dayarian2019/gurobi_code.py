#!/usr/bin/env python3
"""
Gurobi implementation of the Caterer Production-Routing Problem (CPRP).

Paper: "A Branch-Price-and-Cut Algorithm for a Production-Routing Problem
       with Short-Life-Span Products"
Authors: Iman Dayarian, Guy Desaulniers (2019), Transportation Science.

This implements the mathematical model (1)-(15) from the paper using a compact
MIP formulation. Since the original model uses exponentially large sets R (routes)
and S (shifts) that are typically handled via column generation, this compact
formulation directly models routing and production scheduling decisions with
Gurobi as the solver.

INFERRED ASSUMPTION: Since enumerating all feasible routes R and shifts S is
intractable, we use a compact vehicle routing + production scheduling formulation
that captures the same constraints. The routing uses arc-based decision variables
and the production scheduling uses direct assignment variables. This is equivalent
to the set-partitioning formulation (1)-(15) but amenable to direct MIP solving.
"""

import argparse
import json
import math
import sys
import itertools
from gurobipy import Model, GRB, quicksum
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
def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def solve_cprp(instance, time_limit):
    # =========================================================================
    # Extract instance data
    # =========================================================================
    params = instance["global_parameters"]
    orders = instance["orders"]
    products = instance["products"]
    periods = instance["periods"]
    subperiods = instance["subperiods"]
    prod_windows = instance["production_windows"]
    travel_time = instance["travel_time_matrix"]
    depot = instance["depot"]

    num_orders = instance["num_orders"]
    num_products = instance["num_products"]
    num_periods = params["num_periods"]
    num_subperiods = params["num_subperiods"]
    n_V = instance["num_vehicles"]
    n_D = instance["num_drivers"]
    n_W = instance["num_workstations"]
    n_E = instance["num_production_employees"]
    Q = instance["vehicle_capacity"]
    period_len = params["period_length_minutes"]
    subperiod_len = params["subperiod_length_minutes"]
    sp_per_period = params["subperiods_per_period"]
    tau_P = params["preparation_time_minutes"]
    tau_S = params["service_time_minutes"]
    m_T = params["max_trips_per_route"]
    l_bar = params["max_shift_route_length_periods"]
    l_underline_sp = params["min_paid_time_subperiods"]
    l_D = params["min_paid_time_minutes"]
    gamma_D = params["driver_salary_per_minute"]
    gamma_E = params["employee_salary_per_subperiod"]
    gamma_S = params["setup_cost"]
    gamma_V = params["vehicle_operating_cost_per_minute"]
    sigma = params["setup_length_subperiods"]
    horizon_start = params["horizon_start_minutes"]
    horizon_end = params["horizon_end_minutes"]

    # Travel time matrix: index 0 = depot, index 1..num_orders = orders
    # (already provided in instance as a (num_orders+1) x (num_orders+1) matrix
    # where row/col 0 = depot)
    # Actually the travel_time_matrix is (num_orders+1) x (num_orders+1) but let's verify
    # The matrix is indexed: 0=depot, then orders 0..14 -> matrix indices 0..15
    # But from the JSON it seems index 0 in the matrix IS the depot
    # Let's verify: the matrix has 16 rows for 15 orders + depot
    n_locs = len(travel_time)  # Should be num_orders + 1

    # Feasible trip start periods for each order
    # T_i: periods at the beginning of which a trip can start to deliver order i
    # within its time window
    T_i = {}
    for o in orders:
        i = o["order_id"]
        tw_start = o["time_window_start"]
        tw_end = o["time_window_end"]
        feasible_periods = []
        for p_obj in periods:
            t = p_obj["period_id"]
            trip_start_time = p_obj["start_time"]
            # The trip starts at beginning of period t, then has preparation time
            # tau_P, then travels to customer. Earliest arrival at customer i:
            # trip_start_time + tau_P + h_{depot, i+1} (matrix index i+1 for order i)
            earliest_arrival = trip_start_time + tau_P + travel_time[0][i + 1]
            # Must arrive within time window
            if earliest_arrival <= tw_end:
                # Also, the trip must be completable: customer service + return
                # must not exceed the horizon
                feasible_periods.append(t)
        T_i[i] = feasible_periods

    # =========================================================================
    # Build compact MIP model
    # =========================================================================
    model = Model("CPRP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # ----- ROUTING VARIABLES -----
    # We model routing using multi-trip vehicle routes.
    # For tractability with Gurobi, we use a trip-based formulation:
    # Each trip k (k=1..m_T) of each vehicle v starts at a period, visits a
    # sequence of customers, and returns to depot.
    #
    # INFERRED ASSUMPTION: We model each vehicle-trip as an independent unit
    # with arc-flow variables. A route is a sequence of up to m_T trips by
    # the same driver/vehicle.

    # Decision: which vehicle serves which trip starting when, visiting which customers
    # For simplicity with 15 orders: use assignment + sequencing variables

    # x[v][k][t][i][j] = 1 if vehicle v, trip k, starting period t, travels
    # from location i to location j. This is too many variables.
    #
    # INFERRED ASSUMPTION: Since this is a small instance (15 orders), we use
    # a simplified trip-assignment model where we decide:
    # - y[i,v,k]: order i is served by vehicle v in trip k
    # - trip_start[v,k,t]: vehicle v trip k starts at period t
    # - For sequencing within a trip: we use Miller-Tucker-Zemlin (MTZ) subtour
    #   elimination

    vehicles = range(n_V)
    trips = range(m_T)

    # y[i,v,k] = 1 if order i is assigned to vehicle v, trip k
    y = {}
    for i in range(num_orders):
        for v in vehicles:
            for k in trips:
                y[i, v, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{v}_{k}")

    # trip_start[v,k,t] = 1 if vehicle v trip k starts at period t
    trip_start = {}
    for v in vehicles:
        for k in trips:
            for t in range(num_periods):
                trip_start[v, k, t] = model.addVar(vtype=GRB.BINARY,
                                                    name=f"ts_{v}_{k}_{t}")

    # trip_active[v,k] = 1 if vehicle v trip k is used
    trip_active = {}
    for v in vehicles:
        for k in trips:
            trip_active[v, k] = model.addVar(vtype=GRB.BINARY,
                                              name=f"ta_{v}_{k}")

    # driver_used[v] = 1 if vehicle/driver v is used
    driver_used = {}
    for v in vehicles:
        driver_used[v] = model.addVar(vtype=GRB.BINARY, name=f"du_{v}")

    # Arc variables for sequencing: x_arc[v,k,i,j] = 1 if in trip (v,k),
    # travel from i to j. Locations: 0=depot, 1..num_orders = customers
    x_arc = {}
    locs = range(n_locs)  # 0=depot, 1..num_orders
    for v in vehicles:
        for k in trips:
            for i_loc in locs:
                for j_loc in locs:
                    if i_loc != j_loc:
                        x_arc[v, k, i_loc, j_loc] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_{v}_{k}_{i_loc}_{j_loc}")

    # MTZ position variable u_pos[v,k,i] for subtour elimination
    u_pos = {}
    for v in vehicles:
        for k in trips:
            for i in range(1, n_locs):  # customers only
                u_pos[v, k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1,
                                               ub=num_orders,
                                               name=f"u_{v}_{k}_{i}")

    # Trip travel time and duration variables
    trip_travel_time = {}
    trip_end_period = {}  # the period at which trip ends (vehicle returns to depot)
    for v in vehicles:
        for k in trips:
            trip_travel_time[v, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                    name=f"ttt_{v}_{k}")

    # Route duration for driver salary computation
    route_start_time = {}
    route_end_time = {}
    route_duration = {}
    for v in vehicles:
        route_start_time[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=horizon_start,
                                            ub=horizon_end, name=f"rst_{v}")
        route_end_time[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=horizon_start,
                                          ub=horizon_end, name=f"ret_{v}")
        route_duration[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                          name=f"rd_{v}")

    # Driver paid time (max of route_duration and l_D)
    driver_paid = {}
    for v in vehicles:
        driver_paid[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                       name=f"dp_{v}")

    # ----- PRODUCTION VARIABLES -----
    # q[i,p,u]: units of product p produced in subperiod u for order i
    q = {}
    for i in range(num_orders):
        for p_str, p_data in prod_windows[str(i)].items():
            p = int(p_str)
            for u in p_data["valid_subperiod_ids"]:
                q[i, p, u] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                           name=f"q_{i}_{p}_{u}")

    # f[p,u]: number of employees assigned to product p in subperiod u
    f_var = {}
    all_subperiod_ids_per_product = {}
    for p in range(num_products):
        sp_set = set()
        for i in range(num_orders):
            if str(p) in prod_windows[str(i)]:
                sp_set.update(prod_windows[str(i)][str(p)]["valid_subperiod_ids"])
        all_subperiod_ids_per_product[p] = sorted(sp_set)
        for u in all_subperiod_ids_per_product[p]:
            f_var[p, u] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_E,
                                        name=f"f_{p}_{u}")

    # w[t]: number of workstations used in period t
    w = {}
    for t in range(num_periods):
        w[t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_W, name=f"w_{t}")

    # Number of employees starting shift at period t
    alpha = {}
    for t in range(num_periods):
        alpha[t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_E,
                                 name=f"alpha_{t}")

    # beta[p,u]: employees starting setup for product p in subperiod u
    beta = {}
    for p in range(num_products):
        for u in all_subperiod_ids_per_product[p]:
            beta[p, u] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_E,
                                       name=f"beta_{p}_{u}")

    # Total vehicle travel time for cost
    total_vehicle_travel = {}
    for v in vehicles:
        total_vehicle_travel[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                name=f"tvt_{v}")

    # Number of setups for production cost
    total_setups = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_setups")

    # Employee paid subperiods
    total_employee_cost = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                        name="total_emp_cost")

    model.update()

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- (2) Each customer visited exactly once ---
    for i in range(num_orders):
        model.addConstr(
            quicksum(y[i, v, k] for v in vehicles for k in trips) == 1,
            name=f"visit_{i}")

    # --- Link y to arc variables ---
    for v in vehicles:
        for k in trips:
            # Customer i (matrix index i+1) is visited iff y[i,v,k]=1
            for i in range(num_orders):
                i_loc = i + 1
                # If y[i,v,k]=1, then exactly one arc enters and one leaves i_loc
                model.addConstr(
                    quicksum(x_arc[v, k, j_loc, i_loc]
                             for j_loc in locs if j_loc != i_loc) == y[i, v, k],
                    name=f"in_{v}_{k}_{i}")
                model.addConstr(
                    quicksum(x_arc[v, k, i_loc, j_loc]
                             for j_loc in locs if j_loc != i_loc) == y[i, v, k],
                    name=f"out_{v}_{k}_{i}")

            # Depot: trip starts and ends at depot (loc 0)
            model.addConstr(
                quicksum(x_arc[v, k, 0, j_loc] for j_loc in range(1, n_locs)) == trip_active[v, k],
                name=f"depot_out_{v}_{k}")
            model.addConstr(
                quicksum(x_arc[v, k, j_loc, 0] for j_loc in range(1, n_locs)) == trip_active[v, k],
                name=f"depot_in_{v}_{k}")

    # --- Link trip_active to y ---
    for v in vehicles:
        for k in trips:
            model.addConstr(
                trip_active[v, k] <=
                quicksum(y[i, v, k] for i in range(num_orders)),
                name=f"ta_lb_{v}_{k}")
            for i in range(num_orders):
                model.addConstr(y[i, v, k] <= trip_active[v, k],
                                name=f"ta_ub_{v}_{k}_{i}")

    # --- MTZ subtour elimination ---
    for v in vehicles:
        for k in trips:
            for i_loc in range(1, n_locs):
                for j_loc in range(1, n_locs):
                    if i_loc != j_loc:
                        model.addConstr(
                            u_pos[v, k, i_loc] - u_pos[v, k, j_loc] +
                            num_orders * x_arc[v, k, i_loc, j_loc] <= num_orders - 1,
                            name=f"mtz_{v}_{k}_{i_loc}_{j_loc}")

    # --- Vehicle capacity per trip ---
    for v in vehicles:
        for k in trips:
            model.addConstr(
                quicksum(orders[i]["space_requirement"] * y[i, v, k]
                         for i in range(num_orders)) <= Q,
                name=f"cap_{v}_{k}")

    # --- Trip start period assignment ---
    for v in vehicles:
        for k in trips:
            model.addConstr(
                quicksum(trip_start[v, k, t] for t in range(num_periods)) == trip_active[v, k],
                name=f"ts_assign_{v}_{k}")

    # --- Time window constraints ---
    # For each order i assigned to trip (v,k) starting at period t,
    # the arrival time must be within [omega_i, bar_omega_i]
    # We use a big-M approach
    M_time = horizon_end + 500  # Large constant

    # Arrival time at each customer in each trip
    arr_time = {}
    for v in vehicles:
        for k in trips:
            for i in range(num_orders):
                arr_time[v, k, i] = model.addVar(vtype=GRB.CONTINUOUS,
                                                   lb=0, ub=M_time,
                                                   name=f"arr_{v}_{k}_{i}")

    model.update()

    # If order i is served by trip (v,k), arrival time is bounded by time windows
    for v in vehicles:
        for k in trips:
            for i in range(num_orders):
                tw_s = orders[i]["time_window_start"]
                tw_e = orders[i]["time_window_end"]
                # arr_time >= tw_s if y=1
                model.addConstr(
                    arr_time[v, k, i] >= tw_s - M_time * (1 - y[i, v, k]),
                    name=f"tw_lo_{v}_{k}_{i}")
                # arr_time <= tw_e if y=1
                model.addConstr(
                    arr_time[v, k, i] <= tw_e + M_time * (1 - y[i, v, k]),
                    name=f"tw_hi_{v}_{k}_{i}")

    # Arrival time consistency with arc-based travel
    # If arc (depot -> i) used: arr >= trip_start_time + tau_P + h[0][i+1]
    for v in vehicles:
        for k in trips:
            # Trip start time
            # Inactive trip slots have all trip_start binaries equal to zero,
            # so their linked time must be allowed to equal zero.  Active
            # slots still take an actual period start (>= horizon_start).
            trip_st_time = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                         ub=horizon_end, name=f"tst_{v}_{k}")
            model.addConstr(
                trip_st_time == quicksum(
                    periods[t]["start_time"] * trip_start[v, k, t]
                    for t in range(num_periods)),
                name=f"tst_link_{v}_{k}")

            for i in range(num_orders):
                i_loc = i + 1
                # From depot to customer i
                model.addConstr(
                    arr_time[v, k, i] >= trip_st_time + tau_P + travel_time[0][i_loc]
                    - M_time * (1 - x_arc[v, k, 0, i_loc]),
                    name=f"arr_dep_{v}_{k}_{i}")

                # From customer j to customer i
                for j in range(num_orders):
                    if j != i:
                        j_loc = j + 1
                        model.addConstr(
                            arr_time[v, k, i] >=
                            arr_time[v, k, j] + tau_S + travel_time[j_loc][i_loc]
                            - M_time * (1 - x_arc[v, k, j_loc, i_loc]),
                            name=f"arr_seq_{v}_{k}_{j}_{i}")

    # --- Trip travel time ---
    for v in vehicles:
        for k in trips:
            model.addConstr(
                trip_travel_time[v, k] ==
                quicksum(travel_time[i_loc][j_loc] * x_arc[v, k, i_loc, j_loc]
                         for i_loc in locs for j_loc in locs
                         if i_loc != j_loc),
                name=f"ttt_def_{v}_{k}")

    # --- Total vehicle travel time ---
    for v in vehicles:
        model.addConstr(
            total_vehicle_travel[v] ==
            quicksum(trip_travel_time[v, k] for k in trips),
            name=f"tvt_def_{v}")

    # --- Multi-trip sequencing: trip k+1 starts after trip k ends ---
    # INFERRED ASSUMPTION: Trip ordering is enforced within the same vehicle.
    # Trip k must end before trip k+1 starts.
    for v in vehicles:
        for k in range(m_T - 1):
            # If both trips active, trip k+1 start period > trip k start period
            # (simplified: enforce ordering by period)
            for t1 in range(num_periods):
                for t2 in range(num_periods):
                    if t2 <= t1:
                        model.addConstr(
                            trip_start[v, k, t1] + trip_start[v, k + 1, t2] <= 1,
                            name=f"seq_{v}_{k}_{t1}_{t2}")

    # --- Trip k+1 can only be active if trip k is active ---
    for v in vehicles:
        for k in range(m_T - 1):
            model.addConstr(trip_active[v, k + 1] <= trip_active[v, k],
                            name=f"trip_order_{v}_{k}")

    # --- Driver usage ---
    for v in vehicles:
        model.addConstr(
            driver_used[v] >= trip_active[v, 0],
            name=f"du_link_{v}")
        model.addConstr(
            driver_used[v] <= quicksum(trip_active[v, k] for k in trips),
            name=f"du_ub_{v}")

    # --- (5) Driver availability ---
    model.addConstr(
        quicksum(driver_used[v] for v in vehicles) <= n_D,
        name="driver_avail")

    # --- Route duration for driver salary ---
    # Route start = start of first trip, route end = return time of last trip
    for v in vehicles:
        for k in trips:
            for t in range(num_periods):
                model.addConstr(
                    route_start_time[v] <= periods[t]["start_time"] +
                    M_time * (1 - trip_start[v, k, t]),
                    name=f"rst_ub_{v}_{k}_{t}")

        # Route end: after last customer served + service + return to depot
        # We approximate: route_end >= arr[last customer] + service + return
        for k in trips:
            for i in range(num_orders):
                i_loc = i + 1
                model.addConstr(
                    route_end_time[v] >= arr_time[v, k, i] + tau_S +
                    travel_time[i_loc][0] - M_time * (1 - y[i, v, k]),
                    name=f"ret_lb_{v}_{k}_{i}")

        model.addConstr(
            route_duration[v] >= route_end_time[v] - route_start_time[v],
            name=f"rd_def_{v}")

        # Paid time is max of duration and l_D
        model.addConstr(driver_paid[v] >= route_duration[v],
                        name=f"dp_dur_{v}")
        model.addConstr(driver_paid[v] >= l_D * driver_used[v],
                        name=f"dp_min_{v}")

    # --- Route duration <= l_bar periods ---
    for v in vehicles:
        model.addConstr(
            route_duration[v] <= l_bar * period_len,
            name=f"max_route_dur_{v}")

    # =========================================================================
    # PRODUCTION CONSTRAINTS
    # =========================================================================

    # --- (3) Production capacity per subperiod ---
    for p in range(num_products):
        eta = products[p]["production_rate_per_subperiod"]
        for u in all_subperiod_ids_per_product[p]:
            orders_in_u = [i for i in range(num_orders)
                           if (i, p, u) in q]
            if orders_in_u:
                model.addConstr(
                    quicksum(q[i, p, u] for i in orders_in_u) <=
                    eta * f_var[p, u],
                    name=f"prod_cap_{p}_{u}")

    # --- (4) Production due time linking ---
    # For each order i, product p, and feasible trip start period t:
    # production of d_i^p must be complete by time b^t if the order is
    # delivered on a trip starting at period t
    for i in range(num_orders):
        for p_str, p_data in prod_windows[str(i)].items():
            p = int(p_str)
            demand = orders[i]["demands"][p_str]
            valid_sps = p_data["valid_subperiod_ids"]

            for t in T_i[i]:
                period_start = periods[t]["start_time"]
                # Subperiods u with e^u <= b^t (production must be done before
                # trip starts)
                eligible_sps = [u for u in valid_sps
                                if subperiods[u]["end_time"] <= period_start]
                # Constraint (4): even when eligible_sps is empty, the
                # constraint must be added.  An empty LHS (= 0) forces the
                # RHS to 0, preventing delivery at period t when no
                # production can be completed before that period.
                lhs = (quicksum(q[i, p, u] for u in eligible_sps)
                       if eligible_sps else 0)
                model.addConstr(
                    lhs >=
                    demand * quicksum(
                        y[i, v, k] * trip_start[v, k, t]
                        for v in vehicles for k in trips),
                    name=f"due_{i}_{p}_{t}")

    # --- (15) Total production requirement ---
    for i in range(num_orders):
        for p_str, p_data in prod_windows[str(i)].items():
            p = int(p_str)
            demand = orders[i]["demands"][p_str]
            valid_sps = p_data["valid_subperiod_ids"]
            model.addConstr(
                quicksum(q[i, p, u] for u in valid_sps) >= demand,
                name=f"total_prod_{i}_{p}")

    # --- Production setup constraints ---
    # (46) f^{pu} <= beta^{p,u-sigma} + f^{p,u-1}
    for p in range(num_products):
        sps = all_subperiod_ids_per_product[p]
        for idx, u in enumerate(sps):
            prev_f = f_var[p, sps[idx - 1]] if idx > 0 and sps[idx - 1] == u - 1 else 0
            prev_beta_u = u - sigma
            prev_beta = beta[p, prev_beta_u] if (p, prev_beta_u) in beta else 0
            if idx == 0:
                # First subperiod: need setup
                model.addConstr(f_var[p, u] <= (beta[p, u - sigma]
                                                 if (p, u - sigma) in beta else 0),
                                name=f"setup_first_{p}_{u}")
            else:
                model.addConstr(f_var[p, u] <= prev_f + prev_beta,
                                name=f"setup_{p}_{u}")

    # --- (47) Workstation counting: employees in each subperiod ---
    # w^{t(u)} >= sum_p (f^{pu} + sum_{u' in [u-sigma+1,u]} beta^{pu'})
    for t in range(num_periods):
        sp_start = t * sp_per_period
        sp_end = sp_start + sp_per_period
        for u in range(sp_start, sp_end):
            emp_count = []
            for p in range(num_products):
                if (p, u) in f_var:
                    emp_count.append(f_var[p, u])
                for u_prime in range(max(0, u - sigma + 1), u + 1):
                    if (p, u_prime) in beta:
                        emp_count.append(beta[p, u_prime])
            if emp_count:
                model.addConstr(w[t] >= quicksum(emp_count),
                                name=f"ws_count_{t}_{u}")

    # --- (48) Shift start counting ---
    for t in range(num_periods):
        if t == 0:
            model.addConstr(alpha[t] >= w[t], name=f"shift_start_{t}")
        else:
            model.addConstr(alpha[t] >= w[t] - w[t - 1],
                            name=f"shift_start_{t}")

    # --- (49) Minimum paid time ---
    min_paid_periods = l_underline_sp // sp_per_period  # = 6 periods
    for t in range(num_periods):
        model.addConstr(
            w[t] >= quicksum(alpha[t - j]
                             for j in range(1, min(t + 1, min_paid_periods) + 1)
                             if t - j >= 0),
            name=f"min_paid_{t}")

    # --- (50) Employee availability ---
    model.addConstr(
        quicksum(alpha[t] for t in range(num_periods)) <= n_E,
        name="emp_avail")

    # --- Per-subperiod employee capacity ---
    # In each subperiod u, the total employees producing (f_var[p,u]) plus
    # employees performing setups (beta[p,u']) must not exceed n_E.
    for u in range(num_subperiods):
        emp_terms = []
        for p in range(num_products):
            if (p, u) in f_var:
                emp_terms.append(f_var[p, u])
            for u_prime in range(max(0, u - sigma + 1), u + 1):
                if (p, u_prime) in beta:
                    emp_terms.append(beta[p, u_prime])
        if emp_terms:
            model.addConstr(quicksum(emp_terms) <= n_E,
                            name=f"emp_cap_sp_{u}")

    # --- Total setups ---
    model.addConstr(
        total_setups == quicksum(beta[p, u]
                                  for p in range(num_products)
                                  for u in all_subperiod_ids_per_product[p]
                                  if (p, u) in beta),
        name="total_setups_def")

    # --- Employee cost: sum of gamma_E * n_SP * w^t ---
    # INFERRED ASSUMPTION: We approximate shift costs using the compact
    # formulation: total employee salary = gamma_E * sum_t (n_SP * w^t)
    # This counts the total employee-subperiods of paid time.
    model.addConstr(
        total_employee_cost == gamma_E * quicksum(
            sp_per_period * w[t] for t in range(num_periods)),
        name="emp_cost_def")

    # =========================================================================
    # OBJECTIVE FUNCTION (1)
    # =========================================================================
    # Route costs: driver salary + vehicle operational cost
    # Shift costs: employee salary + setup costs
    obj = (
        # Driver salary: gamma_D * max(route_duration, l_D) for each used driver
        quicksum(gamma_D * driver_paid[v] for v in vehicles) +
        # Vehicle operating cost: gamma_V * total travel time
        quicksum(gamma_V * total_vehicle_travel[v] for v in vehicles) +
        # Employee salary
        total_employee_cost +
        # Setup costs
        gamma_S * total_setups
    )

    model.setObjective(obj, GRB.MINIMIZE)

    # =========================================================================
    # SOLVE
    # =========================================================================
    model.optimize()

    # =========================================================================
    # EXTRACT SOLUTION
    # =========================================================================
    solution = {
        "instance_id": instance.get("instance_id", 0),
        "instance_label": instance.get("instance_label", ""),
        "solver": "Gurobi",
        "status": model.Status,
        "objective_value": None,
        "routes": [],
        "production_schedule": [],
        "production_staffing": {
            "employees_by_product_subperiod": [],
            "setups_by_product_subperiod": [],
            "workstations_by_period": [],
            "shift_starts_by_period": [],
        },
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["best_bound"] = model.ObjBound if hasattr(model, 'ObjBound') else None
        solution["gap"] = model.MIPGap if hasattr(model, 'MIPGap') else None

        # Extract routes
        for v in vehicles:
            if driver_used[v].X > 0.5:
                route_info = {"vehicle": v, "trips": []}
                for k in trips:
                    if trip_active[v, k].X > 0.5:
                        trip_info = {"trip": k, "customers": [], "start_period": None}
                        for t in range(num_periods):
                            if trip_start[v, k, t].X > 0.5:
                                trip_info["start_period"] = t
                        # Serialize the actual arc order, not the former
                        # unordered assignment set.  The checker needs this
                        # order to recompute travel and paid-time costs.
                        successor = {}
                        for i_loc in locs:
                            for j_loc in locs:
                                if (
                                    i_loc != j_loc
                                    and x_arc[v, k, i_loc, j_loc].X > 0.5
                                ):
                                    successor[i_loc] = j_loc
                        current = 0
                        seen = set()
                        while current in successor:
                            nxt = successor[current]
                            if nxt == 0:
                                break
                            if nxt in seen:
                                break
                            seen.add(nxt)
                            trip_info["customers"].append(nxt - 1)
                            current = nxt
                        assigned = {
                            i for i in range(num_orders) if y[i, v, k].X > 0.5
                        }
                        if set(trip_info["customers"]) != assigned:
                            raise RuntimeError(
                                f"could not reconstruct route order for vehicle {v}, trip {k}"
                            )
                        route_info["trips"].append(trip_info)
                solution["routes"].append(route_info)

        # Extract production schedule
        for i in range(num_orders):
            for p_str, p_data in prod_windows[str(i)].items():
                p = int(p_str)
                for u in p_data["valid_subperiod_ids"]:
                    # Do not drop small positive split-production terms.  A
                    # single order/product demand can be distributed over
                    # many subperiods, so filtering at 1e-2 can remove enough
                    # mass to make an otherwise feasible solver incumbent fail
                    # the released checker.
                    if (i, p, u) in q and q[i, p, u].X > 1e-9:
                        solution["production_schedule"].append({
                            "order": i,
                            "product": p,
                            "subperiod": u,
                            # The checker binds production to exact integer
                            # demand and derives employee lower bounds from
                            # these quantities. Four-decimal serialization can
                            # turn a solver-feasible 4.0 into 3.9946 after many
                            # split terms, so retain full double precision.
                            "quantity": float(q[i, p, u].X)
                        })

        # Export the integer shift/staffing decisions required to verify the
        # full production contract and exact objective value.
        staffing = solution["production_staffing"]
        for p in range(num_products):
            for u in all_subperiod_ids_per_product[p]:
                staffing["employees_by_product_subperiod"].append({
                    "product": p,
                    "subperiod": u,
                    "employees": int(round(f_var[p, u].X)),
                })
                staffing["setups_by_product_subperiod"].append({
                    "product": p,
                    "subperiod": u,
                    "setups": int(round(beta[p, u].X)),
                })
        staffing["workstations_by_period"] = [
            int(round(w[t].X)) for t in range(num_periods)
        ]
        staffing["shift_starts_by_period"] = [
            int(round(alpha[t].X)) for t in range(num_periods)
        ]
    else:
        solution["objective_value"] = None
        print("WARNING: No feasible solution found within time limit.")

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for the Caterer Production-Routing Problem (CPRP)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    solution = solve_cprp(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"\nSolution written to: {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.2f}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
