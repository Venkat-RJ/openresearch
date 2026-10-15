"""
Gurobi implementation of the Linearized MIP for the Attractive Orienteering Problem (AtOP).

Paper: Freeman, Keskin, Çapar (2018) - "Attractive Orienteering Problem with
       Proximity and Timing Interactions"
European Journal of Operational Research.

This implements the linearized MIP formulation (Section 3.2) using Gurobi as the solver.
The MINLP constraint (11) is replaced by constraints (18)-(22) via McCormick linearization.
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
def load_instance(instance_path):
    """Load the problem instance from a JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def compute_parameters(data):
    """
    Compute derived parameters: s_it, a_kt, alpha_ikt from instance data.

    Formulas from Section 5.1 of the paper:
      s_it   = t_tilde * N_i / avg_N_i
      a_kt   = t_tilde * N_k / avg_N_k
      alpha_ikt = a_kt / (d_p_ik / 1000)^q
    """
    params = data['parameters']
    T = data['num_periods']
    K = data['num_sites']
    I = data['num_customers']
    q = params['q']
    weekly_cycle = params['weekly_attraction_cycle']
    avg_bus_cust = params['avg_businesses_customers']
    avg_bus_sites = params['avg_businesses_sites']

    sites = {s['id']: s for s in data['sites']}
    customers = {cu['id']: cu for cu in data['customers']}

    # Day-of-week attraction (1-indexed periods, cycling through the week)
    t_tilde = {}
    for t in range(1, T + 1):
        t_tilde[t] = weekly_cycle[(t - 1) % 7]

    # Self-attraction s_{it}
    s_it = {}
    for i in range(1, I + 1):
        N_i = customers[i]['num_businesses']
        for t in range(1, T + 1):
            s_it[i, t] = t_tilde[t] * N_i / avg_bus_cust

    # Site attraction a_{kt}
    a_kt = {}
    for k in range(1, K + 1):
        N_k = sites[k]['num_businesses']
        for t in range(1, T + 1):
            a_kt[k, t] = t_tilde[t] * N_k / avg_bus_sites

    # Customer-to-site attraction alpha_{ikt}
    # Parse customer-to-site distances
    d_p = {}
    for key, val in data['customer_to_site_distances'].items():
        parts = key.split('_')
        ci, sk = int(parts[0]), int(parts[1])
        d_p[ci, sk] = val

    alpha = {}
    for i in range(1, I + 1):
        for k in range(1, K + 1):
            dist = d_p[i, k]
            dist_scaled = dist / 1000.0
            if dist_scaled < 1e-10:
                dist_scaled = 1e-10  # Avoid division by zero
            denom = dist_scaled ** q
            for t in range(1, T + 1):
                alpha[i, k, t] = a_kt[k, t] / denom

    return sites, customers, d_p, s_it, a_kt, alpha, t_tilde


def build_and_solve(data, args):
    """Build the linearized MIP model and solve with Gurobi."""
    K = data['num_sites']
    I = data['num_customers']
    T = data['num_periods']
    params = data['parameters']
    delta = params['delta']
    epsilon = params['epsilon']
    L_max = params['L']
    c = params['c']

    sites_dict, customers_dict, d_p, s_it, a_kt, alpha, t_tilde = compute_parameters(data)

    # Site parameters
    theta = {s['id']: s['capacity'] for s in data['sites']}
    r = {s['id']: s['revenue_per_customer'] for s in data['sites']}
    f = {s['id']: s['fixed_cost'] for s in data['sites']}
    m = {s['id']: s['max_events'] for s in data['sites']}
    gamma = {cu['id']: cu['population'] for cu in data['customers']}

    # Parse site-to-site distances (includes dummy site 0)
    d_s = {}
    for key, val in data['site_to_site_distances'].items():
        parts = key.split('_')
        j_id, k_id = int(parts[0]), int(parts[1])
        d_s[j_id, k_id] = val

    site_ids = list(range(1, K + 1))
    all_sites_with_dummy = [0] + site_ids

    # =========================================================================
    # Build Gurobi Model
    # =========================================================================
    model = gp.Model("AtOP_LinearizedMIP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision Variables ---
    # X[j,k,t] in {0,1}: tour moves from site j to site k in period t
    X = {}
    # Period 1: moves from dummy site 0 to any real site
    for k in site_ids:
        X[0, k, 1] = model.addVar(vtype=GRB.BINARY, name=f"X_0_{k}_1")
    # Periods 2..T: moves between real sites
    for t in range(2, T + 1):
        for j in site_ids:
            for k in site_ids:
                X[j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"X_{j}_{k}_{t}")

    # Y[k,t] in {0,1}: event at site k in period t
    Y = {}
    for k in site_ids:
        for t in range(1, T + 1):
            Y[k, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{k}_{t}")

    # P[i,k,t]: expected proportion (auxiliary continuous)
    P = {}
    for i in range(1, I + 1):
        for k in site_ids:
            for t in range(1, T + 1):
                P[i, k, t] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                           name=f"P_{i}_{k}_{t}")

    # A[k,t]: expected attendance (auxiliary continuous)
    A = {}
    for k in site_ids:
        for t in range(1, T + 1):
            A[k, t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"A_{k}_{t}")

    # Z[i,j,k,t,tau]: linearization variables for McCormick relaxation
    Z = {}
    for i in range(1, I + 1):
        for k in site_ids:
            for t in range(1, T + 1):
                tau_lo = max(t - epsilon, 1)
                tau_hi = min(t + epsilon, T)
                for j in site_ids:
                    for tau in range(tau_lo, tau_hi + 1):
                        Z[i, j, k, t, tau] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"Z_{i}_{j}_{k}_{t}_{tau}")

    model.update()

    # --- Objective Function ---
    # Maximize: sum_k sum_t r_k * A_kt - sum_k sum_t f_k * Y_kt
    #           - c * sum_j sum_k sum_t d_s_jk * X_jkt
    obj = gp.LinExpr()
    for k in site_ids:
        for t in range(1, T + 1):
            obj += r[k] * A[k, t]
            obj -= f[k] * Y[k, t]

    # Travel cost for period 1 (from dummy site 0)
    for k in site_ids:
        obj -= c * d_s.get((0, k), 0.0) * X[0, k, 1]
    # Travel cost for periods 2..T
    for t in range(2, T + 1):
        for j in site_ids:
            for k in site_ids:
                if (j, k, t) in X:
                    obj -= c * d_s.get((j, k), 0.0) * X[j, k, t]

    model.setObjective(obj, GRB.MAXIMIZE)

    # --- Constraints ---

    # (2) Tour starts from dummy site 0 in period 1
    model.addConstr(
        gp.quicksum(X[0, k, 1] for k in site_ids) == 1, "C2_start")

    # (3) Exactly one move per period t = 2,...,T
    for t in range(2, T + 1):
        model.addConstr(
            gp.quicksum(X[j, k, t] for j in site_ids for k in site_ids
                        if (j, k, t) in X) == 1,
            f"C3_one_visit_{t}")

    # (4) Flow conservation between period 1 and period 2
    if T >= 2:
        for k in site_ids:
            model.addConstr(
                X[0, k, 1] == gp.quicksum(
                    X[k, j, 2] for j in site_ids if (k, j, 2) in X),
                f"C4_flow_1_2_{k}")

    # (5) Flow conservation for t = 2,...,T-1
    for t in range(2, T):
        for k in site_ids:
            model.addConstr(
                gp.quicksum(X[j, k, t] for j in site_ids if (j, k, t) in X) ==
                gp.quicksum(X[k, j, t + 1] for j in site_ids if (k, j, t + 1) in X),
                f"C5_flow_{t}_{k}")

    # (6) Maximum travel distance per period
    # Period 1
    model.addConstr(
        gp.quicksum(d_s.get((0, k), 0.0) * X[0, k, 1] for k in site_ids) <= L_max,
        "C6_dist_1")
    # Periods 2..T
    for t in range(2, T + 1):
        model.addConstr(
            gp.quicksum(d_s.get((j, k), 0.0) * X[j, k, t]
                        for j in site_ids for k in site_ids
                        if (j, k, t) in X) <= L_max,
            f"C6_dist_{t}")

    # (7) Events only at destination in period 1
    for k in site_ids:
        model.addConstr(Y[k, 1] <= X[0, k, 1], f"C7_event_dest_1_{k}")

    # (8) Events only at destination in periods 2..T
    for k in site_ids:
        for t in range(2, T + 1):
            model.addConstr(
                Y[k, t] <= gp.quicksum(
                    X[j, k, t] for j in site_ids if (j, k, t) in X),
                f"C8_event_dest_{t}_{k}")

    # (9) Maximum number of events at each site
    for k in site_ids:
        model.addConstr(
            gp.quicksum(Y[k, t] for t in range(1, T + 1)) <= m[k],
            f"C9_max_events_{k}")

    # (10) Maximum consecutive events limit
    # As formulated: sum_{k in K} sum_{t'=t}^{t+delta+1} Y_{kt'} <= delta
    # for t = 1,...,T-delta-1
    for t in range(1, T - delta):
        upper = t + delta + 1
        if upper > T:
            upper = T
        model.addConstr(
            gp.quicksum(Y[k, tp] for k in site_ids
                        for tp in range(t, upper + 1)) <= delta,
            f"C10_consec_{t}")

    # (12) Attendance from proportions
    for k in site_ids:
        for t in range(1, T + 1):
            model.addConstr(
                A[k, t] <= gp.quicksum(
                    P[i, k, t] * gamma[i] for i in range(1, I + 1)),
                f"C12_attend_{k}_{t}")

    # (13) Venue capacity constraint
    for k in site_ids:
        for t in range(1, T + 1):
            model.addConstr(A[k, t] <= theta[k], f"C13_cap_{k}_{t}")

    # --- Linearized constraints (18)-(22) replacing (11) ---

    # (18) P_{ikt} * s_{it} = alpha_{ikt} * Y_{kt} - sum_j sum_tau Z_{ijkt,tau}
    for i in range(1, I + 1):
        for k in site_ids:
            for t in range(1, T + 1):
                tau_lo = max(t - epsilon, 1)
                tau_hi = min(t + epsilon, T)
                z_sum = gp.quicksum(
                    Z[i, j, k, t, tau]
                    for j in site_ids
                    for tau in range(tau_lo, tau_hi + 1)
                    if (i, j, k, t, tau) in Z
                )
                model.addConstr(
                    s_it[i, t] * P[i, k, t] == alpha[i, k, t] * Y[k, t] - z_sum,
                    f"C18_{i}_{k}_{t}")

    # (19) Z_{ijkt,tau} <= P_{ikt} * alpha_{ij,tau}
    for i in range(1, I + 1):
        for k in site_ids:
            for t in range(1, T + 1):
                tau_lo = max(t - epsilon, 1)
                tau_hi = min(t + epsilon, T)
                for j in site_ids:
                    for tau in range(tau_lo, tau_hi + 1):
                        if (i, j, k, t, tau) in Z:
                            model.addConstr(
                                Z[i, j, k, t, tau] <= alpha[i, j, tau] * P[i, k, t],
                                f"C19_{i}_{j}_{k}_{t}_{tau}")

    # (20) Z_{ijkt,tau} <= alpha_{ij,tau} * Y_{j,tau}
    for i in range(1, I + 1):
        for k in site_ids:
            for t in range(1, T + 1):
                tau_lo = max(t - epsilon, 1)
                tau_hi = min(t + epsilon, T)
                for j in site_ids:
                    for tau in range(tau_lo, tau_hi + 1):
                        if (i, j, k, t, tau) in Z:
                            model.addConstr(
                                Z[i, j, k, t, tau] <= alpha[i, j, tau] * Y[j, tau],
                                f"C20_{i}_{j}_{k}_{t}_{tau}")

    # (21) Z_{ijkt,tau} >= P_{ikt} * alpha_{ij,tau} - alpha_{ij,tau} * (1 - Y_{j,tau})
    for i in range(1, I + 1):
        for k in site_ids:
            for t in range(1, T + 1):
                tau_lo = max(t - epsilon, 1)
                tau_hi = min(t + epsilon, T)
                for j in site_ids:
                    for tau in range(tau_lo, tau_hi + 1):
                        if (i, j, k, t, tau) in Z:
                            model.addConstr(
                                Z[i, j, k, t, tau] >=
                                alpha[i, j, tau] * P[i, k, t]
                                - alpha[i, j, tau] * (1 - Y[j, tau]),
                                f"C21_{i}_{j}_{k}_{t}_{tau}")

    # (22) Z >= 0 is already handled by lb=0.0 in variable definition.

    # =========================================================================
    # Solve
    # =========================================================================
    model.optimize()

    # =========================================================================
    # Extract solution
    # =========================================================================
    solution = {}
    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal

        # Extract tour: which site is visited each period
        tour = {}
        for t in range(1, T + 1):
            for k in site_ids:
                if t == 1:
                    if X[0, k, 1].X > 0.5:
                        tour[t] = k
                        break
                else:
                    for j in site_ids:
                        if (j, k, t) in X and X[j, k, t].X > 0.5:
                            tour[t] = k
                            break
                    if t in tour:
                        break

        # Extract event plan
        events = {}
        for t in range(1, T + 1):
            for k in site_ids:
                if Y[k, t].X > 0.5:
                    events[t] = k
                    break

        solution['tour'] = {str(t): tour.get(t) for t in range(1, T + 1)}
        solution['events'] = {str(t): (1 if t in events else 0)
                              for t in range(1, T + 1)}
        solution['event_sites'] = {str(t): events.get(t) for t in range(1, T + 1)}

        # Attendance
        attendance = {}
        for k in site_ids:
            for t in range(1, T + 1):
                val = A[k, t].X
                if val > 1e-6:
                    attendance[f"{k}_{t}"] = val
        solution['attendance'] = attendance

        # Travel details
        travel_cost = 0.0
        for k in site_ids:
            if (0, k, 1) in X and X[0, k, 1].X > 0.5:
                travel_cost += c * d_s.get((0, k), 0.0)
        for t in range(2, T + 1):
            for j in site_ids:
                for k in site_ids:
                    if (j, k, t) in X and X[j, k, t].X > 0.5:
                        travel_cost += c * d_s.get((j, k), 0.0)
        solution['total_travel_cost'] = travel_cost

    else:
        solution['objective_value'] = None

    solution['solver_status'] = model.Status
    if model.SolCount > 0:
        solution['mip_gap'] = model.MIPGap
    else:
        solution['mip_gap'] = None
    solution['num_solutions_found'] = model.SolCount

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve the AtOP using Gurobi (Linearized MIP formulation)")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds.')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    solution = build_and_solve(data, args)

    with open(args.solution_path, 'w') as fout:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, fout, indent=2)

    if solution['objective_value'] is not None:
        print(f"Best objective value: {solution['objective_value']:.2f}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == '__main__':
    main()
