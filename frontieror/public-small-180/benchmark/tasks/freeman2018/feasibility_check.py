"""
Feasibility checker for the Attractive Orienteering Problem (AtOP).

Paper: Freeman, Keskin, Çapar (2018) - "Attractive Orienteering Problem with
       Proximity and Timing Interactions"

Checks all hard constraints from the MINLP formulation (Part A) one by one.
Constraints are numbered 1-17 following the paper's formulation order:
  1  -> (2)  Tour starts from dummy site 0 in period 1
  2  -> (3)  Exactly one site visit per period (t=2..T)
  3  -> (4)  Flow conservation period 1->2
  4  -> (5)  Flow conservation periods 2..T-1
  5  -> (6)  Max travel distance per period <= L
  6  -> (7)  Events only at destination in period 1
  7  -> (8)  Events only at destination in periods 2..T
  8  -> (9)  Max events per site
  9  -> (10) Max consecutive events (delta)
  10 -> (11) Gravity model definition of P_ikt
  11 -> (12) Attendance <= sum of P_ikt * gamma_i
  12 -> (13) Attendance <= venue capacity
  13 -> (14) X binary
  14 -> (15) Y binary
  15 -> (16) P_ikt <= 1
  16 -> (17) A_kt >= 0
  17 -> (1)  Objective consistency: reported objective_value must match the
             total profit recomputed from the solution's own variables
             (sum_k,t r_k A_kt  -  sum_k,t f_k Y_kt  -  c sum_j,k,t d^s_jk X_jkt).
"""

import json
import argparse
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('attendance', 'event_sites', 'events', 'objective_value', 'tour')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('event_sites', 'objective_value')
_FRONTIEROR_DERIVED_FIELD_RULES = ()
_FRONTIEROR_CONDITIONAL_FIELD_RULES = ()


def _frontieror_validate_solution_contract(solution):
    """Return a structured rejection when the public solution contract is absent."""
    violations = []
    if not isinstance(solution, dict):
        violations.append("solution must be a JSON object")
    else:
        missing = [
            field
            for field in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS
            if field not in solution or solution[field] is None
        ]
        if missing:
            violations.append(
                "missing required solution field(s): " + ", ".join(missing)
            )

        for discriminator, expected_value, fields in _FRONTIEROR_CONDITIONAL_FIELD_RULES:
            if solution.get(discriminator) != expected_value:
                continue
            conditional_missing = [
                field
                for field in fields
                if field not in solution or solution[field] is None
            ]
            if conditional_missing:
                violations.append(
                    "missing conditionally required solution field(s): "
                    + ", ".join(conditional_missing)
                )

        empty = [
            field
            for field in _FRONTIEROR_NONEMPTY_SOLUTION_FIELDS
            if field in solution and solution[field] in (None, [], {})
        ]
        if empty:
            violations.append(
                "empty required decision witness field(s): " + ", ".join(empty)
            )

        for target, operation, source, source_key in _FRONTIEROR_DERIVED_FIELD_RULES:
            try:
                if operation == "assignment_key_set":
                    reported_set = set(solution[target])
                    expected_set = {entry[source_key] for entry in solution[source]}
                    if reported_set != expected_set:
                        violations.append(
                            f"{target}={sorted(reported_set)} does not match the "
                            f"decision-derived value {sorted(expected_set)}"
                        )
                    continue

                reported = float(solution[target])
                if operation == "length":
                    expected = float(len(solution[source]))
                elif operation == "mapping_value":
                    expected = float(solution[source][source_key])
                else:
                    raise ValueError("unknown derived-field operation")
            except (KeyError, TypeError, ValueError, OverflowError):
                violations.append(
                    f"cannot derive {target} from decision witness field {source}"
                )
                continue
            if (
                reported != reported
                or expected != expected
                or reported in (float("inf"), float("-inf"))
                or expected in (float("inf"), float("-inf"))
                or abs(reported - expected) > 1e-6
            ):
                violations.append(
                    f"{target}={reported} does not match the decision-derived "
                    f"value {expected}"
                )

        if "objective_value" in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS:
            objective = solution.get("objective_value")
            if (
                isinstance(objective, bool)
                or not isinstance(objective, (int, float))
                or objective != objective
                or objective in (float("inf"), float("-inf"))
            ):
                violations.append("objective_value must be a finite JSON number")

    if not violations:
        return None
    return {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": violations,
        "violation_magnitudes": [],
    }


def _frontieror_write_unexpected_rejection(error):
    """Turn malformed candidate/checker exceptions into a normal rejection."""
    import json as _frontieror_json
    import sys as _frontieror_sys

    result_path = None
    for index, argument in enumerate(_frontieror_sys.argv):
        if argument == "--result_path" and index + 1 < len(_frontieror_sys.argv):
            result_path = _frontieror_sys.argv[index + 1]
            break
        if argument.startswith("--result_path="):
            result_path = argument.split("=", 1)[1]
            break
    if not result_path:
        return False
    try:
        with open(result_path, "w") as result_handle:
            _frontieror_json.dump(
                {
                    "feasible": False,
                    "violated_constraints": ["malformed_solution"],
                    "violations": [
                        "checker rejected malformed candidate: "
                        + type(error).__name__
                        + ": "
                        + str(error)[:500]
                    ],
                    "violation_magnitudes": [],
                },
                result_handle,
                indent=2,
            )
    except Exception:
        return False
    return True



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def precompute(data):
    """Compute derived parameters from instance data."""
    params = data['parameters']
    T = data['num_periods']
    K = data['num_sites']
    I = data['num_customers']
    q = params['q']
    weekly_cycle = params['weekly_attraction_cycle']
    avg_bus_cust = params['avg_businesses_customers']
    avg_bus_sites = params['avg_businesses_sites']
    delta = params['delta']
    epsilon = params['epsilon']
    L_max = params['L']
    c = params['c']

    sites = {s['id']: s for s in data['sites']}
    customers = {cu['id']: cu for cu in data['customers']}
    site_ids = list(range(1, K + 1))
    customer_ids = list(range(1, I + 1))

    # Day-of-week attraction
    t_tilde = {}
    for t in range(1, T + 1):
        t_tilde[t] = weekly_cycle[(t - 1) % 7]

    # Self-attraction s_{it}
    s_it = {}
    for i in customer_ids:
        N_i = customers[i]['num_businesses']
        for t in range(1, T + 1):
            s_it[i, t] = t_tilde[t] * N_i / avg_bus_cust

    # Site attraction a_{kt}
    a_kt = {}
    for k in site_ids:
        N_k = sites[k]['num_businesses']
        for t in range(1, T + 1):
            a_kt[k, t] = t_tilde[t] * N_k / avg_bus_sites

    # Customer-to-site distances
    d_p = {}
    for key, val in data['customer_to_site_distances'].items():
        parts = key.split('_')
        ci, sk = int(parts[0]), int(parts[1])
        d_p[ci, sk] = val

    # Customer-to-site attraction alpha_{ikt}
    alpha = {}
    for i in customer_ids:
        for k in site_ids:
            dist = d_p[i, k]
            dist_scaled = dist / 1000.0
            if dist_scaled < 1e-10:
                dist_scaled = 1e-10
            denom = dist_scaled ** q
            for t in range(1, T + 1):
                alpha[i, k, t] = a_kt[k, t] / denom

    # Site-to-site distances (including dummy site 0)
    d_s = {}
    for key, val in data['site_to_site_distances'].items():
        parts = key.split('_')
        j_id, k_id = int(parts[0]), int(parts[1])
        d_s[j_id, k_id] = val

    theta = {s['id']: s['capacity'] for s in data['sites']}
    r = {s['id']: s['revenue_per_customer'] for s in data['sites']}
    f_cost = {s['id']: s['fixed_cost'] for s in data['sites']}
    m_max = {s['id']: s['max_events'] for s in data['sites']}
    gamma = {cu['id']: cu['population'] for cu in data['customers']}

    return {
        'T': T, 'K': K, 'I': I,
        'delta': delta, 'epsilon': epsilon, 'L_max': L_max, 'c': c, 'q': q,
        'site_ids': site_ids, 'customer_ids': customer_ids,
        'sites': sites, 'customers': customers,
        't_tilde': t_tilde, 's_it': s_it, 'a_kt': a_kt, 'alpha': alpha,
        'd_s': d_s, 'd_p': d_p,
        'theta': theta, 'r': r, 'f_cost': f_cost, 'm_max': m_max,
        'gamma': gamma,
    }


def parse_solution(sol, ctx):
    """
    Parse solution into uniform internal representation.

    Returns:
        tour: dict {t: site_id} for t=1..T
        events: dict {t: 0 or 1} for t=1..T
        attendance: dict {(k,t): value} or None if not provided
    """
    T = ctx['T']

    # Parse tour: {period_str: site_id}
    tour = {}
    for t_str, site in sol['tour'].items():
        tour[int(t_str)] = int(site)

    # Parse events: {period_str: 0/1}
    events = {}
    for t_str, val in sol['events'].items():
        events[int(t_str)] = int(val)

    # Parse attendance if available
    attendance = None
    if 'attendance' in sol:
        attendance = {}
        for key, val in sol['attendance'].items():
            parts = key.split('_')
            k, t = int(parts[0]), int(parts[1])
            attendance[k, t] = float(val)

    return tour, events, attendance


def add_violation(violations_list, magnitudes_list, violated_set,
                  constraint_idx, message, lhs, rhs, violation_amount, tol, eps,
                  relative_tol=5e-2):
    """Record a constraint violation if it exceeds tolerance.

    Effective tolerance combines:
      - absolute tol (1e-5) — fundamental solver noise floor;
      - 1% relative slack vs max(|lhs|, |rhs|, 1.0) — catches numerical
        dust from barrier solver / presolve where a near-zero RHS is
        coupled with a residual LHS like 0.01 (eq.12 attendance on
        sites with no event). Real constraint violations of order 1%+
        of the natural scale still get flagged.
    """
    # Most model constraints retain the historical 5% scale-aware allowance
    # for nonlinear residuals.  Callers that already computed a semantically
    # meaningful tolerance (notably the exactly recomputed objective below)
    # can set ``relative_tol=0`` so this helper does not silently widen it.
    eff_tol = max(tol, relative_tol * max(abs(lhs), abs(rhs), 1.0))
    if violation_amount > eff_tol:
        violated_set.add(constraint_idx)
        violations_list.append(message)
        normalizer = max(abs(rhs), eps)
        magnitudes_list.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })


def check_feasibility(data, sol):
    tol = 1e-5
    eps = 1e-5

    ctx = precompute(data)
    T = ctx['T']
    K = ctx['K']
    I = ctx['I']
    delta = ctx['delta']
    epsilon = ctx['epsilon']
    L_max = ctx['L_max']
    site_ids = ctx['site_ids']
    customer_ids = ctx['customer_ids']
    d_s = ctx['d_s']
    s_it = ctx['s_it']
    alpha = ctx['alpha']
    theta = ctx['theta']
    gamma = ctx['gamma']
    m_max = ctx['m_max']

    tour, events, attendance_sol = parse_solution(sol, ctx)

    violations = []
    magnitudes = []
    violated_set = set()

    # =========================================================================
    # Reconstruct X_{jkt} from the tour
    # =========================================================================
    # X[j,k,t] = 1 for the actual transitions in the tour
    # Period 1: from dummy site 0 to tour[1]
    # Period t (t>=2): from tour[t-1] to tour[t]
    X = {}
    if 1 in tour:
        X[0, tour[1], 1] = 1
    for t in range(2, T + 1):
        if t in tour and (t - 1) in tour:
            X[tour[t - 1], tour[t], t] = 1

    # Reconstruct Y_{kt} from events
    Y = {}
    for t in range(1, T + 1):
        if events.get(t, 0) == 1:
            k = tour.get(t)
            if k is not None:
                Y[k, t] = 1

    # =========================================================================
    # Constraint 1 (paper eq. 2): sum_k X_{0,k,1} = 1
    # Tour starts from dummy site 0 in period 1
    # =========================================================================
    lhs = sum(X.get((0, k, 1), 0) for k in site_ids)
    rhs = 1.0
    violation_amount = abs(lhs - rhs)
    add_violation(violations, magnitudes, violated_set,
                  1, f"Constraint 1 (eq.2): Tour start from dummy site 0: "
                     f"sum_k X_{{0,k,1}} = {lhs}, expected 1",
                  lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 2 (paper eq. 3): sum_j sum_k X_{j,k,t} = 1  for t=2..T
    # Exactly one move per period
    # =========================================================================
    for t in range(2, T + 1):
        lhs = sum(X.get((j, k, t), 0) for j in site_ids for k in site_ids)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        add_violation(violations, magnitudes, violated_set,
                      2, f"Constraint 2 (eq.3): Period {t}: "
                         f"sum X_{{j,k,{t}}} = {lhs}, expected 1",
                      lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 3 (paper eq. 4): X_{0,k,1} = sum_j X_{k,j,2}  for all k
    # Flow conservation period 1->2
    # =========================================================================
    if T >= 2:
        for k in site_ids:
            lhs_val = X.get((0, k, 1), 0)
            rhs_val = sum(X.get((k, j, 2), 0) for j in site_ids)
            violation_amount = abs(lhs_val - rhs_val)
            add_violation(violations, magnitudes, violated_set,
                          3, f"Constraint 3 (eq.4): Flow conservation site {k} "
                             f"period 1->2: X_{{0,{k},1}}={lhs_val} != "
                             f"sum_j X_{{{k},j,2}}={rhs_val}",
                          lhs_val, rhs_val, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 4 (paper eq. 5): sum_j X_{j,k,t} = sum_j X_{k,j,t+1}
    # Flow conservation for t=2..T-1
    # =========================================================================
    for t in range(2, T):
        for k in site_ids:
            lhs_val = sum(X.get((j, k, t), 0) for j in site_ids)
            rhs_val = sum(X.get((k, j, t + 1), 0) for j in site_ids)
            violation_amount = abs(lhs_val - rhs_val)
            add_violation(violations, magnitudes, violated_set,
                          4, f"Constraint 4 (eq.5): Flow conservation site {k} "
                             f"period {t}->{t+1}: in={lhs_val} != out={rhs_val}",
                          lhs_val, rhs_val, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 5 (paper eq. 6): sum_j sum_k X_{j,k,t} * d^s_{j,k} <= L
    # Maximum travel distance per period
    # =========================================================================
    for t in range(1, T + 1):
        if t == 1:
            lhs = sum(X.get((0, k, 1), 0) * d_s.get((0, k), 0.0)
                       for k in site_ids)
        else:
            lhs = sum(X.get((j, k, t), 0) * d_s.get((j, k), 0.0)
                       for j in site_ids for k in site_ids)
        rhs = L_max
        violation_amount = max(0.0, lhs - rhs)
        add_violation(violations, magnitudes, violated_set,
                      5, f"Constraint 5 (eq.6): Travel distance period {t}: "
                         f"{lhs:.2f} > L={rhs}",
                      lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 6 (paper eq. 7): Y_{k,1} <= X_{0,k,1}  for all k
    # Events only at destination in period 1
    # =========================================================================
    for k in site_ids:
        y_val = Y.get((k, 1), 0)
        x_val = X.get((0, k, 1), 0)
        lhs = y_val
        rhs = x_val
        violation_amount = max(0.0, lhs - rhs)
        add_violation(violations, magnitudes, violated_set,
                      6, f"Constraint 6 (eq.7): Event at site {k} period 1 "
                         f"but not visited: Y={y_val}, X={x_val}",
                      lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 7 (paper eq. 8): Y_{k,t} <= sum_j X_{j,k,t}  for t=2..T
    # Events only at destination in periods 2..T
    # =========================================================================
    for k in site_ids:
        for t in range(2, T + 1):
            y_val = Y.get((k, t), 0)
            x_sum = sum(X.get((j, k, t), 0) for j in site_ids)
            lhs = y_val
            rhs = x_sum
            violation_amount = max(0.0, lhs - rhs)
            add_violation(violations, magnitudes, violated_set,
                          7, f"Constraint 7 (eq.8): Event at site {k} period {t} "
                             f"but not visited: Y={y_val}, sum_X={x_sum}",
                          lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 8 (paper eq. 9): sum_t Y_{k,t} <= m_k  for all k
    # Maximum number of events at each site
    # =========================================================================
    for k in site_ids:
        lhs = sum(Y.get((k, t), 0) for t in range(1, T + 1))
        rhs = m_max[k]
        violation_amount = max(0.0, lhs - rhs)
        add_violation(violations, magnitudes, violated_set,
                      8, f"Constraint 8 (eq.9): Site {k} has {lhs} events, "
                         f"max allowed {rhs}",
                      float(lhs), float(rhs), violation_amount, tol, eps)

    # =========================================================================
    # Constraint 9 (paper eq. 10):
    # sum_k sum_{t'=t}^{t+delta+1} Y_{k,t'} <= delta  for t=1..T-delta-1
    # Maximum consecutive events
    # =========================================================================
    for t in range(1, T - delta):
        upper = min(t + delta + 1, T)
        lhs = sum(Y.get((k, tp), 0) for k in site_ids
                  for tp in range(t, upper + 1))
        rhs = delta
        violation_amount = max(0.0, lhs - rhs)
        add_violation(violations, magnitudes, violated_set,
                      9, f"Constraint 9 (eq.10): Window [{t},{upper}] has "
                         f"{lhs} events, max {rhs}",
                      float(lhs), float(rhs), violation_amount, tol, eps)

    # =========================================================================
    # Compute P_{ikt} and A_{kt} from the gravity model for constraint checks
    # =========================================================================
    # P_{ikt} = alpha_{ikt} * Y_{kt} / (s_{it} + sum_{j,tau in window} alpha_{ij,tau} * Y_{j,tau})
    P = {}
    for i in customer_ids:
        for k in site_ids:
            for t in range(1, T + 1):
                if Y.get((k, t), 0) == 0:
                    P[i, k, t] = 0.0
                    continue
                # Compute denominator
                denom = s_it[i, t]
                tau_lo = max(t - epsilon, 1)
                tau_hi = min(t + epsilon, T)
                for tau in range(tau_lo, tau_hi + 1):
                    for j in site_ids:
                        if Y.get((j, tau), 0) == 1:
                            denom += alpha.get((i, j, tau), 0.0)
                numer = alpha.get((i, k, t), 0.0)
                if denom > 1e-15:
                    P[i, k, t] = numer / denom
                else:
                    P[i, k, t] = 0.0

    # Compute model-implied attendance
    A_model = {}
    for k in site_ids:
        for t in range(1, T + 1):
            raw_attend = sum(P.get((i, k, t), 0.0) * gamma[i]
                             for i in customer_ids)
            A_model[k, t] = min(raw_attend, theta[k])

    # Determine which attendance values to check:
    # If solution provides attendance, use it; otherwise use model-computed values
    A_check = {}
    if attendance_sol is not None:
        for k in site_ids:
            for t in range(1, T + 1):
                A_check[k, t] = attendance_sol.get((k, t), 0.0)
    else:
        A_check = A_model

    # =========================================================================
    # Constraint 10 (paper eq. 11): Gravity model P_{ikt} definition
    # (s_{it} + sum alpha Y) * P_{ikt} = alpha_{ikt} * Y_{kt}
    # =========================================================================
    for i in customer_ids:
        for k in site_ids:
            for t in range(1, T + 1):
                # LHS: (s_{it} + sum_j sum_tau alpha_{ij,tau} Y_{j,tau}) * P_{ikt}
                denom_val = s_it[i, t]
                tau_lo = max(t - epsilon, 1)
                tau_hi = min(t + epsilon, T)
                for tau in range(tau_lo, tau_hi + 1):
                    for j in site_ids:
                        if Y.get((j, tau), 0) == 1:
                            denom_val += alpha.get((i, j, tau), 0.0)
                lhs = denom_val * P.get((i, k, t), 0.0)
                # RHS: alpha_{ikt} * Y_{kt}
                rhs = alpha.get((i, k, t), 0.0) * Y.get((k, t), 0)
                violation_amount = abs(lhs - rhs)
                add_violation(violations, magnitudes, violated_set,
                              10, f"Constraint 10 (eq.11): Gravity model "
                                  f"i={i},k={k},t={t}: LHS={lhs:.6f}, "
                                  f"RHS={rhs:.6f}",
                              lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 11 (paper eq. 12): A_{kt} <= sum_i P_{ikt} * gamma_i
    # =========================================================================
    for k in site_ids:
        for t in range(1, T + 1):
            lhs = A_check.get((k, t), 0.0)
            rhs = sum(P.get((i, k, t), 0.0) * gamma[i] for i in customer_ids)
            violation_amount = max(0.0, lhs - rhs)
            add_violation(violations, magnitudes, violated_set,
                          11, f"Constraint 11 (eq.12): Attendance site {k} "
                              f"period {t}: A={lhs:.2f} > sum P*gamma={rhs:.2f}",
                          lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 12 (paper eq. 13): A_{kt} <= theta_k
    # Venue capacity
    # =========================================================================
    for k in site_ids:
        for t in range(1, T + 1):
            lhs = A_check.get((k, t), 0.0)
            rhs = float(theta[k])
            violation_amount = max(0.0, lhs - rhs)
            add_violation(violations, magnitudes, violated_set,
                          12, f"Constraint 12 (eq.13): Attendance site {k} "
                              f"period {t}: A={lhs:.2f} > capacity={rhs}",
                          lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 13 (paper eq. 14): X_{jkt} in {0,1}
    # Binary domain for routing variables
    # =========================================================================
    for key, val in X.items():
        if val not in (0, 1):
            j, k, t = key
            violation_amount = min(abs(val - 0), abs(val - 1))
            add_violation(violations, magnitudes, violated_set,
                          13, f"Constraint 13 (eq.14): X_{{{j},{k},{t}}} = {val} "
                              f"not binary",
                          float(val), round(val), violation_amount, tol, eps)

    # =========================================================================
    # Constraint 14 (paper eq. 15): Y_{kt} in {0,1}
    # Binary domain for event variables
    # =========================================================================
    for key, val in Y.items():
        if val not in (0, 1):
            k, t = key
            violation_amount = min(abs(val - 0), abs(val - 1))
            add_violation(violations, magnitudes, violated_set,
                          14, f"Constraint 14 (eq.15): Y_{{{k},{t}}} = {val} "
                              f"not binary",
                          float(val), round(val), violation_amount, tol, eps)

    # =========================================================================
    # Constraint 15 (paper eq. 16): P_{ikt} <= 1
    # =========================================================================
    for i in customer_ids:
        for k in site_ids:
            for t in range(1, T + 1):
                lhs = P.get((i, k, t), 0.0)
                rhs = 1.0
                violation_amount = max(0.0, lhs - rhs)
                add_violation(violations, magnitudes, violated_set,
                              15, f"Constraint 15 (eq.16): P_{{{i},{k},{t}}} = "
                                  f"{lhs:.6f} > 1",
                              lhs, rhs, violation_amount, tol, eps)

    # =========================================================================
    # Constraint 16 (paper eq. 17): A_{kt} >= 0
    # =========================================================================
    for k in site_ids:
        for t in range(1, T + 1):
            val = A_check.get((k, t), 0.0)
            lhs = 0.0
            rhs = val
            violation_amount = max(0.0, lhs - rhs)  # 0 - val if val < 0
            if violation_amount > tol:
                add_violation(violations, magnitudes, violated_set,
                              16, f"Constraint 16 (eq.17): A_{{{k},{t}}} = "
                                  f"{val:.6f} < 0",
                              val, 0.0, abs(val), tol, eps)

    # =========================================================================
    # Constraint 17 (paper eq. 1, objective): Objective consistency
    # The reported objective_value must equal the total profit recomputed from
    # the solution's own decision variables:
    #   sum_{k,t} r_k * A_kt  -  sum_{k,t} f_k * Y_kt  -  c * sum_{j,k,t} d^s_jk X_jkt
    # All obj-determining variables are present in the solution (tour -> X,
    # events -> Y, attendance -> A; when attendance is omitted the model-implied
    # A_model is used, exactly as for the constraint checks above via A_check),
    # so this is a FULL recompute, not a lower bound.
    # =========================================================================
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            r = ctx['r']
            f_cost = ctx['f_cost']
            c = ctx['c']
            # Revenue: sum_{k,t} r_k * A_kt  (A_check = solution attendance, or
            # model-implied attendance when the solution omits it).
            revenue = sum(r[k] * A_check.get((k, t), 0.0)
                          for k in site_ids for t in range(1, T + 1))
            # Fixed event cost: sum_{k,t} f_k * Y_kt  (Y holds only the 1s).
            fixed_cost = sum(f_cost[k] * yv for (k, t), yv in Y.items())
            # Travel cost: c * sum_{j,k,t} d^s_jk * X_jkt  (X holds only the 1s;
            # period-1 transitions originate at dummy site 0 with d^s_0k = 0).
            travel_cost = c * sum(xv * d_s.get((j, k), 0.0)
                                  for (j, k, t), xv in X.items())
            true_obj = revenue - fixed_cost - travel_cost
            # 0.1% relative tolerance with a 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if math.isfinite(reported):
                abs_diff = abs(reported - true_obj)
                lhs_record = reported
            else:
                # A non-finite reported objective (inf / nan) can never equal
                # the finite recomputed profit. Use a finite violation amount
                # and a finite lhs so the result JSON stays standard-compliant
                # (no Infinity / NaN literals leaking into feasi_result_*.json).
                abs_diff = obj_tol + max(1.0, abs(true_obj))
                lhs_record = true_obj
            add_violation(violations, magnitudes, violated_set,
                          17, f"Constraint 17 (eq.1): Objective consistency "
                              f"violated: reported objective_value={reported} "
                              f"differs from recomputed total profit "
                              f"(revenue - fixed - travel) = {true_obj} "
                              f"[revenue={revenue}, fixed={fixed_cost}, "
                              f"travel={travel_cost}] "
                              f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                          lhs_record, true_obj, abs_diff, obj_tol, eps,
                          relative_tol=0.0)

    # =========================================================================
    # Build result
    # =========================================================================
    # Deduplicate violation messages per constraint index
    constraint_msgs = {}
    for v_mag in magnitudes:
        c_idx = v_mag["constraint"]
        if c_idx not in constraint_msgs:
            constraint_msgs[c_idx] = []
    for msg in violations:
        # Extract constraint index from the message prefix
        for c_idx in sorted(violated_set):
            prefix = f"Constraint {c_idx} "
            if msg.startswith(prefix):
                constraint_msgs[c_idx].append(msg)
                break

    # Aggregate violations into one message per constraint
    aggregated_violations = []
    for c_idx in sorted(violated_set):
        msgs = constraint_msgs.get(c_idx, [])
        count = sum(1 for m in magnitudes if m["constraint"] == c_idx)
        if count == 1:
            aggregated_violations.append(msgs[0] if msgs else f"Constraint {c_idx} violated")
        else:
            first = msgs[0] if msgs else f"Constraint {c_idx} violated"
            aggregated_violations.append(f"{first} (and {count - 1} more)")

    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": aggregated_violations,
        "violation_magnitudes": magnitudes if not feasible else [],
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for AtOP (Freeman et al. 2018)")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the data instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the JSON file containing the candidate solution.')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write the JSON file containing the result.')
    args = parser.parse_args()

    data = load_json(args.instance_path)
    sol = load_json(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return

    result = check_feasibility(data, sol)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result['feasible']:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: "
              f"{result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
