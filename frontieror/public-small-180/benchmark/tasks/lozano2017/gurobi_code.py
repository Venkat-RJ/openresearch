"""
Gurobi implementation of Algorithm 1 (REHPP-based sampling) for the
Bilevel Mixed-Integer Linear Program (BMILP) from:

  Lozano & Smith (2017), "A Value-Function-Based Exact Approach for the
  Bilevel Mixed-Integer Programming Problem", Operations Research.

Mathematical model: EHPP / REHPP formulation (Equations 10a-10e).
Solver: Gurobi.

BENCHMARK SPECIALIZATION: The leader objective coefficient vector d^1 is fixed
to zero and recorded by an explicit instance flag (the leader values only
x-variables). This is consistent
with the BMILP testbed where only 'c' and 'd2' are provided.

ASSUMPTION (not specified in paper): Variable upper bounds are set to
2 * max(b^1, b^2), since the paper states variables are bounded but does not
give explicit bound values for the BMILP testbed.
"""

import json
import argparse
import time
import numpy as np
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


def get_var_ub(data):
    """Compute a uniform upper bound for all variables (x and y).
    Valid since all constraint coefficients >= 0, so each variable is
    bounded by max(b1, b2) / min_nonzero_coeff <= 2*max(b1,b2)."""
    b1 = np.array(data['b1'])
    b2 = np.array(data['b2'])
    return float(2.0 * max(np.max(b1), np.max(b2)))


def compute_lf(data, ub):
    """l^f = min d2*y over joint feasible region Omega (LP relaxation).
    Used to compute M^2 values (Table D.2 in paper)."""
    n1, n2, m1, m2 = data['n1'], data['n2'], data['m1'], data['m2']
    A1 = np.array(data['A1'])
    B1 = np.array(data['B1'])
    b1 = np.array(data['b1'])
    A2 = np.array(data['A2'])
    B2 = np.array(data['B2'])
    b2 = np.array(data['b2'])
    d2 = np.array(data['d2'])

    mdl = gp.Model()
    mdl.setParam("Threads", 1)
    mdl.setParam('OutputFlag', 0)
    x = [mdl.addVar(lb=0.0, ub=ub) for _ in range(n1)]
    y = [mdl.addVar(lb=0.0, ub=ub) for _ in range(n2)]
    for i in range(m1):
        mdl.addConstr(
            gp.quicksum(A1[i, k] * x[k] for k in range(n1)) +
            gp.quicksum(B1[i, k] * y[k] for k in range(n2)) <= b1[i])
    for i in range(m2):
        mdl.addConstr(
            gp.quicksum(A2[i, k] * x[k] for k in range(n1)) +
            gp.quicksum(B2[i, k] * y[k] for k in range(n2)) <= b2[i])
    mdl.setObjective(gp.quicksum(d2[k] * y[k] for k in range(n2)), GRB.MINIMIZE)
    mdl.optimize()
    return mdl.ObjVal if mdl.status == GRB.OPTIMAL else -1e9


def solve_follower(x_val, data, ub, time_remaining=None):
    """Solve follower's problem: max d2*y s.t. A2*x + B2*y <= b2, y in H^y.
    Returns (y_val, obj_val) or (None, None)."""
    n2, m2 = data['n2'], data['m2']
    A2 = np.array(data['A2'])
    B2 = np.array(data['B2'])
    b2 = np.array(data['b2'])
    d2 = np.array(data['d2'])
    integer_follower = set(data['integer_follower_vars'])  # 1-indexed

    mdl = gp.Model()
    mdl.setParam("Threads", 1)
    mdl.setParam('OutputFlag', 0)
    if time_remaining is not None:
        mdl.setParam('TimeLimit', max(float(time_remaining), 1.0))

    y = []
    for j in range(n2):
        vtype = GRB.INTEGER if (j + 1) in integer_follower else GRB.CONTINUOUS
        y.append(mdl.addVar(lb=0.0, ub=ub, vtype=vtype))

    rhs = b2 - A2 @ x_val
    for i in range(m2):
        mdl.addConstr(gp.quicksum(B2[i, k] * y[k] for k in range(n2)) <= rhs[i])
    mdl.setObjective(gp.quicksum(d2[k] * y[k] for k in range(n2)), GRB.MAXIMIZE)
    mdl.optimize()

    if mdl.SolCount > 0:
        return np.array([y[k].X for k in range(n2)]), float(mdl.ObjVal)
    return None, None


def compute_gamma(y_hat, data):
    """gamma_{y_hat,j} = floor(b2_j - B2_j * y_hat) + 1 (Proposition 2).
    The leader blocks y_hat via constraint j iff [A2*x]_j >= gamma_{y_hat,j}."""
    B2 = np.array(data['B2'])
    b2 = np.array(data['b2'])
    return (np.floor(b2 - B2 @ y_hat) + 1).astype(int)


def solve_rehpp(Y_hat, data, lf, ub, time_remaining=None):
    """Solve REHPP(Y_hat): relaxed EHPP over sample Y_hat (Equations 10a-10e).

    With M^1_j = 0 (valid since A^2*x >= 0 for non-negative x, A^2; Table D.2):
      Constraint (10b): [A2*x]_j >= sum_{y_hat} gamma_{y_hat,j} * w_{y_hat,j}
      Constraint (10c): d2*y >= d2*y_hat - M2_{y_hat} * sum_{B(y_hat)} w

    M^2_{y_hat} = d2*y_hat - l^f  (Table D.2)

    Returns (x_val, y_val, obj_val, status_str).
    """
    n1, n2, m1, m2 = data['n1'], data['n2'], data['m1'], data['m2']
    c = np.array(data['c'])
    if data.get('leader_follower_objective_coefficients_fixed_zero') is not True:
        raise ValueError('released Lozano 2017 instances must explicitly fix d1=0')
    d1 = np.zeros(n2)
    d2 = np.array(data['d2'])
    A1 = np.array(data['A1'])
    B1 = np.array(data['B1'])
    b1 = np.array(data['b1'])
    A2 = np.array(data['A2'])
    B2 = np.array(data['B2'])
    b2 = np.array(data['b2'])
    integer_follower = set(data['integer_follower_vars'])

    K = len(Y_hat)
    gammas = [compute_gamma(yh, data) for yh in Y_hat]
    # M^2_{y_hat} = max(d2*y_hat - l^f, 0) + epsilon for numerical stability
    M2 = [max(float(d2 @ yh) - lf, 0.0) + 1.0 for yh in Y_hat]

    mdl = gp.Model()
    mdl.setParam("Threads", 1)
    mdl.setParam('OutputFlag', 0)
    if time_remaining is not None:
        mdl.setParam('TimeLimit', max(float(time_remaining), 1.0))

    # Leader variables: all integer (Assumption 3)
    x = [mdl.addVar(lb=0.0, ub=ub, vtype=GRB.INTEGER) for _ in range(n1)]
    # Follower variables: partially integer
    y = []
    for j in range(n2):
        vtype = GRB.INTEGER if (j + 1) in integer_follower else GRB.CONTINUOUS
        y.append(mdl.addVar(lb=0.0, ub=ub, vtype=vtype))

    # Binary w_{y_hat, j} for each y_hat in Y_hat, constraint j
    w = [[mdl.addVar(vtype=GRB.BINARY) for _ in range(m2)] for _ in range(K)]

    # Objective: max c*x + d1*y  (14a)
    mdl.setObjective(
        gp.quicksum(c[k] * x[k] for k in range(n1)) +
        gp.quicksum(d1[k] * y[k] for k in range(n2)),
        GRB.MAXIMIZE)

    # Upper-level constraints: A1*x + B1*y <= b1  (14b)
    for i in range(m1):
        mdl.addConstr(
            gp.quicksum(A1[i, k] * x[k] for k in range(n1)) +
            gp.quicksum(B1[i, k] * y[k] for k in range(n2)) <= b1[i])

    # Lower-level constraints: A2*x + B2*y <= b2  (part of Omega, 10d)
    for i in range(m2):
        mdl.addConstr(
            gp.quicksum(A2[i, k] * x[k] for k in range(n1)) +
            gp.quicksum(B2[i, k] * y[k] for k in range(n2)) <= b2[i])

    if K > 0:
        # Constraint (10b) with M1=0: [A2*x]_j >= sum_i gamma[i][j]*w[i][j]
        for j in range(m2):
            mdl.addConstr(
                gp.quicksum(A2[j, k] * x[k] for k in range(n1)) >=
                gp.quicksum(gammas[i][j] * w[i][j] for i in range(K)))

        # Constraint (10c): d2*y >= d2*y_hat - M2[i]*sum_{B(y_hat_i)} w
        for i, yh in enumerate(Y_hat):
            # B(y_hat_i, Y_hat) = {(ip, q) | gamma[ip][q] >= gamma[i][q]}
            B_sum = gp.quicksum(
                w[ip][q]
                for ip in range(K)
                for q in range(m2)
                if gammas[ip][q] >= gammas[i][q])
            mdl.addConstr(
                gp.quicksum(d2[k] * y[k] for k in range(n2)) >=
                float(d2 @ yh) - M2[i] * B_sum)

    mdl.optimize()

    if mdl.status == GRB.INFEASIBLE:
        return None, None, None, 'infeasible'
    if mdl.SolCount > 0:
        x_val = np.array([x[k].X for k in range(n1)])
        y_val = np.array([y[k].X for k in range(n2)])
        status = 'optimal' if mdl.status == GRB.OPTIMAL else 'time_limit'
        return x_val, y_val, float(mdl.ObjVal), status
    return None, None, None, 'no_solution'


def main():
    parser = argparse.ArgumentParser(
        description='Solve BMILP via REHPP-based Algorithm 1 (Lozano & Smith 2017)')
    parser.add_argument('--instance_path', required=True,
                        help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True,
                        help='Path to write solution JSON (e.g., gurobi_solution_1.json)')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    start_time = time.time()

    ub = get_var_ub(data)
    lf = compute_lf(data, ub)

    n2 = data['n2']
    d2 = np.array(data['d2'])
    c = np.array(data['c'])
    if data.get('leader_follower_objective_coefficients_fixed_zero') is not True:
        raise ValueError('released Lozano 2017 instances must explicitly fix d1=0')
    d1 = np.zeros(n2)
    A1 = np.array(data['A1'])
    B1 = np.array(data['B1'])
    b1 = np.array(data['b1'])

    # Algorithm 1 (Section 2.3)
    UB = float('inf')
    LB = float('-inf')
    Y_hat = []  # REHPP with empty Y_hat = HPP (first iteration)
    best_x = None
    best_y = None
    best_obj = None
    iteration = 0
    status_str = 'running'

    while UB > LB + 1e-8:
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining <= 0:
            status_str = 'time_limit'
            break

        iteration += 1

        # Step 4-8: Solve REHPP(Y_hat^i)
        x_val, y_val, obj_val, rehpp_status = solve_rehpp(
            Y_hat, data, lf, ub, time_remaining=remaining)

        if rehpp_status == 'infeasible':
            # Remark 3: original BMIP is infeasible
            status_str = 'infeasible'
            break

        if x_val is None:
            # Time limit hit inside REHPP
            status_str = 'time_limit'
            break

        UB = obj_val

        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining <= 0:
            status_str = 'time_limit'
            break

        # Step 9: Obtain optimal follower response y_f in Psi(x^i)
        yf, yf_obj = solve_follower(x_val, data, ub, time_remaining=remaining)

        if yf is None:
            status_str = 'time_limit'
            break

        # Step 10: Y_hat^{i+1} = Y_hat^i union {y_f^i}
        Y_hat.append(yf)

        # Step 11: Check bilevel feasibility d2*y_i == d2*y_f
        follower_obj_yi = float(d2 @ y_val)
        if abs(follower_obj_yi - yf_obj) < 1e-6:
            # Step 12: Bilevel feasible => update LB and incumbent
            LB = UB
            best_x = x_val.copy()
            best_y = y_val.copy()
            best_obj = UB
            status_str = 'optimal'
        else:
            # Step 13: Check if x^i in X(y_f): A1*x^i + B1*y_f <= b1
            leader_obj_yf = float(c @ x_val + d1 @ yf)
            in_X_yf = all(
                float(A1[j] @ x_val + B1[j] @ yf) <= b1[j] + 1e-6
                for j in range(data['m1']))

            if in_X_yf and leader_obj_yf > LB + 1e-8:
                # Step 14: Update LB and incumbent
                LB = leader_obj_yf
                best_x = x_val.copy()
                best_y = yf.copy()
                best_obj = leader_obj_yf
                status_str = 'incumbent_updated'
            # Step 16: else LB unchanged

        elapsed = time.time() - start_time
        if elapsed >= args.time_limit:
            status_str = 'time_limit'
            break

    # If no bilevel feasible solution found but UB is finite, return UB as best estimate
    if best_obj is None:
        if LB > float('-inf'):
            best_obj = LB
        elif UB < float('inf'):
            best_obj = UB

    solution = {
        'objective_value': best_obj,
        'x': best_x.tolist() if best_x is not None else None,
        'y': best_y.tolist() if best_y is not None else None,
        'iterations': iteration,
        'status': status_str,
        'upper_bound': UB if UB < float('inf') else None,
        'lower_bound': LB if LB > float('-inf') else None,
    }

    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()
