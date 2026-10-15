"""
Gurobi implementation of the [NetR] (Network Reformulation) for the
Capacitated Facility Location and Production Planning Problem (CFLPP).

Source: Wu, Huang, Liang, Zhang, & Zhang (2022).
        "A supervised learning-driven heuristic for solving the facility
         location and production planning problem."
        European Journal of Operational Research, 301, 785-796.

Model: Appendix A, Equations (2)-(10).
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
def load_instance(path):
    """Load the CFLPP instance from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data


def build_model(data, time_limit):
    """Build the [NetR] Gurobi model from the instance data."""

    # --- Dimensions ---
    J = data['dimensions']['num_items']
    T = data['dimensions']['num_periods']
    K = data['dimensions']['num_plants']
    C = data['dimensions']['num_customers']

    # --- Sets ---
    # K_j[j]: plants capable of producing item j
    K_j = [list(data['plant_capabilities'][j]) for j in range(J)]

    # J_k[k]: items that can be produced at plant k
    J_k = [[] for _ in range(K)]
    for j in range(J):
        for k in K_j[j]:
            J_k[k].append(j)

    # C_j[j]: customers who have ordered item j
    C_j = [[] for _ in range(J)]
    for c in range(C):
        for j in data['customer_item_demands'][c]:
            C_j[j].append(c)

    # --- Parameters (all 0-indexed) ---
    d = data['demand']                  # d[j][c][t]
    sc = data['setup_cost']             # sc[j][k][t]
    pc = data['production_cost']        # pc[j][k][t]
    hc = data['holding_cost']           # hc[j][k][t]
    st = data['setup_time']             # st[j][k][t]
    pt = data['production_time']        # pt[j][k][t]
    cap = data['capacity']              # cap[k][t]
    f_k = data['plant_opening_cost']    # f[k]
    tc = data['transportation_cost']    # tc[j][c][k][t]

    # --- Derived parameters ---
    # Total demand: td[j][t] = sum over c in C_j[j] of d[j][c][t]
    td = [[sum(d[j][c][t] for c in C_j[j]) for t in range(T)] for j in range(J)]

    # Cumulative demand: sd[j][t][ell] = sum_{s=t}^{ell} td[j][s]
    # Used in capacity constraint and cv computation
    sd = {}
    for j in range(J):
        for t in range(T):
            cum = 0
            for ell in range(t, T):
                cum += td[j][ell]
                sd[j, t, ell] = cum

    # cv^k_{jt,ell}: total production and inventory-holding costs
    # cv^k_{jt,ell} = pc^k_{jt} * sd_{j,t,ell}
    #                 + sum_{s=t+1}^{ell} (sum_{p=t}^{s-1} hc^k_{jp}) * td_{js}
    #
    # NOTE: The paper (Appendix A) writes the inner holding cost sum starting
    # from p=1, but this appears to be a notational convention. The standard
    # Wagner-Whitin network reformulation uses p=t (holding cost from the
    # production period t to period s-1). We use p=t here as it represents
    # the actual holding cost incurred. This is an **INFERRED ASSUMPTION**
    # based on standard lot-sizing network reformulation literature.
    cv = {}
    for j in range(J):
        for k in K_j[j]:
            for t in range(T):
                for ell in range(t, T):
                    prod_cost = pc[j][k][t] * sd[j, t, ell]
                    hold_cost = 0
                    for s in range(t + 1, ell + 1):
                        # Holding cost for demand at period s produced at period t:
                        # hold from period t to s-1
                        hold_cost += sum(hc[j][k][p] for p in range(t, s)) * td[j][s]
                    cv[j, k, t, ell] = prod_cost + hold_cost

    # Transportation cost coefficient for U^k_j:
    # trans_coeff[j,k] = sum_t sum_{c in C_j} d[j][c][t] * tc[j][c][k][t]
    trans_coeff = {}
    for j in range(J):
        for k in K_j[j]:
            total = 0
            for t in range(T):
                for c in C_j[j]:
                    total += d[j][c][t] * tc[j][c][k][t]
            trans_coeff[j, k] = total

    # --- Create Gurobi Model ---
    model = gp.Model("CFLPP_NetR")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision Variables ---
    # Y^k_{jt}: setup variable (binary)
    Y = {}
    for j in range(J):
        for k in K_j[j]:
            for t in range(T):
                Y[j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{k}_{t}")

    # W^k_{jt,ell}: fraction variable (continuous >= 0)
    W = {}
    for j in range(J):
        for k in K_j[j]:
            for t in range(T):
                for ell in range(t, T):
                    W[j, k, t, ell] = model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0.0,
                        name=f"W_{j}_{k}_{t}_{ell}"
                    )

    # U^k_j: plant location variable (binary)
    U = {}
    for j in range(J):
        for k in K_j[j]:
            U[j, k] = model.addVar(vtype=GRB.BINARY, name=f"U_{j}_{k}")

    # Z^k: plant-opening variable (binary)
    Z = {}
    for k in range(K):
        Z[k] = model.addVar(vtype=GRB.BINARY, name=f"Z_{k}")

    model.update()

    # --- Objective Function (Equation 2) ---
    # min { sum_j sum_t sum_{k in K_j} (sc * Y + sum_ell cv * W + trans * U)
    #       + sum_k f * Z }
    obj = gp.LinExpr()

    for j in range(J):
        for t in range(T):
            for k in K_j[j]:
                # Setup cost
                obj += sc[j][k][t] * Y[j, k, t]
                # Production and holding cost (via cv)
                for ell in range(t, T):
                    obj += cv[j, k, t, ell] * W[j, k, t, ell]

    # Transportation cost (via U)
    for j in range(J):
        for k in K_j[j]:
            obj += trans_coeff[j, k] * U[j, k]

    # Plant opening cost
    for k in range(K):
        obj += f_k[k] * Z[k]

    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---

    # Constraint (3): Single-sourcing
    # sum_{k in K_j} U^k_j = 1, for all j
    for j in range(J):
        model.addConstr(
            gp.quicksum(U[j, k] for k in K_j[j]) == 1,
            name=f"single_source_{j}"
        )

    # Constraint (4): Plant opening requirement
    # U^k_j <= Z^k, for all j, k in K_j
    for j in range(J):
        for k in K_j[j]:
            model.addConstr(U[j, k] <= Z[k], name=f"plant_open_{j}_{k}")

    # Constraint (5): Setup-location linking
    # Y^k_{jt} <= U^k_j, for all j, t, k in K_j
    for j in range(J):
        for t in range(T):
            for k in K_j[j]:
                model.addConstr(
                    Y[j, k, t] <= U[j, k],
                    name=f"setup_loc_{j}_{k}_{t}"
                )

    # Constraint (6): Demand satisfaction over entire horizon
    # 1 = sum_{t=0}^{T-1} sum_{k in K_j} W^k_{j,0,t}, for all j
    # (0-indexed: production period = 0, "up to" period = t)
    for j in range(J):
        model.addConstr(
            gp.quicksum(
                W[j, k, 0, t] for k in K_j[j] for t in range(T)
            ) == 1,
            name=f"demand_horizon_{j}"
        )

    # Constraint (7): Flow balance / demand satisfaction across periods
    # sum_{ell=0}^{t-1} W^k_{j,ell,t-1} = sum_{ell=t}^{T-1} W^k_{j,t,ell}
    # for all j, t in {1,...,T-1} (0-indexed), k in K_j
    for j in range(J):
        for t in range(1, T):
            for k in K_j[j]:
                lhs = gp.quicksum(W[j, k, ell, t - 1] for ell in range(t))
                rhs = gp.quicksum(W[j, k, t, ell] for ell in range(t, T))
                model.addConstr(lhs == rhs, name=f"flow_{j}_{k}_{t}")

    # Constraint (8): Capacity constraint
    # sum_{j in J_k} (st * Y + sum_ell pt * sd * W) <= cap * Z
    # for all t, k
    for k in range(K):
        for t in range(T):
            lhs = gp.LinExpr()
            for j in J_k[k]:
                lhs += st[j][k][t] * Y[j, k, t]
                for ell in range(t, T):
                    lhs += pt[j][k][t] * sd[j, t, ell] * W[j, k, t, ell]
            model.addConstr(lhs <= cap[k][t] * Z[k], name=f"capacity_{k}_{t}")

    # Constraint (9): Setup forcing
    # sum_{ell=t}^{T-1} W^k_{jt,ell} <= Y^k_{jt}, for all j, t, k in K_j
    for j in range(J):
        for t in range(T):
            for k in K_j[j]:
                model.addConstr(
                    gp.quicksum(W[j, k, t, ell] for ell in range(t, T)) <= Y[j, k, t],
                    name=f"setup_force_{j}_{k}_{t}"
                )

    # Constraint (10): Variable domains are already enforced by variable types

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Solve CFLPP using the [NetR] formulation with Gurobi."
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, default='gurobi_solution_1.json',
                        help='Path for the output solution JSON file.')
    parser.add_argument('--time_limit', type=int, default=300,
                        help='Maximum solver runtime in seconds.')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)

    # Build and solve model
    model = build_model(data, args.time_limit)
    model.optimize()

    # Extract solution
    solution = {}

    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        solution['status'] = model.Status
        solution['mip_gap'] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0

        # Extract variable values
        J = data['dimensions']['num_items']
        T = data['dimensions']['num_periods']
        K = data['dimensions']['num_plants']
        K_j = [list(data['plant_capabilities'][j]) for j in range(J)]

        # Plant opening decisions
        Z_sol = {}
        for k in range(K):
            var = model.getVarByName(f"Z_{k}")
            if var is not None:
                Z_sol[str(k)] = round(var.X)
        solution['Z'] = Z_sol

        # Plant-item assignment decisions
        U_sol = {}
        for j in range(J):
            for k in K_j[j]:
                var = model.getVarByName(f"U_{j}_{k}")
                if var is not None:
                    U_sol[f"{j}_{k}"] = round(var.X)
        solution['U'] = U_sol

        # Setup decisions
        Y_sol = {}
        for j in range(J):
            for k in K_j[j]:
                for t in range(T):
                    var = model.getVarByName(f"Y_{j}_{k}_{t}")
                    if var is not None and var.X > 0.5:
                        Y_sol[f"{j}_{k}_{t}"] = round(var.X)
        solution['Y'] = Y_sol

    else:
        solution['objective_value'] = None
        solution['status'] = model.Status
        solution['mip_gap'] = None

    # Write solution
    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution['objective_value'] is not None:
        print(f"Objective value: {solution['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == '__main__':
    main()
