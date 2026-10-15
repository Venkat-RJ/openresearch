"""
Gurobi implementation of the Two-Stage Stochastic Production Routing Problem (2-SPRP).
Model (1)-(19): 2-BF (Two-Stage Base Formulation).
Reference: Adulyasak, Cordeau, Jans (2015), Operations Research.
"""
import argparse
import json
import math
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
    with open(path) as f:
        return json.load(f)


def solve(instance_path, solution_path, time_limit):
    data = load_instance(instance_path)

    n = data['n']           # number of customers
    T = data['T']           # number of periods
    m = data['m']           # number of vehicles
    Q = data['Q']           # vehicle capacity
    C = data['C']           # production capacity per period
    f_cost = data['f']      # fixed production setup cost
    u = data['u']           # unit production cost
    h = data['h']           # holding costs, h[0]=plant, h[1..n]=customers
    L = data['L']           # inventory capacities, L[0]=plant, L[1..n]=customers
    I0 = data['I0']         # initial inventories
    sigma = data['sigma']   # penalty costs for customers (list of length n)
    c = data['transportation_costs']  # c[i][j], i,j in 0..n
    rho = data['scenario_probabilities']   # rho[omega]
    demand_sc = data['demand_scenarios']   # demand_sc[omega][cust_idx][period_idx] (0-indexed)
    n_omega = data['n_scenarios']

    # Index sets (1-indexed periods, vehicles; 0-indexed customers shifted by 1)
    Nc = list(range(1, n + 1))        # customers
    N = list(range(0, n + 1))         # all nodes (0=plant)
    periods = list(range(1, T + 1))
    vehicles = list(range(1, m + 1))
    scenarios = list(range(n_omega))

    # All undirected edges (i<j)
    edges = [(i, j) for i in N for j in N if i < j]
    edges_Nc = [(i, j) for i in Nc for j in Nc if i < j]

    def d(omega, i, t):
        """Demand of customer i (1-indexed) in period t (1-indexed) under scenario omega."""
        return demand_sc[omega][i - 1][t - 1]

    def M_t(t, omega):
        """M_{t,omega} = min{C, Q, sum_{s>=t, i in Nc} d_{is,omega}}"""
        rem = sum(d(omega, i, s) for i in Nc for s in range(t, T + 1))
        return min(C, Q, rem)

    def M_prime(i, t, omega):
        """M'_{it,omega} = min{L_i, Q, sum_{s>=t} d_{is,omega}}"""
        rem = sum(d(omega, i, s) for s in range(t, T + 1))
        return min(L[i], Q, rem)

    # ----------------------------------------------------------------
    model = gp.Model("2SPRP_2BF")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LazyConstraints", 1)

    # --- First-stage variables ---
    y = {t: model.addVar(vtype=GRB.BINARY, name=f"y_{t}") for t in periods}

    z = {}
    for i in N:
        for k in vehicles:
            for t in periods:
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")

    x = {}
    for (i, j) in edges:
        for k in vehicles:
            for t in periods:
                if i == 0:
                    # Depot edge: can be traversed 0,1,2 times (out-and-back)
                    x[i, j, k, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=2,
                                                  name=f"x_{i}_{j}_{k}_{t}")
                else:
                    x[i, j, k, t] = model.addVar(vtype=GRB.BINARY,
                                                  name=f"x_{i}_{j}_{k}_{t}")

    # --- Second-stage variables ---
    p = {}
    I_var = {}
    q_var = {}
    e_var = {}

    for omega in scenarios:
        for t in periods:
            p[t, omega] = model.addVar(lb=0.0, name=f"p_{t}_{omega}")
        for i in N:
            for t in periods:
                I_var[i, t, omega] = model.addVar(lb=0.0, name=f"I_{i}_{t}_{omega}")
        for i in Nc:
            for t in periods:
                e_var[i, t, omega] = model.addVar(lb=0.0, name=f"e_{i}_{t}_{omega}")
            for k in vehicles:
                for t in periods:
                    q_var[i, k, t, omega] = model.addVar(lb=0.0, name=f"q_{i}_{k}_{t}_{omega}")

    model.update()

    # --- Objective (1) ---
    routing_cost = gp.quicksum(
        c[i][j] * x[i, j, k, t]
        for (i, j) in edges for k in vehicles for t in periods
    )
    setup_cost = gp.quicksum(f_cost * y[t] for t in periods)

    second_stage = gp.quicksum(
        rho[omega] * (
            gp.quicksum(u * p[t, omega] for t in periods)
            + gp.quicksum(h[i] * I_var[i, t, omega] for i in N for t in periods)
            + gp.quicksum(sigma[i - 1] * e_var[i, t, omega] for i in Nc for t in periods)
        )
        for omega in scenarios
    )

    model.setObjective(setup_cost + routing_cost + second_stage, GRB.MINIMIZE)

    # --- Constraint (2): Plant inventory balance ---
    for t in periods:
        for omega in scenarios:
            I_prev = I0[0] if t == 1 else I_var[0, t - 1, omega]
            model.addConstr(
                I_prev + p[t, omega]
                == gp.quicksum(q_var[i, k, t, omega] for i in Nc for k in vehicles)
                + I_var[0, t, omega],
                name=f"plant_bal_{t}_{omega}"
            )

    # --- Constraint (3): Customer inventory balance ---
    for i in Nc:
        for t in periods:
            for omega in scenarios:
                I_prev = I0[i] if t == 1 else I_var[i, t - 1, omega]
                model.addConstr(
                    I_prev + gp.quicksum(q_var[i, k, t, omega] for k in vehicles)
                    + e_var[i, t, omega]
                    == d(omega, i, t) + I_var[i, t, omega],
                    name=f"cust_bal_{i}_{t}_{omega}"
                )

    # --- Constraint (4): Max plant inventory ---
    for t in periods:
        for omega in scenarios:
            model.addConstr(I_var[0, t, omega] <= L[0], name=f"plant_inv_{t}_{omega}")

    # --- Constraint (5): Max customer inventory ---
    for i in Nc:
        for t in periods:
            for omega in scenarios:
                model.addConstr(
                    I_var[i, t, omega] + d(omega, i, t) <= L[i],
                    name=f"cust_inv_{i}_{t}_{omega}"
                )

    # --- Constraint (6): Production capacity ---
    for t in periods:
        for omega in scenarios:
            model.addConstr(
                p[t, omega] <= M_t(t, omega) * y[t],
                name=f"prod_cap_{t}_{omega}"
            )

    # --- Constraint (7): Vehicle load capacity ---
    for k in vehicles:
        for t in periods:
            for omega in scenarios:
                model.addConstr(
                    gp.quicksum(q_var[i, k, t, omega] for i in Nc) <= Q * z[0, k, t],
                    name=f"veh_load_{k}_{t}_{omega}"
                )

    # --- Constraint (8): Delivery only if customer visited ---
    for i in Nc:
        for k in vehicles:
            for t in periods:
                for omega in scenarios:
                    model.addConstr(
                        q_var[i, k, t, omega] <= M_prime(i, t, omega) * z[i, k, t],
                        name=f"del_cap_{i}_{k}_{t}_{omega}"
                    )

    # --- Constraint (9): Each customer visited at most once per period ---
    for i in Nc:
        for t in periods:
            model.addConstr(
                gp.quicksum(z[i, k, t] for k in vehicles) <= 1,
                name=f"visit_{i}_{t}"
            )

    # --- Constraint (10): Degree constraints ---
    for i in N:
        for k in vehicles:
            for t in periods:
                inc = [(a, b) for (a, b) in edges if a == i or b == i]
                model.addConstr(
                    gp.quicksum(x[a, b, k, t] for (a, b) in inc) == 2 * z[i, k, t],
                    name=f"degree_{i}_{k}_{t}"
                )

    # --- Valid inequality (16): z_{ikt} <= z_{0kt} ---
    for i in Nc:
        for k in vehicles:
            for t in periods:
                model.addConstr(z[i, k, t] <= z[0, k, t], name=f"dispatch_{i}_{k}_{t}")

    # --- Valid inequality (17): edge implies both endpoints visited ---
    for (i, j) in edges_Nc:
        for k in vehicles:
            for t in periods:
                model.addConstr(x[i, j, k, t] <= z[i, k, t], name=f"ev_i_{i}_{j}_{k}_{t}")
                model.addConstr(x[i, j, k, t] <= z[j, k, t], name=f"ev_j_{i}_{j}_{k}_{t}")

    # --- Valid inequalities (18)-(19): symmetry breaking for multiple vehicles ---
    if m > 1:
        for k in range(1, m):
            for t in periods:
                model.addConstr(z[0, k, t] >= z[0, k + 1, t], name=f"sym_use_{k}_{t}")
        for j_node in Nc:
            for k in range(1, m):
                for t in periods:
                    model.addConstr(
                        gp.quicksum(2 ** (j_node - i) * z[i, k, t] for i in range(1, j_node + 1))
                        >= gp.quicksum(2 ** (j_node - i) * z[i, k + 1, t] for i in range(1, j_node + 1)),
                        name=f"lex_{j_node}_{k}_{t}"
                    )

    # --- Lazy SEC callback (constraint 11) ---
    def find_subtours(x_val, z_val, k, t):
        """Find customer-only subtours (not connected to depot) in solution."""
        visited = [i for i in Nc if z_val.get((i, k, t), 0) > 0.5]
        if len(visited) < 2:
            return []

        # Build adjacency among visited customers
        adj = {i: [] for i in visited}
        for (a, b) in edges_Nc:
            if a in visited and b in visited:
                val = x_val.get((a, b, k, t), 0)
                if val > 0.5:
                    adj[a].append(b)
                    adj[b].append(a)

        # Find connected components (ignoring depot)
        seen = set()
        components = []
        for start in visited:
            if start not in seen:
                comp = []
                stack = [start]
                while stack:
                    node = stack.pop()
                    if node not in seen:
                        seen.add(node)
                        comp.append(node)
                        for nb in adj.get(node, []):
                            if nb not in seen:
                                stack.append(nb)
                components.append(comp)
        # Only components not connected to depot are subtours
        # Check depot connectivity
        depot_neighbors = set()
        for (a, b) in edges:
            if a == 0 and b in visited and x_val.get((a, b, k, t), 0) > 0.5:
                depot_neighbors.add(b)
            elif b == 0 and a in visited and x_val.get((a, b, k, t), 0) > 0.5:
                depot_neighbors.add(a)

        subtours = []
        for comp in components:
            if not any(v in depot_neighbors for v in comp):
                subtours.append(comp)
        return subtours

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = {key: model.cbGetSolution(x[key]) for key in x}
            z_vals = {key: model.cbGetSolution(z[key]) for key in z}

            for k in vehicles:
                for t in periods:
                    subtours = find_subtours(x_vals, z_vals, k, t)
                    for comp in subtours:
                        S = set(comp)
                        E_S = [(a, b) for (a, b) in edges if a in S and b in S]
                        e_node = comp[0]
                        model.cbLazy(
                            gp.quicksum(x[a, b, k, t] for (a, b) in E_S)
                            <= gp.quicksum(z[i, k, t] for i in S) - z[e_node, k, t]
                        )

    model.optimize(callback)

    # --- Extract and write solution ---
    result = {"objective_value": None}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["gap"] = model.MIPGap
        result["status"] = model.Status

        y_sol = {str(t): round(y[t].X) for t in periods}
        result["y"] = y_sol

        z_sol = {}
        for i in N:
            for k in vehicles:
                for t in periods:
                    v = z[i, k, t].X
                    if v > 0.5:
                        z_sol[f"{i}_{k}_{t}"] = 1
        result["z"] = z_sol

        x_sol = {}
        for (i, j) in edges:
            for k in vehicles:
                for t in periods:
                    v = x[i, j, k, t].X
                    if v > 0.5:
                        x_sol[f"{i}_{j}_{k}_{t}"] = round(v)
        result["x"] = x_sol

    with open(solution_path, 'w') as fp:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, fp, indent=2)

    print(f"Objective value: {result['objective_value']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Solve 2-SPRP with Gurobi (2-BF formulation)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write solution JSON")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Solver time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
