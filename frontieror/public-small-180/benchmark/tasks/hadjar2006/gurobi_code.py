"""
Gurobi implementation of the Multiple Depot Vehicle Scheduling Problem (MDVSP).

Based on: Hadjar, Marcotte, and Soumis (2006),
"A Branch-and-Cut Algorithm for the Multiple Depot Vehicle Scheduling Problem",
Operations Research 54(1), pp. 130-149.

This implements the multicommodity flow formulation (P) from Section 2 of the paper:
  min  sum_{(i,j,k) in A} c_{ij} * X^k_{ij}                          (1)
  s.t. sum_{k in K} sum_{j in delta^+(i)} X^k_{ij} = 1,  i=1,...,n    (2)
       sum_{j=1}^{n} X^k_{n+k,j} <= v_k,  k in K                     (3)
       sum_{j in delta^-(i)} X^k_{ji} - sum_{j in delta^+(i)} X^k_{ij} = 0,
                                           k in K, i=1,...,n, n+k      (4)
       X^k_{ij} in {0,1}  for all (i,j,k) in A
"""

import json
import argparse
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


def load_instance(instance_path):
    """Load an MDVSP instance from a JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_mdvsp(data, time_limit):
    """Solve the MDVSP using Gurobi with the multicommodity flow formulation (P)."""
    num_tasks = data['num_tasks']
    num_depots = data['num_depots']
    depots = data['depots']
    tasks = data['tasks']

    task_ids = [t['id'] for t in tasks]
    depot_ids = [d['id'] for d in depots]
    depot_capacity = {d['id']: d['capacity'] for d in depots}

    # Build adjacency structures for compatible pairs
    # successors[i] = list of j such that (i,j) is a compatible pair
    # predecessors[j] = list of i such that (i,j) is a compatible pair
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    compatible_cost = {}
    for cp in data['compatible_pairs']:
        ti, tj = cp['task_i'], cp['task_j']
        compatible_cost[(ti, tj)] = cp['cost']
        successors[ti].append(tj)
        predecessors[tj].append(ti)

    # Depot-to-task and task-to-depot costs
    depot_to_task_cost = {}
    for dt in data['depot_to_task_costs']:
        depot_to_task_cost[(dt['depot_id'], dt['task_id'])] = dt['cost']

    task_to_depot_cost = {}
    for td in data['task_to_depot_costs']:
        task_to_depot_cost[(td['task_id'], td['depot_id'])] = td['cost']

    # Create Gurobi model
    model = gp.Model("MDVSP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # ---- Decision variables ----
    # X_tt[i,j,k]: task i -> task j with colour k
    X_tt = {}
    for (ti, tj), cost in compatible_cost.items():
        for k in depot_ids:
            X_tt[ti, tj, k] = model.addVar(
                vtype=GRB.BINARY, obj=cost, name=f"x_t{ti}_t{tj}_c{k}"
            )

    # X_dt[k,j]: depot k -> task j (colour k)
    X_dt = {}
    for k in depot_ids:
        for j in task_ids:
            if (k, j) in depot_to_task_cost:
                X_dt[k, j] = model.addVar(
                    vtype=GRB.BINARY, obj=depot_to_task_cost[k, j],
                    name=f"x_d{k}_t{j}"
                )

    # X_td[i,k]: task i -> depot k (colour k)
    X_td = {}
    for i in task_ids:
        for k in depot_ids:
            if (i, k) in task_to_depot_cost:
                X_td[i, k] = model.addVar(
                    vtype=GRB.BINARY, obj=task_to_depot_cost[i, k],
                    name=f"x_t{i}_d{k}"
                )

    model.update()

    # ---- Constraint (2): Assignment ----
    # Each trip assigned to exactly one vehicle
    for i in task_ids:
        outgoing = []
        for j in successors[i]:
            for k in depot_ids:
                outgoing.append(X_tt[i, j, k])
        for k in depot_ids:
            if (i, k) in X_td:
                outgoing.append(X_td[i, k])
        model.addConstr(gp.quicksum(outgoing) == 1, name=f"assign_{i}")

    # ---- Constraint (3): Depot capacity ----
    for k in depot_ids:
        depot_out = [X_dt[k, j] for j in task_ids if (k, j) in X_dt]
        if depot_out:
            model.addConstr(
                gp.quicksum(depot_out) <= depot_capacity[k],
                name=f"depot_cap_{k}"
            )

    # ---- Constraint (4): Flow conservation ----
    for k in depot_ids:
        # Flow conservation at each task node for colour k
        for i in task_ids:
            inflow = []
            outflow = []

            # From depot k to task i
            if (k, i) in X_dt:
                inflow.append(X_dt[k, i])
            # From other tasks to task i (colour k)
            for p in predecessors[i]:
                inflow.append(X_tt[p, i, k])

            # From task i to other tasks (colour k)
            for s in successors[i]:
                outflow.append(X_tt[i, s, k])
            # From task i to depot k
            if (i, k) in X_td:
                outflow.append(X_td[i, k])

            model.addConstr(
                gp.quicksum(inflow) - gp.quicksum(outflow) == 0,
                name=f"flow_t{i}_c{k}"
            )

        # Flow conservation at depot k
        depot_in = [X_td[i, k] for i in task_ids if (i, k) in X_td]
        depot_out = [X_dt[k, j] for j in task_ids if (k, j) in X_dt]
        model.addConstr(
            gp.quicksum(depot_in) - gp.quicksum(depot_out) == 0,
            name=f"flow_d{k}"
        )

    # ---- Optimize ----
    model.optimize()

    # ---- Extract solution ----
    result = {}
    if model.SolCount > 0:
        result['objective_value'] = model.ObjVal
        result['status'] = model.Status
        # Gurobi status 2 means optimal within configured tolerances; it does
        # not imply a mathematically zero serialized gap.
        result['mip_gap'] = model.MIPGap
        result['best_bound'] = model.ObjBound

        # Trace routes from each depot
        routes = []
        for k in depot_ids:
            for j in task_ids:
                if (k, j) in X_dt and X_dt[k, j].X > 0.5:
                    route = {
                        'depot': k,
                        'tasks': [j],
                        'cost': depot_to_task_cost[(k, j)]
                    }
                    current = j
                    while True:
                        found_next = False
                        for s in successors[current]:
                            if X_tt[current, s, k].X > 0.5:
                                route['tasks'].append(s)
                                route['cost'] += compatible_cost[(current, s)]
                                current = s
                                found_next = True
                                break
                        if not found_next:
                            route['cost'] += task_to_depot_cost.get((current, k), 0)
                            break
                    routes.append(route)

        result['num_vehicles'] = len(routes)
        result['routes'] = routes
    else:
        result['objective_value'] = None
        result['status'] = model.Status
        result['mip_gap'] = None
        result['best_bound'] = model.ObjBound
        result['num_vehicles'] = 0
        result['routes'] = []

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    result['status_str'] = status_map.get(model.Status, f"OTHER({model.Status})")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve MDVSP using Gurobi (multicommodity flow formulation)."
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds.')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from: {args.instance_path}")
    data = load_instance(args.instance_path)
    print(f"Instance: {data['num_tasks']} tasks, {data['num_depots']} depots, "
          f"{data['num_compatible_pairs']} compatible pairs")
    print(f"Time limit: {args.time_limit} seconds")

    start_time = time.time()
    result = solve_mdvsp(data, args.time_limit)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Solution Status: {result['status_str']}")
    if result['objective_value'] is not None:
        print(f"Objective Value: {result['objective_value']:.2f}")
        print(f"Number of Vehicles: {result['num_vehicles']}")
    else:
        print("No feasible solution found.")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print(f"{'='*60}")

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)
    print(f"Solution saved to: {args.solution_path}")


if __name__ == '__main__':
    main()
