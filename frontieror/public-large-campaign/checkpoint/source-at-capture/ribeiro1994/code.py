import time
PROCESS_STARTED = time.monotonic()
import threading
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import numpy as np
import highspy
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'time_limit', 'log_path'):
        parser.add_argument('--' + key, required=key != 'log_path')
    args = parser.parse_args()
    budget = float(args.time_limit)
    deadline = PROCESS_STARTED + max(.1, budget - min(8., budget * .2))
    with open(args.instance_path) as stream:
        data = json.load(stream)
    ids = [t['trip_id'] for t in data['trips']]
    n, m = len(ids), data['parameters']['m_depots']
    position = {t: i for i, t in enumerate(ids)}
    pullout = np.full((m, n), np.inf)
    pullin = np.full((m, n), np.inf)
    connections = {}
    for arc in data['depot_to_trip_costs']:
        pullout[arc['depot'] - 1, position[arc['trip']]] = arc['cost']
    for arc in data['trip_to_depot_costs']:
        pullin[arc['depot'] - 1, position[arc['trip']]] = arc['cost']
    for arc in data['compatible_arcs']:
        connections[arc['from_trip'], arc['to_trip']] = arc['cost']
    best = float('inf')
    best_routes = None
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense='minimize')

    def save(routes):
        nonlocal best, best_routes
        value = 0.
        for route in routes:
            k, trips = route['depot'] - 1, route['trips']
            value += pullout[k, position[trips[0]]] + pullin[k, position[trips[-1]]]
            value += sum(connections[u, v] for u, v in zip(trips, trips[1:]))
        if value >= best - 1e-7:
            return
        best, best_routes = float(value), routes
        with open(args.solution_path, 'w') as stream:
            json.dump({'objective_value': best, 'routes': routes, 'num_vehicles': len(routes)}, stream)
        if logger:
            logger.log(best)

    slots = np.repeat(np.arange(m), data['vehicle_capacities'])
    for iteration in range(6):
        if time.monotonic() >= deadline:
            break
        if iteration == 0 or best_routes is None:
            starts, ends = pullout.min(axis=0), pullin.min(axis=0)
        else:
            assigned = np.empty(n, dtype=int)
            for route in best_routes:
                for trip in route['trips']:
                    assigned[position[trip]] = route['depot'] - 1
            blend = .25 + .15 * iteration
            starts = (1. - blend) * pullout.min(axis=0) + blend * pullout[assigned, np.arange(n)]
            ends = (1. - blend) * pullin.min(axis=0) + blend * pullin[assigned, np.arange(n)]
        costs = np.full((n, 2 * n), 1e12)
        costs[:, n:] = 0.
        for (u, v), cost in connections.items():
            costs[position[u], position[v]] = cost - ends[position[u]] - starts[position[v]]
        rows, cols = linear_sum_assignment(costs)
        successor = {ids[i]: ids[j] for i, j in zip(rows, cols) if j < n}
        preceded = set(successor.values())
        paths = []
        for trip in ids:
            if trip in preceded:
                continue
            path = [trip]
            while path[-1] in successor:
                path.append(successor[path[-1]])
            paths.append(path)
        if len(paths) > len(slots):
            continue
        depot_cost = np.asarray([pullout[slots, position[path[0]]] + pullin[slots, position[path[-1]]] for path in paths])
        route_indices, slot_indices = linear_sum_assignment(depot_cost)
        if not np.isfinite(depot_cost[route_indices, slot_indices]).all():
            continue
        save([{'depot': int(slots[j]) + 1, 'trips': paths[i]} for i, j in zip(route_indices, slot_indices)])

    incoming = {(k, t): [] for k in range(1, m + 1) for t in ids}
    outgoing = {(k, t): [] for k in range(1, m + 1) for t in ids}
    starts = {k: [] for k in range(1, m + 1)}
    keys, objective = [], []

    def add(k, u, v, cost):
        j = len(keys)
        keys.append((k, u, v))
        objective.append(cost)
        if u:
            outgoing[k, u].append(j)
        else:
            starts[k].append(j)
        if v:
            incoming[k, v].append(j)

    for arc in data['depot_to_trip_costs']:
        add(arc['depot'], 0, arc['trip'], arc['cost'])
    for arc in data['trip_to_depot_costs']:
        add(arc['depot'], arc['trip'], 0, arc['cost'])
    for k in range(1, m + 1):
        for (u, v), cost in connections.items():
            add(k, u, v, cost)
    rr, cc, vv, lower, upper = [], [], [], [], []

    def row(terms, lo, hi):
        r = len(lower)
        lower.append(lo)
        upper.append(hi)
        for j, value in terms:
            rr.append(r)
            cc.append(j)
            vv.append(value)

    for trip in ids:
        row([(j, 1.) for k in range(1, m + 1) for j in incoming[k, trip]], 1., 1.)
        for k in range(1, m + 1):
            row([(j, 1.) for j in incoming[k, trip]] + [(j, -1.) for j in outgoing[k, trip]], 0., 0.)
    for k in range(1, m + 1):
        row([(j, 1.) for j in starts[k]], -np.inf, data['vehicle_capacities'][k - 1])
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), len(keys))).tocsc()
    del rr, cc, vv
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(keys), len(lower)
    lp.col_cost_ = np.asarray(objective, dtype=float)
    lp.col_lower_, lp.col_upper_ = np.zeros(len(keys)), np.ones(len(keys))
    lp.row_lower_, lp.row_upper_ = np.asarray(lower), np.asarray(upper)
    lp.integrality_ = [highspy.HighsVarType.kInteger] * len(keys)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
    solver = highspy.Highs()
    solver.setOptionValue('output_flag', False)
    solver.setOptionValue('threads', 1)
    solver.setOptionValue('mip_rel_gap', .001)
    solver.passModel(lp)
    if best_routes is not None:
        selected = set()
        for route in best_routes:
            k, trips = route['depot'], route['trips']
            selected.add((k, 0, trips[0]))
            selected.add((k, trips[-1], 0))
            selected.update((k, u, v) for u, v in zip(trips, trips[1:]))
        seed = np.asarray([float(key in selected) for key in keys])
        solver.setSolution(len(keys), np.arange(len(keys), dtype=np.int32), seed)
    if time.monotonic() < deadline:
        solver.setOptionValue('time_limit', max(.01, deadline - time.monotonic()))
        solver.HandleUserInterrupt = True
        expiry = threading.Timer(max(.01, deadline - time.monotonic()), solver.cancelSolve)
        expiry.daemon = True
        expiry.start()
        try:
            solver.run()
        finally:
            expiry.cancel()
        solution = solver.getSolution()
        if solution.value_valid and solver.getInfo().primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
            successor, begin = {}, []
            for value, (k, u, v) in zip(solution.col_value, keys):
                if value > .5:
                    if not u:
                        begin.append((k, v))
                    else:
                        successor[k, u] = v
            routes = []
            for k, trip in begin:
                sequence = []
                while trip:
                    sequence.append(trip)
                    trip = successor[k, trip]
                routes.append({'depot': k, 'trips': sequence})
            save(routes)
    if best_routes is None:
        raise RuntimeError('No complete feasible route incumbent')


if __name__ == '__main__':
    main()
