import argparse
import heapq
import json
import os
import random
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from ortools.sat.python import cp_model


def solve(data, limit, solution_path=None):
    deadline = time.monotonic() + max(.1, limit - 2.)
    arcs = [tuple(a) for a in data['arcs']]
    keys = [f'{u}_{v}' for u, v in arcs]
    commodities = data['commodities']
    n = data['num_nodes']
    count = len(commodities)
    fixed = np.array([data['fixed_costs'][key] for key in keys], dtype=float)
    costs = np.array([data['variable_costs'][key] for key in keys], dtype=float).T
    costs *= np.array([c['demand'] for c in commodities])[:, None]
    adjacency = [[] for _ in range(n)]
    for a, (u, v) in enumerate(arcs):
        adjacency[u].append((v, a))
    pools = [set() for _ in commodities]
    best_routes = None
    best_cost = float('inf')
    best_output = None

    def shortest(k, weights):
        origin = commodities[k]['origin']
        destination = commodities[k]['destination']
        distance = [float('inf')] * n
        parent = [None] * n
        distance[origin] = 0.
        queue = [(0., origin)]
        while queue:
            value, u = heapq.heappop(queue)
            if value != distance[u]:
                continue
            if u == destination:
                break
            for v, a in adjacency[u]:
                new = value + weights[a]
                if new < distance[v]:
                    distance[v] = new
                    parent[v] = (u, a)
                    heapq.heappush(queue, (new, v))
        if distance[destination] == float('inf'):
            raise ValueError('Commodity has no directed origin-to-destination path')
        path = []
        v = destination
        while v != origin:
            u, a = parent[v]
            path.append(a)
            v = u
        path.reverse()
        return tuple(path)

    def save(routes):
        nonlocal best_routes, best_cost, best_output
        opened = {a for path in routes for a in path}
        objective = sum(fixed[a] for a in opened) + sum(
            sum(costs[k, a] for a in path) for k, path in enumerate(routes)
        )
        if objective >= best_cost - 1e-7:
            return
        best_cost = float(objective)
        best_routes = list(routes)
        best_output = {
            'objective_value': best_cost,
            'open_arcs': {key: int(a in opened) for a, key in enumerate(keys)},
            'routings': {str(k): {keys[a]: 1. for a in path}
                         for k, path in enumerate(routes)},
        }
        if solution_path:
            with open(solution_path, 'w') as handle:
                json.dump(best_output, handle, separators=(',', ':'))

    routes = [shortest(k, costs[k]) for k in range(count)]
    for k, path in enumerate(routes):
        pools[k].add(path)
    save(routes)
    if count <= 80 and len(arcs) <= 300:
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
        size = len(arcs)
        columns = (count + 1) * size
        rr, cc, vv, lower, upper = [], [], [], [], []
        for k, commodity in enumerate(commodities):
            base = len(lower)
            for node in range(n):
                balance = int(node == commodity['origin']) - int(node == commodity['destination'])
                lower.append(balance)
                upper.append(balance)
            for a, (u, v) in enumerate(arcs):
                rr.extend((base + u, base + v))
                cc.extend((k * size + a, k * size + a))
                vv.extend((1., -1.))
                row = len(lower)
                lower.append(-np.inf)
                upper.append(0.)
                rr.extend((row, row))
                cc.extend((k * size + a, count * size + a))
                vv.extend((1., -1.))
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), columns)).tocsc()
        remaining = deadline - time.monotonic()
        if remaining > .05:
            result = milp(
                np.r_[costs.ravel(), fixed],
                integrality=np.r_[np.zeros(count * size), np.ones(size)],
                bounds=Bounds(np.zeros(columns), np.ones(columns)),
                constraints=LinearConstraint(matrix, lower, upper),
                options={'time_limit': remaining, 'threads': 1},
            )
            if result.x is not None and result.fun < best_cost:
                values = result.x
                best_output = {
                    'objective_value': float(np.r_[costs.ravel(), fixed] @ values),
                    'open_arcs': {key: int(round(values[count * size + a]))
                                  for a, key in enumerate(keys)},
                    'routings': {str(k): {keys[a]: float(values[k * size + a])
                                         for a in range(size) if values[k * size + a] > 1e-9}
                                 for k in range(count)},
                }
        return best_output

    rng = random.Random(2019)
    search_end = min(deadline - .5, time.monotonic() + min(12., max(.2, limit * .25)))
    usage = np.zeros(len(arcs), dtype=int)
    for path in routes:
        usage[list(path)] += 1
    order = list(range(count))
    rounds = 0
    while time.monotonic() < search_end:
        if rounds:
            rng.shuffle(order)
        changed = False
        for k in order:
            if time.monotonic() >= search_end:
                break
            old = routes[k]
            usage[list(old)] -= 1
            marginal = costs[k] + fixed * (usage == 0)
            path = shortest(k, marginal)
            pools[k].add(path)
            old_cost = sum(marginal[a] for a in old)
            new_cost = sum(marginal[a] for a in path)
            if new_cost < old_cost - 1e-7:
                routes[k] = path
                changed = True
            usage[list(routes[k])] += 1
        save(routes)
        rounds += 1
        if not changed:
            break
    for factor in (.1, .3, 1., 2.):
        for k in range(count):
            if time.monotonic() >= search_end:
                break
            path = shortest(k, costs[k] + factor * fixed / np.maximum(1, usage))
            pools[k].add(path)
    remaining = deadline - time.monotonic()
    if remaining <= .1:
        return best_output

    model = cp_model.CpModel()
    used = sorted({a for pool in pools for path in pool for a in path})
    opened = {a: model.new_bool_var(f'y{a}') for a in used}
    choices = []
    objective = [int(round(1000 * fixed[a])) * opened[a] for a in used]
    incidence = {a: [] for a in used}
    seed_open = {a for path in best_routes for a in path}
    for a in used:
        model.add_hint(opened[a], int(a in seed_open))
    for k, pool in enumerate(pools):
        options = []
        for p, path in enumerate(sorted(pool)):
            variable = model.new_bool_var(f'z{k}_{p}')
            options.append((variable, path))
            objective.append(int(round(1000 * sum(costs[k, a] for a in path))) * variable)
            model.add_hint(variable, int(path == best_routes[k]))
            for a in path:
                model.add_implication(variable, opened[a])
                incidence[a].append(variable)
        model.add_exactly_one(variable for variable, _ in options)
        choices.append(options)
    for a in used:
        model.add(opened[a] <= sum(incidence[a]))
    model.minimize(sum(objective))

    class Incumbent(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            save([next(path for variable, path in options if self.value(variable))
                  for options in choices])

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(.01, deadline - time.monotonic())
    solver.solve(model, Incumbent())
    return best_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=float, default=60)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    with open(args.instance_path) as handle:
        data = json.load(handle)
    output = solve(data, args.time_limit, args.solution_path)
    with open(args.solution_path, 'w') as handle:
        json.dump(output, handle, separators=(',', ':'))
