import time
STARTED = time.monotonic()
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import heapq
import json
import math
import random
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'log_path'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    args = parser.parse_args()
    deadline = STARTED + args.time_limit - min(4.0, max(0.3, args.time_limit * 0.08))
    data = json.load(open(args.instance_path))
    arcs = data['network']['arcs']
    nodes = data['network']['nodes']
    commodities = data['commodities']['commodity_list']
    count = len(commodities)
    edge_count = len(arcs)
    capacity = [arc['capacity'] for arc in arcs]
    costs = [arc['cost'] for arc in arcs]
    adjacency = {node: [] for node in nodes}
    for edge, arc in enumerate(arcs):
        adjacency[arc['from_node']].append((arc['to_node'], edge))
    logger = SolutionLogger(args.log_path, sense='minimize')
    pools = [set() for _ in commodities]
    best = [None] * count
    best_objective = math.inf
    rng = random.Random(2000)

    def unit_cost(path):
        return sum(costs[edge] for edge in path)

    def save(paths):
        nonlocal best, best_objective
        objective = 0
        entries = []
        for k, commodity in enumerate(commodities):
            path = paths[k]
            if path is None:
                objective += commodity['demand'] * commodity['artificial_arc_cost']
                entries.append({'commodity_id': commodity['commodity_id'], 'rejected': True, 'path_arcs': []})
            else:
                objective += commodity['demand'] * unit_cost(path)
                entries.append({'commodity_id': commodity['commodity_id'], 'rejected': False,
                                'path_arcs': [{'from': arcs[edge]['from_node'], 'to': arcs[edge]['to_node'],
                                               'arc_id': arcs[edge]['arc_id']} for edge in path]})
        if objective >= best_objective:
            return
        solution = {'objective_value': objective, 'commodities': entries}
        with open(args.solution_path, 'w') as output:
            json.dump(solution, output)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(objective, solution)
        else:
            logger.log(objective)
        best = list(paths)
        best_objective = objective
        print('incumbent elapsed=%.3f objective=%d routed=%d' % (time.monotonic() - STARTED, objective, sum(path is not None for path in paths)), flush=True)

    save(best)

    def shortest(origin, destination, demand, weights, forbidden_edges=None, forbidden_nodes=None):
        forbidden_edges = forbidden_edges or set()
        forbidden_nodes = forbidden_nodes or set()
        distance = {origin: 0.0}
        previous = {}
        queue = [(0.0, origin)]
        while queue:
            value, node = heapq.heappop(queue)
            if value != distance.get(node):
                continue
            if node == destination:
                result = []
                while node != origin:
                    parent, edge = previous[node]
                    result.append(edge)
                    node = parent
                return tuple(reversed(result)), value
            for neighbor, edge in adjacency[node]:
                if capacity[edge] < demand or edge in forbidden_edges or neighbor in forbidden_nodes:
                    continue
                trial = value + weights[edge]
                if trial < distance.get(neighbor, math.inf):
                    distance[neighbor] = trial
                    previous[neighbor] = (node, edge)
                    heapq.heappush(queue, (trial, neighbor))
        return None, math.inf

    def remember(k, path):
        if path and unit_cost(path) < commodities[k]['artificial_arc_cost']:
            pools[k].add(path)

    for k, commodity in enumerate(commodities):
        if time.monotonic() >= deadline:
            return
        path, _ = shortest(commodity['origin'], commodity['destination'], commodity['demand'], costs)
        remember(k, path)

    def greedy(exponent, randomized=False):
        available = capacity[:]
        paths = [None] * count
        gain = []
        for k, commodity in enumerate(commodities):
            smallest = min((unit_cost(path) for path in pools[k]), default=commodity['artificial_arc_cost'])
            priority = (commodity['artificial_arc_cost'] - smallest) * commodity['demand'] / max(1, commodity['demand'] ** exponent)
            if randomized:
                priority *= rng.uniform(0.5, 1.5)
            gain.append((priority, k))
        for _, k in sorted(gain, reverse=True):
            demand = commodities[k]['demand']
            for path in sorted(pools[k], key=lambda path: (unit_cost(path), len(path), path)):
                if all(available[edge] >= demand for edge in path):
                    paths[k] = path
                    for edge in path:
                        available[edge] -= demand
                    break
        save(paths)

    for exponent in (0.0, 0.5, 1.0, 2.0):
        greedy(exponent)
    pool_deadline = min(deadline, time.monotonic() + min(10.0, args.time_limit * 0.2))
    prices = [0.0] * edge_count
    dual_best = -math.inf
    for iteration in range(35):
        if time.monotonic() >= pool_deadline:
            break
        weights = [costs[edge] + prices[edge] for edge in range(edge_count)]
        use = [0] * edge_count
        dual = -sum(prices[edge] * capacity[edge] for edge in range(edge_count))
        for k, commodity in enumerate(commodities):
            if time.monotonic() >= pool_deadline:
                break
            path, value = shortest(commodity['origin'], commodity['destination'], commodity['demand'], weights)
            remember(k, path)
            rejection = commodity['artificial_arc_cost']
            dual += commodity['demand'] * min(value, rejection)
            if path is not None and value < rejection:
                for edge in path:
                    use[edge] += commodity['demand']
        else:
            dual_best = max(dual_best, dual)
            violation = [use[edge] - capacity[edge] for edge in range(edge_count)]
            norm = sum(value * value for edge, value in enumerate(violation) if value > 0 or prices[edge] > 0)
            if norm == 0:
                break
            step = max(0.001, 1.4 * max(1.0, best_objective - dual) / norm)
            prices = [max(0.0, prices[edge] + step * violation[edge]) for edge in range(edge_count)]
            if iteration % 5 == 0:
                greedy(1.0, True)
            continue
        break

    # Yen deviations retain cheap alternative simple paths beyond the priced columns.
    for k, commodity in enumerate(commodities):
        if time.monotonic() >= pool_deadline:
            break
        initial, _ = shortest(commodity['origin'], commodity['destination'], commodity['demand'], costs)
        if not initial:
            continue
        accepted = [initial]
        pending = []
        seen = {initial}
        for _ in range(7):
            if time.monotonic() >= pool_deadline:
                break
            previous_path = accepted[-1]
            path_nodes = [commodity['origin']] + [arcs[edge]['to_node'] for edge in previous_path]
            for position in range(len(previous_path)):
                prefix = previous_path[:position]
                forbidden = {path[position] for path in accepted if len(path) > position and path[:position] == prefix}
                suffix, _ = shortest(path_nodes[position], commodity['destination'], commodity['demand'], costs,
                                     forbidden, set(path_nodes[:position]))
                if suffix is None:
                    continue
                path = prefix + suffix
                if path not in seen:
                    seen.add(path)
                    heapq.heappush(pending, (unit_cost(path), path))
            if not pending:
                break
            _, path = heapq.heappop(pending)
            accepted.append(path)
            remember(k, path)
    for iteration in range(12):
        if time.monotonic() >= deadline:
            return
        greedy((0.0, 0.5, 1.0, 2.0)[iteration % 4], iteration >= 4)
    print('columns_end elapsed=%.3f columns=%d dual=%.6f objective=%d' %
          (time.monotonic() - STARTED, sum(map(len, pools)), dual_best, best_objective), flush=True)
    model = cp_model.CpModel()
    choices = []
    selected_paths = []
    edge_terms = [[] for _ in arcs]
    terms = []
    for k, commodity in enumerate(commodities):
        if time.monotonic() >= deadline:
            return
        alternatives = [None] + sorted(pools[k], key=lambda path: (unit_cost(path), len(path), path))
        row = [model.NewBoolVar('p%d_%d' % (k, pos)) for pos in range(len(alternatives))]
        model.AddExactlyOne(row)
        for pos, path in enumerate(alternatives):
            cost = commodity['artificial_arc_cost'] if path is None else unit_cost(path)
            terms.append(commodity['demand'] * cost * row[pos])
            model.AddHint(row[pos], int(best[k] == path))
            if path is not None:
                for edge in path:
                    edge_terms[edge].append(commodity['demand'] * row[pos])
        choices.append(row)
        selected_paths.append(alternatives)
    for edge, row in enumerate(edge_terms):
        if row:
            model.Add(sum(row) <= capacity[edge])
    model.Add(sum(terms) <= best_objective)
    model.Minimize(sum(terms))

    class Incumbent(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            paths = [next(selected_paths[k][pos] for pos, variable in enumerate(row) if self.Value(variable))
                     for k, row in enumerate(choices)]
            save(paths)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 2000
    solver.parameters.max_time_in_seconds = remaining
    print('path_master_start elapsed=%.3f budget=%.3f' % (time.monotonic() - STARTED, remaining), flush=True)
    status = solver.Solve(model, Incumbent())
    print('path_master_end elapsed=%.3f status=%s objective=%d' % (time.monotonic() - STARTED, solver.StatusName(status), best_objective), flush=True)


if __name__ == '__main__':
    main()
