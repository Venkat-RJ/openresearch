import argparse
import json
import math
import os
import random
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from ortools.sat.python import cp_model


def solve(data, limit, solution_path=None):
    deadline = time.monotonic() + max(.1, limit - 2.)
    n, stations = data['num_tasks'], data['num_stations']
    processing = data['processing_times']
    forward = data['forward_setup_times']
    backward = data['backward_setup_times']
    edges = [(i - 1, j - 1) for i, j in data['precedence_relations']]
    predecessors = [set() for _ in range(n)]
    successors = [set() for _ in range(n)]
    for i, j in edges:
        predecessors[j].add(i)
        successors[i].add(j)
    indegree = [len(p) for p in predecessors]
    ready = [i for i in range(n) if not indegree[i]]
    topology = []
    while ready:
        i = ready.pop()
        topology.append(i)
        for j in successors[i]:
            indegree[j] -= 1
            if not indegree[j]:
                ready.append(j)
    if len(topology) != n or not 1 <= stations <= n:
        raise RuntimeError('Invalid assembly precedence/station domain')
    ancestors = [set() for _ in range(n)]
    descendants = [set() for _ in range(n)]
    tail = processing.copy()
    for i in topology:
        for j in successors[i]:
            ancestors[j].update(ancestors[i])
            ancestors[j].add(i)
    for i in reversed(topology):
        for j in successors[i]:
            descendants[i].update(descendants[j])
            descendants[i].add(j)
            tail[i] = max(tail[i], processing[i] + tail[j])
    declared_lower = math.ceil(data.get('cycle_time_lower_bound', 0))
    declared_upper = math.floor(data.get('cycle_time_upper_bound', sum(processing) + (n - 1) * max(map(max, forward)) + max(map(max, backward))))
    lower = max(declared_lower, max(processing), math.ceil(sum(processing) / stations))
    best, best_order, best_parts = None, None, None

    def certificate(parts):
        loads = [sum(processing[i] for i in part) + sum(forward[i][j] for i, j in zip(part, part[1:])) + backward[part[-1]][part[0]] for part in parts]
        cycle = max(loads)
        return {'objective_value': cycle, 'cycle_time': cycle,
                'assignment': {str(i + 1): k + 1 for k, part in enumerate(parts) for i in part},
                'station_sequences': {str(k + 1): [i + 1 for i in part] for k, part in enumerate(parts)}}

    def save(parts, order=None):
        nonlocal best, best_parts, best_order
        out = certificate(parts)
        if declared_lower <= out['cycle_time'] <= declared_upper and (best is None or out['cycle_time'] < best['cycle_time']):
            best = out
            best_parts = [part.copy() for part in parts]
            best_order = list(order) if order is not None else [i for part in parts for i in part]
            if solution_path:
                with open(solution_path, 'w') as f:
                    json.dump(out, f, separators=(',', ':'))

    def partition(order):
        prefix = [0]
        setup = [0]
        for i in order:
            prefix.append(prefix[-1] + processing[i])
        for i, j in zip(order, order[1:]):
            setup.append(setup[-1] + forward[i][j])
        costs = [[0] * (n + 1) for _ in range(n)]
        for left in range(n):
            for right in range(left + 1, n + 1):
                costs[left][right] = prefix[right] - prefix[left] + setup[right - 1] - setup[left] + backward[order[right - 1]][order[left]]
        previous = [math.inf] * (n + 1)
        previous[0] = 0
        parents = []
        for k in range(1, stations + 1):
            current = [math.inf] * (n + 1)
            parent = [-1] * (n + 1)
            for right in range(k, n - stations + k + 1):
                for left in range(k - 1, right):
                    value = max(previous[left], costs[left][right])
                    if value < current[right]:
                        current[right], parent[right] = value, left
            parents.append(parent)
            previous = current
        right = n
        parts = []
        for k in range(stations - 1, -1, -1):
            left = parents[k][right]
            parts.append(order[left:right])
            right = left
        return list(reversed(parts))

    rng = random.Random(42)
    save(partition(topology), topology)
    seed_deadline = min(deadline, time.monotonic() + min(1.25, limit * .08))
    iteration = 0
    while time.monotonic() < seed_deadline:
        if best_order is not None and iteration % 3:
            order = best_order.copy()
            i = order.pop(rng.randrange(n))
            positions = {j: p for p, j in enumerate(order)}
            first = max([positions[j] + 1 for j in predecessors[i]] or [0])
            last = min([positions[j] for j in successors[i]] or [n - 1])
            order.insert(rng.randint(first, last), i)
        else:
            degree = [len(p) for p in predecessors]
            available = [i for i in range(n) if not degree[i]]
            order = []
            importance = rng.uniform(0., 1.)
            setup_weight = rng.uniform(0., 12.)
            noise = [rng.uniform(0., max(processing)) for _ in range(n)]
            while available:
                previous = order[-1] if order else None
                i = max(available, key=lambda j: importance * tail[j] + noise[j] - (setup_weight * forward[previous][j] if previous is not None else 0))
                available.remove(i)
                order.append(i)
                for j in successors[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        available.append(j)
        save(partition(order), order)
        iteration += 1
    upper = min(declared_upper, best['cycle_time'] if best is not None else declared_upper)
    if lower > upper:
        raise RuntimeError('No assembly witness within explicit cycle bounds')
    if best is not None and time.monotonic() >= deadline:
        return best
    model = cp_model.CpModel()
    cycle = model.new_int_var(lower, upper, 'cycle')
    domains = []
    for i in range(n):
        earliest = max(0, math.ceil((processing[i] + sum(processing[j] for j in ancestors[i])) / upper) - 1)
        latest = min(stations - 1, stations - math.ceil((processing[i] + sum(processing[j] for j in descendants[i])) / upper))
        domains.append(set(range(earliest, latest + 1)))
    station = [model.new_int_var(min(domains[i]), max(domains[i]), f's{i}') for i in range(n)]
    rank = [model.new_int_var(len(ancestors[i]), n - 1 - len(descendants[i]), f'r{i}') for i in range(n)]
    first = [model.new_bool_var(f'first{i}') for i in range(n)]
    last = [model.new_bool_var(f'last{i}') for i in range(n)]
    assignments, first_at, last_at, contributions = {}, {}, {}, {}
    station_first = [[] for _ in range(stations)]
    station_last = [[] for _ in range(stations)]
    station_work = [[] for _ in range(stations)]
    work = []
    for i in range(n):
        maximum = min(upper, processing[i] + max(max(forward[i]), max(backward[i])))
        work.append(model.new_int_var(processing[i], maximum, f'work{i}'))
        choices, first_choices, last_choices = [], [], []
        for k in sorted(domains[i]):
            x = model.new_bool_var(f'x{i}_{k}')
            f = model.new_bool_var(f'first{i}_{k}')
            l = model.new_bool_var(f'last{i}_{k}')
            w = model.new_int_var(0, maximum, f'work{i}_{k}')
            assignments[i, k], first_at[i, k], last_at[i, k], contributions[i, k] = x, f, l, w
            choices.append(x)
            first_choices.append(f)
            last_choices.append(l)
            station_first[k].append(f)
            station_last[k].append(l)
            station_work[k].append(w)
            model.add(f <= x)
            model.add(l <= x)
            model.add(w == work[i]).only_enforce_if(x)
            model.add(w == 0).only_enforce_if(x.Not())
        model.add_exactly_one(choices)
        model.add(station[i] == sum(k * assignments[i, k] for k in domains[i]))
        model.add(sum(first_choices) == first[i])
        model.add(sum(last_choices) == last[i])
    for i, j in edges:
        model.add(station[i] <= station[j])
        model.add(rank[i] < rank[j])
    outgoing = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]
    backward_out = [[] for _ in range(n)]
    backward_in = [[] for _ in range(n)]
    task_costs = [[] for _ in range(n)]
    forward_arcs, backward_arcs = {}, {}
    for i in range(n):
        for j in range(n):
            if not domains[i].intersection(domains[j]):
                continue
            if i != j and j not in ancestors[i] and not descendants[i].intersection(ancestors[j]) and processing[i] + forward[i][j] <= upper:
                a = model.new_bool_var(f'a{i}_{j}')
                forward_arcs[i, j] = a
                outgoing[i].append((j, a))
                incoming[j].append(a)
                model.add(station[i] == station[j]).only_enforce_if(a)
                model.add(rank[i] < rank[j]).only_enforce_if(a)
                task_costs[i].append(forward[i][j] * a)
            if j not in descendants[i] and processing[i] + backward[i][j] <= upper:
                h = model.new_bool_var(f'b{i}_{j}')
                backward_arcs[i, j] = h
                backward_out[i].append(h)
                backward_in[j].append(h)
                model.add(station[i] == station[j]).only_enforce_if(h)
                if i != j:
                    model.add(rank[i] > rank[j]).only_enforce_if(h)
                task_costs[i].append(backward[i][j] * h)
    for i in range(n):
        model.add(sum(a for j, a in outgoing[i]) + last[i] == 1)
        model.add(sum(incoming[i]) + first[i] == 1)
        model.add(sum(backward_out[i]) == last[i])
        model.add(sum(backward_in[i]) == first[i])
        model.add(work[i] == processing[i] + sum(task_costs[i]))
    loads = []
    for k in range(stations):
        model.add_exactly_one(station_first[k])
        model.add_exactly_one(station_last[k])
        load = model.new_int_var(0, upper, f'load{k}')
        model.add(load == sum(station_work[k]))
        loads.append(load)
    model.add_max_equality(cycle, loads)
    model.minimize(cycle)
    if best_parts is not None:
        chosen_station = {i: k for k, part in enumerate(best_parts) for i in part}
        positions = {i: p for p, i in enumerate(best_order)}
        chosen_forward = {(i, j) for part in best_parts for i, j in zip(part, part[1:])}
        chosen_backward = {(part[-1], part[0]) for part in best_parts}
        chosen_first = {part[0] for part in best_parts}
        chosen_last = {part[-1] for part in best_parts}
        chosen_work = {i: processing[i] + (forward[i][j] if (i, j) in chosen_forward else backward[i][j]) for i, j in chosen_forward | chosen_backward}
        for i in range(n):
            model.add_hint(station[i], chosen_station[i])
            model.add_hint(rank[i], positions[i])
            model.add_hint(first[i], int(i in chosen_first))
            model.add_hint(last[i], int(i in chosen_last))
            model.add_hint(work[i], chosen_work[i])
        for (i, k), x in assignments.items():
            assigned = int(chosen_station[i] == k)
            model.add_hint(x, assigned)
            model.add_hint(first_at[i, k], int(assigned and i in chosen_first))
            model.add_hint(last_at[i, k], int(assigned and i in chosen_last))
            model.add_hint(contributions[i, k], chosen_work[i] if assigned else 0)
        for pair, a in forward_arcs.items():
            model.add_hint(a, int(pair in chosen_forward))
        for pair, h in backward_arcs.items():
            model.add_hint(h, int(pair in chosen_backward))
        for k, part in enumerate(best_parts):
            model.add_hint(loads[k], sum(chosen_work[i] for i in part))
        model.add_hint(cycle, best['cycle_time'])

    def extract(value):
        parts = []
        for k in range(stations):
            i = next(i for i in range(n) if (i, k) in first_at and value(first_at[i, k]))
            part = [i]
            while not value(last[i]):
                i = next(j for j, a in outgoing[i] if value(a))
                part.append(i)
            parts.append(part)
        return parts

    class Incumbent(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            save(extract(self.value))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    remaining = deadline - time.monotonic()
    if remaining > .05:
        solver.parameters.max_time_in_seconds = remaining
        status = solver.solve(model, Incumbent())
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            save(extract(solver.value))
    if best is None:
        raise RuntimeError('No complete assembly witness within cycle bounds')
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=float, default=60)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    with open(args.instance_path) as f:
        instance = json.load(f)
    out = solve(instance, args.time_limit, args.solution_path)
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
