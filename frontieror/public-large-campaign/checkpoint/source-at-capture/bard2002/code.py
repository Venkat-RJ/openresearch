import time
STARTED = time.monotonic()
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import math
import highspy
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'log_path'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    args = parser.parse_args()
    deadline = STARTED + args.time_limit - min(2.0, args.time_limit * 0.1)
    data = json.load(open(args.instance_path))
    nodes = [data['depot']] + data['customers']
    n = len(nodes) - 1
    capacity = data['vehicle_capacity']
    demand = [node['demand'] for node in nodes]
    ready = [node['ready_time'] for node in nodes]
    due = [node['due_date'] for node in nodes]
    distance = data.get('distance_matrix') or [[math.floor(math.hypot(i['x'] - j['x'], i['y'] - j['y']) * 10) / 10 for j in nodes] for i in nodes]
    folded = data.get('travel_time_includes_service_time', False)
    tau = [[max(distance[i][j] + (0 if folded else nodes[i]['service_time']), ready[j] - due[i])
            for j in range(n + 1)] for i in range(n + 1)]
    logger = SolutionLogger(args.log_path, sense='minimize')
    best = None
    best_distance = math.inf

    def route_times(route):
        clock = ready[0]
        previous = 0
        times = {}
        for customer in route:
            clock = max(ready[customer], clock + tau[previous][customer])
            if clock > due[customer] + 1e-8:
                return None
            times[customer] = clock
            previous = customer
        if clock + tau[previous][0] > due[0] + 1e-8:
            return None
        return times

    def admissible(route):
        suffix = 0
        for customer in reversed(route):
            # Nonselected load rows use Q-q_j, not the usual Q.
            if 2 * demand[customer] + suffix > capacity + 1e-8:
                return False
            suffix += demand[customer]
        return suffix <= capacity and route_times(route) is not None

    def route_witness(routes):
        maximum = max(sum(demand[i] for i in route) for route in routes)
        times = {}
        loads = {}
        for route in routes:
            rt = route_times(route)
            if rt is None:
                return None
            times.update(rt)
            previous_load = 0
            for customer in route:
                previous_load = max(previous_load + demand[customer], maximum + 2 * demand[customer] - capacity)
                if previous_load > maximum + 1e-8:
                    return None
                loads[customer] = previous_load
        return times, loads

    def save(routes, times=None, loads=None):
        nonlocal best, best_distance
        total = round(sum(distance[i][j] for route in routes for i, j in zip([0] + route, route + [0])), 1)
        if best is not None and (len(routes), total) >= (len(best), best_distance):
            return
        if times is None:
            witness = route_witness(routes)
            if witness is None:
                return
            times, loads = witness
        solution = {'objective_value': len(routes), 'num_vehicles': len(routes),
                    'routes': [[0] + route + [0] for route in routes], 'total_distance': total,
                    'departure_times': {str(i): float(v) for i, v in times.items()},
                    'loads': {str(i): float(v) for i, v in loads.items()}}
        best = [route[:] for route in routes]
        best_distance = total
        with open(args.solution_path, 'w') as output:
            json.dump(solution, output)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(len(routes), solution)
        else:
            logger.log(len(routes))

    routes = [[i] for i in range(1, n + 1)]
    save(routes)
    heuristic_deadline = min(deadline, time.monotonic() + min(5.0, args.time_limit * 0.12))
    savings = sorted(((distance[i][0] + distance[0][j] - distance[i][j], i, j)
                      for i in range(1, n + 1) for j in range(1, n + 1) if i != j), reverse=True)
    owner = {i: i for i in range(1, n + 1)}
    active = {i: [i] for i in range(1, n + 1)}
    for _, i, j in savings:
        if time.monotonic() >= heuristic_deadline:
            break
        left, right = owner[i], owner[j]
        if left == right or active[left][-1] != i or active[right][0] != j:
            continue
        joined = active[left] + active[right]
        if admissible(joined):
            active[left] = joined
            del active[right]
            for customer in joined:
                owner[customer] = left
            save(list(active.values()))
    routes = [route[:] for route in best] if best is not None else list(active.values())
    changed = True
    while changed and time.monotonic() < heuristic_deadline:
        changed = False
        for source in sorted(range(len(routes)), key=lambda i: len(routes[i])):
            trial = [route[:] for i, route in enumerate(routes) if i != source]
            success = True
            for customer in sorted(routes[source], key=lambda i: due[i]):
                candidate = None
                for r, route in enumerate(trial):
                    for pos in range(len(route) + 1):
                        if time.monotonic() >= heuristic_deadline:
                            success = False
                            break
                        proposal = route[:pos] + [customer] + route[pos:]
                        if not admissible(proposal):
                            continue
                        left = route[pos - 1] if pos else 0
                        right = route[pos] if pos < len(route) else 0
                        increase = distance[left][customer] + distance[customer][right] - distance[left][right]
                        if candidate is None or increase < candidate[0]:
                            candidate = (increase, r, proposal)
                    if not success:
                        break
                if candidate is None or not success:
                    success = False
                    break
                trial[candidate[1]] = candidate[2]
            if success:
                routes = trial
                save(routes)
                changed = True
                break
            if time.monotonic() >= heuristic_deadline:
                break
    print('construction_end elapsed=%.3f vehicles=%s' % (time.monotonic() - STARTED, None if best is None else len(best)), flush=True)
    if time.monotonic() >= deadline:
        return

    highs = highspy.Highs()
    highs.setOptionValue('threads', 1)
    highs.setOptionValue('output_flag', False)
    weight = 2 * n * max(max(row) for row in distance) + 1
    arcs = [(i, j) for i in range(n + 1) for j in range(n + 1)
            if i != j and ready[i] + tau[i][j] <= due[j] + 1e-8
            and (not i or not j or demand[i] + demand[j] <= capacity)]
    x = {(i, j): highs.addVariable(lb=0, ub=1, obj=(weight if i == 0 else 0) + distance[i][j],
                                  type=highspy.HighsVarType.kInteger) for i, j in arcs}
    times = {i: highs.addVariable(lb=ready[i], ub=due[i]) for i in range(1, n + 1)}
    loads = {i: highs.addVariable(lb=demand[i], ub=capacity) for i in range(1, n + 1)}
    outgoing = {i: [] for i in range(n + 1)}
    incoming = {i: [] for i in range(n + 1)}
    for (i, j), variable in x.items():
        outgoing[i].append(variable)
        incoming[j].append(variable)
    for i in range(1, n + 1):
        highs.addConstr(sum(outgoing[i]) == 1)
        highs.addConstr(sum(incoming[i]) == 1)
    highs.addConstr(sum(outgoing[0]) == sum(incoming[0]))
    for i in range(1, n + 1):
        if time.monotonic() >= deadline:
            return
        for j in range(1, n + 1):
            if i == j:
                continue
            variable = x.get((i, j), 0)
            if (i, j) in x:
                big_time = due[i] - ready[j]
                highs.addConstr(times[j] >= times[i] - big_time + (tau[i][j] + big_time) * variable)
            highs.addConstr(loads[j] >= loads[i] + 2 * demand[j] - capacity + (capacity - demand[j]) * variable)
        if (0, i) in x:
            big = max(0, ready[0] + tau[0][i] - ready[i])
            highs.addConstr(times[i] >= ready[0] + tau[0][i] - big * (1 - x[0, i]))
        if (i, 0) in x:
            big = max(0, due[i] + tau[i][0] - due[0])
            highs.addConstr(times[i] + tau[i][0] <= due[0] + big * (1 - x[i, 0]))
    if best is not None:
        witness_times, witness_loads = route_witness(best)
        used = {(i, j) for route in best for i, j in zip([0] + route, route + [0])}
        start = highspy.HighsSolution()
        initial_values = [0.0] * highs.getNumCol()
        for arc, variable in x.items():
            initial_values[int(variable)] = float(arc in used)
        for i in range(1, n + 1):
            initial_values[int(times[i])] = witness_times[i]
            initial_values[int(loads[i])] = witness_loads[i]
        start.col_value = initial_values
        start.value_valid = True
        highs.setSolution(start)

    def emit(values):
        selected = [(i, j) for (i, j), variable in x.items() if values[int(variable)] > 0.5]
        successor = {i: j for i, j in selected if i}
        decoded = []
        covered = set()
        for i, first in selected:
            if i:
                continue
            route = []
            current = first
            while current:
                if current in covered or current not in successor:
                    return
                covered.add(current)
                route.append(current)
                current = successor[current]
            decoded.append(route)
        if len(covered) != n:
            return
        save(decoded, {i: values[int(variable)] for i, variable in times.items()},
             {i: values[int(variable)] for i, variable in loads.items()})

    def interrupt(event):
        if time.monotonic() >= deadline:
            event.interrupt()

    def improving(event):
        emit(event.data_out.mip_solution)

    highs.cbMipInterrupt += interrupt
    highs.cbSimplexInterrupt += interrupt
    highs.cbIpmInterrupt += interrupt
    highs.cbMipImprovingSolution += improving
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    highs.setOptionValue('time_limit', remaining)
    print('mip_start elapsed=%.3f budget=%.3f arcs=%d' % (time.monotonic() - STARTED, remaining, len(x)), flush=True)
    highs.run()
    print('mip_end elapsed=%.3f status=%s' % (time.monotonic() - STARTED, highs.getModelStatus()), flush=True)
    if highs.getSolution().value_valid:
        emit(highs.getSolution().col_value)


if __name__ == '__main__':
    main()
