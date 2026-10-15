import time
STARTED = time.monotonic()
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import numpy as np
import highspy
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'log_path'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    args = parser.parse_args()
    deadline = STARTED + args.time_limit - min(5.0, max(0.3, args.time_limit * 0.1))
    data = json.load(open(args.instance_path))
    items = data['parameters']['NI']
    machines = data['parameters']['NK']
    periods = data['parameters']['NT']
    costs = data['costs']
    capacity = np.asarray(data['machine_data']['capacity_C'], dtype=float)
    startup_time = np.asarray(data['machine_data']['startup_time_sigma'], dtype=float)
    demand = np.asarray(data['demand'], dtype=float)
    initial = np.asarray(data['initial_conditions']['initial_stock_s0'], dtype=float) - np.asarray(data['initial_conditions']['initial_backlog_r0'], dtype=float)
    holding = np.asarray(costs['holding_cost_h'], dtype=float)
    penalty = np.asarray(costs['backlogging_cost_e'], dtype=float)
    setup_cost = costs['setup_cost_f']
    startup_cost = costs['startup_cost_g']
    logger = SolutionLogger(args.log_path, sense='minimize')
    objective = []
    upper = []
    integer = []
    maps = {name: {} for name in ('production', 'stock', 'backlog', 'setup', 'startup')}
    variables = {}

    def variable(name, key, cost=0, bound=highspy.kHighsInf, binary=False):
        index = len(objective)
        variables[name, key] = index
        maps[name][key] = index
        objective.append(cost)
        upper.append(bound)
        integer.append(binary)
        return index

    for i in range(items):
        for t in range(periods):
            variable('stock', f'{i}_{t}', holding[i])
            variable('backlog', f'{i}_{t}', penalty[i])
            for k in range(machines):
                key = f'{i}_{k}_{t}'
                variable('production', key, bound=capacity[k])
                variable('setup', key, setup_cost, 1, True)
                variable('startup', key, startup_cost, 1, True)
    objective = np.asarray(objective, dtype=float)
    upper = np.asarray(upper, dtype=float)
    binary_indices = np.flatnonzero(integer).astype(np.int32)
    best = None
    best_objective = np.inf

    def get(name, i, t, k=None):
        return variables[name, f'{i}_{t}' if k is None else f'{i}_{k}_{t}']

    def save(values):
        nonlocal best, best_objective
        obj = float(objective @ values)
        if obj >= best_objective:
            return
        solution = {name: {key: float(values[index]) for key, index in mapping.items()} for name, mapping in maps.items()}
        solution['objective_value'] = obj
        with open(args.solution_path, 'w') as output:
            json.dump(solution, output)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(obj, solution)
        else:
            logger.log(obj)
        best = np.asarray(values, dtype=float).copy()
        best_objective = obj
        print('incumbent elapsed=%.3f objective=%.6f' % (time.monotonic() - STARTED, obj), flush=True)

    baseline = np.zeros(len(objective))
    for i in range(items):
        net = initial[i]
        for t in range(periods):
            net -= demand[i, t]
            baseline[get('stock', i, t)] = max(0, net)
            baseline[get('backlog', i, t)] = max(0, -net)
    save(baseline)
    starts = [0]
    indices = []
    coefficients = []
    row_lower = []
    row_upper = []

    def add(row, lower=-highspy.kHighsInf, upper_bound=highspy.kHighsInf):
        row_lower.append(lower)
        row_upper.append(upper_bound)
        for index, value in row.items():
            indices.append(index)
            coefficients.append(value)
        starts.append(len(indices))

    for i in range(items):
        for t in range(periods):
            row = {get('stock', i, t): -1, get('backlog', i, t): 1}
            for k in range(machines):
                row[get('production', i, t, k)] = 1
            rhs = demand[i, t]
            if t:
                row[get('stock', i, t - 1)] = 1
                row[get('backlog', i, t - 1)] = -1
            else:
                rhs -= initial[i]
            add(row, rhs, rhs)
            for k in range(machines):
                x = get('production', i, t, k)
                y = get('setup', i, t, k)
                z = get('startup', i, t, k)
                add({x: 1, z: startup_time[k], y: -capacity[k]}, upper_bound=0)
                row = {y: 1, z: -1}
                if t:
                    row[get('setup', i, t - 1, k)] = -1
                add(row, upper_bound=0)
                add({z: 1, y: -1}, upper_bound=0)
                row = {z: 1}
                if t:
                    row[get('setup', i, t - 1, k)] = 1
                for j in range(items):
                    if j != i:
                        row[get('setup', j, t, k)] = 1
                        row[get('startup', j, t, k)] = -1
                add(row, upper_bound=1)
    for k in range(machines):
        for t in range(periods):
            add({get('setup', i, t, k): 1 for i in range(items)}, upper_bound=1)
    starts = np.asarray(starts, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    coefficients = np.asarray(coefficients, dtype=float)
    row_lower = np.asarray(row_lower, dtype=float)
    row_upper = np.asarray(row_upper, dtype=float)

    def build(fixed_schedule=None, window=None):
        solver = highspy.Highs()
        solver.setOptionValue('threads', 1)
        solver.setOptionValue('output_flag', False)
        lower = np.zeros(len(objective))
        bounds = upper.copy()
        if fixed_schedule is not None:
            for i in range(items):
                for k in range(machines):
                    for t in range(periods):
                        if window is not None and window[0] <= t < window[1]:
                            continue
                        index = get('setup', i, t, k)
                        value = float(fixed_schedule[k, t] == i)
                        lower[index] = bounds[index] = value
                        if window is None:
                            index = get('startup', i, t, k)
                            value = float(fixed_schedule[k, t] == i and (t == 0 or fixed_schedule[k, t - 1] != i))
                            lower[index] = bounds[index] = value
        solver.addCols(len(objective), objective, lower, bounds, 0, np.zeros(len(objective) + 1, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0))
        solver.addRows(len(row_lower), row_lower, row_upper, len(indices), starts, indices, coefficients)
        if fixed_schedule is None or window is not None:
            solver.changeColsIntegrality(len(binary_indices), binary_indices, np.ones(len(binary_indices), dtype=np.uint8))
            initial_solution = highspy.HighsSolution()
            initial_solution.col_value = best.tolist()
            initial_solution.value_valid = True
            solver.setSolution(initial_solution)
        return solver

    def run(solver, phase_deadline, label):
        if time.monotonic() >= phase_deadline:
            return

        def interrupt(event):
            if time.monotonic() >= phase_deadline:
                event.interrupt()

        def improving(event):
            save(np.asarray(event.data_out.mip_solution, dtype=float))

        solver.cbMipInterrupt += interrupt
        solver.cbSimplexInterrupt += interrupt
        solver.cbIpmInterrupt += interrupt
        solver.cbMipImprovingSolution += improving
        solver.setOptionValue('time_limit', max(0.01, phase_deadline - time.monotonic()))
        print('%s_start elapsed=%.3f' % (label, time.monotonic() - STARTED), flush=True)
        solver.run()
        print('%s_end elapsed=%.3f status=%s' % (label, time.monotonic() - STARTED, solver.getModelStatus()), flush=True)
        if solver.getSolution().value_valid and solver.getInfo().max_primal_infeasibility <= 1e-6:
            save(np.asarray(solver.getSolution().col_value, dtype=float))

    def dispatch(lookahead):
        schedule = np.full((machines, periods), -1, dtype=np.int32)
        net = initial.copy()
        previous = [-1] * machines
        for t in range(periods):
            horizon = min(periods, t + lookahead)
            for k in range(machines):
                chosen = -1
                chosen_quantity = 0.0
                greatest = 0.0
                for i in range(items):
                    production_capacity = max(0, capacity[k] - (startup_time[k] if previous[k] != i else 0))
                    quantity = min(production_capacity, max(0, float(demand[i, t:horizon].sum()) - net[i]))
                    if quantity <= 0:
                        continue
                    before = net[i]
                    after = before + quantity
                    savings = -setup_cost - (startup_cost if previous[k] != i else 0)
                    for s in range(t, horizon):
                        before -= demand[i, s]
                        after -= demand[i, s]
                        savings += holding[i] * (max(0, before) - max(0, after)) + penalty[i] * (max(0, -before) - max(0, -after))
                    if savings > greatest:
                        greatest = savings
                        chosen = i
                        chosen_quantity = quantity
                if chosen >= 0:
                    schedule[k, t] = chosen
                    net[chosen] += chosen_quantity
                previous[k] = chosen
            net -= demand[:, t]
        return schedule

    heuristic_deadline = min(deadline, time.monotonic() + min(8.0, args.time_limit * 0.2))
    for lookahead in (2, 3, 4, 6, 8, 1):
        if time.monotonic() >= heuristic_deadline:
            break
        schedule = dispatch(lookahead)
        solver = build(schedule)
        run(solver, min(heuristic_deadline, time.monotonic() + max(0.1, args.time_limit * 0.025)), 'dispatch_%d' % lookahead)
    print('construction_end elapsed=%.3f objective=%.6f' % (time.monotonic() - STARTED, best_objective), flush=True)
    if periods > 12:
        neighborhood_deadline = min(deadline, time.monotonic() + min(12.0, args.time_limit * 0.2))
        for start in range(0, periods, 4):
            if time.monotonic() >= neighborhood_deadline:
                break
            schedule = np.full((machines, periods), -1, dtype=np.int32)
            for k in range(machines):
                for t in range(periods):
                    for i in range(items):
                        if best[get('setup', i, t, k)] > 0.5:
                            schedule[k, t] = i
            solver = build(schedule, (start, min(periods, start + 6)))
            run(solver, min(neighborhood_deadline, time.monotonic() + max(0.1, args.time_limit * 0.025)), 'window_%d' % start)
    if time.monotonic() < deadline:
        solver = build()
        run(solver, deadline, 'full_mip')
    print('finished elapsed=%.3f objective=%.6f' % (time.monotonic() - STARTED, best_objective), flush=True)


if __name__ == '__main__':
    main()
