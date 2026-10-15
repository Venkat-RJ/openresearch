import time
STARTED = time.monotonic()
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
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
    jobs = data['jobs']
    count = len(jobs)
    machine_count = data['num_machines']
    machine_ids = sorted({op['machine'] for job in jobs for op in job['operations']})
    machine_index = {value: index for index, value in enumerate(machine_ids)}
    operations = []
    job_operations = []
    duration = []
    machine = []
    job_of = []
    remaining = []
    for j, job in enumerate(jobs):
        ids = []
        rest = sum(op['processing_time'] for op in job['operations'])
        for k, op in enumerate(job['operations']):
            ids.append(len(operations))
            operations.append((j, k))
            job_of.append(j)
            duration.append(op['processing_time'])
            machine.append(machine_index[op['machine']])
            remaining.append(rest)
            rest -= op['processing_time']
        job_operations.append(ids)
    size = len(operations)
    horizon = sum(duration) + max(job['release_date'] for job in jobs)
    predecessor_job = [-1] * size
    successor_job = [-1] * size
    for ids in job_operations:
        for left, right in zip(ids, ids[1:]):
            predecessor_job[right] = left
            successor_job[left] = right
    logger = SolutionLogger(args.log_path, sense='minimize')
    best = None
    best_objective = math.inf
    rng = random.Random(2021)

    def objective(starts):
        return sum(job['weight'] * max(0, starts[job_operations[j][-1]] + duration[job_operations[j][-1]] - job['due_date'])
                   for j, job in enumerate(jobs))

    def save(starts):
        nonlocal best, best_objective
        value = objective(starts)
        if value >= best_objective:
            return
        schedule = []
        for j, job in enumerate(jobs):
            completion = starts[job_operations[j][-1]] + duration[job_operations[j][-1]]
            schedule.append({'job_id': job['job_id'], 'completion_time': completion,
                             'tardiness': max(0, completion - job['due_date']),
                             'operations': [{'machine': job['operations'][k]['machine'], 'start_time': int(starts[op])}
                                            for k, op in enumerate(job_operations[j])]})
        solution = {'objective_value': value, 'schedule': schedule}
        with open(args.solution_path, 'w') as output:
            json.dump(solution, output)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(value, solution)
        else:
            logger.log(value)
        best = list(starts)
        best_objective = value
        print('incumbent elapsed=%.3f objective=%d' % (time.monotonic() - STARTED, value), flush=True)

    starts = [0] * size
    clock = 0
    for j in sorted(range(count), key=lambda j: jobs[j]['due_date'] / jobs[j]['weight']):
        clock = max(clock, jobs[j]['release_date'])
        for op in job_operations[j]:
            starts[op] = clock
            clock += duration[op]
    save(starts)
    heuristic_deadline = min(deadline, time.monotonic() + min(10.0, args.time_limit * 0.2))

    def dispatch(kappa, exponent, randomized):
        next_operation = [0] * count
        job_ready = [job['release_date'] for job in jobs]
        machine_ready = [0] * len(machine_ids)
        result = [0] * size
        multipliers = [rng.uniform(0.75, 1.3) if randomized else 1.0 for _ in jobs]
        average = sum(duration) / max(1, size)
        for _ in range(size):
            available = []
            for j in range(count):
                if next_operation[j] < len(job_operations[j]):
                    op = job_operations[j][next_operation[j]]
                    earliest = max(job_ready[j], machine_ready[machine[op]])
                    available.append((earliest + duration[op], earliest, j, op))
            finish, _, _, pivot = min(available)
            conflict = [entry for entry in available if machine[entry[3]] == machine[pivot] and entry[1] < finish + 1e-8]

            def priority(entry):
                _, earliest, j, op = entry
                slack = max(0, jobs[j]['due_date'] - earliest - remaining[op])
                return math.log(jobs[j]['weight'] * multipliers[j]) - exponent * math.log(max(1, duration[op])) - (1 - exponent) * math.log(max(1, remaining[op])) - slack / (kappa * average)

            _, earliest, j, op = max(conflict, key=priority)
            result[op] = earliest
            job_ready[j] = earliest + duration[op]
            machine_ready[machine[op]] = job_ready[j]
            next_operation[j] += 1
        return result

    for exponent in (0.0, 0.5, 1.0):
        for kappa in (0.25, 0.5, 1.0, 2.0, 4.0):
            for randomized in (False, True):
                if time.monotonic() >= heuristic_deadline:
                    break
                save(dispatch(kappa, exponent, randomized))

    def job_order_schedule(order):
        calendar = [[] for _ in machine_ids]
        result = [0] * size
        for j in order:
            ready = jobs[j]['release_date']
            for op in job_operations[j]:
                entries = calendar[machine[op]]
                position = 0
                clock = ready
                while position < len(entries):
                    left, right = entries[position]
                    if clock + duration[op] <= left:
                        break
                    clock = max(clock, right)
                    position += 1
                result[op] = clock
                ready = clock + duration[op]
                entries.insert(position, (clock, ready))
        return result

    if size > 60:
        job_order_best = None
        job_order_value = math.inf
        total_processing = [sum(duration[op] for op in ids) for ids in job_operations]
        orderings = [sorted(range(count), key=lambda j: total_processing[j] / jobs[j]['weight']),
                     sorted(range(count), key=lambda j: jobs[j]['due_date'] / jobs[j]['weight']),
                     sorted(range(count), key=lambda j: (jobs[j]['due_date'] - jobs[j]['release_date']) / jobs[j]['weight'])]
        for power in (0.5, 1.0, 1.5, 2.0):
            orderings.append(sorted(range(count), key=lambda j: total_processing[j] / (jobs[j]['weight'] ** power)))
        for order in orderings:
            trial = job_order_schedule(order)
            value = objective(trial)
            save(trial)
            if value < job_order_value:
                job_order_best, job_order_value = order[:], value
        order = job_order_best[:]
        current_value = job_order_value
        permutation_deadline = min(heuristic_deadline, time.monotonic() + min(5.0, args.time_limit * 0.1))
        moves = 0
        while time.monotonic() < permutation_deadline:
            proposal = order[:]
            left, right = rng.sample(range(count), 2)
            if rng.random() < 0.8:
                proposal.insert(right, proposal.pop(left))
            else:
                proposal[left], proposal[right] = proposal[right], proposal[left]
            trial = job_order_schedule(proposal)
            value = objective(trial)
            temperature = max(1.0, job_order_value * (0.002 + 0.008 * (1 - (moves % 400) / 400)))
            if value <= current_value or rng.random() < math.exp(min(0, (current_value - value) / temperature)):
                order, current_value = proposal, value
            if value < job_order_value:
                job_order_best, job_order_value = proposal[:], value
                save(trial)
            moves += 1
            if moves % 400 == 0:
                order = job_order_best[:]
                current_value = job_order_value
                for _ in range(3):
                    left, right = rng.sample(range(count), 2)
                    order.insert(right, order.pop(left))
                current_value = objective(job_order_schedule(order))
        print('job_permutation_end elapsed=%.3f objective=%d moves=%d' % (time.monotonic() - STARTED, job_order_value, moves), flush=True)

    def orders_from(starts):
        orders = [[] for _ in machine_ids]
        for op in range(size):
            orders[machine[op]].append(op)
        for order in orders:
            order.sort(key=lambda op: (starts[op], op))
        return orders

    def evaluate(orders):
        previous = [-1] * size
        following = [-1] * size
        for order in orders:
            for left, right in zip(order, order[1:]):
                previous[right] = left
                following[left] = right
        degree = [int(predecessor_job[op] >= 0) + int(previous[op] >= 0) for op in range(size)]
        result = [jobs[job_of[op]]['release_date'] if predecessor_job[op] < 0 else 0 for op in range(size)]
        queue = [op for op in range(size) if degree[op] == 0]
        position = 0
        while position < len(queue):
            op = queue[position]
            position += 1
            finish = result[op] + duration[op]
            for successor in (successor_job[op], following[op]):
                if successor >= 0:
                    result[successor] = max(result[successor], finish)
                    degree[successor] -= 1
                    if degree[successor] == 0:
                        queue.append(successor)
        if position != size:
            return None
        return result, previous

    current_orders = orders_from(best)
    current = best[:]
    tabu = {}
    iteration = 0
    while time.monotonic() < heuristic_deadline and size > 30:
        evaluated = evaluate(current_orders)
        if evaluated is None:
            break
        current, previous = evaluated
        candidates = set()
        traced = set()
        stack = [job_operations[j][-1] for j, job in enumerate(jobs)
                 if current[job_operations[j][-1]] + duration[job_operations[j][-1]] > job['due_date']]
        positions = {op: (m, pos) for m, order in enumerate(current_orders) for pos, op in enumerate(order)}
        while stack:
            op = stack.pop()
            if op in traced:
                continue
            traced.add(op)
            for predecessor in (predecessor_job[op], previous[op]):
                if predecessor >= 0 and current[predecessor] + duration[predecessor] == current[op]:
                    stack.append(predecessor)
                    if predecessor == previous[op]:
                        m, pos = positions[op]
                        candidates.add((m, pos - 1))
        if not candidates:
            candidates = {(m, pos) for m, order in enumerate(current_orders) for pos in range(len(order) - 1)}
        candidates = list(candidates)
        rng.shuffle(candidates)
        choice = None
        for m, pos in candidates[:100]:
            if time.monotonic() >= heuristic_deadline:
                break
            order = current_orders[m]
            left, right = order[pos], order[pos + 1]
            order[pos], order[pos + 1] = right, left
            trial = evaluate(current_orders)
            order[pos], order[pos + 1] = left, right
            if trial is None:
                continue
            value = objective(trial[0])
            if tabu.get((left, right), -1) > iteration and value >= best_objective:
                continue
            if choice is None or value < choice[0]:
                choice = (value, m, pos, trial[0], left, right)
        if choice is None:
            current = dispatch(rng.choice((0.5, 1.0, 2.0)), rng.choice((0, 0.5, 1)), True)
            current_orders = orders_from(current)
        else:
            _, m, pos, current, left, right = choice
            current_orders[m][pos], current_orders[m][pos + 1] = right, left
            tabu[right, left] = iteration + rng.randint(5, 12)
            save(current)
        iteration += 1
    print('construction_end elapsed=%.3f objective=%d iterations=%d' % (time.monotonic() - STARTED, best_objective, iteration), flush=True)

    def solve(phase_deadline, relaxed=None, label='full', free_machines=None):
        if time.monotonic() >= phase_deadline or best_objective == 0:
            return
        model = cp_model.CpModel()
        start_variables = []
        end_variables = []
        intervals = [[] for _ in machine_ids]
        for op in range(size):
            start = model.NewIntVar(0, horizon, 's%d' % op)
            end = model.NewIntVar(0, horizon, 'e%d' % op)
            interval = model.NewIntervalVar(start, duration[op], end, 'iv%d' % op)
            start_variables.append(start)
            end_variables.append(end)
            intervals[machine[op]].append(interval)
            if predecessor_job[op] >= 0:
                model.Add(start >= end_variables[predecessor_job[op]])
            else:
                model.Add(start >= jobs[job_of[op]]['release_date'])
            model.AddHint(start, best[op])
            model.AddHint(end, best[op] + duration[op])
        for machine_intervals in intervals:
            model.AddNoOverlap(machine_intervals)
        if relaxed is not None or free_machines is not None:
            for m, order in enumerate(orders_from(best)):
                if free_machines is not None:
                    retained = [] if m in free_machines else order
                else:
                    retained = [op for op in order if job_of[op] not in relaxed]
                for left, right in zip(retained, retained[1:]):
                    model.Add(start_variables[right] >= end_variables[left])
        terms = []
        for j, job in enumerate(jobs):
            tardiness = model.NewIntVar(0, horizon, 't%d' % j)
            model.AddMaxEquality(tardiness, [0, end_variables[job_operations[j][-1]] - job['due_date']])
            terms.append(job['weight'] * tardiness)
            model.AddHint(tardiness, max(0, best[job_operations[j][-1]] + duration[job_operations[j][-1]] - job['due_date']))
        model.Add(sum(terms) <= best_objective)
        model.Minimize(sum(terms))

        class Incumbent(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                save([self.Value(variable) for variable in start_variables])

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = rng.randrange(1, 1000000)
        solver.parameters.max_time_in_seconds = max(0.01, phase_deadline - time.monotonic())
        print('%s_start elapsed=%.3f budget=%.3f relaxed=%s machines=%s' % (label, time.monotonic() - STARTED, solver.parameters.max_time_in_seconds, None if relaxed is None else len(relaxed), None if free_machines is None else len(free_machines)), flush=True)
        status = solver.Solve(model, Incumbent())
        print('%s_end elapsed=%.3f status=%s' % (label, time.monotonic() - STARTED, solver.StatusName(status)), flush=True)

    if size <= 60:
        solve(deadline)
    else:
        solve(min(deadline, time.monotonic() + min(4.0, args.time_limit * 0.08)), label='initial_full')
        round_number = 0
        while time.monotonic() < deadline and best_objective > 0:
            scores = [(jobs[j]['weight'] * max(0, best[job_operations[j][-1]] + duration[job_operations[j][-1]] - jobs[j]['due_date']), j) for j in range(count)]
            scores.sort(reverse=True)
            group_size = min(count, (6, 10, 14, 18)[round_number % 4])
            if round_number % 3 == 0:
                relaxed = {j for _, j in scores[:group_size // 2]}
                pool = [j for j in range(count) if j not in relaxed]
                relaxed.update(rng.sample(pool, group_size - len(relaxed)))
            else:
                relaxed = set(rng.sample(range(count), group_size))
            if round_number % 3 != 2:
                machine_scores = [0.0] * len(machine_ids)
                for m, order in enumerate(orders_from(best)):
                    for left, right in zip(order, order[1:]):
                        if best[left] + duration[left] == best[right]:
                            j = job_of[right]
                            tardiness = max(0, best[job_operations[j][-1]] + duration[job_operations[j][-1]] - jobs[j]['due_date'])
                            machine_scores[m] += jobs[j]['weight'] * tardiness
                ranked = sorted(range(len(machine_ids)), key=lambda m: -machine_scores[m])
                width = min(len(machine_ids), 1 + (round_number // 3) % 4)
                if round_number % 2:
                    free = set(rng.sample(range(len(machine_ids)), width))
                else:
                    free = set(ranked[:width])
                solve(min(deadline, time.monotonic() + min(2.0, max(0.2, args.time_limit * 0.035))),
                      label='machine_lns_%d' % round_number, free_machines=free)
            else:
                solve(min(deadline, time.monotonic() + min(2.0, max(0.2, args.time_limit * 0.035))), relaxed, 'job_lns_%d' % round_number)
            round_number += 1
    print('finished elapsed=%.3f objective=%d' % (time.monotonic() - STARTED, best_objective), flush=True)


if __name__ == '__main__':
    main()
