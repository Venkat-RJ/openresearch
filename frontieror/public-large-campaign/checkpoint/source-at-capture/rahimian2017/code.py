import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
from ortools.sat.python import cp_model


def add_nurse(model, nurse, data, shifts):
    days = data['num_days']
    i = nurse['id']
    x, work = {}, {}
    for day in range(days):
        work[day] = model.NewBoolVar(f'w{i}_{day}')
        for t in shifts:
            x[day, t] = model.NewBoolVar(f'x{i}_{day}_{t}')
        model.Add(sum(x[day, t] for t in shifts) == work[day])
    for t in shifts:
        model.Add(sum(x[day, t] for day in range(days)) <= nurse['max_shifts_per_type'][str(t)])
    minutes = sum(shifts[t] * x[day, t] for day in range(days) for t in shifts)
    model.Add(minutes >= nurse['min_total_minutes'])
    model.Add(minutes <= nurse['max_total_minutes'])
    for day in nurse['day_off_requests']:
        model.Add(work[day] == 0)
    for rotation in data['forbidden_shift_rotations']:
        for day in range(days - 1):
            model.Add(x[day, rotation['shift_type']] + x[day + 1, rotation['cannot_be_followed_by']] <= 1)
    maximum = nurse['max_consecutive_shifts']
    for day in range(days - maximum):
        model.Add(sum(work[j] for j in range(day, day + maximum + 1)) <= maximum)
    for length in range(1, nurse['min_consecutive_shifts']):
        for day in range(days - length - 1):
            model.Add(work[day] + work[day + length + 1] + length - 1 >= sum(work[j] for j in range(day + 1, day + length + 1)))
    weekends = []
    for week in range(1, data['num_weekends'] + 1):
        saturday, sunday = 7 * week - 2, 7 * week - 1
        if sunday < days:
            variable = model.NewBoolVar(f'week{i}_{week}')
            model.AddMaxEquality(variable, [work[saturday], work[sunday]])
            weekends.append(variable)
    model.Add(sum(weekends) <= nurse['max_weekends'])
    return x, work


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'time_limit', 'log_path'):
        parser.add_argument('--' + key, required=key != 'log_path')
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + max(.1, float(args.time_limit) - 3)
    with open(args.instance_path) as stream:
        data = json.load(stream)
    days = data['num_days']
    shifts = {s['id']: s['length_minutes'] for s in data['shift_types']}
    nurses = data['nurses']
    model = cp_model.CpModel()
    x, work, cost = {}, {}, []
    for nurse in nurses:
        nx, nw = add_nurse(model, nurse, data, shifts)
        x.update({(nurse['id'], day, t): v for (day, t), v in nx.items()})
        work.update({(nurse['id'], day): v for day, v in nw.items()})
    for request in data['shift_on_requests']:
        cost.append(request['penalty'] * (1 - x[request['nurse_id'], request['day'], request['shift_type']]))
    for request in data['shift_off_requests']:
        cost.append(request['penalty'] * x[request['nurse_id'], request['day'], request['shift_type']])
    unders, overs = {}, {}
    coverage = {(r['day'], r['shift_type']): r for r in data['coverage_requirements']}
    for request in data['coverage_requirements']:
        day, t = request['day'], request['shift_type']
        count = sum(x[n['id'], day, t] for n in nurses)
        under = model.NewIntVar(0, max(request['preferred'], len(nurses)), f'under{day}_{t}')
        over = model.NewIntVar(0, len(nurses), f'over{day}_{t}')
        model.AddMaxEquality(under, [request['preferred'] - count, 0])
        model.AddMaxEquality(over, [count - request['preferred'], 0])
        cost.extend([request['under_weight'] * under, request['over_weight'] * over])
        unders[day, t], overs[day, t] = under, over
    model.Minimize(sum(cost))
    best = float('inf')
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense='minimize')

    def save(schedule):
        nonlocal best
        counts = {(day, t): 0 for day in range(days) for t in shifts}
        for nurse in nurses:
            for day, t in schedule[str(nurse['id'])].items():
                if t is not None:
                    counts[int(day), t] += 1
        under, over, objective = {}, {}, 0
        for (day, t), r in coverage.items():
            u, o = max(0, r['preferred'] - counts[day, t]), max(0, counts[day, t] - r['preferred'])
            under[f'{day}_{t}'], over[f'{day}_{t}'] = u, o
            objective += r['under_weight'] * u + r['over_weight'] * o
        for r in data['shift_on_requests']:
            if schedule[str(r['nurse_id'])][str(r['day'])] != r['shift_type']:
                objective += r['penalty']
        for r in data['shift_off_requests']:
            if schedule[str(r['nurse_id'])][str(r['day'])] == r['shift_type']:
                objective += r['penalty']
        if objective < best:
            best = objective
            with open(args.solution_path, 'w') as stream:
                json.dump({'objective_value': objective, 'solution': {'schedule': schedule, 'coverage_under': under, 'coverage_over': over}}, stream)
            if logger:
                logger.log(objective)
        return counts

    if days > 30:
        schedule = {}
        counts = {(day, t): 0 for day in range(days) for t in shifts}
        for index, nurse in enumerate(nurses):
            isolated = cp_model.CpModel()
            nx, nw = add_nurse(isolated, nurse, data, shifts)
            isolated_cost = []
            for r in data['shift_on_requests']:
                if r['nurse_id'] == nurse['id']:
                    isolated_cost.append(r['penalty'] * (1 - nx[r['day'], r['shift_type']]))
            for r in data['shift_off_requests']:
                if r['nurse_id'] == nurse['id']:
                    isolated_cost.append(r['penalty'] * nx[r['day'], r['shift_type']])
            for (day, t), r in coverage.items():
                delta = -r['under_weight'] if counts[day, t] < r['preferred'] else r['over_weight']
                isolated_cost.append(delta * nx[day, t])
            isolated.Minimize(sum(isolated_cost))
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.max_time_in_seconds = max(.01, min(.35, (deadline - time.monotonic()) / max(1, len(nurses) - index + 5)))
            status = solver.Solve(isolated)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                isolated.ClearObjective()
                solver.parameters.max_time_in_seconds = max(.01, min(1., deadline - time.monotonic()))
                status = solver.Solve(isolated)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise RuntimeError('No feasible independent nurse schedule: ' + solver.StatusName(status))
            entry = {}
            for day in range(days):
                t = next((t for t in shifts if solver.Value(nx[day, t])), None)
                entry[str(day)] = t
                if t is not None:
                    counts[day, t] += 1
                model.AddHint(work[nurse['id'], day], int(t is not None))
                for shift in shifts:
                    model.AddHint(x[nurse['id'], day, shift], int(t == shift))
            schedule[str(nurse['id'])] = entry
        counts = save(schedule)
        local_deadline = deadline - min(12., float(args.time_limit) * .2)
        passes = 0
        while time.monotonic() < local_deadline - .4:
            improved = False
            for nurse in nurses[passes % len(nurses):] + nurses[:passes % len(nurses)]:
                if time.monotonic() >= local_deadline - .4:
                    break
                previous = schedule[str(nurse['id'])]
                for day, t in previous.items():
                    if t is not None:
                        counts[int(day), t] -= 1
                isolated = cp_model.CpModel()
                nx, nw = add_nurse(isolated, nurse, data, shifts)
                weights = {(day, t): (-r['under_weight'] if counts[day, t] < r['preferred'] else r['over_weight']) for (day, t), r in coverage.items()}
                for r in data['shift_on_requests']:
                    if r['nurse_id'] == nurse['id']:
                        weights[r['day'], r['shift_type']] -= r['penalty']
                for r in data['shift_off_requests']:
                    if r['nurse_id'] == nurse['id']:
                        weights[r['day'], r['shift_type']] += r['penalty']
                isolated.Minimize(sum(weight * nx[day, t] for (day, t), weight in weights.items()))
                for day in range(days):
                    assigned = previous[str(day)]
                    isolated.AddHint(nw[day], int(assigned is not None))
                    for t in shifts:
                        isolated.AddHint(nx[day, t], int(t == assigned))
                local = cp_model.CpSolver()
                local.parameters.num_search_workers = 1
                local.parameters.cp_model_presolve = False
                local.parameters.max_time_in_seconds = min(.3, max(.01, local_deadline - time.monotonic()))
                status = local.Solve(isolated)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    candidate = {str(day): next((t for t in shifts if local.Value(nx[day, t])), None) for day in range(days)}
                    old_cost = sum(weights[int(day), t] for day, t in previous.items() if t is not None)
                    new_cost = sum(weights[int(day), t] for day, t in candidate.items() if t is not None)
                    if new_cost < old_cost:
                        schedule[str(nurse['id'])] = candidate
                        improved = True
                for day, t in schedule[str(nurse['id'])].items():
                    if t is not None:
                        counts[int(day), t] += 1
                if improved:
                    save(schedule)
            passes += 1
            if not improved:
                break
        pair_deadline = deadline - min(2., float(args.time_limit) * .05)
        pair_index = 0
        while time.monotonic() < pair_deadline - .6:
            first = pair_index % len(nurses)
            second = (first + len(nurses) // 2 + pair_index // len(nurses)) % len(nurses)
            pair_index += 1
            if first == second:
                continue
            pair = [nurses[first], nurses[second]]
            previous = {nurse['id']: schedule[str(nurse['id'])] for nurse in pair}
            for entry in previous.values():
                for day, t in entry.items():
                    if t is not None:
                        counts[int(day), t] -= 1
            isolated = cp_model.CpModel()
            pair_x = {}
            for nurse in pair:
                nx, nw = add_nurse(isolated, nurse, data, shifts)
                pair_x[nurse['id']] = nx
                for day in range(days):
                    assigned = previous[nurse['id']][str(day)]
                    isolated.AddHint(nw[day], int(assigned is not None))
                    for t in shifts:
                        isolated.AddHint(nx[day, t], int(t == assigned))
            pair_cost = []
            for (day, t), r in coverage.items():
                a, b = (pair_x[nurse['id']][day, t] for nurse in pair)
                gap = r['preferred'] - counts[day, t]
                if gap >= 2:
                    pair_cost.append(-r['under_weight'] * (a + b))
                elif gap <= 0:
                    pair_cost.append(r['over_weight'] * (a + b))
                else:
                    both = isolated.NewBoolVar(f'both{day}_{t}')
                    isolated.Add(both <= a)
                    isolated.Add(both <= b)
                    isolated.Add(both >= a + b - 1)
                    isolated.AddHint(both, int(all(previous[nurse['id']][str(day)] == t for nurse in pair)))
                    pair_cost.append(-r['under_weight'] * (a + b) + (r['under_weight'] + r['over_weight']) * both)
            for r in data['shift_on_requests']:
                if r['nurse_id'] in pair_x:
                    pair_cost.append(-r['penalty'] * pair_x[r['nurse_id']][r['day'], r['shift_type']])
            for r in data['shift_off_requests']:
                if r['nurse_id'] in pair_x:
                    pair_cost.append(r['penalty'] * pair_x[r['nurse_id']][r['day'], r['shift_type']])
            isolated.Minimize(sum(pair_cost))
            local = cp_model.CpSolver()
            local.parameters.num_search_workers = 1
            local.parameters.cp_model_presolve = False
            local.parameters.max_time_in_seconds = min(.5, max(.01, pair_deadline - time.monotonic()))
            status = local.Solve(isolated)
            previous_best = best
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                for nurse in pair:
                    schedule[str(nurse['id'])] = {str(day): next((t for t in shifts if local.Value(pair_x[nurse['id']][day, t])), None) for day in range(days)}
                counts = save(schedule)
            if best >= previous_best:
                for nurse in pair:
                    schedule[str(nurse['id'])] = previous[nurse['id']]
                counts = save(schedule)
        model.ClearHints()
        for nurse in nurses:
            for day in range(days):
                assigned = schedule[str(nurse['id'])][str(day)]
                model.AddHint(work[nurse['id'], day], int(assigned is not None))
                for t in shifts:
                    model.AddHint(x[nurse['id'], day, t], int(t == assigned))
        for (day, t), variable in unders.items():
            model.AddHint(variable, max(0, coverage[day, t]['preferred'] - counts[day, t]))
        for (day, t), variable in overs.items():
            model.AddHint(variable, max(0, counts[day, t] - coverage[day, t]['preferred']))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(.01, deadline - time.monotonic())

    class Incumbent(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            schedule = {str(nurse['id']): {str(day): next((t for t in shifts if self.Value(x[nurse['id'], day, t])), None) for day in range(days)} for nurse in nurses}
            save(schedule)

    status = solver.Solve(model, Incumbent())
    if best == float('inf'):
        raise RuntimeError(solver.StatusName(status))


if __name__ == '__main__':
    main()
