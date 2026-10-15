import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
_PROCESS_START = time.monotonic()
import numpy as np
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'log_path'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    args = parser.parse_args()
    deadline = _PROCESS_START + max(0., args.time_limit - 5.)
    with open(args.instance_path) as handle:
        data = json.load(handle)
    n = data['n_clients']
    servers = data['n_servers']
    original_linear = data['linear_costs']
    linear = np.array(original_linear, dtype=float)
    logger = SolutionLogger(args.log_path, sense='minimize')
    pairs = []
    seen = set()
    for i, j, k, l, value in data['quadratic_costs']:
        i, j, k, l = map(int, (i, j, k, l))
        key = tuple(sorted(((i, j), (k, l))))
        if key not in seen:
            seen.add(key)
            pairs.append((i, j, k, l, value))
    best = float('inf')
    best_assignment = None

    def objective(assignment):
        return sum(original_linear[i][assignment[i]] for i in range(n)) + sum(value for i, j, k, l, value in pairs if assignment[i] == j and assignment[k] == l)

    def save(assignment):
        nonlocal best, best_assignment
        value = objective(assignment)
        if value >= best:
            return
        best = value
        best_assignment = np.asarray(assignment, dtype=int).copy()
        solution = {'objective_value': value, 'assignment': {str(i): int(assignment[i]) for i in range(n)}}
        with open(args.solution_path, 'w') as handle:
            json.dump(solution, handle)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(value, solution)
        else:
            logger.log(value)

    save(np.argmin(linear, axis=1))
    interaction = np.zeros((n, servers, n, servers))
    for i, j, k, l, value in pairs:
        if i == k:
            if j == l:
                linear[i, j] += value
        else:
            interaction[i, j, k, l] += value
            interaction[k, l, i, j] += value
    rng = np.random.default_rng(0)
    clients = np.arange(n)
    heuristic_deadline = min(deadline, time.monotonic() + (min(1., args.time_limit * .1) if n <= 10 else min(25., args.time_limit * .45)))
    restart = 0
    moves = 0
    while time.monotonic() < heuristic_deadline and servers > 1:
        if restart == 0:
            assignment = best_assignment.copy()
        elif restart % 4 == 0:
            assignment = rng.integers(servers, size=n)
        else:
            assignment = best_assignment.copy()
            changed = rng.choice(n, size=min(n, 2 + restart % max(3, n // 3)), replace=False)
            assignment[changed] = rng.integers(servers, size=len(changed))
        gradient = linear + interaction[:, :, clients, assignment].sum(axis=2)
        for _ in range(30):
            changed = False
            for i in rng.permutation(n):
                old = assignment[i]
                new = int(np.argmin(gradient[i]))
                if gradient[i, new] < gradient[i, old] - 1e-9:
                    gradient += interaction[:, :, i, new] - interaction[:, :, i, old]
                    assignment[i] = new
                    changed = True
            if not changed or time.monotonic() >= heuristic_deadline:
                break
        value = objective(assignment)
        save(assignment)
        tabu = np.zeros((n, servers), dtype=int)
        last_improvement = 0
        run_best = value
        for step in range(1200):
            if time.monotonic() >= heuristic_deadline:
                break
            delta = gradient - gradient[clients, assignment][:, None]
            delta[clients, assignment] = np.inf
            forbidden = (tabu > step) & (value + delta >= best - 1e-9)
            delta[forbidden] = np.inf
            i, new = np.unravel_index(int(np.argmin(delta)), delta.shape)
            if not np.isfinite(delta[i, new]):
                break
            old = assignment[i]
            value += delta[i, new]
            gradient += interaction[:, :, i, new] - interaction[:, :, i, old]
            assignment[i] = new
            tabu[i, old] = step + 5 + int(rng.integers(9))
            moves += 1
            if value < best - 1e-7:
                save(assignment)
            if value < run_best - 1e-7:
                run_best = value
                last_improvement = step
            if step % 100 == 0:
                value = objective(assignment)
            if step - last_improvement > 250:
                break
        restart += 1
    print('TABU_END elapsed', time.monotonic() - _PROCESS_START, 'restarts', restart, 'moves', moves, 'best', best, flush=True)
    if time.monotonic() >= deadline:
        return

    model = cp_model.CpModel()
    x = [[model.NewBoolVar('x_' + str(i) + '_' + str(j)) for j in range(servers)] for i in range(n)]
    for i in range(n):
        model.AddExactlyOne(x[i])
        for j in range(servers):
            model.AddHint(x[i][j], int(best_assignment[i] == j))
    scale = 100000
    terms = [round(original_linear[i][j] * scale) * x[i][j] for i in range(n) for j in range(servers)]
    for index, (i, j, k, l, value) in enumerate(pairs):
        if index % 500 == 0 and time.monotonic() >= deadline:
            print('MODEL_BUILD_DEADLINE elapsed', time.monotonic() - _PROCESS_START, flush=True)
            return
        if i == k:
            if j == l:
                terms.append(round(value * scale) * x[i][j])
            continue
        product = model.NewBoolVar('q_' + str(index))
        model.Add(product <= x[i][j])
        model.Add(product <= x[k][l])
        model.Add(product >= x[i][j] + x[k][l] - 1)
        model.AddHint(product, int(best_assignment[i] == j and best_assignment[k] == l))
        terms.append(round(value * scale) * product)
    model.Minimize(sum(terms))

    class Callback(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            save([next(j for j in range(servers) if self.Value(x[i][j])) for i in range(n)])

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = remaining
    print('MODEL_BUILD_END elapsed', time.monotonic() - _PROCESS_START, 'solve_budget', remaining, 'best', best, flush=True)
    status = solver.Solve(model, Callback())
    print('SOLVE_END elapsed', time.monotonic() - _PROCESS_START, 'status', solver.StatusName(status), 'best', best, 'bound', solver.BestObjectiveBound() / scale, flush=True)


if __name__ == '__main__':
    main()
