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
    reserve = min(6.0, max(0.3, args.time_limit * 0.12))
    deadline = STARTED + args.time_limit - reserve
    data = json.load(open(args.instance_path))
    n = data['n']
    costs = np.asarray(data['cost_matrices'], dtype=float)
    lam = np.asarray(data['search_direction_lambda'], dtype=float)
    reference = np.asarray(data['reference_point'], dtype=float)
    scaled = costs * lam[:, None, None]
    offset = lam * reference
    criteria = len(lam)
    rows = np.arange(n)
    logger = SolutionLogger(args.log_path, sense='minimize')
    best = None
    best_objective = np.inf
    edges = set()

    def deviations(assignment):
        return scaled[:, rows, assignment].sum(axis=1) - offset

    def save(assignment):
        nonlocal best, best_objective
        assignment = [int(value) for value in assignment]
        objective = float(np.max(deviations(assignment)))
        if objective >= best_objective:
            return
        best = assignment[:]
        best_objective = objective
        matrix = [[0] * n for _ in range(n)]
        for i, j in enumerate(assignment):
            matrix[i][j] = 1
        solution = {'objective_value': objective, 'assignment': assignment, 'assignment_matrix': matrix}
        with open(args.solution_path, 'w') as output:
            json.dump(solution, output)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(objective, solution)
        else:
            logger.log(objective)

    def remember(assignment):
        if assignment is not None:
            edges.update((i, int(j)) for i, j in enumerate(assignment))
            save(assignment)

    remember(list(range(n)))
    combined = np.mean(scaled, axis=0)
    available = np.ones(n, dtype=bool)
    greedy = np.empty(n, dtype=np.int32)
    if n > 1:
        smallest = np.partition(combined, 1, axis=1)[:, :2]
        priority = np.argsort(-(smallest[:, 1] - smallest[:, 0]))
    else:
        priority = rows
    for i in priority:
        j = int(np.argmin(np.where(available, combined[i], np.inf)))
        greedy[i] = j
        available[j] = False
    remember(greedy.tolist())
    heuristic_deadline = min(deadline, time.monotonic() + min(12.0, args.time_limit * 0.25))

    def assignment_minimum(matrix):
        u = np.zeros(n + 1)
        v = np.zeros(n + 1)
        matched = np.zeros(n + 1, dtype=np.int32)
        predecessor = np.zeros(n + 1, dtype=np.int32)
        for row in range(1, n + 1):
            if time.monotonic() >= heuristic_deadline:
                return None
            matched[0] = row
            column = 0
            minimum = np.full(n + 1, np.inf)
            visited = np.zeros(n + 1, dtype=bool)
            while True:
                if time.monotonic() >= heuristic_deadline:
                    return None
                visited[column] = True
                current = matrix[matched[column] - 1] - u[matched[column]] - v[1:]
                improves = (~visited[1:]) & (current < minimum[1:])
                minimum[1:][improves] = current[improves]
                predecessor[1:][improves] = column
                candidates = np.where(visited[1:], np.inf, minimum[1:])
                next_column = int(np.argmin(candidates)) + 1
                delta = minimum[next_column]
                u[matched[visited]] += delta
                v[visited] -= delta
                minimum[~visited] -= delta
                column = next_column
                if matched[column] == 0:
                    break
            while column:
                previous = predecessor[column]
                matched[column] = matched[previous]
                column = previous
        result = np.empty(n, dtype=np.int32)
        result[matched[1:] - 1] = np.arange(n)
        return result.tolist()

    remember(assignment_minimum(np.mean(scaled, axis=0)))
    for k in range(criteria):
        if time.monotonic() >= heuristic_deadline:
            break
        remember(assignment_minimum(scaled[k]))
    if criteria == 2:
        lower, upper = 0.0, 1.0
        for _ in range(14):
            if time.monotonic() >= heuristic_deadline:
                break
            alpha = (lower + upper) / 2
            candidate = assignment_minimum(alpha * scaled[0] + (1 - alpha) * scaled[1])
            if candidate is None:
                break
            remember(candidate)
            deviation = deviations(candidate)
            if deviation[0] > deviation[1]:
                lower = alpha
            else:
                upper = alpha
    else:
        weights = np.ones(criteria) / criteria
        for iteration in range(15):
            if time.monotonic() >= heuristic_deadline:
                break
            candidate = assignment_minimum(np.sum(scaled * weights[:, None, None], axis=0))
            if candidate is None:
                break
            remember(candidate)
            worst = int(np.argmax(deviations(candidate)))
            weights *= 1 - 1 / (iteration + 3)
            weights[worst] += 1 / (iteration + 3)
    for _ in range(3):
        if time.monotonic() >= heuristic_deadline:
            break
        assignment = np.asarray(best, dtype=np.int32)
        selected = scaled[:, rows, assignment]
        total = selected.sum(axis=1) - offset
        changed = False
        for i in range(n):
            if time.monotonic() >= heuristic_deadline:
                break
            delta = scaled[:, i, assignment] + scaled[:, rows, assignment[i]] - selected[:, i, None] - selected
            values = np.max(total[:, None] + delta, axis=0)
            j = int(np.argmin(values))
            if values[j] < np.max(total) - 1e-9:
                assignment[i], assignment[j] = assignment[j], assignment[i]
                selected[:, i] = scaled[:, i, assignment[i]]
                selected[:, j] = scaled[:, j, assignment[j]]
                total += delta[:, j]
                changed = True
        remember(assignment.tolist())
        if not changed:
            break
    print('construction_end elapsed=%.3f objective=%.9f edges=%d' % (time.monotonic() - STARTED, best_objective, len(edges)), flush=True)
    if time.monotonic() >= deadline:
        return
    if n <= 80:
        edges = {(i, j) for i in range(n) for j in range(n)}
    ordered = sorted(edges)
    highs = highspy.Highs()
    highs.setOptionValue('threads', 1)
    highs.setOptionValue('output_flag', False)
    variables = [highs.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger) for _ in ordered]
    epigraph = highs.addVariable(lb=-highspy.kHighsInf, ub=highspy.kHighsInf, obj=1)
    row_variables = [[] for _ in range(n)]
    column_variables = [[] for _ in range(n)]
    for (i, j), variable in zip(ordered, variables):
        row_variables[i].append(variable)
        column_variables[j].append(variable)
    for i in range(n):
        if time.monotonic() >= deadline:
            return
        highs.addConstr(sum(row_variables[i]) == 1)
        highs.addConstr(sum(column_variables[i]) == 1)
    for k in range(criteria):
        highs.addConstr(sum(float(scaled[k, i, j]) * variable for (i, j), variable in zip(ordered, variables)) - epigraph <= float(offset[k]))
    initial_values = [float(best[i] == j) for i, j in ordered] + [best_objective]
    initial = highspy.HighsSolution()
    initial.col_value = initial_values
    initial.value_valid = True
    highs.setSolution(initial)

    def emit(values):
        candidate = [-1] * n
        for (i, j), variable in zip(ordered, variables):
            if values[int(variable)] > 0.5:
                if candidate[i] != -1:
                    return
                candidate[i] = j
        if -1 not in candidate and len(set(candidate)) == n:
            save(candidate)

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
    print('mip_start elapsed=%.3f budget=%.3f binaries=%d' % (time.monotonic() - STARTED, remaining, len(variables)), flush=True)
    highs.run()
    print('mip_end elapsed=%.3f status=%s' % (time.monotonic() - STARTED, highs.getModelStatus()), flush=True)
    if highs.getSolution().value_valid:
        emit(highs.getSolution().col_value)


if __name__ == '__main__':
    main()
