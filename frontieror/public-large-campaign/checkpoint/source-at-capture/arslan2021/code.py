import time
STARTED = time.monotonic()
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import highspy
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'log_path'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    args = parser.parse_args()
    deadline = STARTED + args.time_limit - min(1.0, args.time_limit * 0.05)
    reserve = min(7.0, max(0.3, args.time_limit * 0.12))
    data = json.load(open(args.instance_path))
    items = data['items']
    count = len(items)
    capacity = data['problem_parameters']['knapsack_capacity']
    gamma = data['problem_parameters']['uncertainty_budget']
    logger = SolutionLogger(args.log_path, sense='minimize')
    best_objective = 0.0
    best_selection = [0] * count
    best_values = None

    def save(selection, objective, values=None):
        nonlocal best_objective, best_selection, best_values
        if objective >= best_objective and logger.best_obj is not None:
            return
        best_objective = float(objective)
        best_selection = selection[:]
        best_values = values
        solution = {'objective_value': float(objective),
                    'x': {str(i): int(value) for i, value in enumerate(selection)}}
        with open(args.solution_path, 'w') as output:
            json.dump(solution, output)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(float(objective), solution)
        else:
            logger.log(float(objective))

    save([0] * count, 0.0)

    def build(selection=None, integer=False):
        solver = highspy.Highs()
        solver.setOptionValue('threads', 1)
        solver.setOptionValue('output_flag', False)
        x = []
        for i, item in enumerate(items):
            lo, hi = (0, 1) if selection is None else (selection[i], selection[i])
            x.append(solver.addVariable(lb=lo, ub=hi, obj=item['outsource_penalty'] - item['nominal_profit'],
                                      type=highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous))
        y = [solver.addVariable(lb=0, ub=1, obj=-item['outsource_penalty']) for item in items]
        repair = [solver.addVariable(lb=0, ub=1) for _ in items]
        u = solver.addVariable(lb=0, obj=gamma)
        v = [solver.addVariable(lb=0, obj=1) for _ in items]
        for i, item in enumerate(items):
            solver.addConstr(y[i] <= x[i])
            solver.addConstr(repair[i] <= y[i])
            solver.addConstr(item['max_degradation'] * (y[i] - repair[i]) <= u + v[i])
        solver.addConstr(sum(item['weight'] * y[i] + item['repair_capacity'] * repair[i]
                             for i, item in enumerate(items)) <= capacity)
        return solver, x

    def run(solver, phase_deadline, label):
        remaining = phase_deadline - time.monotonic()
        if remaining <= 0:
            return False

        def interrupt(event):
            if time.monotonic() >= phase_deadline:
                event.interrupt()

        solver.cbMipInterrupt += interrupt
        solver.cbSimplexInterrupt += interrupt
        solver.cbIpmInterrupt += interrupt
        solver.setOptionValue('time_limit', remaining)
        print('%s_start elapsed=%.3f budget=%.3f' % (label, time.monotonic() - STARTED, remaining), flush=True)
        solver.run()
        print('%s_end elapsed=%.3f status=%s' % (label, time.monotonic() - STARTED, solver.getModelStatus()), flush=True)
        return solver.getModelStatus() == highspy.HighsModelStatus.kOptimal

    construction_deadline = min(deadline - reserve, time.monotonic() + min(6.0, args.time_limit * 0.15))
    candidates = []
    mandatory = [int(item['nominal_profit'] > item['outsource_penalty']) for item in items]
    uncertainty_fraction = min(1.0, gamma / max(1, count))
    for degradation_factor, repair_factor in ((0, 0), (uncertainty_fraction, 0), (1, 0), (0, 1), (uncertainty_fraction, 0.5)):
        order = sorted(range(count), key=lambda i: -(items[i]['nominal_profit'] - degradation_factor * items[i]['max_degradation']) /
                       max(1e-9, items[i]['weight'] + repair_factor * items[i]['repair_capacity']))
        selection = mandatory[:]
        used = sum(items[i]['weight'] for i in range(count) if selection[i])
        for i in order:
            if selection[i]:
                continue
            effective_weight = items[i]['weight'] + repair_factor * items[i]['repair_capacity']
            if used + effective_weight <= capacity:
                selection[i] = 1
                used += effective_weight
        candidates.append(selection)
    if time.monotonic() < construction_deadline:
        relaxation, relaxed_x = build()
        if run(relaxation, min(construction_deadline, time.monotonic() + max(0.1, args.time_limit * 0.025)), 'relaxation'):
            values = relaxation.getSolution().col_value
            fractions = [values[int(variable)] for variable in relaxed_x]
            candidates = [[int(mandatory[i] or fractions[i] > threshold) for i in range(count)]
                          for threshold in (0.5, 0.9, 0.1, 0.999, 0.25, 0.75, 1e-8)] + candidates
    seen = set()

    def fixed_selection(selection, phase_deadline, label):
        key = tuple(selection)
        if key in seen or time.monotonic() >= phase_deadline:
            return
        seen.add(key)
        # Each phase gets a fresh native timer; clearSolver does not reset elapsed run time.
        recourse, _ = build(selection)
        if run(recourse, phase_deadline, label):
            save(selection, recourse.getObjectiveValue(), list(recourse.getSolution().col_value))

    for number, selection in enumerate(candidates):
        if time.monotonic() >= construction_deadline:
            break
        fixed_selection(selection, min(construction_deadline, time.monotonic() + max(0.1, args.time_limit * 0.025)), 'construct_%d' % number)
    print('construction_end elapsed=%.3f objective=%.9f selected=%d' %
          (time.monotonic() - STARTED, best_objective, sum(best_selection)), flush=True)
    if time.monotonic() >= deadline - reserve:
        return
    mip, x = build(integer=True)
    if best_values is not None:
        initial = highspy.HighsSolution()
        initial.col_value = best_values
        initial.value_valid = True
        mip.setSolution(initial)
    optimal = run(mip, deadline - reserve, 'mip')
    if not mip.getSolution().value_valid:
        return
    values = mip.getSolution().col_value
    selection = [int(round(values[int(variable)])) for variable in x]
    if optimal:
        save(selection, mip.getObjectiveValue(), list(values))
    else:
        fixed_selection(selection, deadline, 'final_fixed_x')


if __name__ == '__main__':
    main()
