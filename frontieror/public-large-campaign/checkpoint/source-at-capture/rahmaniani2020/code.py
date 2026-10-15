import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
import numpy as np
import highspy
from scipy.sparse import coo_matrix


def engine(cost, lower, upper, row_lower, row_upper, matrix, integer=0):
    lp = highspy.HighsLp()
    lp.num_col_ = len(cost)
    lp.num_row_ = len(row_lower)
    lp.col_cost_ = cost
    lp.col_lower_ = lower
    lp.col_upper_ = upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr
    lp.a_matrix_.index_ = matrix.indices
    lp.a_matrix_.value_ = matrix.data
    if integer:
        lp.integrality_ = [highspy.HighsVarType.kInteger] * integer + [highspy.HighsVarType.kContinuous] * (len(cost) - integer)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('threads', 1)
    h.passModel(lp)
    return h


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'time_limit', 'log_path'):
        parser.add_argument('--' + key, required=key != 'log_path')
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + max(.1, float(args.time_limit) - 3)
    with open(args.instance_path) as stream:
        data = json.load(stream)
    m, n, scenarios = data['num_facilities'], data['num_customers'], data['num_scenarios']
    fixed = np.asarray(data['facilities']['fixed_costs'], dtype=float)
    caps = np.asarray(data['facilities']['capacities'], dtype=float)
    probabilities = np.asarray(data['scenarios']['probabilities'], dtype=float)
    demands = np.asarray(data['scenarios']['demands'], dtype=float)
    costs = np.asarray(data['routing_costs'], dtype=float)
    worst = demands.sum(axis=1).max()
    columns = np.arange(m * n, dtype=np.int32)
    matrix = coo_matrix((np.ones(2 * m * n), (np.r_[np.repeat(np.arange(m), n), m + np.tile(np.arange(n), m)], np.r_[columns, columns])), shape=(m + n, m * n)).tocsc()
    transport = engine(costs.ravel(), np.zeros(m * n), np.full(m * n, np.inf), np.r_[np.full(m, -np.inf), demands[0]], np.r_[caps, demands[0]], matrix)
    transport.setOptionValue('solver', 'simplex')
    transport.setOptionValue('presolve', 'off')
    row_ids = np.arange(m + n, dtype=np.int32)
    cuts = []
    best = np.inf
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense='minimize')

    def evaluate(y):
        nonlocal best
        recourse = np.zeros(scenarios)
        pending = []
        for s in range(scenarios):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            transport.setOptionValue('time_limit', remaining)
            transport.changeRowsBounds(m + n, row_ids, np.r_[np.full(m, -np.inf), demands[s]], np.r_[caps * y, demands[s]])
            transport.run()
            if transport.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                return None
            recourse[s] = transport.getObjectiveValue()
            dual = np.asarray(transport.getSolution().row_dual)
            pending.append((s, dual[:m] * caps, float(dual[m:] @ demands[s])))
        cuts.extend(pending)
        objective = float(fixed @ y + probabilities @ recourse)
        if objective < best - 1e-7:
            best = objective
            with open(args.solution_path, 'w') as stream:
                json.dump({'objective_value': objective, 'y': y.astype(int).tolist()}, stream)
            if logger:
                logger.log(objective)
        return recourse

    y = np.ones(m)
    recourse = evaluate(y)
    if recourse is None:
        raise RuntimeError('Time budget exhausted before full-scenario recourse evaluation')
    seen = {tuple(y)}
    while time.monotonic() < deadline - .2:
        rr, cc, vv = [0] * m, list(range(m)), caps.tolist()
        lower, upper = [worst], [np.inf]
        for s, beta, alpha in cuts:
            r = len(lower)
            lower.append(alpha)
            upper.append(np.inf)
            rr.extend([r] * (m + 1))
            cc.extend(list(range(m)) + [m + s])
            vv.extend((-beta).tolist() + [1.])
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), m + scenarios)).tocsc()
        master = engine(np.r_[fixed, probabilities], np.zeros(m + scenarios), np.r_[np.ones(m), np.full(scenarios, np.inf)], lower, upper, matrix, m)
        master.setOptionValue('mip_rel_gap', .001)
        master.setOptionValue('time_limit', max(.01, min(5., (deadline - time.monotonic()) * .3)))
        master.setSolution(m + scenarios, np.arange(m + scenarios, dtype=np.int32), np.r_[y, recourse])
        master.run()
        info = master.getInfo()
        if info.primal_solution_status != highspy.kSolutionStatusFeasible:
            break
        values = np.asarray(master.getSolution().col_value)
        candidate = np.rint(values[:m])
        if tuple(candidate) in seen:
            break
        seen.add(tuple(candidate))
        evaluated = evaluate(candidate)
        if evaluated is None:
            break
        y, recourse = candidate, evaluated
        if best <= info.mip_dual_bound + max(1e-6, abs(best) * .001):
            break


if __name__ == '__main__':
    main()
