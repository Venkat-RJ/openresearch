import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
import numpy as np
import highspy
from scipy.sparse import coo_matrix


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'time_limit', 'log_path'):
        parser.add_argument('--' + key, required=key != 'log_path')
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + max(.1, float(args.time_limit) - 5)
    with open(args.instance_path) as stream:
        data = json.load(stream)
    I, J, M, K, T, S = (data[key] for key in ('num_demand_sites', 'num_production_sites', 'num_module_sizes', 'units_per_module_size', 'num_time_periods', 'num_scenarios'))
    investment_count = M * K
    block = J * M + I * (J + 1)
    rows_per_cell = M + I + 2 * J
    ordering_rows = M * (K - 1)
    count = investment_count + S * T * block
    row_count = ordering_rows + S * T * rows_per_cell
    nonzeros = 2 * ordering_rows + S * T * (M * (J + K) + I * (J + 1) + J * (2 * I + M))
    costs = np.zeros(count)
    upper = np.ones(count)
    integer = np.zeros(count, dtype=np.uint8)
    integer[:investment_count] = 1
    capacities = np.asarray([size['capacity'] for size in data['module_sizes']], dtype=float)
    production = np.asarray(data['production_costs'], dtype=float)
    penalties = np.asarray(data['penalty_costs'], dtype=float)
    for unit in data['module_units']:
        costs[unit['module_size_id'] * K + unit['unit_id']] = unit['capital_cost']
    rows = np.empty(nonzeros, dtype=np.int32)
    columns = np.empty(nonzeros, dtype=np.int32)
    coefficients = np.empty(nonzeros)
    lower_rows = np.full(row_count, -np.inf)
    upper_rows = np.zeros(row_count)
    cursor = 0

    def emit(rr, cc, vv):
        nonlocal cursor
        size = len(cc)
        rows[cursor:cursor + size] = rr
        columns[cursor:cursor + size] = cc
        coefficients[cursor:cursor + size] = vv
        cursor += size

    r = 0
    for m in range(M):
        for k in range(1, K):
            emit([r, r], [m * K + k, m * K + k - 1], [1., -1.])
            r += 1
    v_indices = np.arange(J * M, dtype=np.int32).reshape(J, M)
    demand_indices = np.arange(I * (J + 1), dtype=np.int32).reshape(I, J + 1) + J * M
    q_indices = demand_indices[:, 0]
    x_by_site = demand_indices[:, 1:].T
    allocation_v_rows = np.tile(np.arange(M, dtype=np.int32), J)
    allocation_y_rows = np.repeat(np.arange(M, dtype=np.int32), K)
    demand_rows = M + np.repeat(np.arange(I, dtype=np.int32), J + 1)
    production_rows = M + I + 2 * np.arange(J, dtype=np.int32)
    for s, scenario in enumerate(data['scenarios']):
        demands = np.asarray(scenario['demands'], dtype=float)
        probability = scenario['probability']
        for t in range(T):
            offset = investment_count + (s * T + t) * block
            row_offset = ordering_rows + (s * T + t) * rows_per_cell
            integer[offset:offset + J * M] = 1
            upper[offset:offset + J * M] = K
            cell_costs = costs[offset + J * M:offset + block].reshape(I, J + 1)
            cell_costs[:, 0] = probability * demands[:, t] * penalties[:, t]
            cell_costs[:, 1:] = probability * demands[:, t, None] * production
            emit(row_offset + allocation_v_rows, offset + v_indices.ravel(), 1.)
            emit(row_offset + allocation_y_rows, np.arange(investment_count, dtype=np.int32), -1.)
            emit(row_offset + demand_rows, offset + demand_indices.ravel(), 1.)
            flow_columns = offset + x_by_site.ravel()
            demand_coefficients = np.tile(demands[:, t], J)
            emit(row_offset + np.repeat(production_rows, I), flow_columns, demand_coefficients)
            emit(row_offset + np.repeat(production_rows + 1, I), flow_columns, demand_coefficients)
            emit(row_offset + np.repeat(production_rows + 1, M), offset + v_indices.ravel(), -np.tile(capacities, J))
            lower_rows[row_offset + M:row_offset + M + I] = 1.
            upper_rows[row_offset + M:row_offset + M + I] = 1.
            upper_rows[row_offset + production_rows] = data['u_max']
    matrix = coo_matrix((coefficients, (rows, columns)), shape=(row_count, count)).tocsc()
    del rows, columns, coefficients
    lp = highspy.HighsLp()
    lp.num_col_ = count
    lp.num_row_ = row_count
    lp.col_cost_ = costs
    lp.col_lower_ = np.zeros(count)
    lp.col_upper_ = upper
    lp.row_lower_ = lower_rows
    lp.row_upper_ = upper_rows
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr
    lp.a_matrix_.index_ = matrix.indices
    lp.a_matrix_.value_ = matrix.data
    solver = highspy.Highs()
    solver.setOptionValue('threads', 1)
    solver.setOptionValue('output_flag', False)
    solver.passModel(lp)
    integer_indices = np.flatnonzero(integer).astype(np.int32)
    solver.changeColsIntegrality(len(integer_indices), integer_indices, np.full(len(integer_indices), highspy.HighsVarType.kInteger, dtype=np.uint8))
    seed = np.zeros(count)
    for cell in range(S * T):
        seed[investment_count + cell * block + q_indices] = 1.
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense='minimize')
    best = float('inf')

    def save(values):
        nonlocal best
        values = np.asarray(values).copy()
        values[integer_indices] = np.rint(values[integer_indices])
        objective = float(costs @ values)
        if objective >= best:
            return
        best = objective
        y = {f'y_{m}_{k}': int(values[m * K + k]) for m in range(M) for k in range(K)}
        witness = {}
        for column in np.flatnonzero(values > 1e-8):
            if column < investment_count:
                m, k = divmod(int(column), K)
                name = f'y[{m},{k}]'
            else:
                cell, position = divmod(int(column) - investment_count, block)
                s, t = divmod(cell, T)
                if position < J * M:
                    j, m = divmod(position, M)
                    name = f'v[{j},{m},{t},{s}]'
                else:
                    i, slot = divmod(position - J * M, J + 1)
                    name = f'q[{i},{t},{s}]' if slot == 0 else f'x[{i},{slot - 1},{t},{s}]'
            witness[name] = float(values[column])
        for s in range(S):
            for m in range(M):
                total_units = sum(y[f'y_{m}_{k}'] for k in range(K))
                witness[f'z[{m},{s}]'] = total_units
                for t in range(T):
                    offset = investment_count + (s * T + t) * block
                    total = 0
                    for j in range(J):
                        current = int(values[offset + j * M + m])
                        previous = int(values[offset - block + j * M + m]) if t else 0
                        delta = current - previous
                        total += current
                        if delta:
                            family = 'w_fd' if delta > 0 else 'w_td'
                            witness[f'{family}[{j},{m},{t},{s}]'] = abs(delta)
                    witness[f'v_depot[{m},{t},{s}]'] = total_units - total
        with open(args.solution_path, 'w') as stream:
            json.dump({'objective_value': objective, 'y': y, 'model_variables': witness}, stream)
        if logger:
            logger.log(objective)

    save(seed)
    solver.setSolution(count, np.arange(count, dtype=np.int32), seed)
    del seed, lp, upper, lower_rows, upper_rows, matrix, integer
    solver.setOptionValue('mip_rel_gap', .001)
    solver.setOptionValue('time_limit', max(.01, deadline - time.monotonic()))
    solver.run()
    if solver.getInfo().primal_solution_status == highspy.kSolutionStatusFeasible:
        save(solver.getSolution().col_value)


if __name__ == '__main__':
    main()
