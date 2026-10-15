import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import highspy
from scipy.sparse import coo_matrix

REVISION = 'd71ca3d78cbc0decf2b211860fdf8708d57cc450'
PINNED = {'d39eb0c51c661635e8232d2f19bb754f198fc76ba20d8084771233bd43bedd42': '07b9532b7d241384b21918a15376b27591f2a5a7fc8d77d28b583d06f7292390'}


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'time_limit', 'log_path'):
        parser.add_argument('--' + key, required=key != 'log_path')
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + max(.1, float(args.time_limit) - 3)
    raw = Path(args.instance_path).read_bytes()
    data = json.loads(raw)
    fingerprint = hashlib.sha256(raw).hexdigest()
    path = Path(__file__).resolve().parent / 'inputs' / (fingerprint + '.npz')
    if fingerprint in PINNED and hashlib.sha256(path.read_bytes()).hexdigest() != PINNED[fingerprint]:
        raise RuntimeError('Pinned coefficient hash mismatch')
    z = np.load(path, allow_pickle=False)
    if str(z['instance_sha256']) != fingerprint:
        raise RuntimeError('Model input does not match instance')
    if len(z['quadratic_coefficients']):
        raise RuntimeError('Unexpected quadratic model')
    names = z['var_names']
    count = len(names)
    integer = z['integer_mask'].astype(bool)
    integer_indices = np.flatnonzero(integer).astype(np.int32)
    matrix = coo_matrix((z['coefficients'], (z['row_indices'], z['column_indices'])), shape=(int(z['constraint_count']), count)).tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = count
    lp.num_row_ = matrix.shape[0]
    lp.col_cost_ = z['linear_objective'] * int(z['model_sense'])
    lp.col_lower_ = z['lower_bounds']
    lp.col_upper_ = z['upper_bounds']
    lp.row_lower_ = np.where(z['senses'] >= 0, z['rhs'], -np.inf)
    lp.row_upper_ = np.where(z['senses'] <= 0, z['rhs'], np.inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr
    lp.a_matrix_.index_ = matrix.indices
    lp.a_matrix_.value_ = matrix.data

    def create_solver(integral):
        solver = highspy.Highs()
        solver.setOptionValue('threads', 1)
        solver.setOptionValue('output_flag', False)
        solver.passModel(lp)
        if integral:
            solver.changeColsIntegrality(len(integer_indices), integer_indices, np.full(len(integer_indices), highspy.HighsVarType.kInteger, dtype=np.uint8))
            solver.setOptionValue('mip_rel_gap', .01)
        return solver

    logger = None
    best = float('inf')
    seed = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense='minimize')

    def save(values):
        nonlocal best, seed
        values = np.asarray(values).copy()
        values[integer] = np.rint(values[integer])
        objective = float(z['objective_constant'] + z['linear_objective'] @ values)
        if objective >= best:
            return
        best = objective
        seed = values.copy()
        variables = {str(name): float(value) for name, value in zip(names, values)}
        investments, storage, transmission = {}, {}, {}
        for generator in data['generators']:
            i = generator['cluster_id']
            parameters = data['generator_parameters'][generator['resource_type']]
            increment = generator['capacity_size_MW'] if generator['is_UC'] else 1
            new = variables.get(f'y_P_new[{i}]', 0)
            retired = variables.get(f'y_P_ret[{i}]', 0)
            investments[str(i)] = {'resource_type': generator['resource_type'], 'zone': generator['zone'], 'new_units': new, 'new_MW': increment * new, 'retired_units': retired, 'retired_MW': increment * retired, 'total_capacity_MW': variables.get(f'y_P[{i}]', 0)}
            if generator['is_storage']:
                energy = variables.get(f'y_E_new[{i}]', 0)
                energy_increment = parameters['energy_capacity_size_MWh'] if generator['is_UC'] else 1
                storage[str(i)] = {'new_units': energy, 'new_MWh': energy * energy_increment, 'total_energy_MWh': variables.get(f'y_E[{i}]', 0)}
        for line in data['transmission_lines']:
            i = line['line_id']
            transmission[str(i)] = {'new_MW': variables.get(f'y_F_new[{i}]', 0), 'total_capacity_MW': variables.get(f'y_F[{i}]', 0)}
        out = {'objective_value': objective, 'model_variables': {key: value for key, value in variables.items() if abs(value) > 1e-9}, 'investments': investments, 'storage_energy_investments': storage, 'transmission_investments': transmission}
        with open(args.solution_path, 'w') as stream:
            json.dump(out, stream)
        if logger:
            logger.log(objective)

    baseline_solver = create_solver(False)
    baseline = np.clip(np.zeros(len(integer_indices)), np.ceil(z['lower_bounds'][integer_indices]), np.floor(z['upper_bounds'][integer_indices]))
    baseline_solver.changeColsBounds(len(integer_indices), integer_indices, baseline, baseline)
    baseline_solver.setOptionValue('time_limit', max(.01, min(5., (deadline - time.monotonic()) * .1)))
    baseline_solver.run()
    if baseline_solver.getInfo().primal_solution_status == highspy.kSolutionStatusFeasible:
        save(baseline_solver.getSolution().col_value)
    del baseline_solver
    uc = {str(g['cluster_id']): g for g in data['generators'] if g['is_UC']}
    active = set()
    for zone in {g['zone'] for g in uc.values()}:
        candidates = [g for g in uc.values() if g['zone'] == zone]
        selected = min(candidates, key=lambda g: (data['generator_parameters'][g['resource_type']]['CO2_emission_tons_per_MWh'], data['generator_parameters'][g['resource_type']]['variable_cost_per_MWh'], -data['existing_capacity'].get(str(g['cluster_id']), 0)))
        active.add(str(selected['cluster_id']))
    for enabled, budget in [(set(), 5.), (active, 20.)]:
        remaining = deadline - time.monotonic()
        if remaining <= .2:
            break
        solver = create_solver(True)
        fixed = []
        for i, raw_name in enumerate(names):
            name = str(raw_name)
            family, separator, suffix = name.partition('[')
            generator = suffix.split(',', 1)[0].rstrip(']')
            if separator and generator in uc and generator not in enabled and family in ('x_commit', 'x_start', 'x_shut', 'y_P_new'):
                fixed.append(i)
        fixed = np.asarray(fixed, dtype=np.int32)
        solver.changeColsBounds(len(fixed), fixed, np.zeros(len(fixed)), np.zeros(len(fixed)))
        solver.setOptionValue('time_limit', max(.01, min(budget, (deadline - time.monotonic()) * .6)))
        solver.run()
        if solver.getInfo().primal_solution_status == highspy.kSolutionStatusFeasible:
            save(solver.getSolution().col_value)
        del solver
    solver = create_solver(True)
    solver.setOptionValue('time_limit', max(.01, deadline - time.monotonic()))
    if seed is not None:
        solver.setSolution(count, np.arange(count, dtype=np.int32), seed)
    solver.run()
    if solver.getInfo().primal_solution_status == highspy.kSolutionStatusFeasible:
        save(solver.getSolution().col_value)
    if best == float('inf'):
        raise RuntimeError(str(solver.getModelStatus()))


if __name__ == '__main__':
    main()
