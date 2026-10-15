import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
_PROCESS_START = time.monotonic()
import numpy as np
from scipy.optimize import minimize
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
    logger = SolutionLogger(args.log_path, sense='minimize')
    bars = data['bars']
    n = len(bars)
    nd = data['num_free_dofs']
    loads = data['num_loading_conditions']
    candidates = np.asarray(data['discrete_areas'], dtype=float)
    order = np.argsort(candidates)
    areas = candidates[order]
    young = data['material_properties']['modulus_of_elasticity']
    density = data['material_properties']['cost_density']
    length = np.array([bar['length'] for bar in bars])
    nodes = {node['node_id']: node for node in data['nodes']}
    dofs = {(dof['node'], dof['direction']): dof['dof_id'] - 1 for dof in data['degrees_of_freedom']}
    B = np.zeros((n, nd))
    P = np.zeros((nd, loads))
    for i, bar in enumerate(bars):
        delta = np.array([nodes[bar['node_j']].get(direction, 0.) - nodes[bar['node_i']].get(direction, 0.) for direction in ('x', 'y', 'z')[:data['dimension']]])
        cosine = delta / np.linalg.norm(delta)
        for direction, value in zip(('x', 'y', 'z'), cosine):
            if (bar['node_i'], direction) in dofs:
                B[i, dofs[bar['node_i'], direction]] = -value
            if (bar['node_j'], direction) in dofs:
                B[i, dofs[bar['node_j'], direction]] = value
    for l, condition in enumerate(data['loading_conditions']):
        for force in condition['loads']:
            P[force['dof_id'] - 1, l] = force['force']
    lower_stress = np.full(n, data['stress_bounds']['lower'], dtype=float)
    upper_stress = np.full(n, data['stress_bounds']['upper'], dtype=float)
    bar_index = {bar['bar_id']: i for i, bar in enumerate(bars)}
    for limits in data.get('bar_specific_stress_bounds', []):
        lower_stress[bar_index[limits['bar_id']]] = limits['lower']
        upper_stress[bar_index[limits['bar_id']]] = limits['upper']
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for linking in data.get('linking_groups', []):
        members = [bar_index[identifier] for identifier in linking['bar_ids']]
        for i in members[1:]:
            parent[find(i)] = find(members[0])
    grouped = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(i)
    groups = list(grouped.values())
    group_of = np.array([next(g for g, members in enumerate(groups) if i in members) for i in range(n)])
    G = len(groups)
    weights = np.array([density * sum(length[i] for i in members) for members in groups])
    stiffness = np.array([B[members].T @ ((young / length[members])[:, None] * B[members]) for members in groups])
    stress_matrix = (young / length)[:, None] * B
    stress_scale = np.maximum(1., np.maximum(abs(lower_stress), abs(upper_stress)))[:, None]
    displacement_lower = data['displacement_bounds']['lower']
    displacement_upper = data['displacement_bounds']['upper']
    best = float('inf')
    best_indices = None

    class BudgetExpired(Exception):
        pass

    cache = {}
    def mechanics(group_areas, derivatives=False):
        if time.monotonic() >= deadline:
            raise BudgetExpired()
        key = tuple(group_areas)
        if cache.get('key') != key:
            K = np.einsum('g,gij->ij', group_areas, stiffness)
            D = np.linalg.solve(K, P)
            stress = stress_matrix @ D
            margins = [((stress - lower_stress[:, None]) / stress_scale).ravel(), ((upper_stress[:, None] - stress) / stress_scale).ravel()]
            if displacement_lower is not None:
                margins.append(((D - displacement_lower) / max(1., abs(displacement_lower))).ravel())
            if displacement_upper is not None:
                margins.append(((displacement_upper - D) / max(1., abs(displacement_upper))).ravel())
            cache.clear()
            cache.update(key=key, K=K, D=D, stress=stress, margins=np.concatenate(margins))
        if derivatives and 'jacobian' not in cache:
            rhs = -np.einsum('gij,jl->igl', stiffness, cache['D'])
            derivative = np.linalg.solve(cache['K'], rhs.reshape(nd, G * loads)).reshape(nd, G, loads)
            ds = np.einsum('ij,jgl->igl', stress_matrix, derivative).transpose(0, 2, 1)
            jacobian = [(ds / stress_scale[:, :, None]).reshape(n * loads, G), (-ds / stress_scale[:, :, None]).reshape(n * loads, G)]
            if displacement_lower is not None:
                jacobian.append(derivative.transpose(0, 2, 1).reshape(nd * loads, G) / max(1., abs(displacement_lower)))
            if displacement_upper is not None:
                jacobian.append(-derivative.transpose(0, 2, 1).reshape(nd * loads, G) / max(1., abs(displacement_upper)))
            cache['jacobian'] = np.vstack(jacobian)
        return cache

    def emit(indices):
        nonlocal best, best_indices
        group_areas = areas[indices]
        try:
            result = mechanics(group_areas)
        except np.linalg.LinAlgError:
            return False
        A = group_areas[group_of]
        D = result['D']
        stress = result['stress']
        forces = A[:, None] * stress
        if np.max(np.abs(B.T @ forces - P)) > 1e-5:
            return False
        if np.any(stress < lower_stress[:, None] - 1e-7) or np.any(stress > upper_stress[:, None] + 1e-7):
            return False
        if displacement_lower is not None and np.any(D < displacement_lower - 1e-7):
            return False
        if displacement_upper is not None and np.any(D > displacement_upper + 1e-7):
            return False
        objective = float(weights @ group_areas)
        if objective < best:
            best = objective
            best_indices = indices.copy()
            solution = {'objective_value': objective,
                        'bar_areas': [{'bar_id': bars[i]['bar_id'], 'area': float(A[i]), 'area_index': int(order[indices[group_of[i]]])} for i in range(n)],
                        'displacements': [{'dof_id': j + 1, 'load': l + 1, 'value': float(D[j, l])} for j in range(nd) for l in range(loads)],
                        'bar_forces': [{'bar_id': bars[i]['bar_id'], 'load': l + 1, 'force': float(forces[i, l])} for i in range(n) for l in range(loads)]}
            with open(args.solution_path, 'w') as handle:
                json.dump(solution, handle)
            if hasattr(logger, 'log_solution'):
                logger.log_solution(objective, solution)
            else:
                logger.log(objective)
        return True

    def discrete_descent(indices):
        indices = indices.copy()
        for _ in range(10):
            changed = False
            for g in np.argsort(-(weights * areas[indices])):
                for k in range(indices[g]):
                    trial = indices.copy()
                    trial[g] = k
                    if emit(trial):
                        indices = trial
                        changed = True
                        break
            if not changed:
                break
        return indices

    def repair(indices):
        indices = indices.copy()
        for _ in range(G * len(areas)):
            if emit(indices):
                return indices
            result = mechanics(areas[indices], True)
            bad = np.minimum(result['margins'], 0.)
            gain = -(bad @ result['jacobian'])
            options = []
            for g in range(G):
                if indices[g] + 1 < len(areas):
                    change = areas[indices[g] + 1] - areas[indices[g]]
                    options.append((gain[g] / weights[g], g, change))
            if not options:
                break
            _, g, _ = max(options)
            indices[g] += 1
        return None

    try:
        emit(np.full(G, len(areas) - 1, dtype=int))
        print('MECHANICAL_BASELINE elapsed', time.monotonic() - _PROCESS_START, 'best', best, 'groups', G, flush=True)
        rng = np.random.default_rng(0)
        nonlinear_deadline = min(deadline, time.monotonic() + min(30., args.time_limit * .55))
        for attempt in range(5):
            if time.monotonic() >= nonlinear_deadline:
                break
            start = areas[best_indices] if best_indices is not None else np.full(G, areas[-1])
            if attempt:
                start = np.clip(start * rng.uniform(.7, 1.3, G), max(areas[0], 1e-6), areas[-1])
            def iteration_callback(value):
                if time.monotonic() >= nonlinear_deadline:
                    raise BudgetExpired()
            result = minimize(lambda value: float(weights @ value / weights.sum()), start,
                              jac=lambda value: weights / weights.sum(),
                              bounds=[(max(areas[0], 1e-6), areas[-1])] * G,
                              constraints={'type': 'ineq', 'fun': lambda value: mechanics(value)['margins'],
                                           'jac': lambda value: mechanics(value, True)['jacobian']},
                              method='SLSQP', callback=iteration_callback,
                              options={'maxiter': 180, 'ftol': 1e-9})
            rounded = np.minimum(len(areas) - 1, np.searchsorted(areas, result.x, side='left'))
            repaired = repair(rounded)
            if repaired is not None:
                discrete_descent(repaired)
            print('NONLINEAR_SIZING attempt', attempt, 'success', result.success, 'continuous_cost', float(weights @ result.x), 'best', best, 'elapsed', time.monotonic() - _PROCESS_START, flush=True)
        if best_indices is not None:
            discrete_descent(best_indices)
            while time.monotonic() < deadline:
                trial = best_indices.copy()
                count = min(G, int(rng.integers(2, min(G, 5) + 1))) if G > 1 else 1
                chosen = rng.choice(G, count, replace=False)
                for g in chosen:
                    trial[g] = min(len(areas) - 1, trial[g] + int(rng.integers(1, 4)))
                if emit(trial):
                    discrete_descent(trial)
    except BudgetExpired:
        pass
    print('SIZING_END elapsed', time.monotonic() - _PROCESS_START, 'best', best, flush=True)
    if best_indices is None:
        raise RuntimeError('No mechanically feasible discrete truss found')


if __name__ == '__main__':
    main()
