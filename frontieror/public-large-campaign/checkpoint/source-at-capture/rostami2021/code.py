import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'time_limit', 'log_path'):
        parser.add_argument('--' + key, required=key != 'log_path')
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + max(.1, float(args.time_limit) - 3)
    with open(args.instance_path) as stream:
        data = json.load(stream)
    n, scenarios = data['n'], data['num_scenarios']
    distances = np.asarray(data['distances'], dtype=float)
    demand = np.asarray([s['demands'] for s in data['scenarios']], dtype=float)
    probability = np.asarray([s['probability'] for s in data['scenarios']], dtype=float)
    outgoing, incoming = demand.sum(axis=2), demand.sum(axis=1)
    parameters = data['cost_parameters']
    chi, alpha, delta = (parameters[key] for key in ('chi', 'alpha', 'delta'))
    fixed = np.asarray(data['fixed_costs'], dtype=float)
    if data['problem_info']['variant'] == 'SApHMP':
        fixed = np.zeros(n)
    capacities = np.asarray(data['hub_capacities'], dtype=float) if data.get('hub_capacities') is not None else np.full(n, np.inf)
    required_hubs = data.get('p_hubs')
    diagonal_demand = np.diagonal(demand, axis1=1, axis2=2).copy()
    pair_demand = demand.copy()
    for s in range(scenarios):
        np.fill_diagonal(pair_demand[s], 0.)
    best = float('inf')
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense='minimize')

    def objective(hubs, allocations):
        value = float(fixed[list(hubs)].sum())
        for s in range(scenarios):
            a = allocations[s]
            cd = chi * outgoing[s] @ distances[np.arange(n), a] + delta * incoming[s] @ distances[a, np.arange(n)]
            transfer = alpha * np.sum(demand[s] * distances[a[:, None], a[None, :]])
            value += probability[s] * (cd + transfer)
        return float(value)

    def save(hubs, allocations):
        nonlocal best
        value = objective(hubs, allocations)
        if value >= best - 1e-7:
            return
        best = value
        hubs = sorted(hubs)
        opened = set(hubs)
        out = {'objective_value': value, 'hubs': hubs, 'allocations': {str(s): {str(i): int(allocations[s, i]) for i in range(n)} for s in range(scenarios)}, 'z_values': [int(i in opened) for i in range(n)], 'x_values': {str(s): {str(i): {str(h): int(h == allocations[s, i]) for h in range(n) if h != i} for i in range(n)} for s in range(scenarios)}}
        with open(args.solution_path, 'w') as stream:
            json.dump(out, stream)
        if logger:
            logger.log(value)

    def conditional(s, i, hubs, a):
        return chi * outgoing[s, i] * distances[i, hubs] + delta * incoming[s, i] * distances[hubs, i] + alpha * (distances[np.ix_(hubs, a)] @ pair_demand[s, i] + pair_demand[s, :, i] @ distances[np.ix_(a, hubs)] + diagonal_demand[s, i] * distances[hubs, hubs])

    def allocate(hubs, previous):
        h = np.asarray(sorted(hubs), dtype=int)
        valid = np.zeros(n, dtype=bool)
        valid[h] = True
        allocations = previous.copy()
        for s in range(scenarios):
            if time.monotonic() >= deadline:
                return None
            a = allocations[s]
            missing = ~valid[a]
            a[h] = h
            missing[h] = False
            loads = np.bincount(a[~missing], weights=outgoing[s, ~missing], minlength=n)
            if np.any(loads[h] > capacities[h] + 1e-7):
                return None
            for i in np.flatnonzero(missing)[np.argsort(-outgoing[s, missing])]:
                feasible = h[loads[h] + outgoing[s, i] <= capacities[h] + 1e-7]
                if not len(feasible):
                    return None
                chosen = int(feasible[np.argmin(conditional(s, i, feasible, a))])
                a[i] = chosen
                loads[chosen] += outgoing[s, i]
            spokes = np.flatnonzero(~valid)
            for repeat in range(3):
                changed = False
                for i in spokes:
                    if time.monotonic() >= deadline:
                        return None
                    old = int(a[i])
                    loads[old] -= outgoing[s, i]
                    feasible = h[loads[h] + outgoing[s, i] <= capacities[h] + 1e-7]
                    values = conditional(s, i, feasible, a)
                    chosen = int(feasible[np.argmin(values)])
                    a[i] = chosen
                    loads[chosen] += outgoing[s, i]
                    changed |= chosen != old
                if not changed:
                    break
        return allocations

    eligible = np.flatnonzero(np.max(outgoing, axis=0) <= capacities + 1e-7)
    hubs = set(map(int, eligible))
    if required_hubs is not None:
        hubs = set(map(int, eligible[np.argsort(fixed[eligible])[:required_hubs]]))
    allocations = allocate(hubs, np.tile(np.arange(n), (scenarios, 1))) if hubs else None
    if required_hubs is not None and len(hubs) != required_hubs:
        allocations = None
    if allocations is not None:
        save(hubs, allocations)
    if n > 20 and required_hubs is None and allocations is not None:
        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            for removed in sorted(hubs, key=lambda i: -fixed[i]):
                candidate = hubs - {removed}
                if not candidate:
                    continue
                trial = allocate(candidate, allocations)
                if trial is not None and objective(candidate, trial) < best - 1e-7:
                    hubs, allocations = candidate, trial
                    save(hubs, allocations)
                    improved = True
                    break
                if time.monotonic() >= deadline:
                    break
    if n <= 20 or allocations is None:
        full_transport = n <= 20
        assignment_count = scenarios * n * n
        transport_count = scenarios * n * n * n if full_transport else 0
        count = n + assignment_count + transport_count
        c = np.zeros(count)
        c[:n] = fixed
        rr, cc, vv, lower, upper = [], [], [], [], []

        def x(s, i, h):
            return n + (s * n + i) * n + h

        def f(s, i, h, k):
            return n + assignment_count + ((s * n + i) * n + h) * n + k

        def row(terms, lo, hi):
            r = len(lower)
            lower.append(lo)
            upper.append(hi)
            for column, value in terms:
                rr.append(r)
                cc.append(column)
                vv.append(value)

        if required_hubs is not None:
            row([(i, 1.) for i in range(n)], required_hubs, required_hubs)
        for s in range(scenarios):
            for i in range(n):
                row([(x(s, i, h), 1.) for h in range(n)], 1., 1.)
                row([(x(s, i, i), 1.), (i, -1.)], 0., 0.)
                for h in range(n):
                    c[x(s, i, h)] = probability[s] * (chi * outgoing[s, i] * distances[i, h] + delta * incoming[s, i] * distances[h, i])
                    row([(x(s, i, h), 1.), (h, -1.)], -np.inf, 0.)
                    if full_transport:
                        row([(f(s, i, h, k), 1.) for k in range(n)] + [(x(s, i, h), -outgoing[s, i])], 0., 0.)
                        for k in range(n):
                            c[f(s, i, h, k)] = probability[s] * alpha * distances[h, k]
                if full_transport:
                    for k in range(n):
                        row([(f(s, i, h, k), 1.) for h in range(n)] + [(x(s, j, k), -demand[s, i, j]) for j in range(n)], 0., 0.)
            if data.get('hub_capacities') is not None:
                for h in range(n):
                    row([(x(s, i, h), outgoing[s, i]) for i in range(n)] + [(h, -capacities[h])], -np.inf, 0.)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), count)).tocsc()
        result = milp(c, integrality=np.r_[np.ones(n + assignment_count), np.zeros(transport_count)], bounds=Bounds(0., np.r_[np.ones(n + assignment_count), np.full(transport_count, np.inf)]), constraints=LinearConstraint(matrix, lower, upper), options={'time_limit': max(.01, deadline - time.monotonic()), 'threads': 1, 'mip_rel_gap': .005})
        if result.x is not None:
            selected = {i for i in range(n) if result.x[i] > .5}
            candidate = np.asarray([[int(np.argmax(result.x[n + (s * n + i) * n:n + (s * n + i + 1) * n])) for i in range(n)] for s in range(scenarios)])
            save(selected, candidate)
    if best == float('inf'):
        raise RuntimeError('No feasible complete scenario allocation found')


if __name__ == '__main__':
    main()
