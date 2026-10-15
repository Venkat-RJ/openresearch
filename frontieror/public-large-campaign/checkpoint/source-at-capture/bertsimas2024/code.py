import time
STARTED = time.monotonic()
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import heapq
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
    reserve = min(8.0, max(0.4, args.time_limit * 0.15))
    deadline = STARTED + args.time_limit - reserve
    data = json.load(open(args.instance_path))
    nodes = data['num_nodes']
    edge_count = data['num_edges']
    commodities = data['num_commodities']
    scenarios = data['num_scenarios']
    gamma = float(data['gamma'])
    edges = data['edges']
    capacity = np.asarray(data['capacities'], dtype=float)
    costs = np.asarray(data['flow_costs'], dtype=float)
    fixed = np.asarray(data['construction_costs'], dtype=float)
    demands = np.asarray(data['demands'], dtype=float)
    destinations = data['commodity_destinations']
    logger = SolutionLogger(args.log_path, sense='minimize')
    best_objective = np.inf
    all_flow = np.zeros((scenarios, commodities, edge_count), dtype=float)
    reverse = [[] for _ in range(nodes)]
    for e, (origin, destination) in enumerate(edges):
        reverse[destination].append((origin, e))

    def save(flow):
        nonlocal best_objective
        total = flow.sum(axis=1)
        design = (np.max(total, axis=0) > 1e-8).astype(np.int32)
        if int(design.sum()) > data['c_0']:
            return False
        objective = float(fixed @ design + (np.sum(flow * costs) + np.sum(total * total) / (2 * gamma)) / scenarios)
        if objective >= best_objective:
            return True
        witness = {f'z[{e}]': int(design[e]) for e in range(edge_count)}
        for r in range(scenarios):
            for k in range(commodities):
                for e in np.flatnonzero(flow[r, k] > 1e-10):
                    witness[f'x[{k},{r},{e}]'] = float(flow[r, k, e])
            for e in np.flatnonzero(total[r] > 1e-10):
                witness[f'tf[{r},{e}]'] = float(total[r, e])
        solution = {'objective_value': objective,
                    'z': {str(e): int(design[e]) for e in range(edge_count)},
                    'model_variables': witness}
        with open(args.solution_path, 'w') as output:
            json.dump(solution, output)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(objective, solution)
        else:
            logger.log(objective)
        best_objective = objective
        print('incumbent elapsed=%.3f objective=%.9f edges=%d' % (time.monotonic() - STARTED, objective, int(design.sum())), flush=True)
        return True

    def shortest_tree(sink, residual, marginal):
        distance = [float('inf')] * nodes
        next_edge = [-1] * nodes
        distance[sink] = 0.0
        queue = [(0.0, sink)]
        while queue:
            value, current = heapq.heappop(queue)
            if value != distance[current]:
                continue
            for origin, edge in reverse[current]:
                if residual[edge] <= 1e-8:
                    continue
                trial = value + float(marginal[edge])
                if trial < distance[origin]:
                    distance[origin] = trial
                    next_edge[origin] = edge
                    heapq.heappush(queue, (trial, origin))
        return next_edge

    def constructive(r):
        flow = np.zeros((commodities, edge_count))
        residual = capacity.copy()
        order = sorted(range(commodities), key=lambda k: -float(np.maximum(demands[r, k], 0).sum()))
        for k in order:
            if time.monotonic() >= deadline:
                return None
            sink = destinations[k]
            marginal = costs + (capacity - residual) / gamma
            next_edge = shortest_tree(sink, residual, marginal)
            for source in np.flatnonzero(demands[r, k] > 1e-9):
                remaining = float(demands[r, k, source])
                while remaining > 1e-8:
                    if time.monotonic() >= deadline:
                        return None
                    current = int(source)
                    path = []
                    while current != sink:
                        edge = next_edge[current]
                        if edge < 0 or len(path) >= nodes:
                            return None
                        path.append(edge)
                        current = edges[edge][1]
                    quantity = min(remaining, min(float(residual[e]) for e in path))
                    if quantity <= 1e-8:
                        next_edge = shortest_tree(sink, residual, costs + (capacity - residual) / gamma)
                        continue
                    for edge in path:
                        flow[k, edge] += quantity
                        residual[edge] -= quantity
                    remaining -= quantity
                    if remaining > 1e-8 or min(residual[e] for e in path) <= 1e-8:
                        next_edge = shortest_tree(sink, residual, costs + (capacity - residual) / gamma)
        return flow

    variable_count = (commodities + 1) * edge_count
    row_count = commodities * (nodes - 1) + edge_count
    starts = [0]
    indices = []
    coefficients = []
    for k in range(commodities):
        for e, (origin, destination) in enumerate(edges):
            if origin < nodes - 1:
                indices.append(k * (nodes - 1) + origin)
                coefficients.append(1.0)
            if destination < nodes - 1:
                indices.append(k * (nodes - 1) + destination)
                coefficients.append(-1.0)
            indices.append(commodities * (nodes - 1) + e)
            coefficients.append(-1.0)
            starts.append(len(indices))
    for e in range(edge_count):
        indices.append(commodities * (nodes - 1) + e)
        coefficients.append(1.0)
        starts.append(len(indices))
    starts = np.asarray(starts, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    coefficients = np.asarray(coefficients, dtype=float)
    lower = np.zeros(variable_count)
    upper = np.tile(capacity, commodities + 1)
    linear = np.r_[np.tile(costs, commodities), np.zeros(edge_count)]
    completed = 0
    native_fallbacks = 0

    def scenario_solve(r, phase_deadline, quadratic=False, seed=None):
        remaining = phase_deadline - time.monotonic()
        if remaining <= 0:
            return None
        solver = highspy.Highs()
        solver.setOptionValue('threads', 1)
        solver.setOptionValue('output_flag', False)
        solver.setOptionValue('primal_feasibility_tolerance', 1e-8)
        rhs = np.r_[demands[r, :, :-1].ravel(), np.zeros(edge_count)]
        solver.addRows(row_count, rhs, rhs, 0, np.zeros(row_count + 1, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0))
        solver.addCols(variable_count, linear, lower, upper, len(indices), starts, indices, coefficients)
        if quadratic:
            hessian = highspy.HighsHessian()
            hessian.dim_ = variable_count
            hessian.format_ = highspy.HessianFormat.kTriangular
            hessian.start_ = [0] * (commodities * edge_count + 1) + list(range(1, edge_count + 1))
            hessian.index_ = list(range(commodities * edge_count, variable_count))
            hessian.value_ = [1.0 / gamma] * edge_count
            solver.passHessian(hessian)
        if seed is not None:
            initial = highspy.HighsSolution()
            initial.col_value = np.r_[seed.ravel(), seed.sum(axis=0)].tolist()
            initial.value_valid = True
            solver.setSolution(initial)

        def interrupt(event):
            if time.monotonic() >= phase_deadline:
                event.interrupt()

        solver.cbSimplexInterrupt += interrupt
        solver.cbIpmInterrupt += interrupt
        solver.setOptionValue('time_limit', max(0.01, phase_deadline - time.monotonic()))
        solver.run()
        solution = solver.getSolution()
        if not solution.value_valid:
            return None
        info = solver.getInfo()
        if info.max_primal_infeasibility > 1e-6:
            return None
        return np.maximum(0, np.asarray(solution.col_value[:commodities * edge_count]).reshape(commodities, edge_count))

    print('routing_start elapsed=%.3f dimensions=%d/%d/%d/%d' % (time.monotonic() - STARTED, nodes, edge_count, commodities, scenarios), flush=True)
    for r in range(scenarios):
        candidate = constructive(r)
        if candidate is None:
            native_fallbacks += 1
            budget = max(0.01, (deadline - time.monotonic()) / max(1, scenarios - r))
            candidate = scenario_solve(r, min(deadline, time.monotonic() + budget))
        if candidate is None:
            print('routing_incomplete elapsed=%.3f completed=%d fallback=%d' % (time.monotonic() - STARTED, completed, native_fallbacks), flush=True)
            return
        all_flow[r] = candidate
        completed += 1
    print('routing_end elapsed=%.3f completed=%d fallback=%d' % (time.monotonic() - STARTED, completed, native_fallbacks), flush=True)
    save(all_flow)
    for r in range(scenarios):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        budget = remaining / max(1, scenarios - r)
        quadratic = variable_count <= 3000
        candidate = scenario_solve(r, min(deadline, time.monotonic() + budget), quadratic=quadratic, seed=all_flow[r])
        if candidate is None:
            continue
        old_total = all_flow[r].sum(axis=0)
        new_total = candidate.sum(axis=0)
        old_cost = float(np.sum(all_flow[r] * costs) + old_total @ old_total / (2 * gamma))
        new_cost = float(np.sum(candidate * costs) + new_total @ new_total / (2 * gamma))
        if new_cost < old_cost - 1e-8:
            previous = all_flow[r].copy()
            all_flow[r] = candidate
            total = all_flow.sum(axis=1)
            design = (np.max(total, axis=0) > 1e-8).astype(np.int32)
            objective = float(fixed @ design + (np.sum(all_flow * costs) + np.sum(total * total) / (2 * gamma)) / scenarios)
            if objective < best_objective:
                save(all_flow)
            else:
                all_flow[r] = previous
        if r % 10 == 0:
            print('refinement elapsed=%.3f scenario=%d quadratic=%s' % (time.monotonic() - STARTED, r, quadratic), flush=True)
    print('finished elapsed=%.3f objective=%.9f' % (time.monotonic() - STARTED, best_objective), flush=True)


if __name__ == '__main__':
    main()
