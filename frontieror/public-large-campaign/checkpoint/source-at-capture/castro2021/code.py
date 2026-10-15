import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import highspy


class Stream:
    def __init__(self, path):
        self.file = open(path, 'rb')
        self.buffer = b''
        self.position = 0
        self.decoder = json.JSONDecoder()

    def fill(self, size=1048576):
        rest = self.buffer[self.position:]
        block = self.file.read(size)
        self.buffer = rest + block
        self.position = 0
        return bool(block)

    def whitespace(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position] in b' \r\n\t':
                self.position += 1
            if self.position < len(self.buffer) or not self.fill():
                return

    def peek(self):
        self.whitespace()
        return self.buffer[self.position:self.position + 1]

    def expect(self, character):
        if self.peek() != character:
            raise ValueError('Unexpected JSON token, expected ' + repr(character))
        self.position += 1

    def value(self):
        self.whitespace()
        while True:
            try:
                text = self.buffer[self.position:].decode('utf-8')
                result, end = self.decoder.raw_decode(text)
                self.position += len(text[:end].encode('utf-8'))
                return result
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not self.fill(max(1048576, len(self.buffer) - self.position)):
                    raise ValueError('Incomplete JSON value')

    def skip_numeric_array(self):
        self.expect(b'[')
        depth = 1
        while depth:
            for match in re.finditer(rb'[\[\]]', self.buffer[self.position:]):
                depth += 1 if match.group() == b'[' else -1
                if depth == 0:
                    self.position += match.end()
                    return
            self.position = len(self.buffer)
            if not self.fill():
                raise ValueError('Incomplete numeric array')

    def matrix(self, n, m, integer=False):
        self.expect(b'[')
        for row in range(n):
            if row:
                self.expect(b',')
            self.expect(b'[')
            while True:
                end = self.buffer.find(b']', self.position)
                if end >= 0:
                    raw = self.buffer[self.position:end]
                    self.position = end + 1
                    break
                if not self.fill(max(1048576, len(self.buffer) - self.position)):
                    raise ValueError('Incomplete coefficient row')
            if integer:
                if re.search(rb'[^0-9,+\-\s]', raw) or re.search(rb'\d{19,}', raw):
                    raise ValueError('Integer coefficient outside exact parser range')
                values = np.fromstring(raw.decode('ascii'), dtype=np.int64, sep=',')
                if values.size and np.max(np.abs(values)) > 2 ** 53:
                    raise ValueError('Integer coefficient cannot be represented exactly in float64')
            else:
                values = np.fromstring(raw.decode('ascii'), dtype=np.float64, sep=',')
            if len(values) != m or not np.isfinite(values).all():
                raise ValueError('Coefficient shape or finite-value contract violated')
            yield values
        self.expect(b']')


def trace(event, **fields):
    print(json.dumps({'event': event, **fields}), file=sys.stderr, flush=True)


def read_instance(path, directory):
    reader = Stream(path)
    data = {}
    costs = quadratic = None
    ranges = {}
    reader.expect(b'{')
    first = True
    while reader.peek() != b'}':
        if not first:
            reader.expect(b',')
        first = False
        key = reader.value()
        reader.expect(b':')
        if key in ('supplier_locations', 'customer_locations'):
            reader.skip_numeric_array()
        elif key in ('linear_costs', 'quadratic_costs'):
            n, m = int(data['n']), int(data['m'])
            integer = key == 'linear_costs' and data.get('cost_type') == 'linear_integer'
            low, high = math.inf, -math.inf
            if key == 'linear_costs':
                costs = np.memmap(Path(directory) / 'costs.bin', dtype=np.float64, mode='w+', shape=(n, m))
            for row, values in enumerate(reader.matrix(n, m, integer)):
                low = min(low, float(values.min()))
                high = max(high, float(values.max()))
                if key == 'linear_costs':
                    costs[row] = values
                else:
                    if np.any(values < 0):
                        raise ValueError('Nonconvex quadratic coefficients are outside this convex-flow solver')
                    if quadratic is None and np.any(values != 0):
                        quadratic = np.memmap(Path(directory) / 'quadratic.bin', dtype=np.float64, mode='w+', shape=(n, m))
                    if quadratic is not None:
                        quadratic[row] = values
            ranges[key] = {'minimum': low, 'maximum': high, 'coefficients_read': n * m}
        else:
            data[key] = reader.value()
    reader.expect(b'}')
    reader.file.close()
    if costs is None or 'quadratic_costs' not in ranges:
        raise ValueError('Both complete coefficient matrices are required')
    supplies = np.asarray(data['supplies'], dtype=np.float64)
    demands = np.asarray(data['demands'], dtype=np.float64)
    n, m = costs.shape
    if len(supplies) != n or len(demands) != m or np.any(supplies < 0) or np.any(demands < 0):
        raise ValueError('Supply/demand shape or sign contract violated')
    if float(supplies.sum()) + 1e-8 < float(demands.sum()):
        raise ValueError('Total public supplier capacity is below total public demand')
    cap = float(data['arc_capacity'])
    if cap < 0 or np.any(demands > n * cap + 1e-8):
        raise ValueError('Public arc capacities cannot carry a customer demand')
    trace('parsed_full_canonical_coefficients', n=n, m=m, ranges=ranges,
          zero_quadratic_verified=quadratic is None)
    return data, costs, quadratic, supplies, demands, cap


def column_oracle(costs, quadratic, demands, prices, flow, cap):
    n, m = costs.shape
    load = np.zeros(n)
    if cap < float(demands.max(initial=0)):
        raise ValueError('Implicit oracle requires verified per-arc capacity at least each customer demand')
    for first in range(0, m, 4096):
        last = min(m, first + 4096)
        demand = demands[first:last]
        adjusted = np.asarray(costs[:, first:last]) + prices[:, None]
        if quadratic is None:
            values = np.zeros(adjusted.shape)
            values[np.argmin(adjusted, axis=0), np.arange(last - first)] = demand
        else:
            q = np.asarray(quadratic[:, first:last])
            inverse = np.zeros(q.shape)
            np.divide(1.0, 2 * q, out=inverse, where=q > 0)
            order = np.argsort(adjusted, axis=0)
            alpha = np.take_along_axis(adjusted, order, axis=0)
            weight = np.take_along_axis(inverse, order, axis=0)
            cumulative = np.cumsum(weight, axis=0)
            numerator = demand[None, :] + np.cumsum(alpha * weight, axis=0)
            beta = np.full(cumulative.shape, np.inf)
            np.divide(numerator, cumulative, out=beta, where=cumulative > 0)
            following = np.vstack((alpha[1:], np.full((1, last - first), np.inf)))
            permitted = beta <= following
            index = np.argmax(permitted, axis=0)
            level = beta[index, np.arange(last - first)]
            linear_price = np.where(q == 0, adjusted, np.inf)
            linear_best = np.argmin(linear_price, axis=0)
            linear_minimum = linear_price[linear_best, np.arange(last - first)]
            level = np.minimum(level, linear_minimum)
            values = np.maximum(0, level[None, :] - adjusted) * inverse
            remainder = demand - np.sum(values, axis=0)
            recipient = np.where(linear_minimum <= level, linear_best, np.argmin(adjusted, axis=0))
            values[recipient, np.arange(last - first)] += remainder
            if np.any(values < -1e-8):
                raise RuntimeError('Convex column oracle produced a negative quantity')
            values = np.maximum(values, 0)
        flow[:, first:last] = values
        load += np.sum(values, axis=1)
    return load


def repair_supply(costs, quadratic, flow, supplies, load, cap):
    spare = supplies - load
    n, m = flow.shape
    for source in np.flatnonzero(spare < -1e-7):
        customers = np.flatnonzero(flow[source] > 1e-10)
        priorities = np.empty(len(customers))
        destinations = np.flatnonzero(spare > 1e-8)
        for first in range(0, len(customers), 2048):
            cols = customers[first:first + 2048]
            marginal = np.asarray(costs[np.ix_(destinations, cols)])
            if quadratic is not None:
                marginal += 2 * quadratic[np.ix_(destinations, cols)] * flow[np.ix_(destinations, cols)]
                base = costs[source, cols] + 2 * quadratic[source, cols] * flow[source, cols]
            else:
                base = costs[source, cols]
            priorities[first:first + len(cols)] = marginal.min(axis=0) - base
        for customer in customers[np.argsort(priorities)]:
            while spare[source] < -1e-7 and flow[source, customer] > 1e-10:
                destinations = np.flatnonzero((spare > 1e-8) & (flow[:, customer] < cap - 1e-8))
                if len(destinations) == 0:
                    raise RuntimeError('No residual capacity for full canonical supply repair')
                marginal = np.asarray(costs[destinations, customer])
                if quadratic is not None:
                    marginal += 2 * quadratic[destinations, customer] * flow[destinations, customer]
                target = int(destinations[np.argmin(marginal)])
                amount = min(-spare[source], float(flow[source, customer]),
                             float(spare[target]), cap - float(flow[target, customer]))
                flow[source, customer] -= amount
                flow[target, customer] += amount
                spare[source] += amount
                spare[target] -= amount
            if spare[source] >= -1e-7:
                break
    if np.any(spare < -1e-6):
        raise RuntimeError('Full supplier capacity repair did not terminate')


def sparse_witness(costs, quadratic, flow):
    n, m = flow.shape
    codes, quantities = [], []
    objective = 0.0
    for first in range(0, m, 4096):
        last = min(m, first + 4096)
        i, j = np.nonzero(flow[:, first:last] > 0)
        j = j + first
        values = np.asarray(flow[i, j])
        codes.append((i.astype(np.int64) * m + j).astype(np.int64))
        quantities.append(values)
        objective += float(np.dot(costs[i, j], values))
        if quadratic is not None:
            objective += float(np.dot(quadratic[i, j], values * values))
    return objective, np.concatenate(codes), np.concatenate(quantities)


def model_for_arcs(costs, quadratic, supplies, demands, cap, codes):
    n, m = costs.shape
    suppliers = codes // m
    customers = codes % m
    count = len(codes)
    model = highspy.HighsModel()
    lp = model.lp_
    lp.num_col_ = count
    lp.num_row_ = n + m
    lp.col_cost_ = np.asarray(costs[suppliers, customers], dtype=np.float64)
    lp.col_lower_ = np.zeros(count)
    lp.col_upper_ = np.full(count, cap)
    lp.row_lower_ = np.concatenate((np.full(n, -highspy.kHighsInf), demands))
    lp.row_upper_ = np.concatenate((supplies, demands))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.arange(0, 2 * count + 1, 2, dtype=np.int32)
    indices = np.empty(2 * count, dtype=np.int32)
    indices[::2] = suppliers
    indices[1::2] = n + customers
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = np.ones(2 * count)
    if quadratic is not None:
        hessian = model.hessian_
        hessian.dim_ = count
        hessian.format_ = highspy.HessianFormat.kTriangular
        hessian.start_ = np.arange(count + 1, dtype=np.int32)
        hessian.index_ = np.arange(count, dtype=np.int32)
        hessian.value_ = 2 * quadratic[suppliers, customers]
    return model


def optimize_active(costs, quadratic, supplies, demands, cap, incumbent, prices, deadline):
    n, m = costs.shape
    if n * m <= 200000:
        codes = np.arange(n * m, dtype=np.int64)
    elif quadratic is not None:
        return incumbent
    else:
        chunks = [incumbent[1]]
        width = min(3, n)
        for first in range(0, m, 4096):
            last = min(m, first + 4096)
            adjusted = costs[:, first:last] + prices[:, None]
            chosen = np.argpartition(adjusted, width - 1, axis=0)[:width]
            cols = np.arange(first, last)[None, :]
            chunks.append((chosen.astype(np.int64) * m + cols).ravel())
        codes = np.unique(np.concatenate(chunks))
    for iteration in range(2):
        if deadline - time.monotonic() < 2:
            break
        model = model_for_arcs(costs, quadratic, supplies, demands, cap, codes)
        solver = highspy.Highs()
        solver.setOptionValue('threads', 1)
        solver.setOptionValue('output_flag', False)
        solver.setOptionValue('time_limit', max(.1, deadline - time.monotonic()))
        solver.passModel(model)
        hints = np.zeros(len(codes))
        hints[np.searchsorted(codes, incumbent[1])] = incumbent[2]
        solver.setSolution(len(codes), np.arange(len(codes), dtype=np.int32), hints)
        solver.run()
        information = solver.getInfo()
        solution = solver.getSolution()
        trace('active_model_result', columns=len(codes), iteration=iteration,
              status=str(solver.getModelStatus()), primal_status=int(information.primal_solution_status))
        if information.primal_solution_status != highspy.SolutionStatus.kSolutionStatusFeasible:
            break
        values = np.asarray(solution.col_value)
        positive = values > 0
        i = codes[positive] // m
        j = codes[positive] % m
        objective = float(np.dot(costs[i, j], values[positive]))
        if quadratic is not None:
            objective += float(np.dot(quadratic[i, j], values[positive] ** 2))
        if objective < incumbent[0]:
            incumbent = objective, codes[positive], values[positive]
        if quadratic is not None or deadline - time.monotonic() < 4:
            break
        dual = np.asarray(solution.row_dual)
        additions = []
        for first in range(0, m, 4096):
            last = min(m, first + 4096)
            reduced = costs[:, first:last] - dual[:n, None] - dual[n + first:n + last][None, :]
            row = np.argmin(reduced, axis=0)
            negative = reduced[row, np.arange(last - first)] < -1e-7
            additions.append(row[negative].astype(np.int64) * m + np.arange(first, last)[negative])
        expanded = np.unique(np.concatenate([codes] + additions))
        trace('full_bipartite_pricing', arcs_examined=n * m, added_arcs=len(expanded) - len(codes))
        if len(expanded) == len(codes):
            break
        codes = expanded
    return incumbent


def write_solution(path, incumbent, m):
    objective, codes, values = incumbent
    with open(path, 'w') as output:
        output.write('{"objective_value":' + repr(float(objective)) + ',"flows":{')
        first = True
        for code, value in zip(codes, values):
            if value <= 0:
                continue
            if not first:
                output.write(',')
            first = False
            i, j = divmod(int(code), m)
            output.write('"x_%d_%d":%s' % (i, j, repr(float(value))))
        output.write('}}')


def main():
    parser = argparse.ArgumentParser()
    for name in ('instance_path', 'solution_path', 'time_limit', 'log_path'):
        parser.add_argument('--' + name, required=name != 'log_path')
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + max(.1, float(args.time_limit) - 3)
    with tempfile.TemporaryDirectory(prefix='castro-canonical-') as directory:
        data, costs, quadratic, supplies, demands, cap = read_instance(args.instance_path, directory)
        n, m = costs.shape
        flow = np.memmap(Path(directory) / 'flow.bin', dtype=np.float64, mode='w+', shape=(n, m))
        prices = np.zeros(n)
        incumbent = None
        iterations = 1 if n * m <= 200000 else (6 if quadratic is None else 10)
        for iteration in range(iterations):
            before = time.monotonic()
            load = column_oracle(costs, quadratic, demands, prices, flow, cap)
            imbalance = load - supplies
            repair_supply(costs, quadratic, flow, supplies, load, cap)
            candidate = sparse_witness(costs, quadratic, flow)
            if incumbent is None or candidate[0] < incumbent[0]:
                incumbent = candidate
            trace('canonical_primal_incumbent', iteration=iteration,
                  nonzero_arcs=len(incumbent[1]), elapsed=time.monotonic() - started)
            elapsed = time.monotonic() - before
            reserve = max(5, 2 + len(incumbent[1]) / 500000)
            if deadline - time.monotonic() < reserve + 2 * elapsed:
                break
            scale = max(1, float(np.mean(supplies)))
            prices = np.maximum(0, prices + 100 * imbalance / (scale * math.sqrt(iteration + 1)))
            if abs(float(supplies.sum()) - float(demands.sum())) < 1e-6:
                prices -= prices.min()
        reserve = max(3, 2 + len(incumbent[1]) / 500000)
        incumbent = optimize_active(costs, quadratic, supplies, demands, cap, incumbent,
                                    prices, deadline - reserve)
        write_solution(args.solution_path, incumbent, m)
        trace('full_canonical_witness_written', objective=incumbent[0],
              nonzero_arcs=len(incumbent[1]), elapsed=time.monotonic() - started)


if __name__ == '__main__':
    main()
