import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from array import array
import json
import time
import warnings
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix
from solution_logger import SolutionLogger


class JSONStream:
    def __init__(self, path):
        self.file = open(path, encoding='utf-8')
        self.buffer = ''
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.position:] + self.file.read(65536)
        self.position = 0
        if not self.buffer:
            self.eof = True

    def peek(self):
        while True:
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                if not char.isspace():
                    return char
                self.position += 1
            self.fill()
            if self.eof:
                raise ValueError('Unexpected end of JSON')

    def take(self, char):
        if self.peek() != char:
            raise ValueError('Expected ' + char)
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError:
                old_length = len(self.buffer) - self.position
                self.fill()
                if len(self.buffer) == old_length:
                    raise ValueError('Incomplete JSON value')


def load(path):
    stream = JSONStream(path)
    profits, weights, ids = array('q'), array('q'), array('q')
    offsets, group_ids = [0], []
    capacities = None
    stream.take('{')
    while stream.peek() != '}':
        key = stream.value()
        stream.take(':')
        if key == 'groups':
            stream.take('[')
            while stream.peek() != ']':
                group = stream.value()
                group_ids.append(group['group_id'])
                for item in group['items']:
                    profits.append(item['profit'])
                    ids.append(item['item_id'])
                    weights.extend(item['weights'])
                offsets.append(len(profits))
                del group
                if stream.peek() == ',':
                    stream.take(',')
            stream.take(']')
        else:
            value = stream.value()
            if key == 'capacities':
                capacities = np.asarray(value, dtype=np.int64)
        if stream.peek() == ',':
            stream.take(',')
    stream.take('}')
    stream.file.close()
    return (np.frombuffer(profits, dtype=np.int64),
            np.frombuffer(weights, dtype=np.int64).reshape(-1, len(capacities)),
            np.frombuffer(ids, dtype=np.int64), np.asarray(offsets, dtype=np.int64),
            group_ids, capacities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + args.time_limit - 0.5
    profits, weights, ids, offsets, groups, capacities = load(args.instance_path)
    n, resources = weights.shape
    count = len(groups)
    logger = SolutionLogger(args.log_path, sense='maximize') if args.log_path else None
    best = None
    best_value = -1

    def emit(chosen):
        nonlocal best, best_value
        if np.any(weights[chosen].sum(axis=0) > capacities):
            return
        value = int(profits[chosen].sum())
        if value <= best_value:
            return
        best, best_value = chosen.copy(), value
        solution = {'objective_value': value, 'selected_items': [
            {'group_id': groups[g], 'item_id': int(ids[j]),
             'profit': int(profits[j]), 'weights': weights[j].tolist()}
            for g, j in enumerate(chosen)]}
        temporary = args.solution_path + '.tmp'
        with open(temporary, 'w') as output:
            json.dump(solution, output)
        os.replace(temporary, args.solution_path)
        if logger:
            logger.log(value)

    scale = np.maximum(capacities, 1).astype(np.float64)
    price = np.zeros(resources)
    normalized_cost = np.einsum('ij,j->i', weights, 1.0 / scale, dtype=np.float64)
    pool = np.empty((count, 4), dtype=np.int64)
    for g, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        pool[g, :] = lo + np.argmin(normalized_cost[lo:hi])
        pool[g, 1] = lo + np.argmax(profits[lo:hi])
    emit(pool[:, 0])
    del normalized_cost
    chosen = np.empty(count, dtype=np.int64)
    step = max(1.0, float(profits.max()) * count / max(resources, 1))
    pricing_deadline = min(deadline, time.monotonic() + max(0.1, (deadline - time.monotonic()) * 0.25))
    scores = None
    for iteration in range(60):
        if time.monotonic() >= pricing_deadline:
            break
        scores = profits - np.einsum('ij,j->i', weights, price / scale, dtype=np.float64)
        for g, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            chosen[g] = lo + np.argmax(scores[lo:hi])
        emit(chosen)
        pool[:, 2 + iteration % 2] = chosen
        violation = weights[chosen].sum(axis=0) / scale - 1.0
        price = np.maximum(0.0, price + step / np.sqrt(iteration + 1) * violation)
    del scores

    # Restricted master retains every original row; pricing never certifies optimality.
    if n <= 5000:
        columns = np.arange(n, dtype=np.int64)
        owners = np.repeat(np.arange(count), np.diff(offsets))
    else:
        columns = np.unique(pool)
        owners = np.searchsorted(offsets[1:], columns, side='right')
    width = resources + 1
    size = len(columns)
    data = np.empty(size * width, dtype=np.float64)
    rows = np.empty(size * width, dtype=np.int32)
    data.reshape(size, width)[:, :resources] = weights[columns]
    data.reshape(size, width)[:, resources] = 1
    rows.reshape(size, width)[:, :resources] = np.arange(resources)
    rows.reshape(size, width)[:, resources] = resources + owners
    matrix = csc_matrix((data, rows, np.arange(0, len(data) + 1, width, dtype=np.int32)),
                        shape=(resources + count, size), copy=False)
    lower = np.concatenate((np.full(resources, -np.inf), np.ones(count)))
    upper = np.concatenate((capacities, np.ones(count)))
    remaining = deadline - time.monotonic()
    if remaining > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected')
            result = milp(-profits[columns].astype(np.float64),
                          integrality=np.ones(size, dtype=np.uint8), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={'time_limit': remaining, 'threads': 1,
                                   'mip_rel_gap': 0.0, 'presolve': True})
        if result.x is not None:
            selected = columns[result.x > 0.5]
            selected_owners = np.searchsorted(offsets[1:], selected, side='right')
            if len(selected) == count and np.array_equal(selected_owners, np.arange(count)):
                emit(selected)


if __name__ == '__main__':
    main()
