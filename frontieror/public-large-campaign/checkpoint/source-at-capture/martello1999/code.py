import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import re
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from solution_logger import SolutionLogger

CHUNK = 65536
CORE = 4096


def load(path):
    decoder = json.JSONDecoder()
    with open(path) as f:
        buf = ''
        while '"items"' not in buf:
            part = f.read(1048576)
            if not part:
                raise ValueError('missing items')
            buf += part
        split = buf.index('"items"')
        header = buf[:split]
        n = int(re.search(r'"n"\s*:\s*(\d+)', header).group(1))
        cap_match = re.search(r'"capacity"\s*:\s*(\d+)', header)
        ids = np.empty(n, dtype=np.int64)
        weights = np.empty(n, dtype=np.int64)
        profits = np.empty(n, dtype=np.int64)
        pos = buf.index('[', split) + 1
        count = 0
        while True:
            while pos < len(buf) and buf[pos] in ' \r\n\t,':
                pos += 1
            if pos < len(buf) and buf[pos] == ']':
                tail = buf[pos + 1:] + f.read()
                break
            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                part = f.read(1048576)
                if not part:
                    raise
                buf = buf[pos:] + part
                pos = 0
                continue
            ids[count] = item['id']
            weights[count] = item['weight']
            profits[count] = item['profit']
            count += 1
            pos = end
        if count != n:
            raise ValueError('item count mismatch')
        if cap_match is None:
            cap_match = re.search(r'"capacity"\s*:\s*(\d+)', tail)
        return ids, weights, profits, int(cap_match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', required=True, type=int)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    start = time.monotonic()
    ids, w, p, capacity = load(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='maximize') if args.log_path else None
    selected = np.zeros(len(w), dtype=np.bool_)
    ratios = p.astype(np.float64)
    np.divide(ratios, w, out=ratios)
    best = -1

    def totals():
        weight = profit = 0
        for lo in range(0, len(w), CHUNK):
            hi = lo + CHUNK
            mask = selected[lo:hi]
            weight += int(w[lo:hi][mask].sum())
            profit += int(p[lo:hi][mask].sum())
        return weight, profit

    def save():
        nonlocal best
        _, value = totals()
        if value <= best:
            return
        temporary = args.solution_path + '.tmp'
        with open(temporary, 'w') as out:
            out.write('{"objective_value":' + str(value) + ',"selected_items":[')
            first = True
            for lo in range(0, len(w), CHUNK):
                values = ids[lo:lo + CHUNK][selected[lo:lo + CHUNK]]
                if len(values):
                    if not first:
                        out.write(',')
                    out.write(','.join(map(str, values)))
                    first = False
            out.write(']}')
        os.replace(temporary, args.solution_path)
        best = value
        if logger:
            logger.log(value)

    low, high = 0.0, float(ratios.max())
    # The high threshold always defines a capacity-feasible fixed prefix.
    for _ in range(40):
        mid = (low + high) / 2
        weight = 0
        for lo in range(0, len(w), CHUNK):
            weight += int(w[lo:lo + CHUNK][ratios[lo:lo + CHUNK] > mid].sum())
        if weight > capacity:
            low = mid
        else:
            high = mid
        if time.monotonic() - start > args.time_limit * .65:
            break
    np.greater(ratios, high, out=selected)
    used, _ = totals()
    remaining = capacity - used
    candidates = np.empty(0, dtype=np.int64)
    distances = np.empty(0, dtype=np.float64)
    for lo in range(0, len(w), CHUNK):
        distance = np.abs(ratios[lo:lo + CHUNK] - high)
        take = min(CORE, len(distance))
        local = np.argpartition(distance, take - 1)[:take]
        candidates = np.concatenate((candidates, local + lo))
        distances = np.concatenate((distances, distance[local]))
        if len(candidates) > CORE:
            keep = np.argpartition(distances, CORE - 1)[:CORE]
            candidates, distances = candidates[keep], distances[keep]
    order = candidates[np.argsort(-ratios[candidates], kind='stable')]
    for i in order:
        if not selected[i] and int(w[i]) <= remaining:
            selected[i] = True
            remaining -= int(w[i])
    save()
    budget = args.time_limit - (time.monotonic() - start) - 1
    if budget > .1:
        used, _ = totals()
        core_capacity = capacity - used + int(w[candidates][selected[candidates]].sum())
        result = milp(-p[candidates].astype(float), integrality=np.ones(len(candidates)),
                      bounds=Bounds(0, 1),
                      constraints=LinearConstraint(w[candidates].astype(float)[None, :], -np.inf, core_capacity),
                      options={'threads': 1, 'time_limit': budget, 'mip_rel_gap': 0.0})
        if result.x is not None:
            proposed = result.x > .5
            # Verify integer arithmetic at the numerical solver boundary.
            if int(w[candidates][proposed].sum()) <= core_capacity:
                selected[candidates] = proposed
                save()


if __name__ == '__main__':
    main()
