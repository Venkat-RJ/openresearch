import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    start = time.monotonic()
    deadline = start + max(0, args.time_limit - 3)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None
    with open(args.instance_path) as handle:
        data = json.load(handle)
    n = data['n']
    A, B, C = [np.asarray(data[k], dtype=np.float64)[:n, :n] for k in ('A', 'B', 'C')]
    rows = np.arange(n)
    best = float('inf')
    best_p = rows.copy()

    def totals(p):
        return A[rows, p].sum(), B[rows, p].sum(), C[rows, p].sum()

    def accept(p):
        nonlocal best, best_p
        a, b, c = totals(p)
        obj = float(a * b + c)
        if obj < best:
            best, best_p = obj, p.copy()
            if logger:
                logger.log(obj)
        return a, b, c

    accept(best_p)
    scale_a = max(1., abs(A.mean()) * n)
    scale_b = max(1., abs(B.mean()) * n)
    for weight in (1., 0.5, 2., 0.1, 10., 0.01, 100.):
        if time.monotonic() >= deadline:
            break
        _, p = linear_sum_assignment(scale_b * weight * A + scale_a / weight * B + C)
        accept(p)
        for _ in range(4):
            if time.monotonic() >= deadline:
                break
            a, b, c = totals(p)
            _, q = linear_sum_assignment(b * A + a * B + C)
            qa, qb, qc = totals(q)
            accept(q)
            if qa * qb + qc >= a * b + c:
                break
            p = q
    if time.monotonic() < deadline:
        _, p = linear_sum_assignment(C)
        accept(p)
    p = best_p.copy()
    for _ in range(100):
        if time.monotonic() >= deadline:
            break
        a, b, c = totals(p)
        da = A[:, p] + A[:, p].T - A[rows, p][:, None] - A[rows, p][None, :]
        db = B[:, p] + B[:, p].T - B[rows, p][:, None] - B[rows, p][None, :]
        dc = C[:, p] + C[:, p].T - C[rows, p][:, None] - C[rows, p][None, :]
        delta = b * da + a * db + da * db + dc
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] >= -1e-7:
            break
        p[i], p[j] = p[j], p[i]
        accept(p)
    solution = {'objective_value': best, 'assignment': best_p.tolist()}
    with open(args.solution_path, 'w') as handle:
        json.dump(solution, handle)
    print(json.dumps({'objective_value': best, 'algorithm_seconds': time.monotonic() - start}))


if __name__ == '__main__':
    main()
