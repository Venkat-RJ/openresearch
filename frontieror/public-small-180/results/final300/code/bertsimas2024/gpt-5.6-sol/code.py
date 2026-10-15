import argparse
import json
import time
import os

if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):
    os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def config(model):
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.OutputFlag = 0


def solve(data, deadline, logger):
    N, K, R, E = (data[name] for name in ('num_nodes', 'num_commodities', 'num_scenarios', 'num_edges'))
    edges = data['edges']
    capacity, cost, fixed = data['capacities'], data['flow_costs'], data['construction_costs']
    demands, gamma = data['demands'], data['gamma']
    big = max(sum(max(0, b) for dem in scen for b in dem) for scen in demands)
    upper = [min(capacity[e], big) for e in range(E)]
    master = gp.Model()
    config(master)
    z = master.addVars(E, vtype=GRB.BINARY)
    theta = master.addVars(R)
    master.addConstr(z.sum() <= data['c_0'])
    master.setObjective(gp.quicksum(fixed[e] * z[e] for e in range(E)) + theta.sum() / R)
    lp = gp.Model()
    config(lp)
    lp.Params.Method = 1
    lp.Params.InfUnbdInfo = 1
    lp.Params.DualReductions = 0
    x = lp.addVars(K, E)
    q = lp.addVars(E, ub={e: upper[e] for e in range(E)})
    eta = lp.addVars(E)
    links, balance = {}, {}
    for e in range(E):
        lp.addConstr(q[e] == gp.quicksum(x[k, e] for k in range(K)))
        links[e] = lp.addConstr(q[e] <= big)
    for k in range(K):
        for n in range(N):
            balance[k, n] = lp.addConstr(gp.quicksum(x[k, e] * (int(i == n) - int(j == n)) for e, (i, j) in enumerate(edges)) == demands[0][k][n])
    lp.setObjective(gp.quicksum(cost[e] * q[e] + eta[e] for e in range(E)))
    points = [dict() for _ in range(E)]

    def tangent(e, t):
        key = round(t, 10)
        if key not in points[e]:
            points[e][key] = lp.addConstr(eta[e] >= t / gamma * q[e] - t * t / (2 * gamma))

    def reset_tangents():
        lp.remove([constraint for pool in points for constraint in pool.values()])
        for pool in points:
            pool.clear()
        for e in range(E):
            for fraction in (0, .1, .3, .6, 1):
                tangent(e, upper[e] * fraction)
        lp.update()

    def prune_tangents(totals):
        for e, pool in enumerate(points):
            keep = set(sorted(pool, key=lambda t: abs(t - totals[e]))[:8])
            keep.update((min(pool), max(pool)))
            for t in list(pool):
                if t not in keep:
                    lp.remove(pool.pop(t))
        lp.update()

    opened = [1] * E
    best, duration = None, 0.
    seen = set()
    for iteration in range(150):
        tick = time.monotonic()
        witness, used = {}, set()
        running, max_gap = 0., 0.
        constants, coefficients = [], []
        complete, infeasible = True, False
        for r in range(R):
            if time.monotonic() >= deadline:
                complete = False
                break
            reset_tangents()
            for k in range(K):
                for n in range(N):
                    balance[k, n].RHS = demands[r][k][n]
            for e in range(E):
                links[e].RHS = big * opened[e]
            local = min(deadline, time.monotonic() + max(.05, (deadline - time.monotonic()) / (R - r)))
            scenario_best = None
            for refine in range(160):
                lp.Params.TimeLimit = max(.01, local - time.monotonic())
                lp.optimize()
                if lp.Status == GRB.INFEASIBLE:
                    ray = [links[e].FarkasDual * big for e in range(E)]
                    master.addConstr(gp.quicksum(ray[e] * z[e] for e in range(E)) >= lp.FarkasProof + sum(ray[e] * opened[e] for e in range(E)))
                    complete, infeasible = False, True
                    break
                if lp.Status != GRB.OPTIMAL:
                    complete = False
                    break
                current = {(k, e): max(0, x[k, e].X) for k in range(K) for e in range(E)}
                totals = [sum(current[k, e] for k in range(K)) for e in range(E)]
                actual = sum(cost[e] * totals[e] + totals[e] ** 2 / (2 * gamma) for e in range(E))
                if scenario_best is None or actual < scenario_best[0]:
                    scenario_best = actual, current, totals
                gap = max(0., scenario_best[0] - lp.ObjVal)
                if gap <= 1e-8 * max(1., scenario_best[0]) or time.monotonic() >= local or refine == 159:
                    break
                violated = [e for e in range(E) if totals[e] ** 2 / (2 * gamma) - eta[e].X > 1e-9]
                if not violated:
                    break
                if lp.NumConstrs + len(violated) > 1650:
                    prune_tangents(totals)
                for e in violated:
                    tangent(e, totals[e])
            if not complete:
                break
            actual, current, totals = scenario_best
            max_gap = max(max_gap, gap)
            coeff = [links[e].Pi * big for e in range(E)]
            constants.append(lp.ObjVal - sum(coeff[e] * opened[e] for e in range(E)))
            coefficients.append(coeff)
            running += actual / R
            print('recourse', iteration, r, 'refinements', refine + 1, 'constraints', lp.NumConstrs, 'gap', gap, flush=True)
            for k in range(K):
                for e in range(E):
                    if current[k, e] > 1e-9:
                        witness[f'x[{k},{r},{e}]'] = current[k, e]
                        used.add(e)
            for e, total in enumerate(totals):
                if total > 1e-9:
                    witness[f'tf[{r},{e}]'] = total
        if not complete and not infeasible:
            break
        if complete and len(used) <= data['c_0']:
            objective = running + sum(fixed[e] for e in used)
            if best is None or objective < best['objective_value'] - 1e-7:
                for e in used:
                    witness[f'z[{e}]'] = 1
                best = {'objective_value': objective, 'z': {str(e): int(e in used) for e in range(E)}, 'model_variables': witness}
                if logger:
                    logger.log_solution(objective, best)
        for r in range(len(constants)):
            master.addConstr(theta[r] >= constants[r] + gp.quicksum(coefficients[r][e] * z[e] for e in range(E)))
        duration = max(duration, time.monotonic() - tick)
        seen.add(tuple(opened))
        if deadline - time.monotonic() < duration + .2:
            break
        for e in range(E):
            z[e].Start = best['z'][str(e)] if best else opened[e]
        master.Params.TimeLimit = max(.01, min(5., deadline - time.monotonic() - duration))
        master.optimize()
        if not master.SolCount:
            break
        if best and best['objective_value'] - master.ObjBound <= 1e-4 * max(1., abs(best['objective_value'])):
            break
        opened = [int(z[e].X > .5) for e in range(E)]
        if tuple(opened) in seen and max_gap <= 1e-6:
            break
    if best is None:
        raise RuntimeError('no complete network design witness')
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path')
    parser.add_argument('--solution_path')
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    start = time.monotonic()
    data = json.load(open(args.instance_path))
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None
    if logger:
        logger.min_interval = 0
        logger.start_time = time.time() - (time.monotonic() - start)
    result = solve(data, start + max(.1, args.time_limit - 2), logger)
    result['runtime'] = time.monotonic() - start
    if logger:
        logger.log_solution(result['objective_value'], result)
    json.dump(result, open(args.solution_path, 'w'))
