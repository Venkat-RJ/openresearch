import argparse
import heapq
import json
import os
import time
from collections import deque

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import highspy
import numpy as np
from scipy.sparse import coo_matrix


def flow_cost(arc, commodity):
    values = arc.get('flow_costs')
    if isinstance(values, dict):
        return values.get(str(commodity), values.get(commodity, arc['flow_cost']))
    if isinstance(values, list) and 0 <= commodity < len(values):
        return values[commodity]
    return arc['flow_cost']


def vehicle_cost(fleet, arc):
    values = fleet.get('arc_fixed_costs')
    if isinstance(values, dict):
        return values.get(str(arc['id']), values.get(arc['id'], fleet['fixed_cost']))
    if isinstance(values, list) and 0 <= arc['id'] < len(values):
        return values[arc['id']]
    return fleet['fixed_cost']


def solve(data, limit, solution_path=None):
    deadline = time.monotonic() + max(.1, limit - 1.5)
    nodes = data['nodes']
    arcs = data['arcs']
    commodities = data['commodities']
    fleets = data['fleet_types']
    node_index = {v: i for i, v in enumerate(nodes)}
    n, a_count = len(nodes), len(arcs)
    tails = np.array([node_index[a['from']] for a in arcs], dtype=np.int32)
    heads = np.array([node_index[a['to']] for a in arcs], dtype=np.int32)
    adjacency = [[] for _ in nodes]
    for a, u in enumerate(tails):
        adjacency[u].append(a)
    fc = np.array([[flow_cost(arc, c['id']) for arc in arcs] for c in commodities], dtype=float)
    vc_all = np.array([[vehicle_cost(f, arc) for arc in arcs] for f in fleets], dtype=float)
    keep = []
    for f, fleet in enumerate(fleets):
        dominated = False
        for g, other in enumerate(fleets):
            if f == g or other['capacity'] < fleet['capacity']:
                continue
            if np.all(vc_all[g] <= vc_all[f]) and (other['capacity'] > fleet['capacity'] or np.any(vc_all[g] < vc_all[f]) or g < f):
                dominated = True
                break
        if not dominated:
            keep.append(f)
    vc = vc_all[keep]
    capacities = np.array([fleets[f]['capacity'] for f in keep], dtype=float)
    f_count = len(keep)
    groups, group_map, members = [], {}, []
    commodity_group = []
    for k, c in enumerate(commodities):
        signature = (c['origin'], tuple(fc[k]))
        if signature not in group_map:
            group_map[signature] = len(groups)
            groups.append((node_index[c['origin']], fc[k]))
            members.append([])
        g = group_map[signature]
        members[g].append(k)
        commodity_group.append(g)
    g_count = len(groups)
    best = None
    best_cost = float('inf')
    best_x, best_y = None, None

    def certificate(x, y):
        flows = {}
        obj = 0.
        for g in range(g_count):
            residual = np.maximum(x[g], 0.).copy()
            origin = groups[g][0]
            for k in members[g]:
                c = commodities[k]
                destination = node_index[c['destination']]
                remaining = float(c['demand'])
                while remaining > 1e-7:
                    parent = np.full(n, -1, dtype=np.int32)
                    seen = {origin}
                    queue = deque([origin])
                    while queue and destination not in seen:
                        u = queue.popleft()
                        for a in adjacency[u]:
                            v = int(heads[a])
                            if residual[a] > 1e-8 and v not in seen:
                                seen.add(v)
                                parent[v] = a
                                queue.append(v)
                    if destination not in seen:
                        if remaining <= 1e-5:
                            break
                        return None
                    path = []
                    v = destination
                    while v != origin:
                        a = int(parent[v])
                        path.append(a)
                        v = int(tails[a])
                    amount = min(remaining, min(residual[a] for a in path))
                    for a in path:
                        key = f'x_{c["id"]}_{arcs[a]["from"]}_{arcs[a]["to"]}'
                        flows[key] = flows.get(key, 0.) + float(amount)
                        obj += amount * fc[k, a]
                        residual[a] -= amount
                    remaining -= amount
        vehicles = {}
        for f, original in enumerate(keep):
            for a, arc in enumerate(arcs):
                count = int(round(y[f, a]))
                if count:
                    vehicles[f'y_{fleets[original]["id"]}_{arc["from"]}_{arc["to"]}'] = count
                    obj += vc[f, a] * count
        return {'objective_value': float(obj), 'flows': flows, 'vehicles': vehicles}

    def save(x, y):
        nonlocal best, best_cost, best_x, best_y
        out = certificate(x, y)
        if out is not None and out['objective_value'] < best_cost - 1e-7:
            best = out
            best_cost = out['objective_value']
            best_x, best_y = x.copy(), y.copy()
            if solution_path:
                with open(solution_path, 'w') as f:
                    json.dump(out, f, separators=(',', ':'))

    def routes(weight):
        x = np.zeros((g_count, a_count))
        cache = {}
        for k, c in enumerate(commodities):
            g = commodity_group[k]
            origin, costs = groups[g]
            if g not in cache:
                distance = [float('inf')] * n
                parent = [-1] * n
                distance[origin] = 0.
                queue = [(0., origin)]
                while queue:
                    value, u = heapq.heappop(queue)
                    if value != distance[u]:
                        continue
                    for a in adjacency[u]:
                        v = int(heads[a])
                        candidate = value + costs[a] + weight[a]
                        if candidate < distance[v] - 1e-10:
                            distance[v] = candidate
                            parent[v] = a
                            heapq.heappush(queue, (candidate, v))
                cache[g] = parent
            parent = cache[g]
            v = node_index[c['destination']]
            while v != origin:
                a = parent[v]
                if a < 0:
                    raise RuntimeError('Unreachable commodity destination')
                x[g, a] += c['demand']
                v = int(tails[a])
        return x

    def circulation(load, f):
        lower = np.ceil(np.maximum(load - 1e-9, 0.) / capacities[f]).astype(np.int64)
        imbalance = np.zeros(n, dtype=np.int64)
        np.add.at(imbalance, tails, lower)
        np.add.at(imbalance, heads, -lower)
        columns = np.arange(a_count, dtype=np.int32)
        incidence = coo_matrix((np.r_[np.ones(a_count), -np.ones(a_count)], (np.r_[tails, heads], np.r_[columns, columns])), shape=(n, a_count)).tocsr()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = a_count, n
        lp.col_cost_ = vc[f]
        lp.col_lower_, lp.col_upper_ = np.zeros(a_count), np.full(a_count, max(1, int(lower.sum())))
        lp.row_lower_, lp.row_upper_ = -imbalance.astype(float), -imbalance.astype(float)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = incidence.indptr, incidence.indices, incidence.data
        network = highspy.Highs()
        network.setOptionValue('threads', 1)
        network.setOptionValue('output_flag', False)
        network.setOptionValue('solver', 'simplex')
        network.passModel(lp)
        network.run()
        if network.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        y = np.zeros((f_count, a_count))
        # Integral supplies and a network matrix make every basic solution integral.
        y[f] = lower + np.rint(network.getSolution().col_value)
        return y

    unit_vehicle = np.min(vc / capacities[:, None], axis=0)
    for factor in [1., 0., 2.]:
        if best is not None and time.monotonic() >= deadline:
            break
        x = routes(factor * unit_vehicle)
        load = x.sum(axis=0)
        for f in range(f_count):
            y = circulation(load, f)
            if y is not None:
                save(x, y)
    if best is None:
        raise RuntimeError('No balanced vehicle circulation for routed commodities')
    if time.monotonic() >= deadline:
        return best
    total = a_count * (g_count + f_count)
    cost = np.r_[np.concatenate([g[1] for g in groups]), vc.ravel()]
    lower = np.zeros(total)
    upper = np.full(total, np.inf)
    for g, group_members in enumerate(members):
        upper[g*a_count:(g+1)*a_count] = sum(commodities[k]['demand'] for k in group_members)
    upper[g_count*a_count:] = np.ceil(best_cost / np.maximum(vc.ravel(), 1e-8))
    rr, cc, vv, lbs, ubs = [], [], [], [], []

    def row(entries, lo, up):
        r = len(lbs)
        lbs.append(lo)
        ubs.append(up)
        for col, value in entries:
            rr.append(r)
            cc.append(col)
            vv.append(value)

    incident = [[] for _ in nodes]
    for a in range(a_count):
        incident[int(tails[a])].append((a, 1))
        incident[int(heads[a])].append((a, -1))
    for g, group_members in enumerate(members):
        supply = np.zeros(n)
        for k in group_members:
            c = commodities[k]
            supply[node_index[c['origin']]] += c['demand']
            supply[node_index[c['destination']]] -= c['demand']
        for v in range(n):
            row([(g*a_count+a, sign) for a, sign in incident[v]], supply[v], supply[v])
    for f in range(f_count):
        for v in range(n):
            row([((g_count+f)*a_count+a, sign) for a, sign in incident[v]], 0., 0.)
    for a in range(a_count):
        row([(g*a_count+a, 1.) for g in range(g_count)] + [((g_count+f)*a_count+a, -capacities[f]) for f in range(f_count)], -np.inf, 0.)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lbs), total)).tocsr()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = total, len(lbs)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lower, upper
    lp.row_lower_, lp.row_upper_ = np.array(lbs), np.array(ubs)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * (g_count*a_count) + [highspy.HighsVarType.kInteger] * (f_count*a_count)
    solver = highspy.Highs()
    solver.setOptionValue('threads', 1)
    solver.setOptionValue('output_flag', False)
    solver.setOptionValue('mip_rel_gap', .002)
    solver.passModel(lp)
    initial = np.r_[best_x.ravel(), best_y.ravel()]
    solver.setSolution(total, np.arange(total, dtype=np.int32), initial)

    def incumbent(event):
        x = np.array(event.data_out.mip_solution)
        save(x[:g_count*a_count].reshape(g_count, a_count), x[g_count*a_count:].reshape(f_count, a_count))

    solver.cbMipImprovingSolution += incumbent
    remaining = deadline - time.monotonic()
    if remaining > .05:
        solver.setOptionValue('time_limit', remaining)
        solver.run()
        solution = solver.getSolution()
        if solution.value_valid and solver.getInfo().primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
            x = np.array(solution.col_value)
            save(x[:g_count*a_count].reshape(g_count, a_count), x[g_count*a_count:].reshape(f_count, a_count))
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=float, default=60)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    with open(args.instance_path) as f:
        instance = json.load(f)
    out = solve(instance, args.time_limit, args.solution_path)
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
