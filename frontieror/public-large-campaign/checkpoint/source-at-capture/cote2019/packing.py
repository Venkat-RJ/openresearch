import itertools
from collections import defaultdict
TOL=1e-5

def compute_expected_area_weight(customers):
    result = {}
    for cust in customers:
        cid = cust['id']
        a_tilde = 0.0
        q_tilde = 0.0
        for item in cust['items']:
            for r in item['realizations']:
                p = r['probability']
                a_tilde += p * r['height'] * r['width']
                q_tilde += p * r['weight']
        result[cid] = (a_tilde, q_tilde)
    return result

def bottom_left_heuristic_with_unloading(items_with_delivery_order, H, W):
    sorted_items = sorted(items_with_delivery_order, key=lambda t: -t[2])
    placements = []
    for ih, iw, dpos in sorted_items:
        if ih > H or iw > W:
            return False
        placed = False
        y_candidates = sorted(set([0] + [p[1] + p[3] for p in placements]))
        x_candidates = sorted(set([0] + [p[0] + p[2] for p in placements]))
        best_pos = None
        for y in y_candidates:
            if y + ih > H:
                continue
            for x in x_candidates:
                if x + iw > W:
                    continue
                overlap = False
                for px, py, pw, ph, pd in placements:
                    if x < px + pw and x + iw > px and (y < py + ph) and (y + ih > py):
                        overlap = True
                        break
                if overlap:
                    continue
                unloading_ok = True
                for px, py, pw, ph, pd in placements:
                    if pd > dpos:
                        if px < x + iw and px + pw > x and (py + ph > y + ih) and (py < H):
                            unloading_ok = False
                            break
                if not unloading_ok:
                    continue
                if best_pos is None or (y, x) < (best_pos[1], best_pos[0]):
                    best_pos = (x, y)
            if best_pos is not None and best_pos[1] == y:
                break
        if best_pos is None:
            return False
        placements.append((best_pos[0], best_pos[1], iw, ih, dpos))
    return True

def check_packing_feasibility_heuristic(items_by_customer_ordered, H, W):
    if not items_by_customer_ordered:
        return True
    total_area = sum((h * w for _, items in items_by_customer_ordered for h, w in items))
    if total_area > H * W:
        return False
    for _, items in items_by_customer_ordered:
        for h, w in items:
            if h > H or w > W:
                return False
    pack_items = []
    for pos_idx, (cid, cust_items) in enumerate(items_by_customer_ordered):
        delivery_pos = pos_idx + 1
        for h, w in cust_items:
            pack_items.append((h, w, delivery_pos))
    return bottom_left_heuristic_with_unloading(pack_items, H, W)

def _solve_2opp_exact(pack_items, H, W):
    if not pack_items:
        return True
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as error:
        raise RuntimeError(f'SciPy MILP unavailable: {error}') from error
    lower = []
    upper = []
    integrality = []

    def variable(lb, ub, integer=False):
        index = len(lower)
        lower.append(float(lb))
        upper.append(float(ub))
        integrality.append(1 if integer else 0)
        return index
    x_index = [variable(0.0, W - item[1]) for item in pack_items]
    y_index = [variable(0.0, H - item[0]) for item in pack_items]
    rows = []
    row_lower = []
    row_upper = []

    def constraint(coefficients, lb=-float('inf'), ub=float('inf')):
        rows.append(dict(coefficients))
        row_lower.append(float(lb))
        row_upper.append(float(ub))
    for i in range(len(pack_items)):
        hi, wi, delivery_i = pack_items[i]
        for j in range(i + 1, len(pack_items)):
            hj, wj, delivery_j = pack_items[j]
            separated = [variable(0.0, 1.0, True) for _ in range(4)]
            constraint({x_index[i]: 1.0, x_index[j]: -1.0, separated[0]: W}, ub=W - wi)
            constraint({x_index[j]: 1.0, x_index[i]: -1.0, separated[1]: W}, ub=W - wj)
            constraint({y_index[i]: 1.0, y_index[j]: -1.0, separated[2]: H}, ub=H - hi)
            constraint({y_index[j]: 1.0, y_index[i]: -1.0, separated[3]: H}, ub=H - hj)
            constraint({index: 1.0 for index in separated}, lb=1.0)
            if delivery_i < delivery_j:
                upper[separated[2]] = 0.0
            elif delivery_j < delivery_i:
                upper[separated[3]] = 0.0
    row_indices = []
    column_indices = []
    values = []
    for row_index, coefficients in enumerate(rows):
        for column_index, value in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(value)
    matrix = coo_matrix((values, (row_indices, column_indices)), shape=(len(rows), len(lower)), dtype=float).tocsr()
    result = milp(c=np.zeros(len(lower), dtype=float), integrality=np.asarray(integrality, dtype=int), bounds=Bounds(np.asarray(lower), np.asarray(upper)), constraints=LinearConstraint(matrix, np.asarray(row_lower), np.asarray(row_upper)), options={'presolve': True, 'time_limit': 10.0, 'threads': 1})
    return bool(result.success)

def _scenario_is_packable(scenario_items, H, W, Q):
    total_weight = sum((weight for _, items in scenario_items for _, _, weight in items))
    if total_weight > Q + TOL:
        return False
    pack_items = []
    for delivery_position, (_, items) in enumerate(scenario_items, start=1):
        for height, width, _ in items:
            if height > H + TOL or width > W + TOL:
                return False
            pack_items.append((height, width, delivery_position))
    if sum((height * width for height, width, _ in pack_items)) > H * W + TOL:
        return False
    if bottom_left_heuristic_with_unloading(pack_items, H, W):
        return True
    return _solve_2opp_exact(pack_items, H, W)

def _count_unserved(delivery_order, scenario_items, H, W, Q):
    by_customer = {customer: items for customer, items in scenario_items}
    served = list(delivery_order)
    for drop_count in range(1, len(delivery_order)):
        to_drop = max(served, key=lambda customer: sum((height * width for height, width, _ in by_customer[customer])))
        served.remove(to_drop)
        remaining = [(customer, by_customer[customer]) for customer in served]
        if _scenario_is_packable(remaining, H, W, Q):
            return drop_count
    return len(delivery_order)

def _route_recourse_for_ordering(delivery_order, customers_by_id, H, W, Q, cf):
    deterministic = all((len(item['realizations']) == 1 for customer in delivery_order for item in customers_by_id[customer]['items']))
    scenarios = enumerate_scenarios_for_route(delivery_order, customers_by_id)
    if deterministic:
        return 0.0 if _scenario_is_packable(scenarios[0][1], H, W, Q) else float(cf) * len(delivery_order)
    recourse = 0.0
    for probability, scenario_items in scenarios:
        if probability < 1e-15:
            continue
        if not _scenario_is_packable(scenario_items, H, W, Q):
            recourse += probability * float(cf) * _count_unserved(delivery_order, scenario_items, H, W, Q)
    return recourse

def compute_route_recourse(route, customers_by_id, H, W, Q, cf):
    if not route:
        return 0.0
    return min((_route_recourse_for_ordering(ordering, customers_by_id, H, W, Q, cf) for ordering in (list(route), list(reversed(route)))))

def enumerate_scenarios_for_route(route_customer_ids, customers_by_id):
    item_realizations = []
    for cid in route_customer_ids:
        cust = customers_by_id[cid]
        for item in cust['items']:
            reals = []
            for r in item['realizations']:
                reals.append((r['height'], r['width'], r['weight'], r['probability'], cid))
            item_realizations.append(reals)
    if not item_realizations:
        return [(1.0, [])]
    scenarios = []
    for combo in itertools.product(*item_realizations):
        prob = 1.0
        cust_items = defaultdict(list)
        for h, w, weight, p, cid in combo:
            prob *= p
            cust_items[cid].append((h, w, weight))
        scenario_data = [(cid, cust_items[cid]) for cid in route_customer_ids if cid in cust_items]
        scenarios.append((prob, scenario_data))
    return scenarios

def is_route_always_infeasible(route, customers_by_id, H, W, Q):
    n_scenarios = 1
    for cid in route:
        cust = customers_by_id[cid]
        for item in cust['items']:
            n_scenarios *= len(item['realizations'])
    if n_scenarios > 50000:
        return False
    scenarios = enumerate_scenarios_for_route(route, customers_by_id)
    for prob, scenario_items in scenarios:
        total_weight = sum((w for _, items in scenario_items for _, _, w in items))
        if total_weight > Q:
            continue
        for ordering in [scenario_items, list(reversed(scenario_items))]:
            items_ordered = [(cid, [(h, w) for h, w, _ in items]) for cid, items in ordering]
            if check_packing_feasibility_heuristic(items_ordered, H, W):
                return False
    return True