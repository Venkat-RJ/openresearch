#!/usr/bin/env python3
"""Feasibility checker for the 2E-CVRPGC problem.

Checks constraints (2)-(23) from the mathematical formulation in math_model.txt,
plus constraint 24: objective-value consistency (recompute the obj from the
decision variables and reject if the reported value disagrees).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('first_echelon_routes', 'group_assignments', 'objective_value', 'second_echelon_routes')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('group_assignments',)
_FRONTIEROR_DERIVED_FIELD_RULES = ()
_FRONTIEROR_CONDITIONAL_FIELD_RULES = ()


def _frontieror_validate_solution_contract(solution):
    """Return a structured rejection when the public solution contract is absent."""
    violations = []
    if not isinstance(solution, dict):
        violations.append("solution must be a JSON object")
    else:
        missing = [
            field
            for field in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS
            if field not in solution or solution[field] is None
        ]
        if missing:
            violations.append(
                "missing required solution field(s): " + ", ".join(missing)
            )

        for discriminator, expected_value, fields in _FRONTIEROR_CONDITIONAL_FIELD_RULES:
            if solution.get(discriminator) != expected_value:
                continue
            conditional_missing = [
                field
                for field in fields
                if field not in solution or solution[field] is None
            ]
            if conditional_missing:
                violations.append(
                    "missing conditionally required solution field(s): "
                    + ", ".join(conditional_missing)
                )

        empty = [
            field
            for field in _FRONTIEROR_NONEMPTY_SOLUTION_FIELDS
            if field in solution and solution[field] in (None, [], {})
        ]
        if empty:
            violations.append(
                "empty required decision witness field(s): " + ", ".join(empty)
            )

        for target, operation, source, source_key in _FRONTIEROR_DERIVED_FIELD_RULES:
            try:
                if operation == "assignment_key_set":
                    reported_set = set(solution[target])
                    expected_set = {entry[source_key] for entry in solution[source]}
                    if reported_set != expected_set:
                        violations.append(
                            f"{target}={sorted(reported_set)} does not match the "
                            f"decision-derived value {sorted(expected_set)}"
                        )
                    continue

                reported = float(solution[target])
                if operation == "length":
                    expected = float(len(solution[source]))
                elif operation == "mapping_value":
                    expected = float(solution[source][source_key])
                else:
                    raise ValueError("unknown derived-field operation")
            except (KeyError, TypeError, ValueError, OverflowError):
                violations.append(
                    f"cannot derive {target} from decision witness field {source}"
                )
                continue
            if (
                reported != reported
                or expected != expected
                or reported in (float("inf"), float("-inf"))
                or expected in (float("inf"), float("-inf"))
                or abs(reported - expected) > 1e-6
            ):
                violations.append(
                    f"{target}={reported} does not match the decision-derived "
                    f"value {expected}"
                )

        if "objective_value" in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS:
            objective = solution.get("objective_value")
            if (
                isinstance(objective, bool)
                or not isinstance(objective, (int, float))
                or objective != objective
                or objective in (float("inf"), float("-inf"))
            ):
                violations.append("objective_value must be a finite JSON number")

    if not violations:
        return None
    return {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": violations,
        "violation_magnitudes": [],
    }


def _frontieror_write_unexpected_rejection(error):
    """Turn malformed candidate/checker exceptions into a normal rejection."""
    import json as _frontieror_json
    import sys as _frontieror_sys

    result_path = None
    for index, argument in enumerate(_frontieror_sys.argv):
        if argument == "--result_path" and index + 1 < len(_frontieror_sys.argv):
            result_path = _frontieror_sys.argv[index + 1]
            break
        if argument.startswith("--result_path="):
            result_path = argument.split("=", 1)[1]
            break
    if not result_path:
        return False
    try:
        with open(result_path, "w") as result_handle:
            _frontieror_json.dump(
                {
                    "feasible": False,
                    "violated_constraints": ["malformed_solution"],
                    "violations": [
                        "checker rejected malformed candidate: "
                        + type(error).__name__
                        + ": "
                        + str(error)[:500]
                    ],
                    "violation_magnitudes": [],
                },
                result_handle,
                indent=2,
            )
    except Exception:
        return False
    return True



def check_feasibility(inst, sol):
    """Run the full feasibility check and return a result dict."""
    tol = 1e-5
    eps = 1e-5

    # ---- Instance data ----
    depot_id = inst['depot']['id']
    sat_ids = [s['id'] for s in inst['satellites']]
    cust_ids = [c['id'] for c in inst['customers']]
    sat_ids_set = set(sat_ids)
    cust_ids_set = set(cust_ids)

    Q1 = inst['Q1']
    Q2 = inst['Q2']
    T1 = inst['T1']
    T2 = inst['T2']
    n_sat = len(sat_ids)

    demand = {c['id']: c['demand'] for c in inst['customers']}
    sat_capacity = {s['id']: s['capacity'] for s in inst['satellites']}
    sat_max_vehicles = {s['id']: s['max_vehicles'] for s in inst['satellites']}
    sat_handling_cost = {s['id']: s['handling_cost'] for s in inst['satellites']}

    groups = {g['group_id']: g['customer_ids'] for g in inst['groups']}
    cust_to_group = {c['id']: c['group'] for c in inst['customers']}

    V0_VS = [depot_id] + sat_ids
    VS_VC = sat_ids + cust_ids

    # Node coordinate lookup (depot + satellites + customers) used by the obj
    # recomputation to recover Euclidean travel costs c_{i,j}.
    coords = {inst['depot']['id']: (inst['depot']['x'], inst['depot']['y'])}
    for s in inst['satellites']:
        coords[s['id']] = (s['x'], s['y'])
    for c in inst['customers']:
        coords[c['id']] = (c['x'], c['y'])

    def edge_cost(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    # ---- Reconstruct decision variables from solution ----

    # First echelon: x[i,j,k] and w[s,k]
    x = {}
    w = {}
    for s in sat_ids:
        for k in range(T1):
            w[s, k] = 0.0

    for route_info in sol.get('first_echelon_routes', []):
        k = route_info['vehicle']
        route = route_info['route']
        for idx in range(len(route) - 1):
            x[route[idx], route[idx + 1], k] = 1
        for s_str, amount in route_info.get('deliveries', {}).items():
            w[int(s_str), k] = float(amount)

    # Second echelon: y[i,j] and f[i,j]
    y = {}
    f_var = {}
    cust_satellite = {}

    for route_info in sol.get('second_echelon_routes', []):
        sat = route_info['satellite']
        route = route_info['route']

        for idx in range(len(route) - 1):
            y[route[idx], route[idx + 1]] = 1

        for node in route:
            if node in cust_ids_set:
                cust_satellite[node] = sat

        # Compute flow along route
        route_custs = [n for n in route if n in cust_ids_set]
        total_demand = sum(demand[c] for c in route_custs)
        current_flow = total_demand
        for idx in range(len(route) - 1):
            i_node, j_node = route[idx], route[idx + 1]
            if i_node in cust_ids_set:
                current_flow -= demand[i_node]
            f_var[i_node, j_node] = current_flow

    # z[i,s]: customer i assigned to satellite s
    z = {}
    for cid in cust_ids:
        for s in sat_ids:
            z[cid, s] = 1 if cust_satellite.get(cid) == s else 0

    # v[h,s]: group h assigned to satellite s
    # Derive v from z using the first member of each group, since constraint 15
    # requires z[i,s] = v[C_h,s] for all i in C_h.  The solution's
    # group_assignments field may be absent or stale, so we reconstruct v from
    # the actual customer-satellite assignments (z) instead.
    v = {}
    for h, members in groups.items():
        representative = members[0]
        for s in sat_ids:
            v[h, s] = z.get((representative, s), 0)

    # u[s,k]: MTZ position variables (derived from route order)
    u = {}
    for s in sat_ids:
        for k in range(T1):
            u[s, k] = 0.0
    for route_info in sol.get('first_echelon_routes', []):
        k = route_info['vehicle']
        route = route_info['route']
        pos = 0
        for node in route:
            if node in sat_ids_set:
                pos += 1
                u[node, k] = float(pos)

    # ---- Check constraints ----
    violations = []
    violation_magnitudes = []
    violated_set = set()

    # Domain check: every node id appearing in any first-echelon route must
    # be in V0_VS (depot + satellites); every node id in any second-echelon
    # route must be in VS_VC (satellites + customers); every group_id and
    # customer_id in group_assignments must be valid.
    valid_V0_VS = set(V0_VS)
    valid_VS_VC = set(VS_VC)
    for r_idx, route_info in enumerate(sol.get('first_echelon_routes', [])):
        for node in route_info.get('route', []):
            if node not in valid_V0_VS:
                violated_set.add(0)
                violations.append(
                    f"Domain: first_echelon_routes[{r_idx}] node {node!r} "
                    f"not depot/satellite"
                )
                violation_magnitudes.append({
                    "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                    "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
                })
        for s_str in route_info.get('deliveries', {}):
            try:
                s_int = int(s_str)
            except (TypeError, ValueError):
                s_int = None
            if s_int not in sat_ids_set:
                violated_set.add(0)
                violations.append(
                    f"Domain: first_echelon_routes[{r_idx}] deliveries key "
                    f"{s_str!r} not a valid satellite id"
                )
                violation_magnitudes.append({
                    "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                    "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
                })
    for r_idx, route_info in enumerate(sol.get('second_echelon_routes', [])):
        if route_info.get('satellite') not in sat_ids_set:
            violated_set.add(0)
            violations.append(
                f"Domain: second_echelon_routes[{r_idx}] satellite "
                f"{route_info.get('satellite')!r} not a valid satellite id"
            )
            violation_magnitudes.append({
                "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
        for node in route_info.get('route', []):
            if node not in valid_VS_VC:
                violated_set.add(0)
                violations.append(
                    f"Domain: second_echelon_routes[{r_idx}] node {node!r} "
                    f"not satellite/customer"
                )
                violation_magnitudes.append({
                    "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                    "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
                })
    if violated_set:
        # Domain failure: skip downstream checks that index node ids
        return {
            "feasible": False,
            "violated_constraints": sorted(violated_set),
            "violations": violations,
            "violation_magnitudes": violation_magnitudes,
        }

    def add_violation(cnum, msg, lhs_val, rhs_val, viol_amount):
        violated_set.add(cnum)
        violations.append(msg)
        normalizer = max(abs(rhs_val), eps)
        violation_magnitudes.append({
            "constraint": cnum,
            "lhs": float(lhs_val),
            "rhs": float(rhs_val),
            "raw_excess": float(viol_amount),
            "normalizer": float(normalizer),
            "ratio": float(viol_amount / normalizer)
        })

    # ------ Constraint 2: Flow conservation first echelon ------
    for i in V0_VS:
        for k in range(T1):
            out_flow = sum(x.get((i, j, k), 0) for j in V0_VS if j != i)
            in_flow = sum(x.get((j, i, k), 0) for j in V0_VS if j != i)
            diff = abs(out_flow - in_flow)
            if diff > tol:
                add_violation(2,
                    f"Flow conservation violated at node {i}, vehicle {k}: out={out_flow}, in={in_flow}",
                    out_flow, in_flow, diff)

    # ------ Constraint 3: Each satellite visited at most once per vehicle ------
    for i in sat_ids:
        for k in range(T1):
            lhs = sum(x.get((i, j, k), 0) for j in V0_VS if j != i)
            rhs_val = 1.0
            if lhs > rhs_val + tol:
                add_violation(3,
                    f"Satellite {i} visited more than once by vehicle {k}: out-degree={lhs}",
                    lhs, rhs_val, lhs - rhs_val)

    # ------ Constraint 4: MTZ subtour elimination ------
    for k in range(T1):
        for i in V0_VS:
            for j in V0_VS:
                if i == j:
                    continue
                xval = x.get((i, j, k), 0)
                if i == depot_id and j in sat_ids_set:
                    lhs = 1.0
                    rhs_val = u.get((j, k), 0) + n_sat * (1 - xval)
                    if lhs > rhs_val + tol:
                        add_violation(4,
                            f"MTZ violated: depot->{j}, vehicle {k}: 1 > {rhs_val}",
                            lhs, rhs_val, lhs - rhs_val)
                elif i in sat_ids_set and j in sat_ids_set:
                    lhs = u.get((i, k), 0) + 1
                    rhs_val = u.get((j, k), 0) + n_sat * (1 - xval)
                    if lhs > rhs_val + tol:
                        add_violation(4,
                            f"MTZ violated: {i}->{j}, vehicle {k}: u[{i}]+1={lhs} > {rhs_val}",
                            lhs, rhs_val, lhs - rhs_val)

    # ------ Constraint 5: Delivery only if satellite is visited ------
    for s in sat_ids:
        for k in range(T1):
            visit = sum(x.get((s, j, k), 0) for j in V0_VS if j != s)
            rhs_val = sat_capacity[s] * visit
            lhs = w.get((s, k), 0)
            if lhs > rhs_val + tol:
                add_violation(5,
                    f"Delivery to satellite {s} by vehicle {k} without visit: w={lhs}, limit={rhs_val}",
                    lhs, rhs_val, lhs - rhs_val)

    # ------ Constraint 6: Vehicle capacity first echelon ------
    for k in range(T1):
        lhs = sum(w.get((s, k), 0) for s in sat_ids)
        rhs_val = float(Q1)
        if lhs > rhs_val + tol:
            add_violation(6,
                f"First echelon vehicle {k} capacity exceeded: {lhs} > {rhs_val}",
                lhs, rhs_val, lhs - rhs_val)

    # ------ Constraint 7: Satellite capacity ------
    for s in sat_ids:
        lhs = sum(w.get((s, k), 0) for k in range(T1))
        rhs_val = float(sat_capacity[s])
        if lhs > rhs_val + tol:
            add_violation(7,
                f"Satellite {s} capacity exceeded: {lhs} > {rhs_val}",
                lhs, rhs_val, lhs - rhs_val)

    # ------ Constraint 8: Each customer visited exactly once (second echelon) ------
    for i in cust_ids:
        out_deg = sum(y.get((i, j), 0) for j in VS_VC
                      if i != j and not (i in sat_ids_set and j in sat_ids_set))
        in_deg = sum(y.get((j, i), 0) for j in VS_VC
                     if i != j and not (j in sat_ids_set and i in sat_ids_set))
        if abs(out_deg - 1) > tol:
            add_violation(8,
                f"Customer {i} out-degree={out_deg}, expected 1",
                float(out_deg), 1.0, abs(out_deg - 1))
        if abs(in_deg - 1) > tol:
            add_violation(8,
                f"Customer {i} in-degree={in_deg}, expected 1",
                float(in_deg), 1.0, abs(in_deg - 1))

    # ------ Constraint 9: Vehicle limit per satellite (second echelon) ------
    for s in sat_ids:
        lhs = sum(y.get((s, j), 0) for j in cust_ids)
        rhs_val = float(sat_max_vehicles[s])
        if lhs > rhs_val + tol:
            add_violation(9,
                f"Satellite {s} second-echelon vehicle limit exceeded: {lhs} > {rhs_val}",
                lhs, rhs_val, lhs - rhs_val)

    # ------ Constraint 10: Total vehicle limit (second echelon) ------
    total_routes = sum(y.get((s, j), 0) for s in sat_ids for j in cust_ids)
    rhs_val = float(T2)
    if total_routes > rhs_val + tol:
        add_violation(10,
            f"Total second-echelon vehicles exceeded: {total_routes} > {rhs_val}",
            float(total_routes), rhs_val, total_routes - rhs_val)

    # ------ Constraint 11: Flow conservation for demand ------
    for i in cust_ids:
        flow_in = sum(f_var.get((j, i), 0) for j in VS_VC
                      if j != i and not (j in sat_ids_set and i in sat_ids_set))
        flow_out = sum(f_var.get((i, j), 0) for j in VS_VC
                       if j != i and not (i in sat_ids_set and j in sat_ids_set))
        lhs = flow_in
        rhs_val = flow_out + demand[i]
        diff = abs(lhs - rhs_val)
        if diff > tol:
            add_violation(11,
                f"Flow conservation violated at customer {i}: flow_in={lhs}, flow_out+demand={rhs_val}",
                lhs, rhs_val, diff)

    # ------ Constraint 12: Flow bounds (Gavish) ------
    for i in VS_VC:
        for j in VS_VC:
            if i == j or (i in sat_ids_set and j in sat_ids_set):
                continue
            yval = y.get((i, j), 0)
            fval = f_var.get((i, j), 0.0)
            d_i = demand.get(i, 0)
            d_j = demand.get(j, 0)

            lb = d_j * yval
            if fval < lb - tol:
                add_violation(12,
                    f"Flow lower bound violated on arc ({i},{j}): f={fval} < d_j*y={lb}",
                    fval, lb, lb - fval)

            ub = (Q2 - d_i) * yval
            if fval > ub + tol:
                add_violation(12,
                    f"Flow upper bound violated on arc ({i},{j}): f={fval} > (Q2-d_i)*y={ub}",
                    fval, ub, fval - ub)

    # ------ Constraint 13: Satellite assignment propagation ------
    for i in VS_VC:
        for j in VS_VC:
            if i == j or (i in sat_ids_set and j in sat_ids_set):
                continue
            yval = y.get((i, j), 0)
            if yval == 0:
                continue
            for s in sat_ids:
                if i in cust_ids_set:
                    z_is = z.get((i, s), 0)
                elif i == s:
                    z_is = 1
                else:
                    z_is = 0

                if j not in cust_ids_set:
                    continue
                z_js = z.get((j, s), 0)

                rhs_val = z_is + yval - 1
                if z_js < rhs_val - tol:
                    add_violation(13,
                        f"Assignment propagation violated: z[{j},{s}]={z_js} < z[{i},{s}]+y[{i},{j}]-1={rhs_val}",
                        float(z_js), rhs_val, rhs_val - z_js)

    # ------ Constraint 14: z[s,s] = 1; route start/end satellite ------
    for r_idx, route_info in enumerate(sol.get('second_echelon_routes', [])):
        sat = route_info['satellite']
        route = route_info['route']
        if not route:
            continue
        if route[0] != sat or route[-1] != sat:
            add_violation(14,
                f"Second-echelon route {r_idx} does not start and end at satellite {sat}: start={route[0]}, end={route[-1]}",
                0.0, 0.0, 1.0)

    # ------ Constraint 15: Grouping constraint ------
    for h, members in groups.items():
        for cid in members:
            for s in sat_ids:
                z_val = z.get((cid, s), 0)
                v_val = v.get((h, s), 0)
                diff = abs(z_val - v_val)
                if diff > tol:
                    add_violation(15,
                        f"Grouping violated: customer {cid} (group {h}), satellite {s}: z={z_val}, v={v_val}",
                        float(z_val), float(v_val), diff)

    # ------ Constraint 16: Linking constraint ------
    for s in sat_ids:
        lhs = sum(w.get((s, k), 0) for k in range(T1))
        rhs_val = sum(f_var.get((s, j), 0) for j in cust_ids)
        diff = abs(lhs - rhs_val)
        if diff > tol:
            add_violation(16,
                f"Linking constraint violated at satellite {s}: sum_w={lhs}, sum_f={rhs_val}",
                lhs, rhs_val, diff)

    # ------ Constraint 17: x binary ------
    for key, val in x.items():
        if val not in (0, 1):
            add_violation(17, f"x{key} not binary: {val}",
                          float(val), round(val), abs(val - round(val)))

    # ------ Constraint 18: w >= 0 ------
    for key, val in w.items():
        if val < -tol:
            add_violation(18, f"w{key} negative: {val}", float(val), 0.0, -val)

    # ------ Constraint 19: u >= 0 ------
    for key, val in u.items():
        if val < -tol:
            add_violation(19, f"u{key} negative: {val}", float(val), 0.0, -val)

    # ------ Constraint 20: f >= 0 ------
    for key, val in f_var.items():
        if val < -tol:
            add_violation(20, f"f{key} negative: {val}", float(val), 0.0, -val)

    # ------ Constraint 21: y binary ------
    for key, val in y.items():
        if val not in (0, 1):
            add_violation(21, f"y{key} not binary: {val}",
                          float(val), round(val), abs(val - round(val)))

    # ------ Constraint 22: z binary ------
    for key, val in z.items():
        if val not in (0, 1):
            add_violation(22, f"z{key} not binary: {val}",
                          float(val), round(val), abs(val - round(val)))

    # ------ Constraint 23: v binary ------
    for key, val in v.items():
        if val not in (0, 1):
            add_violation(23, f"v{key} not binary: {val}",
                          float(val), round(val), abs(val - round(val)))

    # ------ Constraint 24: Objective consistency (Tier C) ------
    # Recompute the total cost from the reconstructed decision variables and
    # reject when the reported objective_value disagrees beyond tolerance.
    # obj = sum_{k,(i,j)} c[i,j] * x[i,j,k]   (echelon 1 routing)
    #     + sum_s h_s * sum_k w[s,k]          (handling at satellites)
    #     + sum_{(i,j) in A2} c[i,j] * y[i,j] (echelon 2 routing)
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            echelon1_cost = sum(edge_cost(i, j) for (i, j, k) in x)
            handling_cost = sum(
                sat_handling_cost[s] * w.get((s, k), 0.0)
                for s in sat_ids for k in range(T1)
            )
            echelon2_cost = sum(edge_cost(i, j) for (i, j) in y)
            true_obj = float(echelon1_cost + handling_cost + echelon2_cost)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                add_violation(24,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed total cost="
                    f"{true_obj} (echelon1={echelon1_cost:.6g}, "
                    f"handling={handling_cost:.6g}, echelon2={echelon2_cost:.6g}; "
                    f"|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    reported, true_obj, abs_diff)
        elif reported is not None:
            # Non-finite reported obj (nan, inf): reject as a Tier-C exploit.
            echelon1_cost = sum(edge_cost(i, j) for (i, j, k) in x)
            handling_cost = sum(
                sat_handling_cost[s] * w.get((s, k), 0.0)
                for s in sat_ids for k in range(T1)
            )
            echelon2_cost = sum(edge_cost(i, j) for (i, j) in y)
            true_obj = float(echelon1_cost + handling_cost + echelon2_cost)
            add_violation(24,
                f"Objective consistency violated: reported objective_value="
                f"{reported} is not finite; recomputed total cost={true_obj}",
                reported, true_obj, float('inf'))

    # ---- Build result ----
    return {
        "feasible": len(violated_set) == 0,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for 2E-CVRPGC")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.instance_path) as fh:
        inst = json.load(fh)
    with open(args.solution_path) as fh:
        sol = json.load(fh)
        _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
        if _frontieror_contract_result is not None:
            with open(args.result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return

    result = check_feasibility(inst, sol)

    with open(args.result_path, 'w') as fh:
        json.dump(result, fh, indent=2)

    if not result['feasible'] and 0 in result['violated_constraints'] and \
            all(c == 0 for c in result['violated_constraints']):
        print("INFEASIBLE: domain violations")
        for v in result['violations']:
            print(f"  - {v}")
        return

    print(f"Feasibility: {result['feasible']}")
    if result['violated_constraints']:
        print(f"Violated constraints: {result['violated_constraints']}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
