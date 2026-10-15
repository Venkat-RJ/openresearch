#!/usr/bin/env python3
"""
Feasibility checker for the Inventory-Routing Problem (IRP).

Checks a candidate solution against the compact arc-flow formulation (Part B)
from math_model.txt (Desaulniers, Rakke, and Coelho 2016).

Hard constraints checked (numbered top-to-bottom in Part B):
  1. Supplier inventory balance
  2. Customer inventory balance
  3. No stockouts (inventory non-negativity)
  4. Inventory capacity (customer)
  5. Supplier inventory capacity
  6. Delivery only if visited
  7. Flow conservation for vehicles
  8. Each vehicle leaves depot at most once
  9. Depot flow balance (leaves = returns)
 10. At most one visit per customer per period
 11. Subtour elimination (MTZ)
 12. Vehicle load constraints
 13. Variable domains (binary x, binary z, non-negative q, non-negative I)
 14. Objective consistency (reported objective_value must equal the
     objective recomputed from the solution variables: total vehicle
     travel cost plus end-of-period holding cost at every location)

NOTE: This is the obj-recompute-augmented variant of feasibility_check.py.
It is identical to the original for constraints 0..13; it ONLY adds the new
constraint 14 (Tier C defense against fabricated objective_value reports).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('inventories', 'objective_value', 'routes')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('inventories',)
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



TOL = 1e-5
EPS = 1e-5


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def load_solution(path):
    with open(path) as f:
        return json.load(f)


def parse_solution(instance, solution):
    """
    Parse solution routes into a uniform structure.
    Returns a list of (period, vehicle_index, route, deliveries) tuples.

    Handles two solution formats:
    - efficient: routes[period] is a list of route objects
    - gurobi:    routes[period] is a dict of vehicle-indexed route objects
    """
    num_periods = instance["num_periods"]
    parsed = []

    raw_routes = solution.get("routes", {})

    for p in range(1, num_periods + 1):
        period_data = raw_routes.get(str(p), {})

        if isinstance(period_data, list):
            # efficient format: list of route dicts
            for k_idx, route_obj in enumerate(period_data):
                route = route_obj["route"]
                deliveries = {}
                for cust_str, qty in route_obj.get("deliveries", {}).items():
                    deliveries[int(cust_str)] = float(qty)
                parsed.append((p, k_idx, route, deliveries))
        elif isinstance(period_data, dict):
            # gurobi format: dict keyed by vehicle index (string)
            for k_str, route_obj in period_data.items():
                route = route_obj["route"]
                deliveries = {}
                for cust_str, qty in route_obj.get("deliveries", {}).items():
                    deliveries[int(cust_str)] = float(qty)
                parsed.append((p, int(k_str), route, deliveries))
        # empty period: nothing to add

    return parsed


def check_feasibility(instance, solution):
    """Check all hard constraints. Return (feasible, violated_constraints, violations, violation_magnitudes)."""
    n = instance["num_customers"]
    T = instance["num_periods"]
    K = instance["num_vehicles"]
    Q = instance["vehicle_capacity"]
    customers = list(range(1, n + 1))
    periods = list(range(1, T + 1))

    dist = instance["distance_matrix"]
    h = instance["holding_costs"]
    demands = instance["customer_demands"]       # demands[i-1][p-1]
    production = instance["supplier_production"]  # production[p-1]
    C = instance["inventory_capacities"]          # C[i] for i in 0..n
    I0 = instance["initial_inventories"]          # I0[i] for i in 0..n

    def d(i, p):
        """Demand of customer i (1-indexed) in period p (1-indexed)."""
        return demands[i - 1][p - 1]

    def d0(p):
        """Production at supplier in period p (1-indexed)."""
        return production[p - 1]

    # Parse solution
    route_entries = parse_solution(instance, solution)

    # Build per-period structures
    # For each period p, collect the list of vehicle routes
    period_vehicles = {p: [] for p in periods}
    for (p, k_idx, route, deliveries) in route_entries:
        period_vehicles[p].append((k_idx, route, deliveries))

    # Aggregate deliveries: total_delivery[i][p] = total quantity delivered to customer i in period p
    total_delivery = {i: {p: 0.0 for p in periods} for i in customers}
    for (p, k_idx, route, deliveries) in route_entries:
        for ci, qty in deliveries.items():
            if ci in total_delivery:
                total_delivery[ci][p] += qty

    # Compute inventories
    # Supplier inventory
    sup_inv = {0: float(I0[0])}  # sup_inv[p] = inventory at end of period p; sup_inv[0] = initial
    # Customer inventory
    cust_inv = {i: {0: float(I0[i])} for i in customers}

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Domain check: every node id appearing in any route must be a valid
    # depot (0) or customer (1..n) id; every delivery key must be a valid
    # customer id.
    # =========================================================================
    valid_nodes = {0} | set(customers)
    for (p, k_idx, route, deliveries) in route_entries:
        for node in route:
            if node not in valid_nodes:
                violated_constraints.add(0)
                violations.append(
                    f"Domain: route in period {p}, vehicle {k_idx} contains "
                    f"invalid node id {node!r} (not depot 0 or customer 1..{n})"
                )
                violation_magnitudes.append({
                    "constraint": 0,
                    "lhs": float(node) if isinstance(node, (int, float)) else 0.0,
                    "rhs": 0.0, "raw_excess": 1.0,
                    "normalizer": 1.0, "ratio": 1.0,
                })
        for cid in deliveries:
            if cid not in customers:
                violated_constraints.add(0)
                violations.append(
                    f"Domain: deliveries in period {p}, vehicle {k_idx} "
                    f"references invalid customer id {cid!r}"
                )
                violation_magnitudes.append({
                    "constraint": 0,
                    "lhs": float(cid) if isinstance(cid, (int, float)) else 0.0,
                    "rhs": 0.0, "raw_excess": 1.0,
                    "normalizer": 1.0, "ratio": 1.0,
                })

    def record_violation(constraint_idx, message, lhs, rhs, op):
        """Record a violation with normalized magnitude."""
        if op == '=':
            violation_amount = abs(lhs - rhs)
        elif op in ('<=', '<'):
            violation_amount = max(0.0, lhs - rhs)
        elif op in ('>=', '>'):
            violation_amount = max(0.0, rhs - lhs)
        else:
            violation_amount = 0.0

        if violation_amount > TOL:
            violated_constraints.add(constraint_idx)
            violations.append(message)
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # =========================================================================
    # Constraint 1: Supplier inventory balance
    # I_{0p} = I_{0,p-1} + d_{0p} - sum_k sum_i q_{ip}^k,  for all p
    # =========================================================================
    for p in periods:
        prev_inv = sup_inv.get(p - 1, float(I0[0]))
        total_shipped = sum(
            sum(deliveries.values())
            for (pp, k_idx, route, deliveries) in route_entries if pp == p
        )
        computed_inv = prev_inv + d0(p) - total_shipped
        sup_inv[p] = computed_inv

        # The constraint is an equality: I_{0p} = I_{0,p-1} + d_{0p} - total_shipped
        # If inventories are provided in the solution, check them; otherwise this is
        # automatically satisfied by our computation.
        # However, we still track sup_inv[p] for later constraint checks.

    # =========================================================================
    # Constraint 2: Customer inventory balance
    # I_{ip} = I_{i,p-1} + sum_k q_{ip}^k - d_{ip},  for all i in N, p in P
    # =========================================================================
    for i in customers:
        for p in periods:
            prev_inv = cust_inv[i].get(p - 1, float(I0[i]))
            delivered = total_delivery[i][p]
            demand = d(i, p)
            computed_inv = prev_inv + delivered - demand
            cust_inv[i][p] = computed_inv

    # Constraints 1 and 2 are balance equations defining inventories.
    # Since we compute inventories from the solution, they hold by construction.
    # However, if the solution provides explicit inventories, we check consistency.
    if "inventories" in solution:
        sol_inv = solution["inventories"]
        # Check supplier inventory
        for p in periods:
            sol_val = float(sol_inv.get("0", {}).get(str(p), sup_inv[p]))
            lhs = sup_inv[p]
            rhs = sol_val
            record_violation(1, f"Supplier inventory mismatch at period {p}: "
                           f"computed={lhs:.4f}, solution={rhs:.4f}",
                           lhs, rhs, '=')
        # Check customer inventories
        for i in customers:
            for p in periods:
                sol_val = float(sol_inv.get(str(i), {}).get(str(p), cust_inv[i][p]))
                lhs = cust_inv[i][p]
                rhs = sol_val
                record_violation(2, f"Customer {i} inventory mismatch at period {p}: "
                               f"computed={lhs:.4f}, solution={rhs:.4f}",
                               lhs, rhs, '=')

    # =========================================================================
    # Constraint 3: No stockouts (inventory non-negativity)
    # I_{ip} >= 0,  for all i in V, p in P
    # =========================================================================
    # Supplier
    for p in periods:
        lhs = 0.0
        rhs = sup_inv[p]
        record_violation(3, f"Supplier stockout at period {p}: inventory={rhs:.4f} < 0",
                        lhs, rhs, '<=')

    # Customers
    for i in customers:
        for p in periods:
            lhs = 0.0
            rhs = cust_inv[i][p]
            record_violation(3, f"Customer {i} stockout at period {p}: inventory={rhs:.4f} < 0",
                            lhs, rhs, '<=')

    # =========================================================================
    # Constraint 4: Inventory capacity (customer)
    # I_{i,p-1} + sum_k q_{ip}^k <= C_i,  for all i in N, p in P
    # =========================================================================
    for i in customers:
        for p in periods:
            prev_inv = cust_inv[i].get(p - 1, float(I0[i]))
            delivered = total_delivery[i][p]
            lhs = prev_inv + delivered
            rhs = float(C[i])
            record_violation(4, f"Customer {i} inventory capacity exceeded at period {p}: "
                           f"prev_inv({prev_inv:.4f}) + delivery({delivered:.4f}) = {lhs:.4f} > capacity({rhs:.4f})",
                           lhs, rhs, '<=')

    # =========================================================================
    # Constraint 5: Supplier inventory capacity
    # 0 <= I_{0p} <= C_0,  for all p in P
    # =========================================================================
    for p in periods:
        inv = sup_inv[p]
        # Lower bound (already checked in constraint 3)
        # Upper bound
        lhs = inv
        rhs = float(C[0])
        record_violation(5, f"Supplier inventory capacity exceeded at period {p}: "
                        f"inventory={inv:.4f} > capacity={rhs:.4f}",
                        lhs, rhs, '<=')

    # =========================================================================
    # Constraint 6: Delivery only if visited
    # q_{ip}^k <= C_i * z_{ip}^k,  for all i in N, k in K, p in P
    # If vehicle k does not visit customer i in period p, delivery must be 0.
    # =========================================================================
    for (p, k_idx, route, deliveries) in route_entries:
        # Customers visited on this route (excluding depot 0)
        visited = set(node for node in route if node != 0)
        for ci, qty in deliveries.items():
            if ci not in visited and qty > TOL:
                lhs = qty
                rhs = 0.0
                record_violation(6, f"Period {p}, vehicle {k_idx}: delivery of {qty:.4f} "
                               f"to customer {ci} who is not on the route {route}",
                               lhs, rhs, '<=')

    # =========================================================================
    # Constraint 7: Flow conservation for vehicles
    # sum_j x_{ijp}^k = sum_j x_{jip}^k = z_{ip}^k,  for all i in N, k, p
    # Each customer node must have equal in-degree and out-degree, matching visit.
    # =========================================================================
    for (p, k_idx, route, deliveries) in route_entries:
        if len(route) < 2:
            continue
        # Count in-degree and out-degree for each customer node
        in_deg = {}
        out_deg = {}
        for idx in range(len(route) - 1):
            u, v = route[idx], route[idx + 1]
            if v != 0:
                in_deg[v] = in_deg.get(v, 0) + 1
            if u != 0:
                out_deg[u] = out_deg.get(u, 0) + 1

        # All customer nodes that appear in the route
        cust_nodes = set(node for node in route if node != 0)
        for ci in cust_nodes:
            ind = in_deg.get(ci, 0)
            outd = out_deg.get(ci, 0)
            if ind != outd:
                lhs = float(ind)
                rhs = float(outd)
                record_violation(7, f"Period {p}, vehicle {k_idx}: flow conservation violated "
                               f"at customer {ci}: in-degree={ind}, out-degree={outd}",
                               lhs, rhs, '=')
            # z_{ip}^k = 1 if visited, so in-degree should be 1
            if ind != 1:
                lhs = float(ind)
                rhs = 1.0
                record_violation(7, f"Period {p}, vehicle {k_idx}: customer {ci} "
                               f"in-degree={ind} != 1 (should visit exactly once per route)",
                               lhs, rhs, '=')

    # =========================================================================
    # Constraint 8: Each vehicle leaves depot at most once
    # sum_j x_{0jp}^k <= 1,  for all k, p
    # This means each route starts from depot at most once.
    # =========================================================================
    for (p, k_idx, route, deliveries) in route_entries:
        # Count arcs leaving depot
        depot_out = sum(1 for idx in range(len(route) - 1) if route[idx] == 0)
        lhs = float(depot_out)
        rhs = 1.0
        record_violation(8, f"Period {p}, vehicle {k_idx}: depot has {depot_out} "
                        f"outgoing arcs (should be <= 1)",
                        lhs, rhs, '<=')

    # =========================================================================
    # Constraint 9: Depot flow balance (leaves = returns)
    # sum_j x_{0jp}^k = sum_j x_{j0p}^k,  for all k, p
    # =========================================================================
    for (p, k_idx, route, deliveries) in route_entries:
        depot_out = sum(1 for idx in range(len(route) - 1) if route[idx] == 0)
        depot_in = sum(1 for idx in range(len(route) - 1) if route[idx + 1] == 0)
        lhs = float(depot_out)
        rhs = float(depot_in)
        record_violation(9, f"Period {p}, vehicle {k_idx}: depot out-arcs={depot_out} "
                        f"!= depot in-arcs={depot_in}",
                        lhs, rhs, '=')

    # =========================================================================
    # Constraint 10: At most one visit per customer per period
    # sum_k z_{ip}^k <= 1,  for all i in N, p in P
    # =========================================================================
    for p in periods:
        visit_count = {i: 0 for i in customers}
        for (pp, k_idx, route, deliveries) in route_entries:
            if pp != p:
                continue
            visited = set(node for node in route if node != 0)
            for ci in visited:
                if ci in visit_count:
                    visit_count[ci] += 1
        for ci in customers:
            lhs = float(visit_count[ci])
            rhs = 1.0
            record_violation(10, f"Customer {ci} visited {visit_count[ci]} times "
                           f"in period {p} (should be <= 1)",
                           lhs, rhs, '<=')

    # =========================================================================
    # Constraint 11: Subtour elimination (MTZ)
    # u_{ip}^k - u_{jp}^k + Q * x_{ijp}^k <= Q - q_{jp}^k
    # for all i,j in N, i != j, k, p
    #
    # We check this structurally: a valid route must be a simple path from depot
    # through customers back to depot (no subtours). If the route starts and ends
    # at depot and visits each customer at most once, there are no subtours.
    # =========================================================================
    for (p, k_idx, route, deliveries) in route_entries:
        if len(route) < 2:
            continue
        # Check route starts and ends at depot
        if route[0] != 0:
            record_violation(11, f"Period {p}, vehicle {k_idx}: route does not start at depot: {route}",
                           1.0, 0.0, '=')
        if route[-1] != 0:
            record_violation(11, f"Period {p}, vehicle {k_idx}: route does not end at depot: {route}",
                           1.0, 0.0, '=')

        # Check no customer is visited more than once within a single route
        cust_in_route = [node for node in route if node != 0]
        if len(cust_in_route) != len(set(cust_in_route)):
            from collections import Counter
            counts = Counter(cust_in_route)
            dupes = {c: cnt for c, cnt in counts.items() if cnt > 1}
            record_violation(11, f"Period {p}, vehicle {k_idx}: subtour/repeated customer "
                           f"in route {route}: duplicates={dupes}",
                           float(max(counts.values())), 1.0, '<=')

        # Check no intermediate depot visits (would indicate subtours)
        interior = route[1:-1]
        depot_in_interior = interior.count(0)
        if depot_in_interior > 0:
            lhs = float(depot_in_interior + 1)  # number of "sub-routes"
            rhs = 1.0
            record_violation(11, f"Period {p}, vehicle {k_idx}: depot appears in interior "
                           f"of route {route} ({depot_in_interior} times), indicating subtour",
                           lhs, rhs, '<=')

    # =========================================================================
    # Constraint 12: Vehicle load constraints
    # sum_i q_{ip}^k <= Q,  for all k, p
    # =========================================================================
    for (p, k_idx, route, deliveries) in route_entries:
        total_load = sum(deliveries.values())
        lhs = total_load
        rhs = float(Q)
        record_violation(12, f"Period {p}, vehicle {k_idx}: load={total_load:.4f} "
                        f"exceeds capacity={Q}",
                        lhs, rhs, '<=')

    # =========================================================================
    # Constraint 13: Variable domains
    # x_{ijp}^k in {0,1}, z_{ip}^k in {0,1}, q_{ip}^k >= 0, I_{ip} >= 0
    # We check non-negativity of delivery quantities.
    # Binary constraints on x and z are structural (routes are sequences of nodes).
    # I_{ip} >= 0 is already checked in constraint 3.
    # =========================================================================
    for (p, k_idx, route, deliveries) in route_entries:
        for ci, qty in deliveries.items():
            if qty < -TOL:
                lhs = 0.0
                rhs = qty
                record_violation(13, f"Period {p}, vehicle {k_idx}: negative delivery "
                               f"quantity {qty:.4f} to customer {ci}",
                               lhs, rhs, '<=')

    # Also check number of vehicles used per period does not exceed K
    # (This is implicit from constraint 8 applied per vehicle, but since solutions
    # may not have explicit vehicle indices, we check the count directly.)
    for p in periods:
        num_routes = len(period_vehicles[p])
        if num_routes > K:
            # This would violate constraint 8 (more routes than vehicles)
            lhs = float(num_routes)
            rhs = float(K)
            record_violation(8, f"Period {p}: {num_routes} routes used but only {K} "
                           f"vehicles available",
                           lhs, rhs, '<=')

    # =========================================================================
    # Constraint 14: Objective consistency (Tier C anti-gaming check)
    #
    # The reported objective_value must equal the objective recomputed from
    # the solution variables. For the compact arc-flow IRP (Part B), the
    # objective is:
    #     min  sum_p sum_k sum_(i,j) c_ij x_ijp^k   +   sum_p sum_i h_i I_ip
    #          \________  travel cost  ________/        \__ holding cost __/
    #
    # Both components are FULLY determined by what the solution carries:
    #   - travel cost  = sum of distance-matrix entries along every route arc.
    #   - holding cost = h_i * I_ip summed over every location i in {0}u N and
    #                    every period p, where I_ip is recomputed from the
    #                    inventory-balance equations (constraints 1 and 2) --
    #                    i.e. the deterministic ground-truth inventory implied
    #                    by the deliveries, NOT the solution's self-reported
    #                    "inventories" field (which an exploit could fabricate
    #                    independently; that field is separately verified by
    #                    constraints 1/2).
    #
    # This is a FULL recompute: no second-stage / scenario variables exist,
    # so the true objective is exactly computable. A candidate that lies
    # about objective_value (e.g. reports 0.0 or sys.float_info.max while the
    # routes/inventories imply a real cost) is rejected here.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            # Travel cost: sum distance-matrix entries along every route arc.
            # Skip arcs touching invalid node ids (already flagged by the
            # domain check) so this never raises IndexError.
            travel_cost = 0.0
            for (p, k_idx, route, deliveries) in route_entries:
                for idx in range(len(route) - 1):
                    a, b = route[idx], route[idx + 1]
                    if a in valid_nodes and b in valid_nodes:
                        travel_cost += float(dist[a][b])

            # Holding cost: h_i * I_ip over all locations (depot 0 and
            # customers 1..n) and all periods, using the inventories
            # recomputed from the balance equations.
            holding_cost = 0.0
            for p in periods:
                holding_cost += float(h[0]) * sup_inv[p]
            for i in customers:
                for p in periods:
                    holding_cost += float(h[i]) * cust_inv[i][p]

            true_obj = travel_cost + holding_cost
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with a 1e-3 absolute floor; this
            # absorbs the 2-decimal rounding used when solutions serialize
            # deliveries/inventories, while still catching any real lie.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                record_violation(
                    14,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed travel_cost+holding_cost="
                    f"{true_obj} (travel_cost={travel_cost}, "
                    f"holding_cost={holding_cost}, |diff|={abs_diff:.6g}, "
                    f"tol={tol:.6g})",
                    reported, true_obj, '=',
                )
        elif reported is not None:
            # Non-finite reported objective (inf / nan) can never equal a
            # finite recomputed cost: reject outright.
            record_violation(
                14,
                f"Objective consistency violated: reported objective_value="
                f"{reported_obj} is not a finite number",
                0.0, 1.0, '=',
            )

    # Compile results
    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    # Deduplicate violation messages per constraint
    unique_violations = []
    seen_msgs = set()
    for msg in violations:
        if msg not in seen_msgs:
            seen_msgs.add(msg)
            unique_violations.append(msg)

    return feasible, sorted_violated, unique_violations, violation_magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for IRP solutions (Desaulniers et al. 2016)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return

    feasible, violated_constraints, violations, violation_magnitudes = check_feasibility(
        instance, solution)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if feasible:
        print(f"FEASIBLE: Solution satisfies all hard constraints.")
    else:
        print(f"INFEASIBLE: {len(violated_constraints)} constraint(s) violated: {violated_constraints}")
        for v in violations:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
