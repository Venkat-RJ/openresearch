#!/usr/bin/env python3
"""
Feasibility checker for the VMIR-OU (Vendor-Managed Inventory Routing
with Order-Up-To level policy) problem.

Source: Archetti, Bertazzi, Laporte, Speranza (2007)
        "A Branch-and-Cut Algorithm for a Vendor-Managed Inventory-Routing Problem"
        Transportation Science 41(3), pp. 382-391.

Checks all hard constraints (2)-(16) from the mathematical formulation,
plus an obj-consistency check (constraint 16) that recomputes the
objective from the solution variables (B, I, y) and rejects when the
reported objective_value disagrees beyond tolerance.
"""

import argparse
import json
import sys

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('deliveries', 'objective_value', 'retailer_inventory', 'routes', 'supplier_inventory')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ()
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


TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    """Check all hard constraints of the VMIR-OU model.

    Returns a dict with feasibility results.
    """
    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # -------------------------------------------------------------------
    # Extract instance data
    # -------------------------------------------------------------------
    n = instance["num_retailers"]
    H = instance["time_horizon"]
    C = instance["vehicle_capacity"]
    dist = instance["distance_matrix"]

    supplier = instance["supplier"]
    B0 = supplier["starting_inventory"]
    r0 = supplier["production_rate"]
    h0 = supplier["inventory_cost"]

    retailers = {ret["id"]: ret for ret in instance["retailers"]}
    r = {s: retailers[s]["demand"] for s in retailers}
    U = {s: retailers[s]["max_inventory"] for s in retailers}
    I0 = {s: retailers[s]["starting_inventory"] for s in retailers}
    h = {s: retailers[s]["inventory_cost"] for s in retailers}

    M = list(range(1, n + 1))
    T = list(range(1, H + 1))
    T_prime = list(range(1, H + 2))  # {1, ..., H+1}
    M_prime = list(range(0, n + 1))  # {0, 1, ..., n}

    # -------------------------------------------------------------------
    # Extract solution data
    # -------------------------------------------------------------------
    # deliveries[t][s] = x_{s,t}  (quantity shipped to retailer s at time t)
    sol_deliveries = solution.get("deliveries", {})
    x = {}
    for t in T:
        t_str = str(t)
        period_del = sol_deliveries.get(t_str, {})
        for s in M:
            s_str = str(s)
            x[s, t] = float(period_del.get(s_str, 0.0))

    # supplier_inventory[t] = B_t
    sol_supplier_inv = solution.get("supplier_inventory", {})
    B = {}
    for t in T_prime:
        B[t] = float(sol_supplier_inv.get(str(t), 0.0))

    # retailer_inventory[s][t] = I_{s,t}
    sol_retailer_inv = solution.get("retailer_inventory", {})
    I = {}
    for s in M:
        s_str = str(s)
        ret_inv = sol_retailer_inv.get(s_str, {})
        for t in T_prime:
            I[s, t] = float(ret_inv.get(str(t), 0.0))

    # routes[t] = list of edges, each edge = [node_i, node_j, value]
    # From these we reconstruct z and y variables.
    sol_routes = solution.get("routes", {})

    # Reconstruct z_{i,t} from solution:
    # z_{s,t} = 1 if retailer s is delivered at time t (x_{s,t} > 0)
    # z_{0,t} = 1 if any delivery happens at time t
    z = {}
    for i in M_prime:
        for t in T:
            z[i, t] = 0

    for t in T:
        t_str = str(t)
        # Mark retailers as visited based on deliveries
        period_del = sol_deliveries.get(t_str, {})
        for s_str, qty in period_del.items():
            s = int(s_str)
            if float(qty) > TOL:
                z[s, t] = 1
        # Also mark retailers from route edges
        edges = sol_routes.get(t_str, [])
        for edge in edges:
            for node in edge[:2]:  # first two elements are the edge endpoints
                if node in range(0, n + 1):
                    if z.get((node, t)) is not None:
                        z[node, t] = 1
        # Depot is visited if any retailer is visited
        if any(z[s, t] == 1 for s in M):
            z[0, t] = 1

    # Reconstruct y variables from route edges.
    # Each edge in routes[t] is [lower_node, higher_node, value].
    # y^t_{i,j} where j < i.
    y = {}
    for t in T:
        for i in M_prime:
            for j in M_prime:
                if j < i:
                    y[i, j, t] = 0

    for t in T:
        t_str = str(t)
        edges = sol_routes.get(t_str, [])
        for edge in edges:
            node_a = edge[0]
            node_b = edge[1]
            val = int(round(edge[2])) if len(edge) > 2 else 1
            i_var = max(node_a, node_b)
            j_var = min(node_a, node_b)
            if (i_var, j_var, t) in y:
                y[i_var, j_var, t] += val

    # -------------------------------------------------------------------
    # Helper to record violations
    # -------------------------------------------------------------------
    def record_violation(constraint_idx, lhs_val, rhs_val, violation_amount, msg):
        if violation_amount > TOL:
            violated_constraint_set.add(constraint_idx)
            normalizer = max(abs(rhs_val), EPS)
            ratio = violation_amount / normalizer
            violations.append(msg)
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ===================================================================
    # Constraint 1 (tag 2): Supplier inventory definition
    # B_t = B_{t-1} + r_{0,t-1} - sum_{s in M} x_{s,t-1}, t in T'
    # where r_{0,0} = 0 and x_{s,0} = 0
    # ===================================================================
    for t in T_prime:
        if t == 1:
            lhs = B[t]
            rhs_val = B0
        else:
            r0_prev = r0  # production at t-1 (for t-1 >= 1)
            total_shipped_prev = sum(x.get((s, t - 1), 0.0) for s in M)
            rhs_val = B[t - 1] + r0_prev - total_shipped_prev
            lhs = B[t]
        violation_amount = abs(lhs - rhs_val)
        record_violation(
            1, lhs, rhs_val, violation_amount,
            f"Constraint 1 (Supplier inventory definition) violated at t={t}: "
            f"B_{t}={lhs:.4f} != {rhs_val:.4f}"
        )

    # ===================================================================
    # Constraint 2 (tag 3): Stockout at supplier
    # B_t >= sum_{s in M} x_{s,t}, t in T
    # ===================================================================
    for t in T:
        total_shipped = sum(x.get((s, t), 0.0) for s in M)
        lhs = B[t]
        rhs_val = total_shipped
        violation_amount = max(0.0, rhs_val - lhs)  # >= constraint
        record_violation(
            2, lhs, rhs_val, violation_amount,
            f"Constraint 2 (Supplier stockout) violated at t={t}: "
            f"B_{t}={lhs:.4f} < sum_x={total_shipped:.4f}"
        )

    # ===================================================================
    # Constraint 3 (tag 4): Retailer inventory definition
    # I_{s,t} = I_{s,t-1} + x_{s,t-1} - r_{s,t-1}, s in M, t in T'
    # where x_{s,0} = r_{s,0} = 0
    # ===================================================================
    for s in M:
        for t in T_prime:
            if t == 1:
                lhs = I[s, t]
                rhs_val = I0[s]
            else:
                x_prev = x.get((s, t - 1), 0.0)
                r_prev = r[s]
                rhs_val = I[s, t - 1] + x_prev - r_prev
                lhs = I[s, t]
            violation_amount = abs(lhs - rhs_val)
            record_violation(
                3, lhs, rhs_val, violation_amount,
                f"Constraint 3 (Retailer inventory definition) violated for "
                f"retailer {s} at t={t}: I_{s},{t}={lhs:.4f} != {rhs_val:.4f}"
            )

    # ===================================================================
    # Constraint 4 (tag 5): Stockout at retailers
    # I_{s,t} >= 0, s in M, t in T'
    # ===================================================================
    for s in M:
        for t in T_prime:
            lhs = I[s, t]
            rhs_val = 0.0
            violation_amount = max(0.0, -lhs)  # >= 0
            record_violation(
                4, lhs, rhs_val, violation_amount,
                f"Constraint 4 (Retailer stockout) violated for retailer {s} "
                f"at t={t}: I_{s},{t}={lhs:.4f} < 0"
            )

    # ===================================================================
    # Constraint 5 (tag 6): Order-up-to lower bound
    # x_{s,t} >= U_s * z_{s,t} - I_{s,t}, s in M, t in T
    # ===================================================================
    for s in M:
        for t in T:
            lhs = x[s, t]
            rhs_val = U[s] * z[s, t] - I[s, t]
            violation_amount = max(0.0, rhs_val - lhs)  # >= constraint
            record_violation(
                5, lhs, rhs_val, violation_amount,
                f"Constraint 5 (Order-up-to lower bound) violated for "
                f"retailer {s} at t={t}: x={lhs:.4f} < U*z - I = {rhs_val:.4f}"
            )

    # ===================================================================
    # Constraint 6 (tag 7): Upper bound on shipping
    # x_{s,t} <= U_s - I_{s,t}, s in M, t in T
    # ===================================================================
    for s in M:
        for t in T:
            lhs = x[s, t]
            rhs_val = U[s] - I[s, t]
            violation_amount = max(0.0, lhs - rhs_val)  # <= constraint
            record_violation(
                6, lhs, rhs_val, violation_amount,
                f"Constraint 6 (Shipping upper bound) violated for "
                f"retailer {s} at t={t}: x={lhs:.4f} > U - I = {rhs_val:.4f}"
            )

    # ===================================================================
    # Constraint 7 (tag 8): Linking x and z
    # x_{s,t} <= U_s * z_{s,t}, s in M, t in T
    # ===================================================================
    for s in M:
        for t in T:
            lhs = x[s, t]
            rhs_val = U[s] * z[s, t]
            violation_amount = max(0.0, lhs - rhs_val)  # <= constraint
            record_violation(
                7, lhs, rhs_val, violation_amount,
                f"Constraint 7 (Linking x and z) violated for "
                f"retailer {s} at t={t}: x={lhs:.4f} > U*z = {rhs_val:.4f}"
            )

    # ===================================================================
    # Constraint 8 (tag 9): Vehicle capacity
    # sum_{s in M} x_{s,t} <= C, t in T
    # ===================================================================
    for t in T:
        total_shipped = sum(x.get((s, t), 0.0) for s in M)
        lhs = total_shipped
        rhs_val = C
        violation_amount = max(0.0, lhs - rhs_val)  # <= constraint
        record_violation(
            8, lhs, rhs_val, violation_amount,
            f"Constraint 8 (Vehicle capacity) violated at t={t}: "
            f"sum_x={lhs:.4f} > C={rhs_val}"
        )

    # ===================================================================
    # Constraint 9 (tag 10): Supplier inclusion in route
    # sum_{s in M} x_{s,t} <= C * z_{0,t}, t in T
    # ===================================================================
    for t in T:
        total_shipped = sum(x.get((s, t), 0.0) for s in M)
        lhs = total_shipped
        rhs_val = C * z[0, t]
        violation_amount = max(0.0, lhs - rhs_val)  # <= constraint
        record_violation(
            9, lhs, rhs_val, violation_amount,
            f"Constraint 9 (Supplier inclusion) violated at t={t}: "
            f"sum_x={lhs:.4f} > C*z0={rhs_val:.4f}"
        )

    # ===================================================================
    # Constraint 10 (tag 11): Degree constraints
    # sum_{j<i} y^t_{ij} + sum_{j>i} y^t_{ji} = 2*z_{i,t}, i in M', t in T
    # ===================================================================
    for i in M_prime:
        for t in T:
            degree = 0
            for j in M_prime:
                if j < i:
                    degree += y.get((i, j, t), 0)
                elif j > i:
                    degree += y.get((j, i, t), 0)
            lhs = degree
            rhs_val = 2 * z[i, t]
            violation_amount = abs(lhs - rhs_val)
            record_violation(
                10, lhs, rhs_val, violation_amount,
                f"Constraint 10 (Degree) violated for node {i} at t={t}: "
                f"degree={lhs} != 2*z={rhs_val}"
            )

    # ===================================================================
    # Constraint 11 (tag 12): Subtour elimination
    # sum_{i in S, j in S, j<i} y^t_{ij} <= sum_{i in S} z_{it} - z_{kt}
    # for S subset of M, k in S, t in T
    #
    # We check this by finding connected components of visited retailers
    # that are not connected to depot.
    # ===================================================================
    for t in T:
        visited = [s for s in M if z[s, t] == 1]
        if len(visited) <= 1:
            continue

        # Build adjacency among visited retailers
        adj = {s: set() for s in visited}
        for i in visited:
            for j in visited:
                if j < i and y.get((i, j, t), 0) >= 1:
                    adj[i].add(j)
                    adj[j].add(i)

        # Find retailers connected to depot
        depot_connected = set()
        for s in visited:
            if y.get((s, 0, t), 0) >= 1:
                depot_connected.add(s)

        # BFS from depot-connected retailers to find full depot component
        full_depot_set = set()
        queue = list(depot_connected)
        while queue:
            node = queue.pop()
            if node in full_depot_set:
                continue
            full_depot_set.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in full_depot_set:
                    queue.append(neighbor)

        # Find components not connected to depot
        remaining = set(visited) - full_depot_set
        while remaining:
            start = next(iter(remaining))
            component = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in component:
                    continue
                component.add(node)
                remaining.discard(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in component:
                        queue.append(neighbor)

            # This is a subtour: compute violation
            S = component
            # LHS: sum of y edges within S
            lhs_val = 0
            for i_s in S:
                for j_s in S:
                    if j_s < i_s:
                        lhs_val += y.get((i_s, j_s, t), 0)
            # RHS: sum z_{it} - max z_{kt} (k in S)
            sum_z = sum(z[i_s, t] for i_s in S)
            max_z = max(z[k_s, t] for k_s in S)
            rhs_val = sum_z - max_z

            violation_amount = max(0.0, lhs_val - rhs_val)
            record_violation(
                11, float(lhs_val), float(rhs_val), violation_amount,
                f"Constraint 11 (Subtour elimination) violated at t={t}: "
                f"subtour among retailers {sorted(S)}"
            )

    # ===================================================================
    # Constraint 12 (tag 13): Nonnegativity of x
    # x_{s,t} >= 0, s in M, t in T
    # ===================================================================
    for s in M:
        for t in T:
            lhs = x[s, t]
            rhs_val = 0.0
            violation_amount = max(0.0, -lhs)
            record_violation(
                12, lhs, rhs_val, violation_amount,
                f"Constraint 12 (Nonnegativity x) violated for "
                f"retailer {s} at t={t}: x={lhs:.4f} < 0"
            )

    # ===================================================================
    # Constraint 13 (tag 14): Binary y_{ij} for retailer-retailer edges
    # y^t_{ij} in {0, 1}, i in M, j in M, j < i, t in T
    # ===================================================================
    for i in M:
        for j in M:
            if j < i:
                for t in T:
                    val = y.get((i, j, t), 0)
                    if val not in (0, 1):
                        dist_from_binary = min(abs(val - 0), abs(val - 1))
                        record_violation(
                            13, float(val), 0.0, dist_from_binary,
                            f"Constraint 13 (Binary y_ij) violated for "
                            f"edge ({j},{i}) at t={t}: y={val} not in {{0,1}}"
                        )

    # ===================================================================
    # Constraint 14 (tag 15): y_{i0} in {0, 1, 2}
    # y^t_{i0} in {0, 1, 2}, i in M, t in T
    # ===================================================================
    for i in M:
        for t in T:
            val = y.get((i, 0, t), 0)
            if val not in (0, 1, 2):
                dist_from_domain = min(abs(val - 0), abs(val - 1), abs(val - 2))
                record_violation(
                    14, float(val), 0.0, dist_from_domain,
                    f"Constraint 14 (Domain y_i0) violated for "
                    f"edge (0,{i}) at t={t}: y={val} not in {{0,1,2}}"
                )

    # ===================================================================
    # Constraint 15 (tag 16): Binary z
    # z_{i,t} in {0, 1}, i in M', t in T
    # ===================================================================
    for i in M_prime:
        for t in T:
            val = z[i, t]
            if val not in (0, 1):
                dist_from_binary = min(abs(val - 0), abs(val - 1))
                record_violation(
                    15, float(val), 0.0, dist_from_binary,
                    f"Constraint 15 (Binary z) violated for "
                    f"node {i} at t={t}: z={val} not in {{0,1}}"
                )

    # ===================================================================
    # Constraint 16 (objective consistency, Tier C anti-exploit):
    # The reported objective_value must equal the recomputed
    #   sum_{t in T'} h_0 * B_t
    # + sum_{s in M} sum_{t in T'} h_s * I_{s,t}
    # + sum_{i,j in M', j<i} sum_{t in T} c_{ij} * y^t_{ij}
    # within tolerance. All variables driving the objective are present
    # in the solution (B, I, y reconstructed from routes), so we can do a
    # full recompute. Tolerance: 0.1% relative with 1e-3 absolute floor.
    # ===================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            supplier_inv_cost = sum(h0 * B[t] for t in T_prime)
            retailer_inv_cost = sum(
                h[s] * I[s, t] for s in M for t in T_prime
            )
            transport_cost = 0.0
            for i in M_prime:
                for j in M_prime:
                    if j < i:
                        cij = dist[i][j]
                        for t in T:
                            transport_cost += cij * y.get((i, j, t), 0)
            true_obj = supplier_inv_cost + retailer_inv_cost + transport_cost
            abs_diff = abs(reported - true_obj)
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraint_set.add(16)
                normalizer = max(abs(true_obj), EPS)
                ratio = abs_diff / normalizer
                msg = (
                    f"Constraint 16 (Objective consistency) violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed h0*sum_B + sum_s h_s*sum_I + sum_t sum_ij c_ij*y={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
                violations.append(msg)
                violation_magnitudes.append({
                    "constraint": 16,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })

    # -------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------
    violated_constraints = sorted(violated_constraint_set)
    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for VMIR-OU (Archetti et al. 2007)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
    _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
    if _frontieror_contract_result is not None:
        with open(args.result_path, "w") as _frontieror_result_handle:
            json.dump(
                _frontieror_contract_result,
                _frontieror_result_handle,
                indent=2,
            )
        return

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
