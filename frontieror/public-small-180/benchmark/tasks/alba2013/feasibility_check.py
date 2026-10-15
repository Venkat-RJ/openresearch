#!/usr/bin/env python3
"""
Feasibility checker for the Double Traveling Salesman Problem with Multiple Stacks (DTSPMS).

Checks constraints from the mathematical formulation (Alba Martinez et al. 2013):
  Constraint 1 (Eq 2): Out-degree assignment — each vertex has exactly one successor in each tour.
  Constraint 2 (Eq 3): In-degree assignment — each vertex has exactly one predecessor in each tour.
  Constraint 3 (Eq 4): Subtour elimination — each tour is a single Hamiltonian cycle.
  Constraint 4 (Eq 5): Binary domain — x variables are 0 or 1.
  Constraint 5 (Eq 6): Loading feasibility — LIFO constraint across stacks linking pickup/delivery tours.
  Constraint 6 (Eq 1): Objective consistency — reported objective_value must equal the recomputed
    total routing cost (pickup distance + delivery distance) summed over the tours.

Constraints 7-10 are strengthened valid inequalities (cuts) implied by constraint 5 and are not
checked separately.
"""

import argparse
import json
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('delivery_tour', 'objective_value', 'pickup_tour')
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



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_x_from_tour(tour, n):
    """
    Build the x variable dictionary from a tour sequence.
    tour is a list of vertices [v0, v1, ..., v_{n}] representing the order of visit.
    The tour is a cycle: v_{n} -> v0 is implied.
    Returns dict {(i,j): 1} for arcs used.
    """
    x = {}
    for idx in range(len(tour)):
        i = tour[idx]
        j = tour[(idx + 1) % len(tour)]
        x[(i, j)] = 1
    return x


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance['parameters']['num_requests_n']
    s = instance['parameters']['num_stacks_s']
    l_cap = instance['parameters']['stack_height_l']

    V = list(range(n + 1))  # {0, 1, ..., n}

    pickup_tour = solution['pickup_tour']
    delivery_tour = solution['delivery_tour']

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # Build x variables from tours
    x_P = build_x_from_tour(pickup_tour, n)
    x_D = build_x_from_tour(delivery_tour, n)

    # =========================================================================
    # Constraint 1 (Eq 2): Out-degree — sum_j x^T_{ij} = 1 for each i in V^T
    # =========================================================================
    for label, x_vals, tour in [("pickup", x_P, pickup_tour), ("delivery", x_D, delivery_tour)]:
        for i in V:
            out_sum = sum(v for (a, b), v in x_vals.items() if a == i)
            rhs = 1.0
            violation_amount = abs(out_sum - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                violated_constraint_set.add(1)
                violations.append(
                    f"Constraint 1 (out-degree): Vertex {i} in {label} tour has "
                    f"out-degree {out_sum}, expected 1"
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": float(out_sum),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer)
                })

    # =========================================================================
    # Constraint 2 (Eq 3): In-degree — sum_i x^T_{ij} = 1 for each j in V^T
    # =========================================================================
    for label, x_vals, tour in [("pickup", x_P, pickup_tour), ("delivery", x_D, delivery_tour)]:
        for j in V:
            in_sum = sum(v for (a, b), v in x_vals.items() if b == j)
            rhs = 1.0
            violation_amount = abs(in_sum - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                violated_constraint_set.add(2)
                violations.append(
                    f"Constraint 2 (in-degree): Vertex {j} in {label} tour has "
                    f"in-degree {in_sum}, expected 1"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(in_sum),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer)
                })

    # =========================================================================
    # Constraint 3 (Eq 4): Subtour elimination — tour must be a single cycle
    # visiting all vertices (i.e., a Hamiltonian cycle on V^T).
    # We check: (a) tour visits all n+1 vertices, (b) no repeated vertices,
    # (c) forms a single connected cycle.
    # =========================================================================
    for label, tour in [("pickup", pickup_tour), ("delivery", delivery_tour)]:
        tour_set = set(tour)
        v_set = set(V)

        # Check all vertices are visited
        missing = v_set - tour_set
        extra = tour_set - v_set
        repeated = len(tour) - len(tour_set)

        if missing:
            # The tour doesn't visit all vertices => subtours exist
            # Violation: the tour only covers |tour_set| vertices out of n+1
            # This means at least one subtour exists among the missing vertices
            # For the subtour constraint: sum_{i in S, j in S} x_{ij} <= |S| - 1
            # If S = tour_set (without depot potentially), a subtour of size |S|
            # uses |S| arcs inside S, violating |S| - 1 bound by 1
            violation_amount = 1.0  # at minimum one extra arc in subtour
            rhs_val = len(tour_set) - 1.0 if len(tour_set) >= 2 else 0.0
            lhs_val = rhs_val + 1.0  # subtour uses |S| arcs = |S|-1+1
            normalizer = max(abs(rhs_val), eps)
            violated_constraint_set.add(3)
            violations.append(
                f"Constraint 3 (subtour elimination): {label} tour missing vertices "
                f"{sorted(missing)}; tour has subtour(s)"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

        if repeated > 0:
            violation_amount = float(repeated)
            rhs_val = float(n + 1)  # expected number of unique vertices
            lhs_val = float(len(tour))  # actual tour length
            normalizer = max(abs(rhs_val), eps)
            violated_constraint_set.add(3)
            violations.append(
                f"Constraint 3 (subtour elimination): {label} tour has {repeated} "
                f"repeated vertex visit(s); tour length {len(tour)} but only "
                f"{len(tour_set)} unique vertices"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

        if extra:
            violated_constraint_set.add(3)
            violations.append(
                f"Constraint 3 (subtour elimination): {label} tour visits invalid "
                f"vertices {sorted(extra)} not in V^T"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(len(extra)),
                "rhs": 0.0,
                "raw_excess": float(len(extra)),
                "normalizer": eps,
                "ratio": float(len(extra) / eps)
            })

        # Check single-cycle property (no disconnected subtours)
        if not missing and not extra and repeated == 0:
            # Build adjacency from tour
            adj = {}
            for idx in range(len(tour)):
                adj[tour[idx]] = tour[(idx + 1) % len(tour)]
            # Follow from depot
            visited = set()
            current = 0
            while current not in visited:
                visited.add(current)
                current = adj.get(current, -1)
            if len(visited) < len(V):
                # Multiple cycles detected
                num_subtours = 0
                remaining = set(V) - visited
                while remaining:
                    start = next(iter(remaining))
                    cycle = set()
                    cur = start
                    while cur not in cycle:
                        cycle.add(cur)
                        cur = adj.get(cur, -1)
                    remaining -= cycle
                    num_subtours += 1

                violation_amount = float(num_subtours)
                rhs_val = 0.0
                normalizer = eps
                violated_constraint_set.add(3)
                violations.append(
                    f"Constraint 3 (subtour elimination): {label} tour has "
                    f"{num_subtours + 1} disconnected cycle(s) instead of 1"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(num_subtours + 1),
                    "rhs": 1.0,
                    "raw_excess": float(num_subtours),
                    "normalizer": max(1.0, eps),
                    "ratio": float(num_subtours / max(1.0, eps))
                })

    # =========================================================================
    # Constraint 4 (Eq 5): Binary domain — x^T_{ij} in {0,1}
    # Also: x^T_{ii} = 0 (variable fixing from comment 4)
    # =========================================================================
    for label, x_vals in [("pickup", x_P), ("delivery", x_D)]:
        for (i, j), val in x_vals.items():
            if val not in (0, 1):
                violation_amount = min(abs(val - 0), abs(val - 1))
                if violation_amount > tol:
                    violated_constraint_set.add(4)
                    violations.append(
                        f"Constraint 4 (binary domain): x^{label[0].upper()}"
                        f"_{{{i},{j}}} = {val}, not binary"
                    )
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": float(val),
                        "rhs": 1.0,
                        "raw_excess": float(violation_amount),
                        "normalizer": max(1.0, eps),
                        "ratio": float(violation_amount / max(1.0, eps))
                    })
            if i == j and val > tol:
                violated_constraint_set.add(4)
                violations.append(
                    f"Constraint 4 (binary domain): Self-loop x^{label[0].upper()}"
                    f"_{{{i},{i}}} = {val}, must be 0"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(val),
                    "rhs": 0.0,
                    "raw_excess": float(val),
                    "normalizer": eps,
                    "ratio": float(val / eps)
                })

    # =========================================================================
    # Constraint 5 (Eq 6): Loading feasibility — LIFO constraint
    # For a given pickup tour order and delivery tour order, there must exist
    # a feasible assignment of items to s stacks (each with capacity l) such
    # that LIFO is respected: if item i is picked up before item j and both
    # are in the same stack, then j must be delivered before i.
    # =========================================================================
    # Extract customer ordering from tours (exclude depot 0)
    p_customers = [v for v in pickup_tour if v != 0]
    d_customers = [v for v in delivery_tour if v != 0]

    # Check basic: both should contain exactly the customer set {1..n}
    p_set = set(p_customers)
    d_set = set(d_customers)
    expected_customers = set(range(1, n + 1))

    if p_set != expected_customers or d_set != expected_customers:
        # Already caught by constraints 1-3, but note for loading
        pass
    else:
        # Build position maps
        p_pos = {v: i for i, v in enumerate(p_customers)}
        d_pos = {v: i for i, v in enumerate(d_customers)}

        # Try to assign items to stacks respecting LIFO
        # Use backtracking to check feasibility
        items_in_pickup_order = list(p_customers)  # items in pickup order

        feasible = _check_loading_feasibility(items_in_pickup_order, p_pos, d_pos, s, l_cap)

        if not feasible:
            # Compute violation: the loading is infeasible
            # For the path constraint (6): LHS = |A(Q)| + |A(F)|, RHS = |A(Q)| + |A(F)| - 1
            # With Q = full pickup path, F = full delivery path
            q = len(p_customers)  # arcs in pickup path
            f = len(d_customers)  # arcs in delivery path
            lhs_val = float(q + f)  # all arcs are used (integer solution)
            rhs_val = float(q + f - 1)
            violation_amount = lhs_val - rhs_val  # = 1.0
            normalizer = max(abs(rhs_val), eps)
            violated_constraint_set.add(5)
            violations.append(
                f"Constraint 5 (loading feasibility): No feasible LIFO loading exists "
                f"for the given pickup order {pickup_tour} and delivery order "
                f"{delivery_tour} with {s} stack(s) of capacity {l_cap}"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

    # =========================================================================
    # Constraint 6 (Eq 1): Objective consistency — reported objective_value must
    # equal recomputed total routing cost (sum of pickup arc costs + sum of
    # delivery arc costs). All variables that determine the obj (pickup_tour,
    # delivery_tour) are in the solution, so a full recompute is exact.
    # Tier C defense against LLM solutions that fabricate objective_value.
    # =========================================================================
    reported_obj_raw = solution.get("objective_value")
    pickup_dm = instance.get("pickup_region", {}).get("distance_matrix")
    delivery_dm = instance.get("delivery_region", {}).get("distance_matrix")
    if reported_obj_raw is not None and pickup_dm is not None and delivery_dm is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None:
            try:
                pickup_cost = sum(
                    float(pickup_dm[pickup_tour[i]][pickup_tour[(i + 1) % len(pickup_tour)]])
                    for i in range(len(pickup_tour))
                )
                delivery_cost = sum(
                    float(delivery_dm[delivery_tour[i]][delivery_tour[(i + 1) % len(delivery_tour)]])
                    for i in range(len(delivery_tour))
                )
                true_obj = pickup_cost + delivery_cost
            except (IndexError, TypeError, ValueError):
                true_obj = None

            if true_obj is not None:
                abs_diff = abs(reported_obj - true_obj)
                # 0.1% relative tolerance with 1e-3 absolute floor.
                obj_tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    normalizer = max(abs(true_obj), eps)
                    violated_constraint_set.add(6)
                    violations.append(
                        f"Constraint 6 (objective consistency): reported objective_value="
                        f"{reported_obj} differs from recomputed pickup_cost+delivery_cost="
                        f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                    )
                    violation_magnitudes.append({
                        "constraint": 6,
                        "lhs": float(reported_obj),
                        "rhs": float(true_obj),
                        "raw_excess": float(abs_diff),
                        "normalizer": float(normalizer),
                        "ratio": float(abs_diff / normalizer)
                    })

    # =========================================================================
    # Build result
    # =========================================================================
    violated_list = sorted(violated_constraint_set)
    feasible = len(violated_list) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_list,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }

    return result


def _check_loading_feasibility(items_in_pickup_order, p_pos, d_pos, s, l_cap):
    """
    Check if items can be assigned to s stacks (each capacity l_cap) respecting LIFO.

    LIFO rule: if item i is picked up before item j and both are in the same stack,
    then item j must be delivered before item i (j is on top).

    Uses backtracking.
    """
    stacks = [[] for _ in range(s)]

    def can_place(item, stack_idx):
        stack = stacks[stack_idx]
        if len(stack) >= l_cap:
            return False
        # Check LIFO: for each item w already in the stack,
        # if w was picked up before item (p_pos[w] < p_pos[item]),
        #   then item must be delivered before w (d_pos[item] < d_pos[w])
        # if item was picked up before w (shouldn't happen since we process in pickup order),
        #   then w must be delivered before item
        for w in stack:
            if p_pos[w] < p_pos[item]:
                # w picked before item => item on top => item delivered before w
                if d_pos[item] > d_pos[w]:
                    return False
            elif p_pos[item] < p_pos[w]:
                # item picked before w => w on top => w delivered before item
                if d_pos[w] > d_pos[item]:
                    return False
        return True

    def backtrack(idx):
        if idx == len(items_in_pickup_order):
            return True
        item = items_in_pickup_order[idx]
        for k in range(s):
            if can_place(item, k):
                stacks[k].append(item)
                if backtrack(idx + 1):
                    return True
                stacks[k].pop()
            # Pruning: if stack k is empty, no point trying further empty stacks
            if len(stacks[k]) == 0:
                break
        return False

    return backtrack(0)


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for DTSPMS solutions"
    )
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE: Solution satisfies all constraints.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
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
