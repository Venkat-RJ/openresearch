#!/usr/bin/env python3
"""
Feasibility Checker for VRPDDP (Vehicle Routing Problem with Divisible Deliveries and Pickups)

Checks all hard constraints from the mathematical formulation in math_model.txt:
  Constraint 1:  Objective function (not checked -- it is the objective, not a constraint)
  Constraint 2:  In-degree = 1 for every customer j in L ∪ B
  Constraint 3:  Out-degree = 1 for every customer j in L ∪ B
  Constraint 4:  Delivery flow conservation at linehaul customers
  Constraint 5:  Delivery flow conservation at backhaul customers
  Constraint 6:  Pickup flow conservation at linehaul customers
  Constraint 7:  Pickup flow conservation at backhaul customers
  Constraint 8:  Vehicles start with zero pickup load (P leaving depot = 0)
  Constraint 9:  Vehicles finish with zero delivery load (R returning to depot = 0)
  Constraint 10: Capacity constraint  R_ij + P_ij <= C * x_ij
  Constraint 11: Binary domain for x_ij
  Constraint 12: Non-negativity of R_ij
  Constraint 13: Non-negativity of P_ij
  Constraint 14: No self-loops  x_ii = 0
  Constraint 15: No arc from backhaul n+i to its linehaul i
  Constraint 16: Minimum vehicles  sum x_{0,i} >= (1/C) * sum q_i  (i in L)
  Constraint 17: No 2-cycles  x_ij + x_ji <= 1  for i,j in L ∪ B
  Constraint 18: Objective consistency -- reported objective_value must equal
                 sum_{i,j in V} d_ij * x_ij  (Tier C anti-exploit check)
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'routes', 'routes_detailed')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('routes_detailed',)
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


def build_vrpmdp_from_instance(instance):
    """
    Build the VRPMDP formulation sets and parameters from the instance.

    Returns:
        n: number of original customers
        C: vehicle capacity
        V: set of all vertex indices {0, 1, ..., 2n}  (0 = depot)
        L: set of linehaul nodes {1, ..., n}
        B: set of backhaul nodes {n+1, ..., 2n}
        q: demand dict  q[i] for i in L ∪ B
        d: distance matrix dict  d[(i,j)] for i,j in V
        customers: list of customer dicts from instance
    """
    n = instance["num_customers"]
    C = instance["vehicle_capacity"]
    customers = instance["customers"]
    dist_raw = instance["distance_matrix"]

    # In the VRPMDP formulation:
    #   Depot = node 0
    #   Linehaul customer i (i=1..n) corresponds to original customer i
    #   Backhaul customer n+i (i=1..n) is co-located with customer i
    V = set(range(0, 2 * n + 1))
    L = set(range(1, n + 1))
    B = set(range(n + 1, 2 * n + 1))

    # Demands: q[i] = delivery demand for linehaul i, pickup demand for backhaul n+i
    cust_map = {c["id"]: c for c in customers}
    q = {}
    for i in range(1, n + 1):
        q[i] = cust_map[i]["delivery_demand"]      # linehaul
        q[n + i] = cust_map[i]["pickup_demand"]     # backhaul

    # Distance matrix in VRPMDP space.
    # dist_raw is (n+1) x (n+1) indexed by original ids 0..n.
    # Backhaul n+i is co-located with linehaul i, so distance to/from n+i
    # equals distance to/from i (the original customer).
    def orig(node):
        """Map VRPMDP node to original customer id for distance lookup."""
        if node == 0:
            return 0
        if node <= n:
            return node
        return node - n  # backhaul n+i -> original i

    d = {}
    for i in V:
        for j in V:
            d[(i, j)] = dist_raw[orig(i)][orig(j)]

    return n, C, V, L, B, q, d, customers


def extract_arcs_and_flows(instance, solution):
    """
    Extract arcs (x), delivery flow (R), and pickup flow (P) from the solution.

    Handles two solution formats:
      1. Efficient algorithm format: routes with 'visits' and 'detail'
      2. Gurobi format: routes with node IDs directly (possibly with routes_detailed)

    Returns:
        x: dict  x[(i,j)] = 1 for active arcs
        R: dict  R[(i,j)] = delivery goods on arc (i,j)
        P: dict  P[(i,j)] = pickup goods on arc (i,j)
    """
    n = instance["num_customers"]
    C = instance["vehicle_capacity"]

    x = {}
    R = {}
    P = {}

    # Detect solution format
    if "routes" in solution and len(solution["routes"]) > 0:
        first_route = solution["routes"][0]
        if isinstance(first_route, dict) and "visits" in first_route:
            # Efficient algorithm format
            return _extract_efficient_format(instance, solution, n, C)
        elif isinstance(first_route, list):
            # Gurobi format: routes are lists of node IDs
            return _extract_gurobi_format(instance, solution, n, C)

    raise ValueError("Unrecognized solution format")


def _extract_efficient_format(instance, solution, n, C):
    """
    Extract arcs and flows from the efficient algorithm solution format.

    In this format, each route has:
      - 'visits': list of original customer IDs (with depot 0 at start/end)
      - 'detail': list of dicts with customer_id, entity_type, delivery, pickup

    We need to map these to the VRPMDP node space:
      - 'linehaul' entity for customer i -> node i
      - 'backhaul' entity for customer i -> node n+i
      - 'combined' entity for customer i -> we treat it as visiting both node i
        and node n+i consecutively (linehaul then backhaul, co-located)
    """
    x = {}
    R = {}
    P = {}

    for route_data in solution["routes"]:
        detail = route_data["detail"]

        # Build the sequence of VRPMDP nodes for this route
        nodes = [0]  # start at depot
        for entry in detail:
            cid = entry["customer_id"]
            etype = entry["entity_type"]
            if etype == "linehaul":
                nodes.append(cid)          # linehaul node
            elif etype == "backhaul":
                nodes.append(n + cid)      # backhaul node
            elif etype == "combined":
                # Combined: visit linehaul i then backhaul n+i
                nodes.append(cid)          # linehaul node
                nodes.append(n + cid)      # backhaul node
        nodes.append(0)  # return to depot

        # Set arcs
        for k in range(len(nodes) - 1):
            arc = (nodes[k], nodes[k + 1])
            x[arc] = x.get(arc, 0) + 1

        # Compute flows along arcs.
        # Delivery flow (R): starts with total delivery demand on route,
        # decreases at each linehaul node.
        # Pickup flow (P): starts at 0, increases at each backhaul node.

        # Collect demands for each VRPMDP node in this route
        cust_map = {c["id"]: c for c in instance["customers"]}
        node_delivery = {}  # delivery consumed at node
        node_pickup = {}    # pickup added at node
        for entry in detail:
            cid = entry["customer_id"]
            etype = entry["entity_type"]
            if etype == "linehaul":
                node_delivery[cid] = entry["delivery"]
                # backhaul node not visited in this entry
            elif etype == "backhaul":
                node_pickup[n + cid] = entry["pickup"]
            elif etype == "combined":
                node_delivery[cid] = entry["delivery"]
                node_pickup[n + cid] = entry["pickup"]

        # Compute R and P along arcs
        delivery_load = sum(node_delivery.get(nd, 0) for nd in nodes if nd != 0)
        pickup_load = 0.0

        for k in range(len(nodes) - 1):
            arc = (nodes[k], nodes[k + 1])
            # At node nodes[k], if it's a linehaul node, delivery is dropped off
            # before departing (flow decreases). If backhaul, pickup is loaded.
            nd = nodes[k]
            if nd != 0:
                if nd in node_delivery:
                    delivery_load -= node_delivery[nd]
                if nd in node_pickup:
                    pickup_load += node_pickup[nd]
            R[arc] = R.get(arc, 0) + delivery_load
            P[arc] = P.get(arc, 0) + pickup_load

    return x, R, P


def _extract_gurobi_format(instance, solution, n, C):
    """
    Extract arcs and flows from the Gurobi solution format.

    Routes are lists of node IDs in VRPMDP space (0=depot, 1..n=linehaul, n+1..2n=backhaul).
    """
    x = {}
    R = {}
    P = {}

    cust_map = {c["id"]: c for c in instance["customers"]}
    q = {}
    for i in range(1, n + 1):
        q[i] = cust_map[i]["delivery_demand"]
        q[n + i] = cust_map[i]["pickup_demand"]

    for route_nodes in solution["routes"]:
        # route_nodes is e.g. [0, 12, 27, 1, 4, 19, 0]
        # Set arcs
        for k in range(len(route_nodes) - 1):
            arc = (route_nodes[k], route_nodes[k + 1])
            x[arc] = x.get(arc, 0) + 1

        # Compute delivery and pickup flows
        # Delivery load starts at sum of delivery demands of linehaul nodes on route
        inner_nodes = route_nodes[1:-1]  # exclude depot at start and end
        delivery_load = sum(q.get(nd, 0) for nd in inner_nodes if 1 <= nd <= n)
        pickup_load = 0.0

        for k in range(len(route_nodes) - 1):
            nd = route_nodes[k]
            # At each node, consume delivery (if linehaul) or add pickup (if backhaul)
            if 1 <= nd <= n:
                delivery_load -= q.get(nd, 0)
            elif n + 1 <= nd <= 2 * n:
                pickup_load += q.get(nd, 0)
            arc = (route_nodes[k], route_nodes[k + 1])
            R[arc] = R.get(arc, 0) + delivery_load
            P[arc] = P.get(arc, 0) + pickup_load

    return x, R, P


def check_feasibility(instance, solution):
    """
    Check all hard constraints of the VRPDDP formulation.

    Returns:
        result dict with feasible, violated_constraints, violations, violation_magnitudes
    """
    n, C, V, L, B, q, d, customers = build_vrpmdp_from_instance(instance)
    x, R, P = extract_arcs_and_flows(instance, solution)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs, rhs, operator):
        """Record a constraint violation with normalized magnitude."""
        if operator in ("<=", "<"):
            violation_amount = lhs - rhs
        elif operator in (">=", ">"):
            violation_amount = rhs - lhs
        elif operator == "=":
            violation_amount = abs(lhs - rhs)
        else:
            violation_amount = abs(lhs - rhs)

        if violation_amount > TOL:
            violated_constraints.add(constraint_idx)
            violations.append(msg)
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # Helper: get x value
    def get_x(i, j):
        return x.get((i, j), 0)

    def get_R(i, j):
        return R.get((i, j), 0.0)

    def get_P(i, j):
        return P.get((i, j), 0.0)

    # -----------------------------------------------------------------------
    # Constraint 2: sum_{i in V} x_{ij} = 1  for j in L ∪ B
    # In-degree = 1 for every customer node
    # -----------------------------------------------------------------------
    for j in sorted(L | B):
        lhs = sum(get_x(i, j) for i in V)
        rhs = 1
        if abs(lhs - rhs) > TOL:
            if j <= n:
                node_type = "linehaul"
                orig_cust = j
            else:
                node_type = "backhaul"
                orig_cust = j - n
            record_violation(
                2,
                f"Constraint 2: In-degree of {node_type} node {j} (customer {orig_cust}) is {lhs}, expected 1",
                lhs, rhs, "="
            )

    # -----------------------------------------------------------------------
    # Constraint 3: sum_{i in V} x_{ji} = 1  for j in L ∪ B
    # Out-degree = 1 for every customer node
    # -----------------------------------------------------------------------
    for j in sorted(L | B):
        lhs = sum(get_x(j, i) for i in V)
        rhs = 1
        if abs(lhs - rhs) > TOL:
            if j <= n:
                node_type = "linehaul"
                orig_cust = j
            else:
                node_type = "backhaul"
                orig_cust = j - n
            record_violation(
                3,
                f"Constraint 3: Out-degree of {node_type} node {j} (customer {orig_cust}) is {lhs}, expected 1",
                lhs, rhs, "="
            )

    # -----------------------------------------------------------------------
    # Constraint 4: sum_{i in V} R_{ij} - q_j = sum_{i in V} R_{ji}  for j in L
    # Delivery flow conservation at linehaul customers
    # Rewritten: sum R_{ij} - sum R_{ji} = q_j  (equality)
    # -----------------------------------------------------------------------
    for j in sorted(L):
        lhs = sum(get_R(i, j) for i in V) - sum(get_R(j, i) for i in V)
        rhs = q[j]
        if abs(lhs - rhs) > TOL:
            record_violation(
                4,
                f"Constraint 4: Delivery flow conservation at linehaul node {j} "
                f"(customer {j}): R_in - R_out = {lhs}, expected q_{j} = {rhs}",
                lhs, rhs, "="
            )

    # -----------------------------------------------------------------------
    # Constraint 5: sum_{i in V} R_{ij} = sum_{i in V} R_{ji}  for j in B
    # Delivery flow conservation at backhaul customers (no delivery consumed)
    # Rewritten: sum R_{ij} - sum R_{ji} = 0
    # -----------------------------------------------------------------------
    for j in sorted(B):
        lhs = sum(get_R(i, j) for i in V) - sum(get_R(j, i) for i in V)
        rhs = 0.0
        if abs(lhs - rhs) > TOL:
            orig_cust = j - n
            record_violation(
                5,
                f"Constraint 5: Delivery flow conservation at backhaul node {j} "
                f"(customer {orig_cust}): R_in - R_out = {lhs}, expected 0",
                lhs, rhs, "="
            )

    # -----------------------------------------------------------------------
    # Constraint 6: sum_{i in V} P_{ij} = sum_{i in V} P_{ji}  for j in L
    # Pickup flow conservation at linehaul customers (no pickup added)
    # Rewritten: sum P_{ij} - sum P_{ji} = 0
    # -----------------------------------------------------------------------
    for j in sorted(L):
        lhs = sum(get_P(i, j) for i in V) - sum(get_P(j, i) for i in V)
        rhs = 0.0
        if abs(lhs - rhs) > TOL:
            record_violation(
                6,
                f"Constraint 6: Pickup flow conservation at linehaul node {j} "
                f"(customer {j}): P_in - P_out = {lhs}, expected 0",
                lhs, rhs, "="
            )

    # -----------------------------------------------------------------------
    # Constraint 7: sum_{i in V} P_{ij} + q_j = sum_{i in V} P_{ji}  for j in B
    # Pickup flow conservation at backhaul customers
    # Rewritten: sum P_{ji} - sum P_{ij} = q_j
    # -----------------------------------------------------------------------
    for j in sorted(B):
        lhs = sum(get_P(j, i) for i in V) - sum(get_P(i, j) for i in V)
        rhs = q[j]
        if abs(lhs - rhs) > TOL:
            orig_cust = j - n
            record_violation(
                7,
                f"Constraint 7: Pickup flow conservation at backhaul node {j} "
                f"(customer {orig_cust}): P_out - P_in = {lhs}, expected q_{j} = {rhs}",
                lhs, rhs, "="
            )

    # -----------------------------------------------------------------------
    # Constraint 8: sum_{i in L ∪ B} P_{0,i} = 0
    # Vehicles start with zero pickup load
    # -----------------------------------------------------------------------
    lhs_8 = sum(get_P(0, i) for i in (L | B))
    rhs_8 = 0.0
    if abs(lhs_8 - rhs_8) > TOL:
        record_violation(
            8,
            f"Constraint 8: Total pickup flow leaving depot = {lhs_8}, expected 0",
            lhs_8, rhs_8, "="
        )

    # -----------------------------------------------------------------------
    # Constraint 9: sum_{i in L ∪ B} R_{i,0} = 0
    # Vehicles finish with zero delivery load
    # -----------------------------------------------------------------------
    lhs_9 = sum(get_R(i, 0) for i in (L | B))
    rhs_9 = 0.0
    if abs(lhs_9 - rhs_9) > TOL:
        record_violation(
            9,
            f"Constraint 9: Total delivery flow returning to depot = {lhs_9}, expected 0",
            lhs_9, rhs_9, "="
        )

    # -----------------------------------------------------------------------
    # Constraint 10: R_{ij} + P_{ij} <= C * x_{ij}  for all i,j in V
    # Capacity constraint on each arc
    # -----------------------------------------------------------------------
    all_arcs = set(x.keys()) | set(R.keys()) | set(P.keys())
    for (i, j) in sorted(all_arcs):
        lhs_val = get_R(i, j) + get_P(i, j)
        rhs_val = C * get_x(i, j)
        if lhs_val - rhs_val > TOL:
            record_violation(
                10,
                f"Constraint 10: Capacity on arc ({i},{j}): R+P = {lhs_val} > C*x = {rhs_val}",
                lhs_val, rhs_val, "<="
            )

    # -----------------------------------------------------------------------
    # Constraint 11: x_{ij} in {0, 1}  for all i,j in V
    # Binary domain
    # -----------------------------------------------------------------------
    for (i, j), val in sorted(x.items()):
        if val not in (0, 1):
            record_violation(
                11,
                f"Constraint 11: x({i},{j}) = {val}, not binary",
                float(val), 1.0, "<="
            )

    # -----------------------------------------------------------------------
    # Constraint 12: R_{ij} >= 0  for all i,j in V
    # Non-negativity of delivery flow
    # -----------------------------------------------------------------------
    for (i, j), val in sorted(R.items()):
        if val < -TOL:
            record_violation(
                12,
                f"Constraint 12: R({i},{j}) = {val} < 0",
                val, 0.0, ">="
            )

    # -----------------------------------------------------------------------
    # Constraint 13: P_{ij} >= 0  for all i,j in V
    # Non-negativity of pickup flow
    # -----------------------------------------------------------------------
    for (i, j), val in sorted(P.items()):
        if val < -TOL:
            record_violation(
                13,
                f"Constraint 13: P({i},{j}) = {val} < 0",
                val, 0.0, ">="
            )

    # -----------------------------------------------------------------------
    # Constraint 14: x_{ii} = 0  for all i in V
    # No self-loops
    # -----------------------------------------------------------------------
    for i in sorted(V):
        val = get_x(i, i)
        if abs(val) > TOL:
            record_violation(
                14,
                f"Constraint 14: Self-loop x({i},{i}) = {val}, expected 0",
                float(val), 0.0, "="
            )

    # -----------------------------------------------------------------------
    # Constraint 15: x_{(n+i), i} = 0  for i in L
    # No arc from backhaul n+i to its corresponding linehaul i
    # -----------------------------------------------------------------------
    for i in sorted(L):
        val = get_x(n + i, i)
        if abs(val) > TOL:
            record_violation(
                15,
                f"Constraint 15: Arc from backhaul {n+i} to linehaul {i} "
                f"(customer {i}): x = {val}, expected 0",
                float(val), 0.0, "="
            )

    # -----------------------------------------------------------------------
    # Constraints 16 and 17 SKIPPED: valid inequalities added by the paper
    # for LP-relaxation strengthening.
    #   (16) min-vehicles  sum x_{0,i} >= (1/C)*sum q_i  (valid for any
    #         feasible routing that respects capacity).
    #   (17) no 2-cycles   x_{ij} + x_{ji} <= 1  (redundant under single-visit
    #         VRPMDP structure).
    # Neither appears in the business description; any feasible routing that
    # satisfies capacity and customer-visit rules automatically satisfies them.
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Constraint 18: Objective consistency (Tier C anti-exploit check).
    # Recompute Z = sum_{i,j in V} d_ij * x_ij from the solution's arcs and
    # the instance's (integer, rounded-Euclidean) distance matrix. Reject when
    # the reported objective_value disagrees beyond a 0.1% relative tolerance
    # with a 1e-3 absolute floor. The recompute is exact (integer arithmetic)
    # because all obj-determining variables (x_ij) are derivable from
    # `routes`, so there is no missing-variable slack to absorb.
    # -----------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(d[(i, j)] * xv for (i, j), xv in x.items()))
            abs_diff = abs(reported - true_obj)
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                record_violation(
                    18,
                    f"Constraint 18: Objective consistency violated -- "
                    f"reported objective_value={reported} differs from "
                    f"recomputed sum_(i,j) d_ij*x_ij = {true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})",
                    reported, true_obj, "="
                )

    # Build result
    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    # Aggregate violation messages per constraint
    aggregated_violations = []
    for cidx in sorted_violated:
        msgs = [v for v, m in zip(violations, violation_magnitudes) if m["constraint"] == cidx]
        if not msgs:
            msgs = [v for v in violations if v.startswith(f"Constraint {cidx}:")]
        count = len(msgs)
        if count <= 3:
            aggregated_violations.extend(msgs)
        else:
            aggregated_violations.extend(msgs[:2])
            aggregated_violations.append(
                f"... and {count - 2} more violations of constraint {cidx}"
            )

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": aggregated_violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for VRPDDP solutions"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
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
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
        for v in result["violations"][:10]:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
