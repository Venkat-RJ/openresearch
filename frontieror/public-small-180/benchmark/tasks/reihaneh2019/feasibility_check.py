#!/usr/bin/env python3
"""
Feasibility checker for the VRDAP (Vehicle Routing with Demand Allocation Problem).

Checks constraints (4b)-(4l) from Reihaneh & Ghoniem (2019), Appendix A.
Numbered as Constraint 1 through Constraint 11 (top to bottom in the formulation).

Constraint 12: Objective-value consistency (Tier C defense against LLM
score-gaming exploits). Recomputes the obj from the routing arcs and
customer-to-site assignments in the solution and rejects when the reported
objective_value disagrees with the recomputed total cost.
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignments', 'customer_tours', 'objective_value', 'routes')
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
    with open(path, "r") as f:
        return json.load(f)


def build_solution_variables(instance, solution):
    """
    Reconstruct the decision variables (s, e, theta, q) from the solution JSON.

    Supports two solution formats:
      - Gurobi format: has 'customer_tours' with {vehicle, site} per customer
      - Efficient algorithm format: routes contain 'customers' mapping site -> [customer IDs]
    """
    num_sites = instance["num_delivery_sites"]
    num_customers = instance["num_customers"]
    Q = instance["vehicle_capacity"]
    num_vehicles = instance["num_vehicles"]

    customers = instance["customers"]
    d = {c["id"]: c["demand"] for c in customers}

    S = list(range(1, num_sites + 1))
    K = list(range(1, num_customers + 1))
    N = [0] + S

    # --- Determine solution format and reconstruct variables ---

    # s[i, k]: customer k assigned to delivery site i
    s = {}
    for i in S:
        for k in K:
            s[i, k] = 0

    assignments = solution.get("assignments", {})
    for k_str, site in assignments.items():
        k = int(k_str)
        s[site, k] = 1

    # e[v, i, j]: arc (i,j) in vehicle tour v
    e = {}
    # Collect all vehicle IDs used in routes
    route_vehicles = [int(v) for v in solution["routes"].keys()]
    V = list(range(1, num_vehicles + 1))

    for v in V:
        for i in N:
            for j in N:
                if i != j:
                    e[v, i, j] = 0

    for v_str, route_data in solution["routes"].items():
        v = int(v_str)
        if "arcs" in route_data:
            # Gurobi format
            for arc in route_data["arcs"]:
                i, j = arc[0], arc[1]
                e[v, i, j] = 1
        else:
            # Efficient algorithm format: reconstruct arcs from sequence
            seq = route_data["sequence"]
            for idx in range(len(seq) - 1):
                i, j = seq[idx], seq[idx + 1]
                e[v, i, j] = 1

    # theta[v, i, k]: site i visited by tour v AND customer k assigned to site i
    theta = {}
    for v in V:
        for i in S:
            for k in K:
                theta[v, i, k] = 0

    if "customer_tours" in solution:
        # Gurobi format: customer_tours maps customer -> {vehicle, site}
        for k_str, info in solution["customer_tours"].items():
            k = int(k_str)
            v = info["vehicle"]
            i = info["site"]
            theta[v, i, k] = 1
    else:
        # Efficient algorithm format: routes contain 'customers' mapping
        for v_str, route_data in solution["routes"].items():
            v = int(v_str)
            if "customers" in route_data:
                for site_str, cust_list in route_data["customers"].items():
                    site = int(site_str)
                    for k in cust_list:
                        theta[v, site, k] = 1

    # q[v, i]: cumulative deliveries upon serving site i in tour v
    # Reconstruct from route sequence and demand assignments
    q = {}
    for v in V:
        for i in S:
            q[v, i] = 0.0

    for v_str, route_data in solution["routes"].items():
        v = int(v_str)
        seq = route_data["sequence"]
        cumulative = 0.0
        for node in seq:
            if node == 0:
                continue
            # Add demand of all customers assigned to this site on this tour
            demand_at_site = sum(d[k] for k in K if theta[v, node, k] > 0.5)
            cumulative += demand_at_site
            q[v, node] = cumulative

    return s, e, theta, q, S, K, V, N, d, Q


def check_feasibility(instance, solution):
    """Check all hard constraints (4b)-(4l), numbered 1-11, plus obj
    consistency (Constraint 12)."""
    tol = 1e-5
    eps = 1e-5

    s, e, theta, q, S, K, V, N, d, Q = build_solution_variables(instance, solution)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Domain check: every node id appearing in any route's sequence/arcs
    # must be in N (= [0] + delivery sites); every assignments site value
    # must be in S; every customer key must be in K; every vehicle key
    # must be in V.
    valid_N = set(N)
    valid_S = set(S)
    valid_K = set(K)
    valid_V = set(V)
    for v_str, route_data in solution.get("routes", {}).items():
        try:
            v_int = int(v_str)
        except (TypeError, ValueError):
            v_int = None
        if v_int not in valid_V:
            violated_constraints.add(0)
            violations.append(
                f"Domain: routes vehicle id {v_str!r} not in valid range [1, {len(V)}]"
            )
            violation_magnitudes.append({
                "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
        for arc in route_data.get("arcs", []):
            for node in arc[:2]:
                if node not in valid_N:
                    violated_constraints.add(0)
                    violations.append(
                        f"Domain: route v={v_str} arc node {node!r} not in valid set"
                    )
                    violation_magnitudes.append({
                        "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                        "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
                    })
        for node in route_data.get("sequence", []):
            if node not in valid_N:
                violated_constraints.add(0)
                violations.append(
                    f"Domain: route v={v_str} sequence node {node!r} not in valid set"
                )
                violation_magnitudes.append({
                    "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                    "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
                })
    for k_str, site in solution.get("assignments", {}).items():
        try:
            k_int = int(k_str)
        except (TypeError, ValueError):
            k_int = None
        if k_int not in valid_K:
            violated_constraints.add(0)
            violations.append(
                f"Domain: assignments customer id {k_str!r} not a valid customer"
            )
            violation_magnitudes.append({
                "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
        if site not in valid_S:
            violated_constraints.add(0)
            violations.append(
                f"Domain: assignments[{k_str}] = {site!r} not a valid delivery site"
            )
            violation_magnitudes.append({
                "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })

    def record_violation(constraint_idx, msg, lhs, rhs, operator="<="):
        violated_constraints.add(constraint_idx)
        violations.append(msg)
        if operator == "<=":
            violation_amount = max(lhs - rhs, 0.0)
        elif operator == ">=":
            violation_amount = max(rhs - lhs, 0.0)
        elif operator == "=":
            violation_amount = abs(lhs - rhs)
        else:
            violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio,
        })

    # ---------------------------------------------------------------
    # Constraint 1 (4b): Each delivery site visited by at most one tour
    # sum_v sum_{i in N\{j}} e^v_{ij} <= 1, forall j in S
    # ---------------------------------------------------------------
    for j in S:
        lhs = sum(e[v, i, j] for v in V for i in N if i != j)
        rhs = 1.0
        if lhs - rhs > tol:
            record_violation(1, f"Constraint 1 (4b): Site {j} visited by {lhs} tours (max 1)", lhs, rhs, "<=")

    # ---------------------------------------------------------------
    # Constraint 2 (4c): Each tour has at most one arc leaving depot
    # sum_{j in S} e^v_{0j} <= 1, forall v in V
    # ---------------------------------------------------------------
    for v in V:
        lhs = sum(e[v, 0, j] for j in S)
        rhs = 1.0
        if lhs - rhs > tol:
            record_violation(2, f"Constraint 2 (4c): Vehicle {v} has {lhs} arcs leaving depot (max 1)", lhs, rhs, "<=")

    # ---------------------------------------------------------------
    # Constraint 3 (4d): Flow balance at each node
    # sum_{i in N\{j}} e^v_{ij} - sum_{i in N\{j}} e^v_{ji} = 0, forall v, j in N
    # ---------------------------------------------------------------
    for v in V:
        for j in N:
            inflow = sum(e[v, i, j] for i in N if i != j)
            outflow = sum(e[v, j, i] for i in N if i != j)
            lhs = inflow - outflow
            rhs = 0.0
            if abs(lhs - rhs) > tol:
                record_violation(3, f"Constraint 3 (4d): Flow imbalance at node {j} on vehicle {v}: in={inflow}, out={outflow}", lhs, rhs, "=")

    # ---------------------------------------------------------------
    # Constraint 4 (4e): Subtour elimination / cumulative delivery tracking
    # q^v_j >= q^v_i + sum_k d_k s_{jk} - 2Q(1 - e^v_{ij}) + Q e^v_{ji}
    # forall v, i,j in S, i != j
    # Rewrite as: q^v_j - q^v_i - sum_k d_k s_{jk} + 2Q(1 - e^v_{ij}) - Q e^v_{ji} >= 0
    # i.e., LHS = q^v_j, RHS = q^v_i + sum_k d_k s_{jk} - 2Q(1-e^v_{ij}) + Q e^v_{ji}
    # ---------------------------------------------------------------
    for v in V:
        for i in S:
            for j in S:
                if i != j:
                    demand_j = sum(d[k] * s[j, k] for k in K)
                    rhs_val = q[v, i] + demand_j - 2 * Q * (1 - e[v, i, j]) + Q * e[v, j, i]
                    lhs_val = q[v, j]
                    if rhs_val - lhs_val > tol:
                        record_violation(
                            4,
                            f"Constraint 4 (4e): Subtour elimination violated for vehicle {v}, arc ({i},{j}): "
                            f"q[{v},{j}]={lhs_val:.4f} < RHS={rhs_val:.4f}",
                            lhs_val, rhs_val, ">="
                        )

    # ---------------------------------------------------------------
    # Constraint 5 (4f): Linking theta and (s, e) — lower bound
    # theta^v_{ik} >= s_{ik} + sum_{j in N\{i}} e^v_{ij} - 1
    # ---------------------------------------------------------------
    for v in V:
        for i in S:
            for k in K:
                rhs_val = s[i, k] + sum(e[v, i, j] for j in N if j != i) - 1
                lhs_val = theta[v, i, k]
                if rhs_val - lhs_val > tol:
                    record_violation(
                        5,
                        f"Constraint 5 (4f): theta[{v},{i},{k}]={lhs_val} < s[{i},{k}]+out_edges-1={rhs_val}",
                        lhs_val, rhs_val, ">="
                    )

    # ---------------------------------------------------------------
    # Constraint 6 (4g): Linking theta and e — upper bound
    # theta^v_{ik} <= sum_{j in N\{i}} e^v_{ij}
    # ---------------------------------------------------------------
    for v in V:
        for i in S:
            for k in K:
                lhs_val = theta[v, i, k]
                rhs_val = sum(e[v, i, j] for j in N if j != i)
                if lhs_val - rhs_val > tol:
                    record_violation(
                        6,
                        f"Constraint 6 (4g): theta[{v},{i},{k}]={lhs_val} > out_edges={rhs_val}",
                        lhs_val, rhs_val, "<="
                    )

    # ---------------------------------------------------------------
    # Constraint 7 (4h): Linking theta and s — upper bound
    # theta^v_{ik} <= s_{ik}
    # ---------------------------------------------------------------
    for v in V:
        for i in S:
            for k in K:
                lhs_val = theta[v, i, k]
                rhs_val = s[i, k]
                if lhs_val - rhs_val > tol:
                    record_violation(
                        7,
                        f"Constraint 7 (4h): theta[{v},{i},{k}]={lhs_val} > s[{i},{k}]={rhs_val}",
                        lhs_val, rhs_val, "<="
                    )

    # ---------------------------------------------------------------
    # Constraint 8 (4i): Upper bound on cumulative delivery
    # q^v_i <= Q * sum_{j in N\{i}} e^v_{ij}
    # ---------------------------------------------------------------
    for v in V:
        for i in S:
            lhs_val = q[v, i]
            rhs_val = Q * sum(e[v, i, j] for j in N if j != i)
            if lhs_val - rhs_val > tol:
                record_violation(
                    8,
                    f"Constraint 8 (4i): q[{v},{i}]={lhs_val} > Q*out_edges={rhs_val}",
                    lhs_val, rhs_val, "<="
                )

    # ---------------------------------------------------------------
    # Constraint 9 (4j): Lower bound on cumulative delivery
    # sum_k d_k theta^v_{ik} <= q^v_i
    # ---------------------------------------------------------------
    for v in V:
        for i in S:
            lhs_val = sum(d[k] * theta[v, i, k] for k in K)
            rhs_val = q[v, i]
            if lhs_val - rhs_val > tol:
                record_violation(
                    9,
                    f"Constraint 9 (4j): demand on site {i} tour {v} = {lhs_val} > q[{v},{i}]={rhs_val}",
                    lhs_val, rhs_val, "<="
                )

    # ---------------------------------------------------------------
    # Constraint 10 (4k): Every customer assigned to exactly one site and tour
    # sum_v sum_i theta^v_{ik} = 1, forall k
    # ---------------------------------------------------------------
    for k in K:
        lhs_val = sum(theta[v, i, k] for v in V for i in S)
        rhs_val = 1.0
        if abs(lhs_val - rhs_val) > tol:
            record_violation(
                10,
                f"Constraint 10 (4k): Customer {k} assigned to {lhs_val} (site,tour) pairs (must be 1)",
                lhs_val, rhs_val, "="
            )

    # ---------------------------------------------------------------
    # Constraint 11 (4l): Variable domains
    # e^v_{ij} in {0,1}, s_{ik} in {0,1}, theta^v_{ik} >= 0, q^v_i >= 0
    # ---------------------------------------------------------------
    # Check binary domains for e
    for v in V:
        for i in N:
            for j in N:
                if i != j:
                    val = e[v, i, j]
                    if val < -tol or val > 1 + tol or (abs(val - round(val)) > tol):
                        record_violation(
                            11,
                            f"Constraint 11 (4l): e[{v},{i},{j}]={val} not binary",
                            val, round(val), "="
                        )

    # Check binary domains for s
    for i in S:
        for k in K:
            val = s[i, k]
            if val < -tol or val > 1 + tol or (abs(val - round(val)) > tol):
                record_violation(
                    11,
                    f"Constraint 11 (4l): s[{i},{k}]={val} not binary",
                    val, round(val), "="
                )

    # Check non-negativity of theta
    for v in V:
        for i in S:
            for k in K:
                val = theta[v, i, k]
                if val < -tol:
                    record_violation(
                        11,
                        f"Constraint 11 (4l): theta[{v},{i},{k}]={val} < 0",
                        val, 0.0, ">="
                    )

    # Check non-negativity of q
    for v in V:
        for i in S:
            val = q[v, i]
            if val < -tol:
                record_violation(
                    11,
                    f"Constraint 11 (4l): q[{v},{i}]={val} < 0",
                    val, 0.0, ">="
                )

    # ---------------------------------------------------------------
    # Constraint 12: Objective-value consistency (Tier C)
    # Recompute obj = sum_v sum_{(i,j) in E} c_{ij} e^v_{ij}
    #               + sum_{i in S, k in K} f_{ik} s_{ik}
    # using the routing-cost matrix and assignment-cost matrix from the
    # instance. Reject when |reported - recomputed| exceeds tolerance.
    # All obj-determining variables (routes -> e, assignments -> s) are
    # present in the solution, so full recompute is exact.
    # ---------------------------------------------------------------
    c_mat = instance.get("routing_cost_matrix")
    f_mat = instance.get("assignment_cost_matrix")
    reported_obj = solution.get("objective_value")
    if c_mat is not None and f_mat is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            routing_cost = 0.0
            for v in V:
                for i in N:
                    for j in N:
                        if i != j and e[v, i, j]:
                            routing_cost += float(c_mat[i][j])
            assign_cost = 0.0
            for i in S:
                for k in K:
                    if s[i, k]:
                        # assignment_cost_matrix is shape [num_customers][num_sites],
                        # rows 0-indexed by customer (k-1), cols 0-indexed by site (i-1).
                        assign_cost += float(f_mat[k - 1][i - 1])
            true_obj = routing_cost + assign_cost
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                record_violation(
                    12,
                    f"Constraint 12: Objective consistency violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"routing_cost+assign_cost={true_obj} "
                    f"(routing={routing_cost}, assign={assign_cost}, "
                    f"|diff|={abs_diff:.3g}, tol={tol_obj:.3g})",
                    reported, true_obj, "="
                )

    # --- Build result ---
    feasible = len(violated_constraints) == 0
    # Aggregate violations by constraint index: keep unique messages per constraint
    sorted_violated = sorted(violated_constraints)

    # Build aggregated violation messages (one per constraint index)
    aggregated_violations = []
    for c_idx in sorted_violated:
        msgs = [v for v, m in zip(violations, violation_magnitudes) if m["constraint"] == c_idx]
        if len(msgs) <= 3:
            aggregated_violations.append("; ".join(msgs))
        else:
            aggregated_violations.append(f"{msgs[0]}; ... ({len(msgs)} total violations for constraint {c_idx})")

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": aggregated_violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for VRDAP (Reihaneh & Ghoniem 2019)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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
        print(f"FEASIBLE - no constraint violations found. Result saved to {args.result_path}")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}. Result saved to {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
