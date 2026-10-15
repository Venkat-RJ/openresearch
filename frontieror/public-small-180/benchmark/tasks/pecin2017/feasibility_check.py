#!/usr/bin/env python3
"""
feasibility_check.py

Feasibility checker for VRPTW solutions based on Pecin et al. (2017).
Checks all hard constraints from the mathematical formulation in math_model.txt.

Constraint numbering (top to bottom in formulation):
  1:  Objective (min total route cost) — not a constraint, skip
  2:  Customer coverage: each customer visited exactly once
  3:  Binary lambda_r — satisfied by construction (route list)
  4:  Pricing objective — not a constraint, skip
  5:  Depot flow: route starts at depot 0, ends at depot n+1
  6:  Flow conservation at customer nodes
  7:  Vehicle capacity per route
  8:  Time propagation on active arcs
  9:  Time windows at all nodes
  10: Binary x_{ij} — satisfied by construction
  11-18: Valid inequalities (RCC, 2PC, SRC, rank-1, elementary) — implied by
         integer feasibility of (2)-(3) with feasible routes; checking all
         subsets is computationally intractable (exponential).
         Constraint 13 is a reduced-cost formula definition, not a constraint.
  19: Objective consistency — reported objective_value must equal the
      Euclidean travel cost summed over all arcs in the given routes
      (Tier C defense against LLM score-gaming exploits that fabricate
      objective_value while keeping the route structure feasible).
"""

import argparse
import json
import math


def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    schema_errors = []
    if not isinstance(solution, dict):
        schema_errors.append("solution must be a JSON object")
    else:
        if "objective_value" not in solution or solution["objective_value"] is None:
            schema_errors.append("missing required solution field: objective_value")
        objective = solution.get("objective_value")
        if (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(float(objective))
        ):
            schema_errors.append("objective_value must be a finite number")
        if "routes" not in solution or not isinstance(solution["routes"], list):
            schema_errors.append("routes must be a list")
    if schema_errors:
        return {
            "feasible": False,
            "violated_constraints": ["solution_schema"],
            "violations": schema_errors,
            "violation_magnitudes": [],
        }

    nodes = instance["nodes"]
    n_nodes = instance["problem_parameters"]["n_nodes"]
    n_customers = instance["problem_parameters"]["n_customers"]
    Q = instance["problem_parameters"]["vehicle_capacity"]
    H = instance["problem_parameters"]["planning_horizon"]

    depot_source = 0
    depot_sink = n_nodes - 1
    customers = list(range(1, n_customers + 1))

    # Extract node data
    node_x = [0.0] * n_nodes
    node_y = [0.0] * n_nodes
    demand = [0.0] * n_nodes
    tw_open = [0.0] * n_nodes
    tw_close = [float(H)] * n_nodes
    svc = [0.0] * n_nodes

    for node in nodes:
        i = node["id"]
        node_x[i] = node["x"]
        node_y[i] = node["y"]
        demand[i] = node["demand"]
        tw_open[i] = node["time_window_open"]
        tw_close[i] = node["time_window_close"]
        svc[i] = node["service_time"]

    def dist(i, j):
        return euclidean(node_x[i], node_y[i], node_x[j], node_y[j])

    routes = solution.get("routes", [])

    violations = []
    violation_magnitudes = []
    violated_constraints = set()

    # =========================================================================
    # Constraint 2: Customer coverage (equality)
    #   sum_r a^r_i * lambda_r = 1, for all i in V+
    # =========================================================================
    visit_count = {c: 0 for c in customers}
    for route in routes:
        for node_id in route:
            if 1 <= node_id <= n_customers:
                visit_count[node_id] += 1

    for c in customers:
        cnt = visit_count[c]
        violation_amount = abs(cnt - 1)
        if violation_amount > tol:
            violated_constraints.add(2)
            rhs = 1.0
            normalizer = max(abs(rhs), eps)
            if cnt == 0:
                violations.append(
                    f"Constraint 2: Customer {c} is not visited by any route")
            else:
                violations.append(
                    f"Constraint 2: Customer {c} is visited {cnt} times "
                    f"instead of exactly once")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(cnt),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # =========================================================================
    # Constraint 3: Binary lambda_r
    #   Satisfied by construction — routes are either in the list or not.
    # =========================================================================

    # =========================================================================
    # Constraint 5: Depot flow
    #   sum_{(0,j)} x_{0j} = sum_{(i,n+1)} x_{i,n+1} = 1  (per route)
    #   Each route must start at depot_source and end at depot_sink.
    # =========================================================================
    for r_idx, route in enumerate(routes):
        if len(route) < 2:
            violated_constraints.add(5)
            violations.append(
                f"Constraint 5: Route {r_idx} has fewer than 2 nodes")
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })
            continue

        if route[0] != depot_source:
            violated_constraints.add(5)
            violations.append(
                f"Constraint 5: Route {r_idx} does not start at depot source "
                f"(node {depot_source}), starts at node {route[0]}")
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })

        if route[-1] != depot_sink:
            violated_constraints.add(5)
            violations.append(
                f"Constraint 5: Route {r_idx} does not end at depot sink "
                f"(node {depot_sink}), ends at node {route[-1]}")
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })

    # =========================================================================
    # Constraint 6: Flow conservation at customer nodes
    #   sum_{(i,j)} x_{ij} - sum_{(j,i)} x_{ji} = 0, for all i in V+
    #   In route representation: no depot node should appear in route interior.
    # =========================================================================
    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route) - 1):
            node_id = route[pos]
            if node_id == depot_source or node_id == depot_sink:
                violated_constraints.add(6)
                violations.append(
                    f"Constraint 6: Route {r_idx} has depot node {node_id} "
                    f"at interior position {pos}")
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": 1.0,
                    "rhs": 0.0,
                    "raw_excess": 1.0,
                    "normalizer": eps,
                    "ratio": 1.0 / eps
                })

    # =========================================================================
    # Constraint 7: Vehicle capacity (<=)
    #   sum_{(i,j) in A} d_i * x_{ij} <= Q  (per route)
    # =========================================================================
    for r_idx, route in enumerate(routes):
        total_demand = sum(demand[node_id] for node_id in route)
        lhs = total_demand
        rhs = float(Q)
        violation_amount = max(0.0, lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(7)
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 7: Route {r_idx} total demand {lhs:.1f} "
                f"exceeds vehicle capacity {rhs:.1f}")
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # =========================================================================
    # Constraint 8: Time propagation on active arcs (<=)
    #   x_{ij}(T_i + s_i + t_{ij}) <= T_j, for all (i,j) in A
    #   For active arcs: T_i + s_i + t_{ij} <= T_j
    #
    # Constraint 9: Time windows (bounds)
    #   e_i <= T_i <= l_i, for all i in V
    #
    # We compute T_k = max(e_{route[k]}, T_{k-1} + svc[route[k-1]] + dist(...))
    # By construction, constraint 8 (T_i + s_i + t_{ij} <= T_j) is always
    # satisfied. The only possible violations are in constraint 9 (T_i > l_i).
    # =========================================================================
    for r_idx, route in enumerate(routes):
        if len(route) < 2:
            continue

        # Compute service start times along the route
        T = [0.0] * len(route)
        T[0] = tw_open[route[0]]

        for k in range(1, len(route)):
            i_node = route[k - 1]
            j_node = route[k]
            arrival = T[k - 1] + svc[i_node] + dist(i_node, j_node)
            T[k] = max(tw_open[j_node], arrival)

        # Check constraint 8: T_i + s_i + t_{ij} <= T_j for active arcs
        for k in range(1, len(route)):
            i_node = route[k - 1]
            j_node = route[k]
            lhs = T[k - 1] + svc[i_node] + dist(i_node, j_node)
            rhs_val = T[k]
            violation_amount = max(0.0, lhs - rhs_val)
            if violation_amount > tol:
                violated_constraints.add(8)
                normalizer = max(abs(rhs_val), eps)
                violations.append(
                    f"Constraint 8: Route {r_idx}, arc ({i_node},{j_node}): "
                    f"arrival {lhs:.4f} > service start {rhs_val:.4f}")
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": lhs,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

        # Check constraint 9: e_i <= T_i <= l_i
        for k in range(len(route)):
            node_id = route[k]

            # Lower bound: T_i >= e_i (>= constraint)
            lb_violation = max(0.0, tw_open[node_id] - T[k])
            if lb_violation > tol:
                violated_constraints.add(9)
                normalizer = max(abs(tw_open[node_id]), eps)
                violations.append(
                    f"Constraint 9: Route {r_idx}, node {node_id}: "
                    f"service start {T[k]:.4f} < earliest {tw_open[node_id]:.4f}")
                violation_magnitudes.append({
                    "constraint": 9,
                    "lhs": tw_open[node_id],
                    "rhs": T[k],
                    "raw_excess": lb_violation,
                    "normalizer": normalizer,
                    "ratio": lb_violation / normalizer
                })

            # Upper bound: T_i <= l_i (<= constraint)
            ub_violation = max(0.0, T[k] - tw_close[node_id])
            if ub_violation > tol:
                violated_constraints.add(9)
                normalizer = max(abs(tw_close[node_id]), eps)
                violations.append(
                    f"Constraint 9: Route {r_idx}, node {node_id}: "
                    f"service start {T[k]:.4f} > latest {tw_close[node_id]:.4f}")
                violation_magnitudes.append({
                    "constraint": 9,
                    "lhs": T[k],
                    "rhs": tw_close[node_id],
                    "raw_excess": ub_violation,
                    "normalizer": normalizer,
                    "ratio": ub_violation / normalizer
                })

    # =========================================================================
    # Constraint 10: Binary x_{ij} — satisfied by construction.
    # =========================================================================

    # =========================================================================
    # Constraints 11-18: Valid inequalities (RCC, 2PC, rank-1, elementary).
    # These are cutting planes parameterized by customer subsets S ⊆ V+.
    # For integer feasible solutions satisfying constraints (2)-(10), these
    # are automatically satisfied. Enumerating all subsets is intractable.
    # Constraint 13 is a reduced-cost formula, not a constraint.
    # =========================================================================

    # =========================================================================
    # Constraint 19: Objective consistency (Tier C anti-score-gaming check).
    #   Recompute the total Euclidean travel cost from the given routes and
    #   verify it matches the reported objective_value. The solution contains
    #   every variable the objective formula c_r = sum c_{ij} depends on
    #   (the full route sequences), so a full recompute is exact.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            true_obj = 0.0
            for route in routes:
                for k in range(1, len(route)):
                    true_obj += dist(route[k - 1], route[k])
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(19)
                normalizer = max(abs(true_obj), eps)
                violations.append(
                    f"Constraint 19: Objective consistency violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed total route cost={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})")
                violation_magnitudes.append({
                    "constraint": 19,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer
                })
        elif reported is not None:
            # Non-finite reported obj (inf/nan): always reject.
            true_obj = 0.0
            for route in routes:
                for k in range(1, len(route)):
                    true_obj += dist(route[k - 1], route[k])
            violated_constraints.add(19)
            normalizer = max(abs(true_obj), eps)
            violations.append(
                f"Constraint 19: Objective consistency violated: "
                f"reported objective_value={reported} is not finite; "
                f"recomputed total route cost={true_obj}")
            violation_magnitudes.append({
                "constraint": 19,
                "lhs": reported,
                "rhs": true_obj,
                "raw_excess": float("inf"),
                "normalizer": normalizer,
                "ratio": float("inf")
            })

    # Build result
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }
    return result


def _write_error_result(result_path, message):
    result = {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": [message],
        "violation_magnitudes": [],
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="VRPTW Feasibility Checker")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    try:
        with open(args.instance_path) as f:
            instance = json.load(f)
        with open(args.solution_path) as f:
            solution = json.load(f)
        result = check_feasibility(instance, solution)
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as exc:
        result = _write_error_result(
            args.result_path,
            f"checker could not parse or validate solution: {type(exc).__name__}: {exc}",
        )

    if result["feasible"]:
        print(f"Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: "
              f"{result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
