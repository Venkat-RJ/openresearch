"""
Feasibility checker for the Split-Delivery Vehicle Routing Problem with
Time Windows and Linear Weight-Related Cost (SDVRPTWL).

Based on: Luo, Qin, Zhu, Lim (2017), Transportation Science 51(2):668-687.

Checks constraints (2)-(9) from the arc-flow formulation in Section 3.

This is the "_new" variant: identical to feasibility_check.py except that
it adds constraint 10 (objective consistency, Tier C defense) — recomputes
the reported objective from the routes/edge-loads and rejects when the
program's self-reported objective_value disagrees with the recomputation.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'routes')
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
    with open(path) as f:
        return json.load(f)


def build_problem(instance):
    """Build problem data structures from instance JSON."""
    n = instance["num_customers"]
    Q = instance["vehicle_capacity"]
    a_cost = instance["cost_function"]["a"]
    b_cost = instance["cost_function"]["b"]
    depot = instance["depot"]
    customers = instance["customers"]
    dm = instance["distance_matrix"]
    n1 = n + 1  # depot return vertex

    # Build extended distance matrix (n+2 vertices: 0..n, n+1)
    nv = n + 2
    c = [[0.0] * nv for _ in range(nv)]
    for i in range(n + 1):
        for j in range(n + 1):
            c[i][j] = dm[i][j]
        c[i][n1] = dm[i][0]  # n+1 is at depot location
    for j in range(n + 1):
        c[n1][j] = dm[0][j]
    c[0][n1] = 0.0
    c[n1][n1] = 0.0

    # Customer data
    d = [0] * nv  # demand
    s = [0] * nv  # service time
    e = [0] * nv  # earliest time
    l = [0] * nv  # latest time

    s[0] = depot["service_time"]
    e[0] = depot["time_window"][0]
    l[0] = depot["time_window"][1]

    for cu in customers:
        i = cu["id"]
        d[i] = cu["demand"]
        s[i] = cu["service_time"]
        e[i] = cu["time_window"][0]
        l[i] = cu["time_window"][1]

    d[n1] = 0
    s[n1] = 0
    e[n1] = depot["time_window"][0]
    l[n1] = depot["time_window"][1]

    return {
        "n": n, "Q": Q, "a": a_cost, "b": b_cost,
        "nv": nv, "n1": n1,
        "c": c, "d": d, "s": s, "e": e, "l": l,
    }


def parse_solution(solution, P):
    """
    Parse solution JSON into a uniform structure.
    Returns list of vehicle dicts, each with:
      - route: list of vertex ids [0, ..., n+1]
      - deliveries: dict {customer_id(int): quantity(float)}
      - edges: list of (i, j, load_weight) if available, else None
    """
    n = P["n"]
    n1 = P["n1"]
    vehicles = []

    for r in solution.get("routes", []):
        route = list(r["route"])
        # Remove duplicate depot at end (gurobi format sometimes has it)
        while len(route) > 1 and route[-1] == route[-2]:
            route.pop()

        deliveries = {}
        for k, v in r.get("deliveries", {}).items():
            deliveries[int(k)] = float(v)

        # Parse edges if available (gurobi format)
        edges = None
        if "edges" in r:
            edges = []
            for ed in r["edges"]:
                edges.append((int(ed["from"]), int(ed["to"]), float(ed["load_weight"])))

        vehicles.append({
            "route": route,
            "deliveries": deliveries,
            "edges": edges,
        })

    return vehicles


def compute_edge_loads(vehicle, P):
    """
    Compute load on each edge for a vehicle.
    This is a delivery problem: vehicle leaves depot loaded, load decreases.
    Load on edge (i,j) = total remaining delivery after node i.

    For the first edge (0, first_customer), load = sum of all deliveries.
    After delivering at node i, load decreases by delivery[i].
    """
    route = vehicle["route"]
    deliveries = vehicle["deliveries"]

    # If explicit edges with loads are provided, use them
    if vehicle["edges"] is not None:
        return vehicle["edges"]

    # Otherwise compute from route and deliveries
    # Total load leaving depot
    total_load = sum(deliveries.values())
    edge_loads = []
    current_load = total_load

    for idx in range(len(route) - 1):
        i = route[idx]
        j = route[idx + 1]
        edge_loads.append((i, j, current_load))
        # After arriving at j and delivering
        if j in deliveries:
            current_load -= deliveries[j]

    return edge_loads


def check_feasibility(instance, solution):
    """
    Check all hard constraints (2)-(9) from the SDVRPTWL formulation, plus
    constraint (10) — objective consistency (Tier C exploit defense).
    """
    # large_31 had violation amount 2e-4 (delivered=49.9998 vs 50.0); MIP
    # rounding artifacts on continuous flow vars can exceed 1e-4 -- relax to
    # 1e-3 (still 0.002% relative for typical demand ~50).
    tol = 1e-3
    eps = 1e-5

    P = build_problem(instance)
    vehicles = parse_solution(solution, P)
    n = P["n"]
    n1 = P["n1"]
    Q = P["Q"]
    c = P["c"]
    d = P["d"]
    s = P["s"]
    e = P["e"]
    l = P["l"]
    a_cost = P["a"]
    b_cost = P["b"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    m = len(vehicles)  # number of vehicles used

    # Pre-compute edge loads for all vehicles
    all_edge_loads = []
    for k, veh in enumerate(vehicles):
        all_edge_loads.append(compute_edge_loads(veh, P))

    # =========================================================================
    # Constraint (2): Demand fulfillment
    # sum_k (sum_{j in V^-(i)} w_{j,i,k} - sum_{j in V^+(i)} w_{i,j,k}) >= d_i
    # For each customer i, the total delivered must be >= d_i.
    # The LHS is the net flow into each customer = total delivered to customer i.
    # =========================================================================
    for i in range(1, n + 1):
        total_delivered = 0.0
        for k, veh in enumerate(vehicles):
            total_delivered += veh["deliveries"].get(i, 0.0)

        lhs = total_delivered
        rhs = float(d[i])
        violation_amount = rhs - lhs  # >= constraint: violation when rhs > lhs

        if violation_amount > tol:
            violated_constraints.add(2)
            violations.append(
                f"Constraint (2): Customer {i} demand not fulfilled: "
                f"delivered={lhs:.4f}, required={rhs:.4f}"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (3) SKIPPED: min-visits valid inequality
    # `sum_k sum_{j in V^-(i)} x_{j,i,k} >= ceil(d_i / Q)` is a strengthening
    # VI (the paper's math_model notes it is "redundant, strengthens LP
    # relaxation"). Any feasible routing that fulfills customer demand
    # satisfies this inequality automatically, so it is not checked.
    # =========================================================================

    # =========================================================================
    # Constraint (4): Each vehicle departs from depot (vertex 0)
    # sum_{i in V_C ∪ {n+1}} x_{0,i,k} = 1 for each vehicle k
    # Each vehicle's route must start at vertex 0 and have exactly one arc
    # leaving vertex 0.
    # =========================================================================
    for k, veh in enumerate(vehicles):
        route = veh["route"]
        # Count arcs leaving vertex 0
        departures_from_depot = 0
        for idx in range(len(route) - 1):
            if route[idx] == 0:
                departures_from_depot += 1

        lhs = float(departures_from_depot)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)  # = constraint

        if violation_amount > tol:
            violated_constraints.add(4)
            violations.append(
                f"Constraint (4): Vehicle {k} has {departures_from_depot} "
                f"departures from depot, expected 1"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (5): Flow conservation and single visit per vehicle
    # sum_{j in V^+(i)} x_{i,j,k} = sum_{j in V^-(i)} x_{j,i,k} <= 1
    # For each vehicle k and customer i:
    #   (a) in-degree = out-degree (flow conservation)
    #   (b) in-degree <= 1 (visit at most once)
    # =========================================================================
    for k, veh in enumerate(vehicles):
        route = veh["route"]
        for i in range(1, n + 1):
            # Count arcs into and out of customer i for this vehicle
            in_degree = 0
            out_degree = 0
            for idx in range(len(route) - 1):
                if route[idx + 1] == i:
                    in_degree += 1
                if route[idx] == i:
                    out_degree += 1

            # (5a) Flow conservation: out_degree = in_degree
            lhs_a = float(out_degree)
            rhs_a = float(in_degree)
            violation_amount_a = abs(lhs_a - rhs_a)

            if violation_amount_a > tol:
                violated_constraints.add(5)
                violations.append(
                    f"Constraint (5): Vehicle {k}, customer {i}: "
                    f"out-degree={out_degree} != in-degree={in_degree} "
                    f"(flow conservation violated)"
                )
                normalizer = max(abs(rhs_a), eps)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs_a,
                    "rhs": rhs_a,
                    "raw_excess": violation_amount_a,
                    "normalizer": normalizer,
                    "ratio": violation_amount_a / normalizer,
                })

            # (5b) in_degree <= 1 (each vehicle visits each customer at most once)
            lhs_b = float(in_degree)
            rhs_b = 1.0
            violation_amount_b = lhs_b - rhs_b  # <= constraint

            if violation_amount_b > tol:
                violated_constraints.add(5)
                violations.append(
                    f"Constraint (5): Vehicle {k}, customer {i}: "
                    f"visited {in_degree} times, at most 1 allowed"
                )
                normalizer = max(abs(rhs_b), eps)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs_b,
                    "rhs": rhs_b,
                    "raw_excess": violation_amount_b,
                    "normalizer": normalizer,
                    "ratio": violation_amount_b / normalizer,
                })

    # =========================================================================
    # Constraint (6): Each vehicle returns to depot (vertex n+1)
    # sum_{i in V_C ∪ {0}} x_{i,n+1,k} = 1 for each vehicle k
    # =========================================================================
    for k, veh in enumerate(vehicles):
        route = veh["route"]
        arrivals_at_depot = 0
        for idx in range(len(route) - 1):
            if route[idx + 1] == n1:
                arrivals_at_depot += 1

        lhs = float(arrivals_at_depot)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)  # = constraint

        if violation_amount > tol:
            violated_constraints.add(6)
            violations.append(
                f"Constraint (6): Vehicle {k} has {arrivals_at_depot} "
                f"arrivals at depot (n+1), expected 1"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (7): Vehicle capacity on each edge
    # w_{i,j,k} <= Q * x_{i,j,k}
    # For each edge traversed (x=1), load must be <= Q.
    # For edges not traversed (x=0), load must be 0.
    # =========================================================================
    for k, veh in enumerate(vehicles):
        edge_loads = all_edge_loads[k]
        for (i, j, w_ij) in edge_loads:
            lhs = float(w_ij)
            rhs = float(Q)  # x_{i,j,k}=1 for traversed edges
            violation_amount = lhs - rhs  # <= constraint

            if violation_amount > tol:
                violated_constraints.add(7)
                violations.append(
                    f"Constraint (7): Vehicle {k}, edge ({i},{j}): "
                    f"load={w_ij:.4f} exceeds capacity Q={Q}"
                )
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # =========================================================================
    # Constraint (8): Time window precedence (big-M)
    # a_{j,k} >= a_{i,k} + s_i + t_{i,j} + M(x_{i,j,k} - 1)
    # For traversed edges (x=1): a_{j,k} >= a_{i,k} + s_i + t_{i,j}
    # i.e., arrival at j must be no earlier than departure from i + travel time.
    # We check this by simulating the schedule along each route.
    # =========================================================================
    for k, veh in enumerate(vehicles):
        route = veh["route"]
        if len(route) < 2:
            continue

        # Simulate schedule
        a_k = {}  # service start times
        t_cur = float(e[route[0]])  # start at depot earliest time
        a_k[route[0]] = t_cur

        for idx in range(1, len(route)):
            i = route[idx - 1]
            j = route[idx]
            t_ij = c[i][j]  # travel time = distance
            arrival = t_cur + s[i] + t_ij
            start_j = max(float(e[j]), arrival)  # wait if early
            a_k[j] = start_j

            # Check: a_{j,k} >= a_{i,k} + s_i + t_{i,j}
            # Which means arrival <= start_j (satisfied by max with e[j])
            # But start_j must also be <= l[j] (checked in constraint 9)
            # The precedence itself: start_j >= a_k[i] + s[i] + t_ij
            lhs = start_j
            rhs_val = a_k[route[idx - 1]] + s[i] + t_ij
            violation_amount = rhs_val - lhs  # >= constraint (reversed: rhs > lhs means violation)

            if violation_amount > tol:
                violated_constraints.add(8)
                violations.append(
                    f"Constraint (8): Vehicle {k}, edge ({i},{j}): "
                    f"service start at {j} is {lhs:.4f}, but must be >= {rhs_val:.4f} "
                    f"(time precedence violated)"
                )
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": lhs,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

            t_cur = start_j

    # =========================================================================
    # Constraint (9): Time window bounds
    # e_i <= a_{i,k} <= l_i for each vehicle k, vertex i
    # =========================================================================
    for k, veh in enumerate(vehicles):
        route = veh["route"]
        if len(route) < 2:
            continue

        # Simulate schedule
        t_cur = float(e[route[0]])
        for idx in range(len(route)):
            vertex = route[idx]
            if idx == 0:
                a_ik = t_cur
            else:
                prev = route[idx - 1]
                t_ij = c[prev][vertex]
                arrival = t_cur + s[prev] + t_ij
                a_ik = max(float(e[vertex]), arrival)
                t_cur = a_ik

            # Check e_i <= a_{i,k}
            lhs_lower = float(e[vertex])
            rhs_lower = a_ik
            violation_lower = lhs_lower - rhs_lower  # e_i <= a_ik means violation if e_i > a_ik

            if violation_lower > tol:
                violated_constraints.add(9)
                violations.append(
                    f"Constraint (9): Vehicle {k}, vertex {vertex}: "
                    f"service start {a_ik:.4f} < earliest time {e[vertex]}"
                )
                normalizer = max(abs(lhs_lower), eps)
                violation_magnitudes.append({
                    "constraint": 9,
                    "lhs": rhs_lower,
                    "rhs": lhs_lower,
                    "raw_excess": violation_lower,
                    "normalizer": normalizer,
                    "ratio": violation_lower / normalizer,
                })

            # Check a_{i,k} <= l_i
            lhs_upper = a_ik
            rhs_upper = float(l[vertex])
            violation_upper = lhs_upper - rhs_upper  # <= constraint

            if violation_upper > tol:
                violated_constraints.add(9)
                violations.append(
                    f"Constraint (9): Vehicle {k}, vertex {vertex}: "
                    f"service start {a_ik:.4f} > latest time {l[vertex]}"
                )
                normalizer = max(abs(rhs_upper), eps)
                violation_magnitudes.append({
                    "constraint": 9,
                    "lhs": lhs_upper,
                    "rhs": rhs_upper,
                    "raw_excess": violation_upper,
                    "normalizer": normalizer,
                    "ratio": violation_upper / normalizer,
                })

    # =========================================================================
    # Constraint (10): Objective consistency (Tier C exploit defense).
    # Recompute the obj (Eq. 1) from the solution variables and reject when
    # the reported value disagrees. All variables the obj depends on
    # (x_{i,j,k} via the traversed edges, w_{i,j,k} via load_weight) are
    # carried in the solution, so a full recompute is exact.
    #
    #   obj = sum_k sum_{(i,j) traversed by k} c_{i,j} * (a * w_{i,j,k} + b)
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and not (math.isnan(reported) or math.isinf(reported)):
            true_obj = 0.0
            for k, veh in enumerate(vehicles):
                for (i, j, w_ij) in all_edge_loads[k]:
                    true_obj += c[i][j] * (a_cost * float(w_ij) + b_cost)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor; Gurobi obj
            # for large instances is ~1e4-1e5 so this is far tighter than
            # any meaningful exploit gap (LLM exploits return 0, 1.8e308,
            # or 1e6 x truth -- all >> 0.1%).
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_constraints.add(10)
                violations.append(
                    f"Constraint (10): Objective consistency violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed sum c_ij*(a*w_ijk + b*x_ijk)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 10,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })
        elif reported is not None:
            # NaN / Inf reported obj -- always a lie since the true obj is
            # bounded by sum |c_ij| * (a*Q + b) over all edges. Treat as
            # objective-consistency violation.
            violated_constraints.add(10)
            violations.append(
                f"Constraint (10): Objective consistency violated: "
                f"reported objective_value={reported_obj} is NaN or Inf"
            )
            violation_magnitudes.append({
                "constraint": 10,
                "lhs": reported,
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": eps,
                "ratio": float("inf"),
            })

    # Build result
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SDVRPTWL (Luo et al. 2017)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the candidate solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file")
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

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")
    print(f"Written to: {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
