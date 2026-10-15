#!/usr/bin/env python3
"""
Feasibility checker for the Multi-Depot Periodic Vehicle Routing Problem (MDPVRP).

Checks constraints from Vidal et al. (2011), Annex 7.1:
  Constraint 1 (eq 6):  Each customer assigned exactly one pattern and one depot
  Constraint 2 (eq 7):  Visits match assigned pattern/depot
  Constraint 3 (eq 8):  Vehicle leaves depot at most once per period
  Constraint 4 (eq 9):  Vehicles from depot o cannot leave from other depots
  Constraint 5 (eq 10): Flow conservation
  Constraint 6 (eq 11): Vehicle capacity
  Constraint 7 (eq 12): Route duration
  Constraint 8 (eq 13): Subtour elimination
  Constraint 9 (eq 14): Binary domain for x
  Constraint 10 (eq 15): Binary domain for y
  Constraint 11 (obj): Objective-value consistency (reported objective_value
                       must match recomputed total travel cost summed over
                       all routes' depot-customer-...-depot tours).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignments', 'objective_value', 'routes')
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


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def add_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs):
    violation_amount = abs(lhs - rhs)
    if constraint_idx not in [v["constraint"] for v in magnitudes_list]:
        pass  # we add all instances
    violations_list.append({"constraint": constraint_idx, "message": message})
    normalizer = max(abs(rhs), EPS)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(violation_amount / normalizer),
    })


def check_feasibility(instance, solution):
    # Handle no-solution cases (solver timeout, infeasible, etc.)
    obj = solution.get("objective_value")
    status_str = solution.get("status_str", "")
    has_routes = bool(solution.get("routes"))
    has_assignments = bool(solution.get("assignments"))
    if ((obj is None or obj == float("inf")) and not has_routes and not has_assignments) or \
            status_str in ("NO_SOLUTION_FOUND", "INFEASIBLE", "INF_OR_UNBD"):
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": [
                f"No solution to check (status={status_str!r}, objective_value={obj!r})"
            ],
            "violation_magnitudes": [],
        }

    n = instance["n"]
    d = instance["d"]
    t = instance["t"]
    m = instance["m"]
    Q = instance["vehicle_capacity"]
    T = instance["route_duration_limit"]

    depots = instance["depots"]
    customers = instance["customers"]

    # Build coordinate lookup
    coords = {}
    for dep in depots:
        coords[dep["id"]] = (dep["x"], dep["y"])
    for cust in customers:
        coords[d + cust["id"]] = (cust["x"], cust["y"])

    depot_indices = list(range(d))
    customer_indices = list(range(d, d + n))

    # Customer data indexed by vertex id (d + cust_id)
    demand = {}
    service = {}
    freq = {}
    patterns = {}
    for cust in customers:
        vi = d + cust["id"]
        demand[vi] = cust["demand"]
        service[vi] = cust["service_duration"]
        freq[vi] = cust["frequency"]
        patterns[vi] = cust["pattern_list"]
    for dep_idx in depot_indices:
        demand[dep_idx] = 0
        service[dep_idx] = 0

    # Distance matrix
    all_indices = list(range(d + n))
    dist = {}
    for i in all_indices:
        for j in all_indices:
            xi, yi = coords[i]
            xj, yj = coords[j]
            dist[i, j] = euclidean_distance(xi, yi, xj, yj)

    # Parse solution
    routes = solution.get("routes", [])
    assignments = solution.get("assignments", {})

    violations_list = []
    magnitudes_list = []

    # Domain check: every depot in routes/assignments must be a valid depot
    # id; every customer in routes/assignments must be a valid customer id;
    # every period must be in [1, instance['p']]; every vehicle index must
    # be in [0, m).
    valid_depot_ids = {dep["id"] for dep in depots}
    valid_customer_ids = {cust["id"] for cust in customers}
    P = instance.get("p", 0)
    valid_periods = set(range(1, P + 1)) if P else set()
    domain_violations = []
    for r_idx, route in enumerate(routes):
        if route.get("depot") not in valid_depot_ids:
            domain_violations.append(
                f"routes[{r_idx}] depot {route.get('depot')!r} not a valid depot id"
            )
        if valid_periods and route.get("period") not in valid_periods:
            domain_violations.append(
                f"routes[{r_idx}] period {route.get('period')!r} not in [1, {P}]"
            )
        veh = route.get("vehicle")
        if not isinstance(veh, int) or not (1 <= veh <= m):
            domain_violations.append(
                f"routes[{r_idx}] vehicle {veh!r} not in [1, {m}]"
            )
        for cid in route.get("customers", []):
            if cid not in valid_customer_ids:
                domain_violations.append(
                    f"routes[{r_idx}] customer {cid!r} not a valid customer id"
                )
    for k_str, asgn in assignments.items():
        try:
            cid = int(k_str)
        except (TypeError, ValueError):
            cid = None
        if cid not in valid_customer_ids:
            domain_violations.append(
                f"assignments key {k_str!r} not a valid customer id"
            )
        if isinstance(asgn, dict):
            if asgn.get("depot") not in valid_depot_ids:
                domain_violations.append(
                    f"assignments[{k_str}].depot = {asgn.get('depot')!r} "
                    f"not a valid depot id"
                )
            if valid_periods:
                for per in asgn.get("pattern", []):
                    if per not in valid_periods:
                        domain_violations.append(
                            f"assignments[{k_str}].pattern contains period "
                            f"{per!r} not in [1, {P}]"
                        )

    if domain_violations:
        for msg in domain_violations:
            violations_list.append({"constraint": 0, "message": f"Domain: {msg}"})
            magnitudes_list.append({
                "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
        violated = sorted({v["constraint"] for v in violations_list})
        return {
            "feasible": False,
            "violated_constraints": violated,
            "violations": [v["message"] for v in violations_list],
            "violation_magnitudes": magnitudes_list,
        }

    # =========================================================================
    # Reconstruct assignment variables y_{ipo} from assignments dict
    # assignments: {cust_id_str: {"depot": o, "pattern": [list of periods]}}
    # =========================================================================
    # Map each customer to (pattern_index, depot)
    cust_assignment = {}  # cust_id -> (pattern_index, depot)
    for cust in customers:
        cid = cust["id"]
        key = str(cid)
        if key in assignments:
            asgn = assignments[key]
            assigned_depot = asgn["depot"]
            assigned_pattern = tuple(asgn["pattern"])
            # Find pattern index
            pat_idx = None
            for pi, pat in enumerate(cust["pattern_list"]):
                if tuple(pat) == assigned_pattern:
                    pat_idx = pi
                    break
            if pat_idx is None:
                # Pattern not in feasible list - this is a constraint 1 issue
                # but we still record the assignment for other checks
                pat_idx = -1
            cust_assignment[cid] = (pat_idx, assigned_depot)
        elif int(cid) in assignments:
            # Handle integer keys
            asgn = assignments[int(cid)]
            assigned_depot = asgn["depot"]
            assigned_pattern = tuple(asgn["pattern"])
            pat_idx = None
            for pi, pat in enumerate(cust["pattern_list"]):
                if tuple(pat) == assigned_pattern:
                    pat_idx = pi
                    break
            if pat_idx is None:
                pat_idx = -1
            cust_assignment[cid] = (pat_idx, assigned_depot)

    # =========================================================================
    # Reconstruct route structure
    # Each route: {depot, period, vehicle (optional), customers: [cust_ids]}
    # =========================================================================
    # Build: for each (depot, period, vehicle_idx) -> list of customer ids in visit order
    # If vehicle not given, assign sequentially
    route_structs = []
    vehicle_counter = {}  # (depot, period) -> next vehicle idx
    for r in routes:
        dep = r["depot"]
        per = r["period"]
        veh = r.get("vehicle", None)
        custs = r["customers"]
        if veh is None:
            key = (dep, per)
            veh = vehicle_counter.get(key, 1)
            vehicle_counter[key] = veh + 1
        route_structs.append({
            "depot": dep,
            "period": per,
            "vehicle": veh,
            "customers": custs,
        })

    # =========================================================================
    # Constraint 1 (eq 6): Each customer assigned exactly one pattern+depot
    # sum_{p in L_i} sum_{o in V^DEP} y_{ipo} = 1 for all customers
    # =========================================================================
    for cust in customers:
        cid = cust["id"]
        lhs = 1.0 if cid in cust_assignment else 0.0
        rhs = 1.0
        if cid in cust_assignment:
            pat_idx, dep = cust_assignment[cid]
            if pat_idx == -1:
                # Assigned pattern not in feasible list
                add_violation(violations_list, magnitudes_list, 1,
                              f"Customer {cid}: assigned pattern not in feasible pattern list L_i",
                              0.0, 1.0)
        else:
            add_violation(violations_list, magnitudes_list, 1,
                          f"Customer {cid}: not assigned to any pattern/depot (sum=0, need 1)",
                          lhs, rhs)

    # =========================================================================
    # Constraint 2 (eq 7): Visit consistency
    # sum_j sum_k x_{ijklo} - sum_p a_{pl} y_{ipo} = 0
    # For each customer i, depot o, period l
    # Meaning: customer i is visited from depot o on day l iff assigned pattern includes day l at depot o
    # =========================================================================
    # Count actual visits per (customer, depot, period)
    visit_count = {}  # (cust_id, depot, period) -> count of routes visiting this customer
    for rs in route_structs:
        dep = rs["depot"]
        per = rs["period"]
        for cid in rs["customers"]:
            key = (cid, dep, per)
            visit_count[key] = visit_count.get(key, 0) + 1

    for cust in customers:
        cid = cust["id"]
        for o in depot_indices:
            for l in range(1, t + 1):
                # LHS: number of times customer cid visited from depot o on day l
                actual_visits = visit_count.get((cid, o, l), 0)
                # RHS: expected visits based on assignment
                expected = 0.0
                if cid in cust_assignment:
                    pat_idx, assigned_depot = cust_assignment[cid]
                    if assigned_depot == o and pat_idx >= 0:
                        assigned_pattern = cust["pattern_list"][pat_idx]
                        if l in assigned_pattern:
                            expected = 1.0
                lhs = float(actual_visits) - expected
                rhs = 0.0
                violation_amount = abs(lhs - rhs)
                if violation_amount > TOL:
                    add_violation(violations_list, magnitudes_list, 2,
                                  f"Customer {cid}, depot {o}, period {l}: "
                                  f"actual visits={actual_visits}, expected={expected}",
                                  lhs, rhs)

    # =========================================================================
    # Constraint 3 (eq 8): Vehicle leaves depot at most once per period
    # sum_j x_{ojklo} <= 1 for each depot o, vehicle k, period l
    # In the solution, each route from depot o on period l with vehicle k means
    # the vehicle leaves once. So count routes per (depot, vehicle, period).
    # =========================================================================
    route_count_per_veh = {}  # (depot, vehicle, period) -> count
    for rs in route_structs:
        key = (rs["depot"], rs["vehicle"], rs["period"])
        route_count_per_veh[key] = route_count_per_veh.get(key, 0) + 1

    for key, count in route_count_per_veh.items():
        dep, veh, per = key
        lhs = float(count)
        rhs = 1.0
        if lhs - rhs > TOL:
            add_violation(violations_list, magnitudes_list, 3,
                          f"Depot {dep}, vehicle {veh}, period {per}: "
                          f"vehicle departs {count} times (max 1)",
                          lhs, rhs)

    # Also check that vehicle indices don't exceed m
    for rs in route_structs:
        if rs["vehicle"] > m:
            add_violation(violations_list, magnitudes_list, 3,
                          f"Depot {rs['depot']}, period {rs['period']}: "
                          f"vehicle index {rs['vehicle']} exceeds m={m}",
                          float(rs["vehicle"]), float(m))

    # =========================================================================
    # Constraint 4 (eq 9): Vehicles from depot o cannot operate from other depots
    # In route representation: each route starts/ends at its assigned depot.
    # A customer assigned to depot o should only appear in routes from depot o.
    # =========================================================================
    for rs in route_structs:
        route_depot = rs["depot"]
        for cid in rs["customers"]:
            if cid in cust_assignment:
                _, assigned_depot = cust_assignment[cid]
                if route_depot != assigned_depot:
                    # This customer is visited from a depot that's not its assigned depot
                    add_violation(violations_list, magnitudes_list, 4,
                                  f"Customer {cid} assigned to depot {assigned_depot} "
                                  f"but visited from depot {route_depot} on period {rs['period']}",
                                  1.0, 0.0)

    # =========================================================================
    # Constraint 5 (eq 10): Flow conservation
    # In a valid route representation (depot -> c1 -> c2 -> ... -> depot),
    # flow is automatically conserved. We verify each route forms a valid
    # path starting and ending at the depot (no repeated customers within a route).
    # =========================================================================
    for rs in route_structs:
        custs = rs["customers"]
        if len(custs) != len(set(custs)):
            from collections import Counter
            counts = Counter(custs)
            duplicates = {c: cnt for c, cnt in counts.items() if cnt > 1}
            for c, cnt in duplicates.items():
                add_violation(violations_list, magnitudes_list, 5,
                              f"Depot {rs['depot']}, period {rs['period']}, "
                              f"vehicle {rs['vehicle']}: customer {c} appears {cnt} times "
                              f"in same route (flow conservation violated)",
                              float(cnt), 1.0)

    # =========================================================================
    # Constraint 6 (eq 11): Vehicle capacity
    # sum_i sum_j q_i * x_{ijklo} <= Q
    # For each route, total demand of customers served <= Q
    # =========================================================================
    for rs in route_structs:
        total_demand = sum(demand[d + cid] for cid in rs["customers"])
        lhs = float(total_demand)
        rhs = float(Q)
        if lhs - rhs > TOL:
            add_violation(violations_list, magnitudes_list, 6,
                          f"Depot {rs['depot']}, period {rs['period']}, "
                          f"vehicle {rs['vehicle']}: demand={total_demand} exceeds capacity={Q}",
                          lhs, rhs)

    # =========================================================================
    # Constraint 7 (eq 12): Route duration
    # sum_i sum_j (c_{ij} + tau_i) * x_{ijklo} <= T
    # Route duration = travel time + service time of visited customers
    # =========================================================================
    for rs in route_structs:
        dep = rs["depot"]
        custs = rs["customers"]
        if not custs:
            continue
        # Route: depot -> custs[0] -> custs[1] -> ... -> custs[-1] -> depot
        # Travel time
        travel = dist[dep, d + custs[0]]
        for idx in range(len(custs) - 1):
            travel += dist[d + custs[idx], d + custs[idx + 1]]
        travel += dist[d + custs[-1], dep]
        # Service time
        svc = sum(service[d + cid] for cid in custs)
        duration = travel + svc
        lhs = float(duration)
        rhs = float(T)
        if lhs - rhs > TOL:
            add_violation(violations_list, magnitudes_list, 7,
                          f"Depot {dep}, period {rs['period']}, "
                          f"vehicle {rs['vehicle']}: duration={duration:.2f} exceeds limit={T}",
                          lhs, rhs)

    # =========================================================================
    # Constraint 8 (eq 13): Subtour elimination
    # In the route-list representation, subtours are impossible if each route
    # is a simple path from depot through customers back to depot.
    # We already check for duplicates in constraint 5.
    # Additionally verify no customer appears in multiple routes for same
    # (depot, period) - which would indicate a structural issue.
    # =========================================================================
    cust_in_routes = {}  # (cust_id, depot, period) -> list of vehicle indices
    for rs in route_structs:
        for cid in rs["customers"]:
            key = (cid, rs["depot"], rs["period"])
            cust_in_routes.setdefault(key, []).append(rs["vehicle"])
    for (cid, dep, per), vehs in cust_in_routes.items():
        if len(vehs) > 1:
            lhs = float(len(vehs))
            rhs = 1.0
            add_violation(violations_list, magnitudes_list, 8,
                          f"Customer {cid} visited by {len(vehs)} vehicles "
                          f"from depot {dep} on period {per} (subtour/duplication issue)",
                          lhs, rhs)

    # =========================================================================
    # Constraint 11 (obj): Objective-value consistency
    # Recompute total travel cost from the routes (sum of euclidean arc lengths
    # along each depot->c1->...->ck->depot tour, over all routes, depots,
    # periods, vehicles) and reject when the reported objective_value differs
    # by more than 0.1% relative (with a 1e-3 absolute floor).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            true_obj = 0.0
            for rs in route_structs:
                dep = rs["depot"]
                custs = rs["customers"]
                if not custs:
                    continue
                true_obj += dist[dep, d + custs[0]]
                for idx in range(len(custs) - 1):
                    true_obj += dist[d + custs[idx], d + custs[idx + 1]]
                true_obj += dist[d + custs[-1], dep]
            abs_diff = abs(reported - true_obj)
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                add_violation(
                    violations_list, magnitudes_list, 11,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed total travel cost="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol:.6g})",
                    reported, true_obj,
                )

    # =========================================================================
    # Aggregate results
    # =========================================================================
    violated_constraint_indices = sorted(set(v["constraint"] for v in violations_list))
    # Aggregate messages per constraint
    messages = []
    for ci in violated_constraint_indices:
        msgs = [v["message"] for v in violations_list if v["constraint"] == ci]
        if len(msgs) <= 3:
            messages.extend(msgs)
        else:
            messages.extend(msgs[:3])
            messages.append(f"  ... and {len(msgs) - 3} more violation(s) for constraint {ci}")

    feasible = len(violated_constraint_indices) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraint_indices,
        "violations": messages,
        "violation_magnitudes": magnitudes_list if not feasible else [],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="MDPVRP Feasibility Checker")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write feasibility result JSON file")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    with open(args.solution_path, 'r') as f:
        raw = f.read()
        # Handle non-standard JSON values (Infinity, NaN) that some solvers emit
        raw = raw.replace(': Infinity', ': 1e9999')
        raw = raw.replace(':Infinity', ':1e9999')
        raw = raw.replace(': -Infinity', ': -1e9999')
        raw = raw.replace(':-Infinity', ':-1e9999')
        raw = raw.replace(': NaN', ': null')
        raw = raw.replace(':NaN', ':null')
        solution = json.loads(raw)
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

    if result["feasible"] is None:
        status = "NO_SOLUTION"
    elif result["feasible"]:
        status = "FEASIBLE"
    else:
        status = "INFEASIBLE"
    print(f"Result: {status}")
    if result["feasible"] is None:
        print(f"  {result['violations'][0] if result['violations'] else 'No solution to check'}")
    elif not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")
    print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
