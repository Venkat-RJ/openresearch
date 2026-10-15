"""
Feasibility checker for PDPTWL (Pickup and Delivery Problem with Time Windows
and Last-in-First-Out Loading).

Based on: Cherkesly, Desaulniers, and Laporte (2015), Transportation Science.

Checks constraints (2)-(13) from the mathematical formulation, plus
constraint (14) -- objective-consistency check (Tier C defense against
LLM score-gaming exploits that fabricate objective_value).
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



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_instance(data):
    """Preprocess instance data into convenient structures."""
    n = data["n_requests"]
    n_nodes = data["n_nodes"]
    Q = data["vehicle_capacity"]

    nodes = {}
    for nd in data["nodes"]:
        nodes[nd["node_id"]] = nd

    P = set(range(1, n + 1))
    D = set(range(n + 1, 2 * n + 1))
    depot_o = 0
    depot_d = 2 * n + 1

    arc_cost = {}
    arc_time = {}
    arc_set = set()
    for arc in data["arcs"]:
        i, j = arc["from"], arc["to"]
        arc_set.add((i, j))
        arc_cost[(i, j)] = arc["cost"]
        arc_time[(i, j)] = arc["travel_time"]

    return {
        "n": n,
        "n_nodes": n_nodes,
        "Q": Q,
        "nodes": nodes,
        "P": P,
        "D": D,
        "depot_o": depot_o,
        "depot_d": depot_d,
        "arc_set": arc_set,
        "arc_cost": arc_cost,
        "arc_time": arc_time,
        "vehicle_fixed_cost": data.get("vehicle_fixed_cost", 0),
    }


def check_feasibility(inst, solution):
    """
    Check all hard constraints (2)-(13) from the formulation plus
    constraint (14) objective-consistency check.
    Returns (feasible, violated_constraints, violations, violation_magnitudes).
    """
    tol = 1e-5
    eps = 1e-5

    n = inst["n"]
    Q = inst["Q"]
    P = inst["P"]
    D = inst["D"]
    depot_o = inst["depot_o"]
    depot_d = inst["depot_d"]
    nodes = inst["nodes"]
    arc_set = inst["arc_set"]
    arc_cost = inst["arc_cost"]
    arc_time = inst["arc_time"]

    routes = solution["routes"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs, rhs, op):
        """Record a constraint violation with normalized magnitude."""
        if op == "eq":
            violation_amount = abs(lhs - rhs)
        elif op in ("le", "lt"):
            violation_amount = lhs - rhs
        elif op in ("ge", "gt"):
            violation_amount = rhs - lhs
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            violated_constraints.add(constraint_idx)
            violations.append(msg)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # =========================================================================
    # Build route structures: for each vehicle k, extract x^k, and later T^k, Q^k
    # =========================================================================
    # Each route is a sequence [depot_o, ..., depot_d].
    # x^k_{ij} = 1 iff arc (i,j) is consecutive in route k.

    vehicle_routes = []  # list of dicts with route info
    for route_data in routes:
        seq = route_data["route_sequence"]
        arcs_used = set()
        for idx in range(len(seq) - 1):
            arcs_used.add((seq[idx], seq[idx + 1]))
        vehicle_routes.append({
            "seq": seq,
            "arcs": arcs_used,
            "vehicle_id": route_data["vehicle_id"],
        })

    num_vehicles = len(vehicle_routes)

    # =========================================================================
    # Constraint (2): Each pickup visited exactly once
    # sum_{k in K} sum_{j in N} x^k_{ij} = 1, for all i in P
    # =========================================================================
    pickup_visit_count = {i: 0 for i in P}
    for vr in vehicle_routes:
        seq = vr["seq"]
        for node in seq:
            if node in P:
                pickup_visit_count[node] += 1

    for i in sorted(P):
        count = pickup_visit_count[i]
        if abs(count - 1) > tol:
            record_violation(
                2,
                f"Constraint (2): Pickup node {i} visited {count} time(s) (expected exactly 1)",
                float(count), 1.0, "eq"
            )

    # =========================================================================
    # Constraint (3): Pairing - pickup i and delivery n+i on same vehicle
    # sum_{j} x^k_{ij} - sum_{j} x^k_{n+i,j} = 0, for all i in P, k in K
    # =========================================================================
    for vr in vehicle_routes:
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        nodes_in_route = set(seq)
        for i in sorted(P):
            pickup_in = 1 if i in nodes_in_route else 0
            delivery_in = 1 if (n + i) in nodes_in_route else 0
            diff = pickup_in - delivery_in
            if abs(diff) > tol:
                record_violation(
                    3,
                    f"Constraint (3): Vehicle {vid} has pickup {i} (present={pickup_in}) "
                    f"but delivery {n + i} (present={delivery_in})",
                    float(pickup_in), float(delivery_in), "eq"
                )

    # =========================================================================
    # Constraint (4): Each vehicle leaves the origin depot
    # sum_{j in N} x^k_{0,j} = 1, for all k in K
    # =========================================================================
    for vr in vehicle_routes:
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        if len(seq) < 2 or seq[0] != depot_o:
            count_from_depot = 0
        else:
            # Count arcs from depot_o used by this vehicle
            count_from_depot = sum(1 for (a, b) in vr["arcs"] if a == depot_o)
        if abs(count_from_depot - 1) > tol:
            record_violation(
                4,
                f"Constraint (4): Vehicle {vid} has {count_from_depot} arc(s) from "
                f"origin depot (expected 1)",
                float(count_from_depot), 1.0, "eq"
            )

    # =========================================================================
    # Constraint (5): Flow conservation at P ∪ D nodes
    # sum_{j} x^k_{ji} - sum_{j} x^k_{ij} = 0, for all i in P ∪ D, k in K
    # =========================================================================
    for vr in vehicle_routes:
        vid = vr["vehicle_id"]
        seq = vr["seq"]
        # Count in-degree and out-degree for each node in P ∪ D
        in_deg = {}
        out_deg = {}
        for (a, b) in vr["arcs"]:
            out_deg[a] = out_deg.get(a, 0) + 1
            in_deg[b] = in_deg.get(b, 0) + 1

        all_nodes_in_route = set(seq)
        for node in sorted(all_nodes_in_route):
            if node in P or node in D:
                ind = in_deg.get(node, 0)
                outd = out_deg.get(node, 0)
                diff = ind - outd
                if abs(diff) > tol:
                    record_violation(
                        5,
                        f"Constraint (5): Vehicle {vid}, node {node}: "
                        f"in-degree={ind}, out-degree={outd} (should be equal)",
                        float(ind), float(outd), "eq"
                    )

    # =========================================================================
    # Constraint (6): Each vehicle enters destination depot
    # sum_{i in N} x^k_{i,2n+1} = 1, for all k in K
    # =========================================================================
    for vr in vehicle_routes:
        vid = vr["vehicle_id"]
        count_to_depot = sum(1 for (a, b) in vr["arcs"] if b == depot_d)
        if abs(count_to_depot - 1) > tol:
            record_violation(
                6,
                f"Constraint (6): Vehicle {vid} has {count_to_depot} arc(s) to "
                f"destination depot (expected 1)",
                float(count_to_depot), 1.0, "eq"
            )

    # =========================================================================
    # Constraint (7): LIFO loading constraints
    # The LIFO policy requires that deliveries happen in reverse order of
    # pickups. If pickup i is picked up after pickup j on the same route,
    # then i must be delivered before j.
    # =========================================================================
    for vr in vehicle_routes:
        vid = vr["vehicle_id"]
        seq = vr["seq"]
        # Track pickup order using a stack
        stack = []
        for node in seq:
            if node in P:
                stack.append(node)
            elif node in D:
                req = node - n
                if req in stack:
                    if stack[-1] != req:
                        # LIFO violation: req is not on top of stack
                        # Find its position
                        pos = stack.index(req)
                        top = stack[-1]
                        # The LHS is the number of arcs used in the infeasible
                        # sub-path; RHS is |N(R)| - 2. We approximate the
                        # violation: LHS = 1 (violation exists), RHS = 0 (should
                        # not happen).
                        record_violation(
                            7,
                            f"Constraint (7): Vehicle {vid}, LIFO violation: "
                            f"request {req} delivered but request {top} "
                            f"(picked up later) is still on board",
                            1.0, 0.0, "le"
                        )
                    # Remove from stack wherever it is
                    stack.remove(req)

    # =========================================================================
    # Compute T^k_i and Q^k_i for each vehicle route
    # We simulate the route to get arrival times and loads.
    # =========================================================================
    route_times = []  # list of dicts: node_id -> arrival time
    route_loads = []  # list of dicts: node_id -> load upon leaving

    for vr in vehicle_routes:
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        times = {}
        loads = {}

        current_time = nodes[depot_o]["tw_early"]
        current_load = 0
        times[seq[0]] = current_time
        loads[seq[0]] = current_load  # load upon leaving depot is 0

        for idx in range(1, len(seq)):
            prev = seq[idx - 1]
            curr = seq[idx]
            s_prev = nodes[prev]["service_time"]
            t_arc = arc_time.get((prev, curr), None)

            if t_arc is None:
                # Compute Euclidean distance as fallback
                xi, yi = nodes[prev]["x"], nodes[prev]["y"]
                xj, yj = nodes[curr]["x"], nodes[curr]["y"]
                t_arc = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

            arrival = current_time + s_prev + t_arc
            # Respect earliest time window
            service_start = max(arrival, nodes[curr]["tw_early"])
            times[curr] = service_start

            current_load += nodes[curr]["load"]
            loads[curr] = current_load
            current_time = service_start

        route_times.append(times)
        route_loads.append(loads)

    # =========================================================================
    # Constraint (8): Load propagation
    # Q^k_j >= Q^k_i + q_j (when arc (i,j) is used by vehicle k)
    # Linearized: Q^k_j >= Q^k_i + q_j - M*(1 - x^k_{ij})
    # When x^k_{ij} = 1: Q^k_j >= Q^k_i + q_j
    # =========================================================================
    for k, vr in enumerate(vehicle_routes):
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        for idx in range(len(seq) - 1):
            i_node = seq[idx]
            j_node = seq[idx + 1]
            Qk_i = route_loads[k].get(i_node, 0)
            Qk_j = route_loads[k].get(j_node, 0)
            q_j = nodes[j_node]["load"]
            rhs_val = Qk_i + q_j
            # Qk_j >= rhs_val
            if rhs_val - Qk_j > tol:
                record_violation(
                    8,
                    f"Constraint (8): Vehicle {vid}, arc ({i_node},{j_node}): "
                    f"Q[{j_node}]={Qk_j} < Q[{i_node}]+q[{j_node}]={rhs_val}",
                    float(Qk_j), float(rhs_val), "ge"
                )

    # =========================================================================
    # Constraint (9): Load bounds
    # max(0, q_i) <= Q^k_i <= min(Q, Q + q_i), for all i in N, k in K
    # =========================================================================
    for k, vr in enumerate(vehicle_routes):
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        for node in seq:
            Qk_i = route_loads[k].get(node, 0)
            q_i = nodes[node]["load"]
            lb = max(0, q_i)
            ub = min(Q, Q + q_i)
            if Qk_i < lb - tol:
                record_violation(
                    9,
                    f"Constraint (9): Vehicle {vid}, node {node}: "
                    f"load={Qk_i} < lower bound={lb}",
                    float(Qk_i), float(lb), "ge"
                )
            if Qk_i > ub + tol:
                record_violation(
                    9,
                    f"Constraint (9): Vehicle {vid}, node {node}: "
                    f"load={Qk_i} > upper bound={ub}",
                    float(Qk_i), float(ub), "le"
                )

    # =========================================================================
    # Constraint (10): Time propagation
    # T^k_j >= T^k_i + s_i + t_{ij} (when arc (i,j) is used by vehicle k)
    # Linearized: T^k_j >= T^k_i + s_i + t_{ij} - M*(1 - x^k_{ij})
    # When x^k_{ij} = 1: T^k_j >= T^k_i + s_i + t_{ij}
    # =========================================================================
    for k, vr in enumerate(vehicle_routes):
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        for idx in range(len(seq) - 1):
            i_node = seq[idx]
            j_node = seq[idx + 1]
            Tk_i = route_times[k].get(i_node, 0)
            Tk_j = route_times[k].get(j_node, 0)
            s_i = nodes[i_node]["service_time"]
            t_ij = arc_time.get((i_node, j_node), None)
            if t_ij is None:
                xi, yi = nodes[i_node]["x"], nodes[i_node]["y"]
                xj, yj = nodes[j_node]["x"], nodes[j_node]["y"]
                t_ij = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            rhs_val = Tk_i + s_i + t_ij
            if rhs_val - Tk_j > tol:
                record_violation(
                    10,
                    f"Constraint (10): Vehicle {vid}, arc ({i_node},{j_node}): "
                    f"T[{j_node}]={Tk_j:.2f} < T[{i_node}]+s[{i_node}]+t({i_node},{j_node})={rhs_val:.2f}",
                    float(Tk_j), float(rhs_val), "ge"
                )

    # =========================================================================
    # Constraint (11): Time window bounds
    # w_i <= T^k_i <= w_bar_i, for all i in N, k in K
    # =========================================================================
    for k, vr in enumerate(vehicle_routes):
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        for node in seq:
            Tk_i = route_times[k].get(node, 0)
            w_early = nodes[node]["tw_early"]
            w_late = nodes[node]["tw_late"]
            if Tk_i < w_early - tol:
                record_violation(
                    11,
                    f"Constraint (11): Vehicle {vid}, node {node}: "
                    f"T={Tk_i:.2f} < earliest={w_early}",
                    float(Tk_i), float(w_early), "ge"
                )
            if Tk_i > w_late + tol:
                record_violation(
                    11,
                    f"Constraint (11): Vehicle {vid}, node {node}: "
                    f"T={Tk_i:.2f} > latest={w_late}",
                    float(Tk_i), float(w_late), "le"
                )

    # =========================================================================
    # Constraint (12): Pickup before delivery with minimum travel time
    # T^k_i + s_i + t_{i,n+i} <= T^k_{n+i}, for all i in P, k in K
    # (Only applies on vehicles that serve request i)
    # =========================================================================
    for k, vr in enumerate(vehicle_routes):
        seq = vr["seq"]
        vid = vr["vehicle_id"]
        nodes_in_route = set(seq)
        for i in sorted(P):
            if i in nodes_in_route and (n + i) in nodes_in_route:
                Tk_i = route_times[k].get(i, 0)
                Tk_ni = route_times[k].get(n + i, 0)
                s_i = nodes[i]["service_time"]
                t_i_ni = arc_time.get((i, n + i), None)
                if t_i_ni is None:
                    xi, yi = nodes[i]["x"], nodes[i]["y"]
                    xj, yj = nodes[n + i]["x"], nodes[n + i]["y"]
                    t_i_ni = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                lhs_val = Tk_i + s_i + t_i_ni
                rhs_val = Tk_ni
                if lhs_val - rhs_val > tol:
                    record_violation(
                        12,
                        f"Constraint (12): Vehicle {vid}, request {i}: "
                        f"T[{i}]+s[{i}]+t({i},{n + i})={lhs_val:.2f} > T[{n + i}]={rhs_val:.2f}",
                        float(lhs_val), float(rhs_val), "le"
                    )

    # =========================================================================
    # Constraint (13): Integrality / arc validity
    # x^k_{ij} in {0, 1}, for all (i,j) in A, k in K
    # Since the solution provides route sequences, x values are implicitly
    # binary. We check that all arcs used actually belong to A.
    # =========================================================================
    for vr in vehicle_routes:
        vid = vr["vehicle_id"]
        seq = vr["seq"]
        for idx in range(len(seq) - 1):
            i_node = seq[idx]
            j_node = seq[idx + 1]
            if (i_node, j_node) not in arc_set:
                record_violation(
                    13,
                    f"Constraint (13): Vehicle {vid} uses arc ({i_node},{j_node}) "
                    f"which is not in the arc set A",
                    1.0, 0.0, "le"
                )

    # =========================================================================
    # Constraint (14): Objective-consistency check (Tier C defense).
    # The objective (1) is min sum_{k} sum_{(i,j) in A} c_{ij} * x^k_{ij}.
    # Every variable that determines this sum (the per-vehicle route
    # sequences) is present in the solution, so we can do a FULL recompute.
    # For arcs not in arc_cost we fall back to Euclidean distance with the
    # vehicle-fixed-cost surcharge on depot-out arcs, mirroring how the
    # instance generator builds c_{ij}; this only kicks in if a route uses
    # an arc not in A, which constraint (13) would already flag.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            vehicle_fixed_cost = float(inst.get("vehicle_fixed_cost", 0) or 0)
            # vehicle_fixed_cost lives on the raw instance dict; pull it from
            # the build_instance input via a stash on inst if available,
            # otherwise default to 0 (the cost is already embedded in
            # arc_cost for arcs in A, so the fallback only matters for
            # missing arcs).
            true_obj = 0.0
            for vr in vehicle_routes:
                seq = vr["seq"]
                for idx in range(len(seq) - 1):
                    a, b = seq[idx], seq[idx + 1]
                    c = arc_cost.get((a, b))
                    if c is None:
                        xi, yi = nodes[a]["x"], nodes[a]["y"]
                        xj, yj = nodes[b]["x"], nodes[b]["y"]
                        c = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                        if a == depot_o:
                            c += vehicle_fixed_cost
                    true_obj += float(c)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(
                    14,
                    f"Constraint (14): Objective consistency violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_k sum_(i,j) c[i,j]*x^k[i,j]={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})",
                    float(reported), float(true_obj), "eq"
                )

    # =========================================================================
    # Compile results
    # =========================================================================
    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    return feasible, sorted_violated, violations, violation_magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for PDPTWL solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the candidate solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file")
    args = parser.parse_args()

    data = load_json(args.instance_path)
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
    inst = build_instance(data)

    feasible, violated_constraints, violations, violation_magnitudes = \
        check_feasibility(inst, solution)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {feasible}")
    if not feasible:
        print(f"Violated constraints: {violated_constraints}")
        for v in violations:
            print(f"  - {v}")
    print(f"Result written to: {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
