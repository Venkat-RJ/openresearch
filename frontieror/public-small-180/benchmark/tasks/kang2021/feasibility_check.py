"""
Feasibility checker for the Heterogeneous Drone-Truck Routing Problem (HDTRP).

Checks all hard constraints from the mathematical formulation (P) in:
  Kang & Lee (2021), Transportation Science, 55(5):1088-1112.

Constraints numbered (2)-(12) as listed in math_model.txt, plus:

  Constraint (13): Objective-value consistency (Tier C defense).
    Recomputes the objective
        obj = sum_{(i,j) in A} t^v_{ij} x_{ij} + sum_{i in N_s} w_i
    directly from the solution variables (truck_route -> x, waiting_times -> w)
    and rejects the solution as infeasible when the self-reported
    `objective_value` disagrees with the recomputed value. This blocks
    score-gaming exploits where a candidate returns a fabricated objective
    (e.g. 0.0 or sys.float_info.max) on a route whose constraints all pass.
"""

import argparse
import json
import sys

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('drone_assignments', 'objective_value', 'truck_route', 'waiting_times')
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


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # ----------------------------------------------------------------
    # Extract instance data
    # ----------------------------------------------------------------
    num_nodes = instance["num_nodes"]
    depot = instance["depot_index"]
    customers = instance["customer_indices"]
    N = list(customers)
    n_cust = len(N)

    s = depot  # source depot
    t = num_nodes  # virtual sink (duplicated depot)

    N_s = N + [s]
    N_t = N + [t]
    N_st = N + [s, t]

    A = set((i, j) for i in N_s for j in N_t if i != j)

    num_drones = instance["parameters"]["num_drones"]
    L = list(range(num_drones))
    B_cap = instance["parameters"]["drone_battery_capacities"]

    tv = instance["truck_travel_times"]
    b_raw = instance["drone_battery_consumption"]
    tau_raw = instance["drone_delivery_time"]

    def get_tv(i, j):
        ri = i if i != t else depot
        rj = j if j != t else depot
        return tv[ri][rj]

    def get_b(l, i, j):
        ri = i if i != t else depot
        rj = j if j != t else depot
        return b_raw[l][ri][rj]

    def get_tau(l, i, j):
        ri = i if i != t else depot
        rj = j if j != t else depot
        return tau_raw[l][ri][rj]

    # ----------------------------------------------------------------
    # Parse solution: build x_{ij} and h^l_{ij}
    # ----------------------------------------------------------------
    x = {}  # (i, j) -> 1
    for arc in solution.get("truck_route", []):
        i, j = arc[0], arc[1]
        x[(i, j)] = 1

    h = {}  # (l, i, j) -> 1
    for da in solution.get("drone_assignments", []):
        l = da["drone"]
        i = da["dispatch_node"]
        j = da["delivery_node"]
        h[(l, i, j)] = 1

    # Parse waiting times.
    # Two formats observed:
    #   {"total": X} — only aggregate total provided
    #   {"1": val, "9": val, ...} — per-node waiting times
    raw_wt = solution.get("waiting_times", {})
    w = {}  # node -> waiting time
    if "total" in raw_wt and len(raw_wt) == 1:
        # Only total is given; we cannot reconstruct per-node values.
        # We'll set w to None and handle constraint (9) with total only.
        w_total_only = raw_wt["total"]
        w = None
    elif "total" in raw_wt and len(raw_wt) > 1:
        # Both total and per-node are present
        w = {}
        for k, v in raw_wt.items():
            if k != "total":
                w[int(k)] = v
        w_total_only = None
    else:
        # All keys are per-node
        w = {}
        for k, v in raw_wt.items():
            w[int(k)] = v
        w_total_only = None

    # Helper to record a violation
    def record_violation(constraint_idx, msg, lhs_val, rhs_val, op):
        """Record a constraint violation.
        op: 'le' for <=, 'ge' for >=, 'eq' for =
        """
        if op == "le":
            violation_amount = lhs_val - rhs_val
        elif op == "ge":
            violation_amount = rhs_val - lhs_val
        elif op == "eq":
            violation_amount = abs(lhs_val - rhs_val)
        else:
            violation_amount = abs(lhs_val - rhs_val)

        if violation_amount > tol:
            violated_constraint_set.add(constraint_idx)
            violations.append(msg)
            normalizer = max(abs(rhs_val), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # ================================================================
    # Constraint (2): sum_{j in N} x_{sj} = 1
    # Truck departs depot exactly once.
    # ================================================================
    lhs_c2 = sum(x.get((s, j), 0) for j in N)
    rhs_c2 = 1
    if abs(lhs_c2 - rhs_c2) > tol:
        record_violation(2,
            f"Constraint (2): sum x_{{s,j}} for j in N = {lhs_c2}, expected 1",
            lhs_c2, rhs_c2, "eq")

    # ================================================================
    # Constraint (3): sum_{i in N} x_{it} = 1
    # Truck returns to depot exactly once.
    # ================================================================
    lhs_c3 = sum(x.get((i, t), 0) for i in N)
    rhs_c3 = 1
    if abs(lhs_c3 - rhs_c3) > tol:
        record_violation(3,
            f"Constraint (3): sum x_{{i,t}} for i in N = {lhs_c3}, expected 1",
            lhs_c3, rhs_c3, "eq")

    # ================================================================
    # Constraint (4): Flow balance at each customer node
    # sum_{j in N_t: j!=i} x_{ij} = sum_{j in N_s: j!=i} x_{ji},  for all i in N
    # ================================================================
    for i in N:
        out_flow = sum(x.get((i, j), 0) for j in N_t if j != i)
        in_flow = sum(x.get((j, i), 0) for j in N_s if j != i)
        if abs(out_flow - in_flow) > tol:
            record_violation(4,
                f"Constraint (4): Flow imbalance at node {i}: out={out_flow}, in={in_flow}",
                out_flow, in_flow, "eq")

    # ================================================================
    # Constraint (5): Subtour elimination (MTZ)
    # v_i - v_j <= M(1 - x_{ij}) - 1,  for all (i,j) in A
    #
    # For arcs with x_{ij}=1: v_i - v_j <= -1, i.e. v_j >= v_i + 1.
    # This means arcs with x=1 must form a DAG (no directed cycles).
    # We verify this by checking for directed cycles among active arcs.
    # ================================================================
    M_val = n_cust + 1
    active_arcs = [(i, j) for (i, j), val in x.items() if val > 0.5]

    # Build adjacency for active arcs (excluding self-loops)
    truck_adj = {}
    all_active_nodes = set()
    for (i, j) in active_arcs:
        truck_adj.setdefault(i, []).append(j)
        all_active_nodes.add(i)
        all_active_nodes.add(j)

    # Detect directed cycles using DFS coloring
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in all_active_nodes}
    has_cycle = False
    cycle_nodes = []

    def dfs_cycle(u, path):
        nonlocal has_cycle
        if has_cycle:
            return
        color[u] = GRAY
        path.append(u)
        for v in truck_adj.get(u, []):
            if v not in color:
                color[v] = WHITE
            if color[v] == GRAY:
                # Found a cycle
                has_cycle = True
                # Extract cycle
                cycle_start = path.index(v)
                cycle_nodes.extend(path[cycle_start:])
                return
            if color[v] == WHITE:
                dfs_cycle(v, path)
                if has_cycle:
                    return
        path.pop()
        color[u] = BLACK

    for node in all_active_nodes:
        if color.get(node, WHITE) == WHITE:
            dfs_cycle(node, [])
            if has_cycle:
                break

    if has_cycle:
        record_violation(5,
            f"Constraint (5): Subtour detected — directed cycle among active truck arcs "
            f"involving nodes {cycle_nodes}",
            1.0, 0.0, "le")

    # ================================================================
    # Constraint (6): Each customer served exactly once
    # sum_{i in N_s: i!=j} x_{ij} + sum_{i in N_s: i!=j} sum_{l in L} h^l_{ij} = 1
    # for all j in N
    # ================================================================
    for j in N:
        truck_serve = sum(x.get((i, j), 0) for i in N_s if i != j)
        drone_serve = sum(h.get((l, i, j), 0) for l in L for i in N_s if i != j)
        lhs_c6 = truck_serve + drone_serve
        rhs_c6 = 1
        if abs(lhs_c6 - rhs_c6) > tol:
            if lhs_c6 < rhs_c6 - tol:
                record_violation(6,
                    f"Constraint (6): Customer {j} not served (truck={truck_serve}, drone={drone_serve})",
                    lhs_c6, rhs_c6, "eq")
            else:
                record_violation(6,
                    f"Constraint (6): Customer {j} served {lhs_c6} times (truck={truck_serve}, drone={drone_serve})",
                    lhs_c6, rhs_c6, "eq")

    # ================================================================
    # Constraint (7): Drones dispatched only from truck-visited nodes
    # M * sum_{j in N_t: j!=i} x_{ij} >= sum_{j in N: j!=i} sum_{l in L} h^l_{ij}
    # for all i in N_s
    # ================================================================
    M_dispatch = n_cust
    for i in N_s:
        lhs_c7_raw = M_dispatch * sum(x.get((i, j), 0) for j in N_t if j != i)
        rhs_c7_raw = sum(h.get((l, i, j), 0) for l in L for j in N if j != i)
        # Constraint: lhs >= rhs, i.e., check rhs > lhs
        if rhs_c7_raw > lhs_c7_raw + tol:
            record_violation(7,
                f"Constraint (7): Drones dispatched from node {i} which truck does not visit "
                f"(M*truck_out={lhs_c7_raw}, drone_dispatches={rhs_c7_raw})",
                lhs_c7_raw, rhs_c7_raw, "ge")

    # ================================================================
    # Constraint (8): Drone battery capacity
    # sum_{i in N_s} sum_{j in N: j!=i} b^l_{ij} h^l_{ij} <= B^l
    # for all l in L
    # ================================================================
    for l in L:
        total_battery = sum(
            get_b(l, i, j) * h.get((l, i, j), 0)
            for i in N_s for j in N if j != i
        )
        cap = B_cap[l]
        if total_battery > cap + tol:
            record_violation(8,
                f"Constraint (8): Drone {l} battery usage {total_battery} exceeds capacity {cap}",
                total_battery, cap, "le")

    # ================================================================
    # Constraint (9): Truck waiting time at each node
    # w_i >= sum_{j in N: j!=i} tau^l_{ij} h^l_{ij}
    # for all i in N_s, l in L
    #
    # If per-node waiting times are available, check directly.
    # If only total is available, we compute the minimum required total
    # waiting time (max over drones at each dispatch node) and check that.
    # ================================================================
    # Compute minimum required waiting time at each dispatch node
    min_required_w = {}
    for i in N_s:
        max_drone_time = 0.0
        for l in L:
            drone_time_at_i_l = sum(
                get_tau(l, i, j) * h.get((l, i, j), 0)
                for j in N if j != i
            )
            if drone_time_at_i_l > max_drone_time:
                max_drone_time = drone_time_at_i_l
        min_required_w[i] = max_drone_time

    if w is not None:
        # Per-node waiting times available — check each constraint instance
        for i in N_s:
            w_i = w.get(i, 0.0)
            for l in L:
                drone_time = sum(
                    get_tau(l, i, j) * h.get((l, i, j), 0)
                    for j in N if j != i
                )
                if drone_time > w_i + tol:
                    record_violation(9,
                        f"Constraint (9): Waiting time at node {i} = {w_i} "
                        f"< drone {l} delivery time {drone_time}",
                        w_i, drone_time, "ge")
    else:
        # Only total waiting time available
        # The total waiting time must be >= sum of minimum required per-node
        # waiting times (since w_i >= max_l drone_time(i, l) for each i,
        # we have sum w_i >= sum max_l drone_time(i, l))
        total_w = w_total_only if w_total_only is not None else 0.0
        min_required_total = sum(min_required_w[i] for i in N_s)
        if min_required_total > total_w + tol:
            record_violation(9,
                f"Constraint (9): Total waiting time {total_w} "
                f"< minimum required total waiting {min_required_total}",
                total_w, min_required_total, "ge")

    # ================================================================
    # Constraint (10): v_s = 0
    # Already enforced in our subtour check above (v_order[s] = 0).
    # No separate check needed as the visiting order variable v_s is
    # not in the solution — we always set it to 0.
    # ================================================================
    # This is automatically satisfied by construction; no violation possible.

    # ================================================================
    # Constraint (11): x_{ij} in {0, 1}
    # Check that all x values are binary.
    # ================================================================
    for (i, j), val in x.items():
        if (i, j) not in A:
            record_violation(11,
                f"Constraint (11): Truck arc ({i},{j}) not in arc set A",
                val, 0.0, "eq")
        if abs(val - round(val)) > tol:
            record_violation(11,
                f"Constraint (11): x_({i},{j}) = {val} is not binary",
                abs(val - round(val)), 0.0, "le")

    # ================================================================
    # Constraint (12): h^l_{ij} in {0, 1}
    # Check that all h values are binary and within valid index ranges.
    # ================================================================
    for (l, i, j), val in h.items():
        if l not in L:
            record_violation(12,
                f"Constraint (12): Drone index {l} not in L={L}",
                val, 0.0, "eq")
        if i not in N_s:
            record_violation(12,
                f"Constraint (12): Dispatch node {i} not in N_s for h^{l}_({i},{j})",
                val, 0.0, "eq")
        if j not in N:
            record_violation(12,
                f"Constraint (12): Delivery node {j} not in N for h^{l}_({i},{j})",
                val, 0.0, "eq")
        if abs(val - round(val)) > tol:
            record_violation(12,
                f"Constraint (12): h^{l}_({i},{j}) = {val} is not binary",
                abs(val - round(val)), 0.0, "le")

    # ================================================================
    # Constraint (13): Objective-value consistency  (Tier C defense)
    # Recompute the objective directly from the solution variables:
    #   obj = sum_{(i,j) in A} t^v_{ij} x_{ij} + sum_{i in N_s} w_i
    # The truck route (x) and the waiting times (w) are BOTH present in
    # the solution, so a full exact recompute is possible. Reject the
    # solution when the self-reported `objective_value` disagrees with the
    # recomputed value beyond a small numerical tolerance. This catches
    # score-gaming exploits that fabricate `objective_value` (e.g. 0.0 or
    # sys.float_info.max) on an otherwise constraint-feasible route.
    # ================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Truck travel time component: sum of t^v over traversed arcs.
            # Guard each lookup so a malformed arc (already flagged by
            # constraint (11)) cannot crash the checker.
            truck_travel = 0.0
            for (i, j), val in x.items():
                try:
                    truck_travel += get_tv(i, j) * val
                except (IndexError, KeyError, TypeError):
                    # Out-of-range / malformed arc — constraint (11)
                    # already records it; skip it for the obj recompute.
                    pass

            # Total waiting-time component: sum_{i in N_s} w_i.
            # waiting_times may be per-node, {"total": X}, or both.
            raw_wt2 = solution.get("waiting_times", {}) or {}
            per_node_items = [
                (k, v) for k, v in raw_wt2.items() if k != "total"
            ]
            if per_node_items:
                total_waiting = sum(float(v) for _, v in per_node_items)
            elif "total" in raw_wt2:
                total_waiting = float(raw_wt2["total"])
            else:
                total_waiting = 0.0

            true_obj = float(truck_travel) + float(total_waiting)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with a 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(13,
                    f"Constraint (13): Objective consistency violated — "
                    f"reported objective_value={reported} differs from "
                    f"recomputed sum_(i,j) tv_ij*x_ij + sum_i w_i = {true_obj} "
                    f"(truck_travel={truck_travel}, total_waiting={total_waiting}, "
                    f"|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    reported, true_obj, "eq")

    # ================================================================
    # Build result
    # ================================================================
    feasible = len(violated_constraint_set) == 0
    violated_constraints = sorted(violated_constraint_set)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for HDTRP solutions."
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance.",
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution.",
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result.",
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
        print(f"FEASIBLE — no constraint violations found.")
    else:
        print(f"INFEASIBLE — violated constraints: {result['violated_constraints']}")
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
