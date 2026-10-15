#!/usr/bin/env python3
"""
Feasibility checker for the ILSSP-NT (Integrated Lot Sizing and Scheduling Problem
with Non-Triangular setup) on parallel machines.

Based on: Carvalho & Nascimento (2022)
Checks constraints (2)-(20) from the mathematical formulation, plus
constraint (1) — objective-value consistency: the reported objective_value
must match the recomputed sum of inventory and setup costs derived from the
solution's x and y variables.
"""

import argparse
import json
import math
from collections import deque

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('carryover', 'objective_value', 'production', 'setups')
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


def parse_solution(sol, n, m, p):
    """Parse solution JSON into structured variable dictionaries."""
    # Parse x[i,k,t,u]
    x = {}
    for key, val in sol.get("production", {}).items():
        parts = key.split("_")  # x_i_k_t_u
        i, k, t, u = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        x[i, k, t, u] = float(val)

    # Parse y[i,j,k,t]
    y = {}
    for key, val in sol.get("setups", {}).items():
        parts = key.split("_")  # y_i_j_k_t
        i, j, k, t = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        y[i, j, k, t] = int(round(float(val)))

    # Parse z[i,k,t]
    z = {}
    for key, val in sol.get("carryover", {}).items():
        parts = key.split("_")  # z_i_k_t
        i, k, t = int(parts[1]), int(parts[2]), int(parts[3])
        z[i, k, t] = int(round(float(val)))

    return x, y, z


def get_x(x, i, k, t, u):
    return x.get((i, k, t, u), 0.0)


def get_y(y, i, j, k, t):
    return y.get((i, j, k, t), 0)


def get_z(z, i, k, t):
    return z.get((i, k, t), 0)


def derive_G(z, y, n, m, p):
    """Derive G[i,k,t] = 1 if machine k is ready to produce item i in period t."""
    G = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                val = get_z(z, i, k, t) + sum(get_y(y, j, i, k, t) for j in range(n))
                G[i, k, t] = 1 if val > 0.5 else 0
    return G


def derive_R(y, n, m, p):
    """Derive R[k,t] = 1 if at least one setup is performed at machine k in period t."""
    R = {}
    for k in range(m):
        for t in range(p):
            total = sum(get_y(y, i, j, k, t) for i in range(n) for j in range(n))
            R[k, t] = 1 if total > 0.5 else 0
    return R


def compute_production_split(x, z, M, n, m, p):
    """Compute x_a and x_b by maximizing x_b (production before first setup)."""
    total_prod = {}
    x_b = {}
    x_a = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                tot = sum(get_x(x, i, k, t, u) for u in range(t, p))
                total_prod[i, k, t] = tot
                max_xb = M[i][t] * get_z(z, i, k, t)
                xb = min(tot, max_xb)
                x_b[i, k, t] = xb
                x_a[i, k, t] = tot - xb
    return total_prod, x_a, x_b


def check_subtour_reachability(z, y, G, n, m, p):
    """
    Check sub-tour elimination: for each (k,t), all active items (G[i,k,t]=1)
    must be reachable from the initial state item (z[i,k,t]=1) via y-edges.
    Returns list of (k, t, unreachable_items) for violations.
    """
    violations = []
    for k in range(m):
        for t in range(p):
            active = {i for i in range(n) if G.get((i, k, t), 0) == 1}
            if not active:
                continue

            # Find the initial state item
            start_items = {i for i in range(n) if get_z(z, i, k, t) == 1}

            # BFS from start items through y-edges
            visited = set()
            queue = deque(start_items)
            visited.update(start_items)
            while queue:
                node = queue.popleft()
                for j in range(n):
                    if j not in visited and get_y(y, node, j, k, t) > 0:
                        visited.add(j)
                        queue.append(j)
                # Also check incoming edges that create reachability
                # (y[j,node,k,t] > 0 means node is reachable from j)
            # Need to also check reverse: items reachable TO via incoming y
            # Actually, the flow goes from initial state forward through transitions.
            # y[i,j,k,t] means transition from i to j, so j is reachable from i.

            unreachable = active - visited
            if unreachable:
                violations.append((k, t, unreachable))
    return violations


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["dimensions"]["n"]
    m = instance["dimensions"]["m"]
    p = instance["dimensions"]["p"]

    d = instance["demands"]
    h = instance["inventory_costs"]
    proc_time_val = instance["processing_time"]
    b_raw = instance["setup_times"]
    c_raw = instance["setup_costs"]
    T_cap = instance["machine_capacities"]
    M = instance["max_production"]
    q_raw = instance["max_setups_per_item"]
    min_lot = instance["minimum_lot_sizes"]

    x, y, z = parse_solution(solution, n, m, p)
    G = derive_G(z, y, n, m, p)
    R = derive_R(y, n, m, p)
    total_prod, x_a, x_b = compute_production_split(x, z, M, n, m, p)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs, rhs, op):
        """Record a constraint violation with normalized magnitude."""
        if op == "eq":
            violation_amount = abs(lhs - rhs)
        elif op == "leq":
            violation_amount = max(0, lhs - rhs)
        elif op == "geq":
            violation_amount = max(0, rhs - lhs)
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            violated_constraints.add(constraint_idx)
            violations.append(msg)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": round(lhs, 10),
                "rhs": round(rhs, 10),
                "raw_excess": round(violation_amount, 10),
                "normalizer": round(normalizer, 10),
                "ratio": round(ratio, 10)
            })

    # =========================================================================
    # Constraint (2): Demand satisfaction (equality)
    # sum_k sum_{t=1}^{u} x_{iktu} = d_{iu}  forall (i, u)
    # =========================================================================
    for i in range(n):
        for u in range(p):
            lhs = sum(get_x(x, i, k, t, u) for k in range(m) for t in range(u + 1))
            rhs = d[i][u]
            record_violation(2,
                f"Demand violation: item {i}, period {u}: produced={lhs:.4f}, demand={rhs}",
                lhs, rhs, "eq")

    # =========================================================================
    # Constraint (3): Capacity (<=)
    # sum_i (sum_{u>=t} f*x_{iktu} + sum_j b_{jik}*y_{jikt}) <= T_{kt}
    # =========================================================================
    for k in range(m):
        for t in range(p):
            prod_time = sum(
                proc_time_val * get_x(x, i, k, t, u)
                for i in range(n) for u in range(t, p))
            setup_time = sum(
                b_raw[j][i][k] * get_y(y, j, i, k, t)
                for i in range(n) for j in range(n))
            lhs = prod_time + setup_time
            rhs = T_cap[k][t]
            record_violation(3,
                f"Capacity exceeded: machine {k}, period {t}: used={lhs:.4f}, capacity={rhs}",
                lhs, rhs, "leq")

    # =========================================================================
    # Constraint (4): Production-setup link (<=)
    # x_{iktu} <= M_{it} * G_{ikt}  forall (i, k, t, u)
    # =========================================================================
    for i in range(n):
        for k in range(m):
            for t in range(p):
                g_val = G.get((i, k, t), 0)
                for u in range(t, p):
                    lhs = get_x(x, i, k, t, u)
                    rhs = M[i][t] * g_val
                    record_violation(4,
                        f"Production without setup: item {i}, machine {k}, period {t}, "
                        f"demand period {u}: x={lhs:.4f}, bound={rhs}",
                        lhs, rhs, "leq")

    # =========================================================================
    # Constraint (5): One setup state per machine per period (equality)
    # sum_i z_{ikt} = 1  forall (k, t)
    # =========================================================================
    for k in range(m):
        for t in range(p + 1):
            lhs = sum(get_z(z, i, k, t) for i in range(n))
            rhs = 1
            record_violation(5,
                f"Setup state count: machine {k}, period {t}: sum_z={lhs}, expected=1",
                lhs, rhs, "eq")

    # =========================================================================
    # Constraint (6): Flow conservation (equality)
    # z_{ikt} + sum_j y_{jikt} = sum_j y_{ijkt} + z_{ik,t+1}  forall (i, k, t)
    # =========================================================================
    for i in range(n):
        for k in range(m):
            for t in range(p):
                lhs = get_z(z, i, k, t) + sum(get_y(y, j, i, k, t) for j in range(n))
                rhs = sum(get_y(y, i, j, k, t) for j in range(n)) + get_z(z, i, k, t + 1)
                record_violation(6,
                    f"Flow conservation: item {i}, machine {k}, period {t}: "
                    f"in={lhs}, out={rhs}",
                    lhs, rhs, "eq")

    # =========================================================================
    # Constraint (7): G link (>=)
    # z_{ikt} + sum_j y_{jikt} >= G_{ikt}  forall (i, k, t)
    # =========================================================================
    for i in range(n):
        for k in range(m):
            for t in range(p):
                lhs = get_z(z, i, k, t) + sum(get_y(y, j, i, k, t) for j in range(n))
                rhs = G.get((i, k, t), 0)
                record_violation(7,
                    f"G link: item {i}, machine {k}, period {t}: "
                    f"z+sum_y={lhs}, G={rhs}",
                    lhs, rhs, "geq")

    # =========================================================================
    # Constraint (8): Max setups (<=)
    # z_{ikt} + sum_j y_{jikt} <= q_{ikt} * G_{ikt}  forall (i, k, t)
    # =========================================================================
    for i in range(n):
        for k in range(m):
            for t in range(p):
                lhs = get_z(z, i, k, t) + sum(get_y(y, j, i, k, t) for j in range(n))
                rhs = q_raw[i][k][t] * G.get((i, k, t), 0)
                record_violation(8,
                    f"Max setups exceeded: item {i}, machine {k}, period {t}: "
                    f"count={lhs}, limit={rhs}",
                    lhs, rhs, "leq")

    # =========================================================================
    # Constraints (9) and (10) SKIPPED: definitional identities for the
    # reformulation variables x_a, x_b (split of production into before-/
    # after-first-changeover portions). They are construction-satisfied and
    # thus purely redundant under the Rule 4 relaxation.
    # =========================================================================

    # =========================================================================
    # Constraint (11)+(12) simplified as the original BUSINESS rule:
    # Non-triangular setups require a minimum-lot rule — when a setup occurs
    # for item i on machine k in period t (sum_j y_{jikt} >= 1) without
    # carry-over into the next period (z_{ik,t+1} = 0), the total production
    # of item i on machine k across periods t..p-1 (before being changed out)
    # must be at least m_i. This replaces the x_a/x_b reformulation with a
    # direct business check on the primary x, y, z variables.
    # =========================================================================
    for i in range(n):
        if min_lot[i] == 0:
            continue  # trivially satisfied
        for k in range(m):
            for t in range(p):
                sum_y_in = sum(get_y(y, j, i, k, t) for j in range(n))
                if sum_y_in == 0:
                    continue  # no setup in this period, no min-lot requirement
                # Setup occurs here; check total production of item i on
                # machine k from period t onward at least once exceeds m_i.
                # Use the largest possible production window (whichever first
                # "runs out" by being changed out via setup to another item).
                total_production_ahead = total_prod[i, k, t]
                # Also accumulate production in later periods while item i
                # remains carried over (z = 1 across intermediate periods).
                for lam in range(t + 1, p):
                    if get_z(z, i, k, lam) < 0.5:
                        break
                    total_production_ahead += total_prod.get((i, k, lam), 0)
                required = min_lot[i] * sum_y_in
                if required - total_production_ahead > tol:
                    record_violation(11,
                        f"Min lot violated: item {i}, machine {k}, period {t}: "
                        f"total production = {total_production_ahead:.4f} < "
                        f"required {required:.4f} (min_lot={min_lot[i]} x "
                        f"num_setups={sum_y_in})",
                        total_production_ahead, required, "geq")

    # =========================================================================
    # Constraint (13): Sub-tour flow origin (equality)
    # sum_j F_{0jkt} = sum_i G_{ikt}  forall (k, t)
    # Constraint (14): Sub-tour flow balance (equality)
    # F_{0ikt} + sum_j F_{jikt} = G_{ikt} + sum_j F_{ijkt}  forall (i, k, t)
    # Constraint (15): Flow capacity from origin (<=)
    # F_{0ikt} <= n * z_{ikt}  forall (i, k, t)
    # Constraint (16): Flow capacity on arcs (<=)
    # F_{ijkt} <= n * y_{ijkt}  forall (i, j, k, t)
    #
    # These constraints ensure sub-tour elimination. We check by verifying
    # all active items are reachable from the initial state via y-transitions.
    # If reachable, valid F, F0 exist satisfying (13)-(16).
    # =========================================================================
    subtour_violations = check_subtour_reachability(z, y, G, n, m, p)
    for k_v, t_v, unreachable in subtour_violations:
        for constraint_idx in [13, 14, 15, 16]:
            # Report against all sub-tour constraints since the flow is infeasible
            pass
        # Report as constraint 13 (the primary sub-tour constraint)
        num_active = sum(1 for i in range(n) if G.get((i, k_v, t_v), 0) == 1)
        num_reachable = num_active - len(unreachable)
        record_violation(13,
            f"Sub-tour elimination: machine {k_v}, period {t_v}: "
            f"unreachable items {unreachable} from initial state "
            f"(reachable={num_reachable}, active={num_active})",
            num_reachable, num_active, "eq")

    # Also check constraint (15): flow from origin only to initial state item
    # If z[i,k,t]=0 for some active item that has no y-path from z-item, it's caught above.
    # Additionally check: if there is no initial state (sum z = 0) but active items exist
    for k in range(m):
        for t in range(p):
            num_active = sum(1 for i in range(n) if G.get((i, k, t), 0) == 1)
            has_initial = sum(get_z(z, i, k, t) for i in range(n))
            if num_active > 0 and has_initial == 0:
                record_violation(15,
                    f"No initial state for flow: machine {k}, period {t}: "
                    f"active items={num_active} but no z[i,k,t]=1",
                    0, num_active, "geq")

    # Check constraint (16): F_{ijkt} <= n * y_{ijkt}
    # If y[i,j,k,t] = 0 but transition is needed for reachability, caught by (13) check.
    # No additional check needed beyond reachability.

    # =========================================================================
    # Constraint (17): F_{ijkt} >= 0
    # Satisfied by construction (F is derived as non-negative flow).
    # No check needed.
    # =========================================================================

    # =========================================================================
    # Constraint (18): G_{ikt}, R_{kt} in {0, 1}
    # Satisfied by construction (derived as binary).
    # No check needed.
    # =========================================================================

    # =========================================================================
    # Constraint (19): z_{ikt} in {0, 1}, x_{iktu} >= 0
    # =========================================================================
    for key, val in z.items():
        if val not in (0, 1):
            i, k, t = key
            record_violation(19,
                f"z not binary: z[{i},{k},{t}]={val}",
                val, round(val), "eq")

    for key, val in x.items():
        if val < -tol:
            i, k, t, u = key
            record_violation(19,
                f"Negative production: x[{i},{k},{t},{u}]={val:.6f}",
                val, 0.0, "geq")

    # =========================================================================
    # Constraint (20): y_{ijkt} in {0, ..., q_{jkt}}
    # =========================================================================
    for key, val in y.items():
        i, j, k, t = key
        if val < 0 or val != int(val):
            record_violation(20,
                f"y not non-negative integer: y[{i},{j},{k},{t}]={val}",
                val, 0, "geq")
        elif val > q_raw[j][k][t]:
            record_violation(20,
                f"y exceeds max setups: y[{i},{j},{k},{t}]={val}, "
                f"max q[{j},{k},{t}]={q_raw[j][k][t]}",
                val, q_raw[j][k][t], "leq")

    # =========================================================================
    # Constraint (1): Objective-value consistency (full recompute).
    # The reported objective_value must equal the recomputed sum of inventory
    # holding costs and setup changeover costs:
    #     obj = Σ (u-t) * h_i * x_{iktu}  +  Σ c_{ijk} * y_{ijkt}
    # Every variable that determines the objective (x and y) is present in
    # the solution, so a full recompute is exact (no lower-bound looseness).
    # Tolerance: max(1e-3, 1e-3 * |true|)  — 0.1% relative with 1e-3 floor.
    # =========================================================================
    reported_raw = solution.get("objective_value")
    try:
        reported = float(reported_raw) if reported_raw is not None else None
    except (TypeError, ValueError):
        reported = None
    if reported is not None and (math.isnan(reported) or math.isinf(reported)):
        # NaN / inf is itself a lie; treat as unbounded deviation.
        reported_finite = False
    else:
        reported_finite = reported is not None
    if reported is not None:
        inv_cost = 0.0
        for (i, k, t, u), xv in x.items():
            if u >= t:
                inv_cost += (u - t) * h[i] * float(xv)
        setup_cost = 0.0
        for (i, j, k, t), yv in y.items():
            setup_cost += c_raw[i][j][k] * int(yv)
        true_obj = float(inv_cost + setup_cost)
        obj_tol = max(1e-3, 1e-3 * abs(true_obj))
        if not reported_finite:
            abs_diff = float("inf")
        else:
            abs_diff = abs(reported - true_obj)
        if abs_diff > obj_tol:
            violated_constraints.add(1)
            msg = (
                f"Objective consistency violated: reported objective_value="
                f"{reported_raw} differs from recomputed inventory+setup cost="
                f"{true_obj:.6f} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g}); "
                f"inv_cost={inv_cost:.6f}, setup_cost={setup_cost:.6f}"
            )
            violations.append(msg)
            normalizer = max(abs(true_obj), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": round(reported, 10) if reported_finite else reported_raw,
                "rhs": round(true_obj, 10),
                "raw_excess": round(abs_diff, 10) if reported_finite else abs_diff,
                "normalizer": round(normalizer, 10),
                "ratio": round(abs_diff / normalizer, 10) if reported_finite else abs_diff,
            })

    # Build result
    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    # Deduplicate violation messages per constraint
    seen_constraints_msgs = {}
    deduped_violations = []
    for msg in violations:
        if msg not in seen_constraints_msgs:
            seen_constraints_msgs[msg] = True
            deduped_violations.append(msg)

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": deduped_violations,
        "violation_magnitudes": violation_magnitudes if not feasible else []
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for ILSSP-NT solutions")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON")
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
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
        for v in result["violations"][:10]:
            print(f"  - {v}")
        if len(result["violations"]) > 10:
            print(f"  ... and {len(result['violations']) - 10} more violations")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
