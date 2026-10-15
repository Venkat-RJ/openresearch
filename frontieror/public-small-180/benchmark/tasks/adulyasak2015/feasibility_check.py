#!/usr/bin/env python3
"""
Feasibility checker for the Two-Stage Stochastic Production Routing Problem (2-SPRP).
Reference: Adulyasak, Cordeau, Jans (2015), Operations Research.

Checks all hard constraints from the mathematical formulation.
Constraint numbering follows the paper's formulation section, counted from top to bottom:
  Constraint 1  = Eq.(2):  Plant inventory flow balance (=)
  Constraint 2  = Eq.(3):  Customer inventory flow balance (=)
  Constraint 3  = Eq.(4):  Maximum inventory at plant (<=)
  Constraint 4  = Eq.(5):  Maximum inventory at customers (<=)
  Constraint 5  = Eq.(6):  Production capacity / setup (<=)
  Constraint 6  = Eq.(7):  Vehicle load capacity (<=)
  Constraint 7  = Eq.(8):  Delivery only if customer visited (<=)
  Constraint 8  = Eq.(9):  Each customer visited at most once per period (<=)
  Constraint 9  = Eq.(10): Degree constraints (=)
  Constraint 10 = Eq.(11): Subtour elimination (<=)
  Constraint 11 = Eq.(12): Nonnegativity of second-stage variables (>=)
  Constraint 12 = Eq.(13): Binary y, z (domain)
  Constraint 13 = Eq.(14): Binary x for non-depot edges (domain)
  Constraint 14 = Eq.(15): x at depot edges in {0,1,2} (domain)
  Constraint 15 = Eq.(16): Customer visit requires vehicle dispatch (<=)
  Constraint 16 = Eq.(17): Edge implies both endpoints visited (<=)
  Constraint 17 = Eq.(18): Vehicle symmetry breaking (>=)
  Constraint 18 = Eq.(19): Lexicographic ordering of visits (>=)
  Constraint 19 = Eq.(20): Nonanticipativity for p (=, M-SPRP only)
  Constraint 20 = Eq.(21): Nonanticipativity for I (=, M-SPRP only)
  Constraint 21 = Eq.(22): Nonanticipativity for e (=, M-SPRP only)
  Constraint 22 = Eq.(23): Nonanticipativity for q (=, M-SPRP only)
  Constraint 23 = Objective consistency (lower-bound check on reported obj)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solverless_witness import require_linked_variables, validate_complete_witness

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('model_variables', 'objective_value', 'x', 'y', 'z')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('model_variables',)
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




# FRONTIEROR_ADULYASAK_EXACT_OBJECTIVE_V1
import math
from typing import Any

def _binary(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    number = float(value)
    rounded = round(number)
    if not math.isfinite(number) or abs(number - rounded) > 1e-6 or rounded not in (0, 1):
        raise ValueError(f"{key} must be binary")
    return int(rounded)

def _edge_value(mapping: dict[str, Any], key: str, depot_edge: bool) -> int:
    value = mapping.get(key, 0)
    if isinstance(value, bool):
        value = int(value)
    number = float(value)
    rounded = round(number)
    upper = 2 if depot_edge else 1
    if (
        not math.isfinite(number)
        or abs(number - rounded) > 1e-6
        or rounded < 0
        or rounded > upper
    ):
        raise ValueError(f"{key} must be an integer in [0, {upper}]")
    return int(rounded)

def _frontieror_solverless_objective(instance, solution, instance_path):
    """Validate complete recourse and return its directly evaluated cost."""
    n = int(instance["n"])
    horizon = int(instance["T"])
    vehicle_count = int(instance["m"])
    nodes = range(n + 1)
    periods = range(1, horizon + 1)
    vehicles = range(1, vehicle_count + 1)
    edges = [(i, j) for i in nodes for j in nodes if i < j]
    expected = {
        f"y_{t}": _binary(solution["y"], str(t)) for t in periods
    }
    expected.update({
        f"z_{i}_{k}_{t}": _binary(solution["z"], f"{i}_{k}_{t}")
        for i in nodes for k in vehicles for t in periods
    })
    expected.update({
        f"x_{i}_{j}_{k}_{t}": _edge_value(
            solution["x"], f"{i}_{j}_{k}_{t}", i == 0
        )
        for i, j in edges for k in vehicles for t in periods
    })
    objective, errors, _ = validate_complete_witness(
        instance_path, solution, Path(__file__).resolve().parent
    )
    errors.extend(require_linked_variables(solution, expected))
    if errors:
        raise ValueError("; ".join(errors[:20]))
    if objective is None:
        raise ValueError("complete witness objective is unavailable")
    return objective


def check_feasibility(data, sol, instance_path=None):
    tol = 1e-5
    eps = 1e-5

    n = data['n']
    T = data['T']
    m = data['m']
    Q = data['Q']
    C = data['C']
    L = data['L']
    I0 = data['I0']
    demand_sc = data['demand_scenarios']
    n_omega = data['n_scenarios']
    problem_type = data.get('problem_type', '2-SPRP')

    Nc = list(range(1, n + 1))
    N = list(range(0, n + 1))
    periods = list(range(1, T + 1))
    vehicles = list(range(1, m + 1))
    scenarios = list(range(n_omega))

    edges = [(i, j) for i in N for j in N if i < j]
    edges_Nc = [(i, j) for i in Nc for j in Nc if i < j]

    def d(omega, i, t):
        return demand_sc[omega][i - 1][t - 1]

    def M_t_val(t, omega):
        rem = sum(d(omega, ci, s) for ci in Nc for s in range(t, T + 1))
        return min(C, Q, rem)

    def M_prime_val(i, t, omega):
        rem = sum(d(omega, i, s) for s in range(t, T + 1))
        return min(L[i], Q, rem)

    # --- Parse solution ---
    y = {}
    for key, val in sol.get('y', {}).items():
        y[int(key)] = val
    for t in periods:
        y.setdefault(t, 0)

    z = {}
    for key, val in sol.get('z', {}).items():
        parts = key.split('_')
        i, k, t = int(parts[0]), int(parts[1]), int(parts[2])
        z[i, k, t] = val
    for i in N:
        for k in vehicles:
            for t in periods:
                z.setdefault((i, k, t), 0)

    x = {}
    for key, val in sol.get('x', {}).items():
        parts = key.split('_')
        i, j, k, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        x[i, j, k, t] = val
    for (i, j) in edges:
        for k in vehicles:
            for t in periods:
                x.setdefault((i, j, k, t), 0)

    violations_list = []  # list of (constraint_idx, message, lhs, rhs, raw_excess, normalizer, ratio)
    violated_set = set()

    def record_violation(constraint_idx, lhs, rhs, op, msg):
        if op == '<=':
            violation_amount = max(0.0, lhs - rhs)
        elif op == '>=':
            violation_amount = max(0.0, rhs - lhs)
        elif op == '=':
            violation_amount = abs(lhs - rhs)
        else:
            return

        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_set.add(constraint_idx)
            violations_list.append({
                'constraint': constraint_idx,
                'msg': msg,
                'lhs': float(lhs),
                'rhs': float(rhs),
                'raw_excess': float(violation_amount),
                'normalizer': float(normalizer),
                'ratio': float(ratio),
            })

    # =====================================================================
    # Constraints 1-7 (Eqs. 2-8): Second-stage constraints
    # These involve second-stage variables (p, I, q, e) not present in the
    # solution. We construct feasible second-stage variables to verify.
    # For any first-stage (y, z, x), feasible second-stage always exists
    # because unmet demand variable e acts as slack.
    # =====================================================================

    # Construct feasible second-stage variables for each scenario
    for omega in scenarios:
        # Determine maximum production and delivery given first-stage decisions
        # Greedy: produce as much as allowed, deliver as much as possible
        # to minimize unmet demand (which always makes constraints feasible).

        # Production capacity per period
        p_max = {}
        for t in periods:
            p_max[t] = M_t_val(t, omega) * y[t]

        # Delivery capacity per customer/vehicle/period
        q_max = {}
        for i_c in Nc:
            for k in vehicles:
                for t in periods:
                    q_max[i_c, k, t] = M_prime_val(i_c, t, omega) * z[i_c, k, t]

        # Vehicle load limit per vehicle/period
        veh_cap = {}
        for k in vehicles:
            for t in periods:
                veh_cap[k, t] = Q * z[0, k, t]

        # Forward simulation: allocate production and deliveries
        I_plant = {0: float(I0[0])}
        I_cust = {}
        for i_c in Nc:
            I_cust[i_c, 0] = float(I0[i_c])

        p_val = {}
        q_val = {}
        e_val = {}

        for t in periods:
            # Determine total demand and delivery needs
            total_need = 0.0
            for i_c in Nc:
                need = max(0.0, d(omega, i_c, t) - I_cust[i_c, t - 1])
                total_need += need

            # Production: produce up to capacity, limited by plant storage
            produce = min(p_max[t], total_need, L[0] - I_plant[t - 1] + total_need)
            produce = max(0.0, produce)
            p_val[t] = produce

            # Available at plant for delivery
            available = I_plant[t - 1] + produce

            # Allocate deliveries: respect vehicle capacity and customer visit
            total_delivered = 0.0
            for k in vehicles:
                veh_remaining = veh_cap[k, t]
                for i_c in Nc:
                    cap = min(q_max[i_c, k, t], veh_remaining,
                              max(0.0, available - total_delivered))
                    # Deliver up to demand shortfall
                    need = max(0.0, d(omega, i_c, t) - I_cust[i_c, t - 1])
                    deliver = min(cap, need)
                    q_val[i_c, k, t] = deliver
                    veh_remaining -= deliver
                    total_delivered += deliver

            # Update plant inventory
            I_plant[t] = I_plant[t - 1] + p_val[t] - total_delivered

            # Update customer inventories and unmet demand
            for i_c in Nc:
                delivery = sum(q_val[i_c, k, t] for k in vehicles)
                dem = d(omega, i_c, t)
                prev = I_cust[i_c, t - 1]
                # Balance: prev + delivery + e = dem + I_new
                I_cust[i_c, t] = max(0.0, prev + delivery - dem)
                e_val[i_c, t] = max(0.0, dem - prev - delivery)

            # --- Constraint 1 (Eq.2): Plant inventory balance ---
            lhs = I_plant[t - 1] + p_val[t]
            rhs = total_delivered + I_plant[t]
            record_violation(1, lhs, rhs, '=',
                f"Constraint 1 (Eq.2): Plant balance violated in period {t}, scenario {omega}: "
                f"LHS={lhs:.6f} != RHS={rhs:.6f}")

            # --- Constraint 3 (Eq.4): Plant inventory capacity ---
            record_violation(3, I_plant[t], float(L[0]), '<=',
                f"Constraint 3 (Eq.4): Plant inventory {I_plant[t]:.4f} > capacity {L[0]} "
                f"in period {t}, scenario {omega}")

            # --- Constraint 5 (Eq.6): Production capacity ---
            record_violation(5, p_val[t], p_max[t], '<=',
                f"Constraint 5 (Eq.6): Production {p_val[t]:.4f} > capacity {p_max[t]:.4f} "
                f"in period {t}, scenario {omega}")

            for i_c in Nc:
                delivery = sum(q_val[i_c, k, t] for k in vehicles)
                dem = d(omega, i_c, t)

                # --- Constraint 2 (Eq.3): Customer inventory balance ---
                lhs2 = I_cust[i_c, t - 1] + delivery + e_val[i_c, t]
                rhs2 = dem + I_cust[i_c, t]
                record_violation(2, lhs2, rhs2, '=',
                    f"Constraint 2 (Eq.3): Customer {i_c} balance violated in period {t}, "
                    f"scenario {omega}: LHS={lhs2:.6f} != RHS={rhs2:.6f}")

                # --- Constraint 4 (Eq.5): Customer inventory capacity ---
                record_violation(4, I_cust[i_c, t] + dem, float(L[i_c]), '<=',
                    f"Constraint 4 (Eq.5): Customer {i_c} inventory+demand "
                    f"{I_cust[i_c, t] + dem:.4f} > capacity {L[i_c]} "
                    f"in period {t}, scenario {omega}")

                # --- Constraint 7 (Eq.8): Delivery only if visited ---
                for k in vehicles:
                    mp = M_prime_val(i_c, t, omega) * z[i_c, k, t]
                    record_violation(7, q_val[i_c, k, t], mp, '<=',
                        f"Constraint 7 (Eq.8): Delivery to customer {i_c} by vehicle {k} "
                        f"in period {t}, scenario {omega}: {q_val[i_c, k, t]:.4f} > {mp:.4f}")

            # --- Constraint 6 (Eq.7): Vehicle load capacity ---
            for k in vehicles:
                total_load = sum(q_val[i_c, k, t] for i_c in Nc)
                cap_rhs = Q * z[0, k, t]
                record_violation(6, total_load, float(cap_rhs), '<=',
                    f"Constraint 6 (Eq.7): Vehicle {k} load {total_load:.4f} > capacity "
                    f"{cap_rhs:.4f} in period {t}, scenario {omega}")

        # --- Constraint 11 (Eq.12): Nonnegativity ---
        for t in periods:
            record_violation(11, 0.0, p_val[t], '<=',
                f"Constraint 11 (Eq.12): p_{t} negative ({p_val[t]:.6f}) in scenario {omega}")
            record_violation(11, 0.0, I_plant[t], '<=',
                f"Constraint 11 (Eq.12): I_plant_{t} negative ({I_plant[t]:.6f}) in scenario {omega}")
            for i_c in Nc:
                record_violation(11, 0.0, I_cust[i_c, t], '<=',
                    f"Constraint 11 (Eq.12): I_{i_c}_{t} negative ({I_cust[i_c, t]:.6f}) "
                    f"in scenario {omega}")
                record_violation(11, 0.0, e_val[i_c, t], '<=',
                    f"Constraint 11 (Eq.12): e_{i_c}_{t} negative ({e_val[i_c, t]:.6f}) "
                    f"in scenario {omega}")
                for k in vehicles:
                    record_violation(11, 0.0, q_val[i_c, k, t], '<=',
                        f"Constraint 11 (Eq.12): q_{i_c}_{k}_{t} negative "
                        f"({q_val[i_c, k, t]:.6f}) in scenario {omega}")

    # =====================================================================
    # Constraint 8 (Eq.9): Each customer visited at most once per period
    # sum_k z_{ikt} <= 1  for all i in Nc, t in T
    # =====================================================================
    for i_c in Nc:
        for t in periods:
            lhs = sum(z[i_c, k, t] for k in vehicles)
            record_violation(8, float(lhs), 1.0, '<=',
                f"Constraint 8 (Eq.9): Customer {i_c} visited {lhs} times "
                f"(> 1) in period {t}")

    # =====================================================================
    # Constraint 9 (Eq.10): Degree constraints
    # sum_{(j,j') in delta(i)} x_{jj'kt} = 2 * z_{ikt}
    # =====================================================================
    for i_node in N:
        for k in vehicles:
            for t in periods:
                inc_edges = [(a, b) for (a, b) in edges if a == i_node or b == i_node]
                lhs = sum(x[a, b, k, t] for (a, b) in inc_edges)
                rhs = 2.0 * z[i_node, k, t]
                record_violation(9, float(lhs), float(rhs), '=',
                    f"Constraint 9 (Eq.10): Degree of node {i_node} by vehicle {k} "
                    f"in period {t}: sum_edges={lhs} != 2*z={rhs}")

    # =====================================================================
    # Constraint 10 (Eq.11): Subtour elimination
    # For each vehicle k and period t, check that visited customers form
    # a connected component through the depot (no customer-only subtours).
    # =====================================================================
    for k in vehicles:
        for t in periods:
            visited = [i_c for i_c in Nc if z[i_c, k, t] > 0.5]
            if len(visited) < 2:
                continue

            # Build adjacency among all nodes with edges
            adj = defaultdict(set)
            for (a, b) in edges:
                if x[a, b, k, t] > 0.5:
                    adj[a].add(b)
                    adj[b].add(a)

            # BFS from depot to find reachable customers
            reachable = set()
            stack = [0]
            seen = set()
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                if node in visited:
                    reachable.add(node)
                for nb in adj[node]:
                    if nb not in seen:
                        stack.append(nb)

            # Customers visited but not reachable from depot form subtours
            subtour_nodes = set(visited) - reachable
            if subtour_nodes:
                # Find the subtour components
                remaining = set(subtour_nodes)
                while remaining:
                    start = next(iter(remaining))
                    comp = set()
                    stack = [start]
                    while stack:
                        node = stack.pop()
                        if node in comp:
                            continue
                        comp.add(node)
                        for nb in adj[node]:
                            if nb in remaining and nb not in comp:
                                stack.append(nb)
                    remaining -= comp
                    S = comp
                    # Compute violation: sum x_{ij} for (i,j) in E(S) <= sum z - z_e
                    E_S = [(a, b) for (a, b) in edges if a in S and b in S]
                    sum_x = sum(x[a, b, k, t] for (a, b) in E_S)
                    for e_node in S:
                        rhs_val = sum(z[i_c, k, t] for i_c in S) - z[e_node, k, t]
                        violation_amount = max(0.0, sum_x - rhs_val)
                        if violation_amount > tol:
                            normalizer = max(abs(rhs_val), eps)
                            ratio = violation_amount / normalizer
                            violated_set.add(10)
                            violations_list.append({
                                'constraint': 10,
                                'msg': (f"Constraint 10 (Eq.11): Subtour among customers "
                                        f"{sorted(S)} by vehicle {k} in period {t}"),
                                'lhs': float(sum_x),
                                'rhs': float(rhs_val),
                                'raw_excess': float(violation_amount),
                                'normalizer': float(normalizer),
                                'ratio': float(ratio),
                            })
                            break  # one violation per component suffices

    # =====================================================================
    # Constraint 12 (Eq.13): Binary y_t, z_{ikt}
    # =====================================================================
    for t in periods:
        val = y[t]
        if abs(val - round(val)) > tol or round(val) not in (0, 1):
            record_violation(12, float(val), float(round(val)), '=',
                f"Constraint 12 (Eq.13): y_{t} = {val} is not binary")

    for i_node in N:
        for k in vehicles:
            for t in periods:
                val = z[i_node, k, t]
                if abs(val - round(val)) > tol or round(val) not in (0, 1):
                    record_violation(12, float(val), float(round(val)), '=',
                        f"Constraint 12 (Eq.13): z_{i_node}_{k}_{t} = {val} is not binary")

    # =====================================================================
    # Constraint 13 (Eq.14): Binary x for non-depot edges (i != 0)
    # =====================================================================
    for (i_e, j_e) in edges:
        if i_e == 0:
            continue
        for k in vehicles:
            for t in periods:
                val = x[i_e, j_e, k, t]
                if abs(val - round(val)) > tol or round(val) not in (0, 1):
                    record_violation(13, float(val), float(round(val)), '=',
                        f"Constraint 13 (Eq.14): x_{i_e}_{j_e}_{k}_{t} = {val} "
                        f"is not binary (non-depot edge)")

    # =====================================================================
    # Constraint 14 (Eq.15): x at depot edges in {0, 1, 2}
    # =====================================================================
    for j_e in Nc:
        for k in vehicles:
            for t in periods:
                val = x[0, j_e, k, t]
                if abs(val - round(val)) > tol or round(val) not in (0, 1, 2):
                    record_violation(14, float(val), float(round(val)), '=',
                        f"Constraint 14 (Eq.15): x_0_{j_e}_{k}_{t} = {val} "
                        f"not in {{0,1,2}} (depot edge)")

    # =====================================================================
    # Constraint 15 (Eq.16): z_{ikt} <= z_{0kt} for i in Nc
    # =====================================================================
    for i_c in Nc:
        for k in vehicles:
            for t in periods:
                lhs = float(z[i_c, k, t])
                rhs = float(z[0, k, t])
                record_violation(15, lhs, rhs, '<=',
                    f"Constraint 15 (Eq.16): z_{i_c}_{k}_{t}={lhs} > "
                    f"z_0_{k}_{t}={rhs} (customer visited without vehicle dispatch)")

    # =====================================================================
    # Constraint 16 (Eq.17): x_{ijkt} <= z_{ikt} and x_{ijkt} <= z_{jkt}
    # for (i,j) in E(Nc)
    # =====================================================================
    for (i_e, j_e) in edges_Nc:
        for k in vehicles:
            for t in periods:
                val = float(x[i_e, j_e, k, t])
                zi = float(z[i_e, k, t])
                zj = float(z[j_e, k, t])
                record_violation(16, val, zi, '<=',
                    f"Constraint 16 (Eq.17): x_{i_e}_{j_e}_{k}_{t}={val} > "
                    f"z_{i_e}_{k}_{t}={zi}")
                record_violation(16, val, zj, '<=',
                    f"Constraint 16 (Eq.17): x_{i_e}_{j_e}_{k}_{t}={val} > "
                    f"z_{j_e}_{k}_{t}={zj}")

    # =====================================================================
    # Constraint 17 (Eq.18): z_{0kt} >= z_{0,k+1,t} (vehicle symmetry)
    # Only applies when m > 1
    # =====================================================================
    if m > 1:
        for k in range(1, m):
            for t in periods:
                lhs = float(z[0, k, t])
                rhs = float(z[0, k + 1, t])
                record_violation(17, lhs, rhs, '>=',
                    f"Constraint 17 (Eq.18): z_0_{k}_{t}={lhs} < "
                    f"z_0_{k+1}_{t}={rhs} (vehicle ordering violated)")

    # =====================================================================
    # Constraint 18 (Eq.19): Lexicographic ordering
    # sum_{i=1}^{j} 2^{j-i} z_{ikt} >= sum_{i=1}^{j} 2^{j-i} z_{i,k+1,t}
    # Only applies when m > 1
    # =====================================================================
    if m > 1:
        for j_node in Nc:
            for k in range(1, m):
                for t in periods:
                    lhs = sum(2 ** (j_node - i_c) * z[i_c, k, t]
                              for i_c in range(1, j_node + 1))
                    rhs = sum(2 ** (j_node - i_c) * z[i_c, k + 1, t]
                              for i_c in range(1, j_node + 1))
                    record_violation(18, float(lhs), float(rhs), '>=',
                        f"Constraint 18 (Eq.19): Lexicographic ordering violated "
                        f"for j={j_node}, k={k}, t={t}: {lhs} < {rhs}")

    # =====================================================================
    # Constraints 19-22 (Eqs.20-23): Nonanticipativity (M-SPRP only)
    # These only apply to multistage problems. For 2-SPRP, first-stage
    # variables are scenario-independent by construction, so these are
    # trivially satisfied.
    # =====================================================================
    # (Skipped for 2-SPRP; the solution format already has scenario-independent y, z, x.)

    # =====================================================================
    # Constraint 23: exact objective consistency.
    # The public witness contains first-stage routing decisions but omits
    # scenario recourse variables. Fix those decisions and solve the continuous
    # recourse LP so the reported expected cost is certified in both directions.
    try:
        reported = float(sol["objective_value"])
        if not math.isfinite(reported):
            raise ValueError("objective_value must be finite")
        if instance_path is None:
            raise ValueError("instance_path is required for solverless validation")
        certified = _frontieror_solverless_objective(data, sol, instance_path)
        objective_tolerance = max(
            1e-4,
            1e-8 * max(abs(reported), abs(certified)),
        )
        objective_difference = abs(reported - certified)
        if objective_difference > objective_tolerance:
            violated_set.add(23)
            violations_list.append({
                "constraint": 23,
                "msg": (
                    "Constraint 23 (exact objective consistency): reported "
                    f"objective_value={reported:.10g} differs from fixed-decision "
                    f"recourse value={certified:.10g}; "
                    f"difference={objective_difference:.6g}, "
                    f"tolerance={objective_tolerance:.6g}"
                ),
                "lhs": float(reported),
                "rhs": float(certified),
                "raw_excess": float(objective_difference),
                "normalizer": float(max(abs(certified), eps)),
                "ratio": float(objective_difference / max(abs(certified), eps)),
            })
    except Exception as exc:
        violated_set.add(23)
        violations_list.append({
            "constraint": 23,
            "msg": (
                "Constraint 23 (exact objective consistency) could not certify "
                f"the candidate: {type(exc).__name__}: {exc}"
            ),
            "lhs": 1.0,
            "rhs": 0.0,
            "raw_excess": 1.0,
            "normalizer": 1.0,
            "ratio": 1.0,
        })

    # =====================================================================
    # Aggregate results
    # =====================================================================
    # Build unique violation messages per constraint
    msg_per_constraint = defaultdict(list)
    for v in violations_list:
        msg_per_constraint[v['constraint']].append(v['msg'])

    # Deduplicate messages: summarize if too many
    final_messages = []
    for c_idx in sorted(violated_set):
        msgs = msg_per_constraint[c_idx]
        if len(msgs) <= 3:
            final_messages.extend(msgs)
        else:
            final_messages.append(msgs[0])
            final_messages.append(f"  ... and {len(msgs) - 1} more violations of Constraint {c_idx}")

    # Build violation_magnitudes (all entries)
    final_magnitudes = []
    for v in violations_list:
        final_magnitudes.append({
            'constraint': v['constraint'],
            'lhs': v['lhs'],
            'rhs': v['rhs'],
            'raw_excess': v['raw_excess'],
            'normalizer': v['normalizer'],
            'ratio': v['ratio'],
        })

    result = {
        'feasible': len(violated_set) == 0,
        'violated_constraints': sorted(violated_set),
        'violations': final_messages,
        'violation_magnitudes': final_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Stochastic Production Routing Problem"
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument('--solution_path', type=str, required=True,
                        help="Path to JSON solution file")
    parser.add_argument('--result_path', type=str, required=True,
                        help="Path to write JSON feasibility result")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)
    with open(args.solution_path) as f:
        sol = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
        if _frontieror_contract_result is not None:
            with open(args.result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return

    result = check_feasibility(data, sol, args.instance_path)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result['feasible']:
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} constraint(s) violated: "
              f"{result['violated_constraints']}")
        for msg in result['violations']:
            print(f"  - {msg}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
