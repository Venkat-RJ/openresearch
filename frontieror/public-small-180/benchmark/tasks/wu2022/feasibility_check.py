"""
Feasibility checker for the CFLPP (Capacitated Facility Location and
Production Planning Problem).

Solution variables stored: Y (setup per item/plant/period), U (item-plant
assignment over horizon), Z (plant-open indicator). Production quantities X
and inventory levels I are not stored directly — they are DERIVED internally
by solving a small LP that attempts to find a consistent production plan
respecting:
  - Cumulative demand coverage (production by period t covers demand by t)
  - Plant capacity in each period (setup time + production time ≤ capacity)
  - Setup forcing (production only when setup is active)
  - Single-sourcing (U) and plant-open (Z) rules

If the LP is infeasible, the primary (Y, U, Z) solution cannot be extended
to a complete CFLPP plan, so feasibility is rejected with a business-facing
violation message.

Constraints checked directly on (Y, U, Z):
  1. Single-sourcing: each item assigned to exactly one plant
  2. Plant opening: items may only be assigned to open plants
  3. Setup-location linking: setups may only occur at the assigned plant
  8. Variable domain (binary Y, U, Z; non-negative implied)

Constraints checked indirectly via internally-derived production plan:
  4. Demand coverage is possible
  5. Cumulative production balance (inventory non-negative) is possible
  6. Plant capacity respected
  7. Setup forcing (production only with setup) respected

The LP-derived production plan is an auxiliary computation: an LLM algorithm
that outputs (Y, U, Z) matching a valid CFLPP solution will pass this check;
any inconsistency triggers violations tied to the underlying CFLPP rule.
"""

import json
import argparse
import math
import sys
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('U', 'Y', 'Z', 'objective_value')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('objective_value',)
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



def check_objective_consistency(inst, solution, Y, U, Z, w_cost, new_idx=9):
    """Bind the reported score to the minimum W completion of (Y,U,Z)."""
    violations = []
    magnitudes = []
    if w_cost is None:
        return violations, magnitudes
    reported_obj = float(solution["objective_value"])
    setup_cost = sum(
        inst['sc'][j][k][t] * float(value)
        for (j, k, t), value in Y.items()
    )
    opening_cost = sum(
        inst['f_k'][k] * float(Z[k]) for k in range(inst['K'])
    )
    transportation_cost = 0.0
    for j in range(inst['J']):
        for k in inst['K_j'][j]:
            coefficient = sum(
                inst['d'][j][customer][t]
                * inst['tc'][j][customer][k][t]
                for t in range(inst['T'])
                for customer in inst['C_j'][j]
            )
            transportation_cost += coefficient * float(U[j, k])
    expected = (
        setup_cost + float(w_cost) + transportation_cost + opening_cost
    )
    difference = abs(reported_obj - expected)
    # The historical large MIPs were solved with scaled coefficients; allow
    # one part per million for cross-solver LP numerical drift (the largest
    # observed reference difference is 5 on a 7.17M objective).
    tolerance = max(1e-3, 1e-6 * abs(expected))
    if difference > tolerance:
        violations.append(
            f"Objective consistency violated: objective_value={reported_obj} "
            f"does not match the minimum-cost completion {expected} "
            f"(setup={setup_cost}, production/holding={w_cost}, "
            f"transport={transportation_cost}, opening={opening_cost}, "
            f"|diff|={difference:.4g}, tol={tolerance:.4g})."
        )
        normalizer = max(abs(expected), 1e-5)
        magnitudes.append({
            "constraint": new_idx,
            "lhs": reported_obj,
            "rhs": expected,
            "raw_excess": difference,
            "normalizer": normalizer,
            "ratio": difference / normalizer,
        })
    return violations, magnitudes


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_instance(data):
    """Parse instance data into convenient structures."""
    J = data['dimensions']['num_items']
    T = data['dimensions']['num_periods']
    K = data['dimensions']['num_plants']
    C = data['dimensions']['num_customers']

    K_j = [list(data['plant_capabilities'][j]) for j in range(J)]

    J_k = [[] for _ in range(K)]
    for j in range(J):
        for k in K_j[j]:
            J_k[k].append(j)

    C_j = [[] for _ in range(J)]
    for c in range(C):
        for j in data['customer_item_demands'][c]:
            C_j[j].append(c)

    d = data['demand']
    sc = data['setup_cost']
    pc = data['production_cost']
    hc = data['holding_cost']
    st = data['setup_time']
    pt = data['production_time']
    cap = data['capacity']
    f_k = data['plant_opening_cost']
    tc = data['transportation_cost']

    td = [[sum(d[j][c][t] for c in C_j[j]) for t in range(T)] for j in range(J)]

    sd = {}
    for j in range(J):
        for t_val in range(T):
            cum = 0
            for ell in range(t_val, T):
                cum += td[j][ell]
                sd[j, t_val, ell] = cum

    return {
        'J': J, 'T': T, 'K': K, 'C': C,
        'K_j': K_j, 'J_k': J_k, 'C_j': C_j,
        'd': d, 'sc': sc, 'pc': pc, 'hc': hc,
        'st': st, 'pt': pt, 'cap': cap, 'f_k': f_k, 'tc': tc,
        'td': td, 'sd': sd,
    }


def extract_variables(solution, inst):
    """
    Extract Y, U, Z from the solution.
    If U or Z are missing (efficient solution), infer them from Y.
    """
    J, T, K = inst['J'], inst['T'], inst['K']
    K_j = inst['K_j']

    # Parse Y
    Y = {}
    y_raw = solution.get('Y', {})
    for key_str, val in y_raw.items():
        parts = key_str.split('_')
        j, k, t = int(parts[0]), int(parts[1]), int(parts[2])
        Y[j, k, t] = val

    # Fill missing Y entries with 0
    for j in range(J):
        for k in K_j[j]:
            for t in range(T):
                if (j, k, t) not in Y:
                    Y[j, k, t] = 0

    # Parse or infer U
    U = {}
    if 'U' in solution:
        for key_str, val in solution['U'].items():
            parts = key_str.split('_')
            j, k = int(parts[0]), int(parts[1])
            U[j, k] = val
        # Fill missing U entries with 0
        for j in range(J):
            for k in K_j[j]:
                if (j, k) not in U:
                    U[j, k] = 0
    else:
        # Infer U from Y: U^k_j = 1 if any Y^k_{jt} = 1 for some t
        for j in range(J):
            for k in K_j[j]:
                U[j, k] = 1 if any(Y.get((j, k, t), 0) > 0.5 for t in range(T)) else 0

    # Parse or infer Z
    Z = {}
    if 'Z' in solution:
        for key_str, val in solution['Z'].items():
            k = int(key_str)
            Z[k] = val
        for k in range(K):
            if k not in Z:
                Z[k] = 0
    else:
        # Infer Z from U: Z^k = 1 if any U^k_j = 1 for some j
        for k in range(K):
            Z[k] = 1 if any(U.get((j, k), 0) > 0.5 for j in range(J)) else 0

    return Y, U, Z


def solve_w_feasibility(inst, Y, U, Z):
    """
    Internally derive a consistent production plan from (Y, U, Z).

    Solves a small feasibility LP to determine whether the primary
    (Y, U, Z) can be extended to a complete CFLPP solution (production
    quantities satisfying demand, capacity, and setup forcing). The LP
    uses path-share variables W as a convenient encoding — this is an
    internal derivation, not a constraint on the solution itself.

    Returns the derived plan (as a W dict) and infeasibility messages.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    import numpy as np

    J, T, K = inst['J'], inst['T'], inst['K']
    K_j = inst['K_j']
    J_k = inst['J_k']
    sd = inst['sd']
    st = inst['st']
    pt = inst['pt']
    cap = inst['cap']

    # Index W variables and build their production/holding objective
    # coefficients in the same order.  Keeping this objective (rather than a
    # zero feasibility objective) makes the omitted W completion canonical.
    w_indices = {}
    objective = []
    for j in range(J):
        for k in K_j[j]:
            for t in range(T):
                cumulative_holding_rate = 0.0
                holding_cost = 0.0
                for ell in range(t, T):
                    if ell > t:
                        cumulative_holding_rate += inst['hc'][j][k][ell - 1]
                        holding_cost += (
                            cumulative_holding_rate * inst['td'][j][ell]
                        )
                    w_indices[j, k, t, ell] = len(objective)
                    objective.append(
                        inst['pc'][j][k][t] * sd[j, t, ell]
                        + holding_cost
                    )
    n_vars = len(objective)

    if n_vars == 0:
        return {}, [], 0.0

    # Build sparse equality and inequality constraints directly.  The former
    # dense construction required roughly 180 GiB on each published large
    # instance before HiGHS was even called.
    eq_rows = []
    eq_columns = []
    eq_values = []
    b_eq_rows = []
    ub_rows = []
    ub_columns = []
    ub_values = []
    b_ub_rows = []

    def add_eq(coefficients, rhs):
        row = len(b_eq_rows)
        for column, value in coefficients.items():
            eq_rows.append(row)
            eq_columns.append(column)
            eq_values.append(value)
        b_eq_rows.append(rhs)

    def add_ub(coefficients, rhs):
        row = len(b_ub_rows)
        for column, value in coefficients.items():
            ub_rows.append(row)
            ub_columns.append(column)
            ub_values.append(value)
        b_ub_rows.append(rhs)

    # Constraint 4 (Eq. 6): sum_{t=0}^{T-1} sum_{k in K_j} W^k_{j,0,t} = 1, for all j
    for j in range(J):
        row = {}
        for k in K_j[j]:
            for t in range(T):
                key = (j, k, 0, t)
                if key in w_indices:
                    row[w_indices[key]] = 1.0
        add_eq(row, 1.0)

    # Constraint 5 (Eq. 7): Flow balance
    # sum_{ell=0}^{t-1} W^k_{j,ell,t-1} = sum_{ell=t}^{T-1} W^k_{j,t,ell}
    # for all j, t in {1,...,T-1}, k in K_j
    for j in range(J):
        for t in range(1, T):
            for k in K_j[j]:
                row = {}
                # LHS: sum_{ell=0}^{t-1} W^k_{j,ell,t-1}
                for ell in range(t):
                    key = (j, k, ell, t - 1)
                    if key in w_indices:
                        row[w_indices[key]] = 1.0
                # RHS: sum_{ell=t}^{T-1} W^k_{j,t,ell}
                for ell in range(t, T):
                    key = (j, k, t, ell)
                    if key in w_indices:
                        row[w_indices[key]] = -1.0
                add_eq(row, 0.0)

    # Constraint 6 (Eq. 8): Capacity
    # sum_{j in J_k} (st * Y + sum_ell pt * sd * W) <= cap * Z
    for k in range(K):
        for t in range(T):
            row = {}
            rhs_val = cap[k][t] * Z.get(k, 0)
            fixed_part = 0.0
            for j in J_k[k]:
                fixed_part += st[j][k][t] * Y.get((j, k, t), 0)
                for ell in range(t, T):
                    key = (j, k, t, ell)
                    if key in w_indices:
                        row[w_indices[key]] = pt[j][k][t] * sd[j, t, ell]
            add_ub(row, rhs_val - fixed_part)

    # Constraint 7 (Eq. 9): Setup forcing
    # sum_{ell=t}^{T-1} W^k_{jt,ell} <= Y^k_{jt}
    for j in range(J):
        for t in range(T):
            for k in K_j[j]:
                row = {}
                for ell in range(t, T):
                    key = (j, k, t, ell)
                    if key in w_indices:
                        row[w_indices[key]] = 1.0
                add_ub(row, float(Y.get((j, k, t), 0)))

    A_eq = coo_matrix(
        (eq_values, (eq_rows, eq_columns)),
        shape=(len(b_eq_rows), n_vars),
    ).tocsr() if b_eq_rows else None
    b_eq = np.array(b_eq_rows) if b_eq_rows else None
    A_ub = coo_matrix(
        (ub_values, (ub_rows, ub_columns)),
        shape=(len(b_ub_rows), n_vars),
    ).tocsr() if b_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    result = linprog(
        np.asarray(objective, dtype=float),
        A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(0, None), method='highs', options={"presolve": True},
    )

    W = {}
    if result.success:
        for (j, k, t, ell), i in w_indices.items():
            W[j, k, t, ell] = result.x[i]
        return W, [], float(result.fun)
    else:
        return None, [result.message], None


def check_feasibility(inst, Y, U, Z):
    """
    Check all hard constraints (1-8, corresponding to Eq. 3-10).
    Returns (violations_list, violation_magnitudes_list).
    """
    tol = 1e-5
    eps = 1e-5

    J, T, K = inst['J'], inst['T'], inst['K']
    K_j = inst['K_j']
    J_k = inst['J_k']
    sd = inst['sd']
    st = inst['st']
    pt = inst['pt']
    cap = inst['cap']

    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 (Eq. 3): Single-sourcing
    # sum_{k in K_j} U^k_j = 1, for all j
    # =========================================================================
    for j in range(J):
        lhs_val = sum(U.get((j, k), 0) for k in K_j[j])
        rhs_val = 1.0
        viol = abs(lhs_val - rhs_val)
        if viol > tol:
            violations.append(
                f"Constraint 1 (Single-sourcing): item j={j}, "
                f"sum_k U^k_j = {lhs_val}, expected 1"
            )
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": viol,
                "normalizer": normalizer,
                "ratio": viol / normalizer,
            })

    # =========================================================================
    # Constraint 2 (Eq. 4): Plant opening requirement
    # U^k_j <= Z^k, for all j, k in K_j
    # =========================================================================
    for j in range(J):
        for k in K_j[j]:
            lhs_val = U.get((j, k), 0)
            rhs_val = Z.get(k, 0)
            viol = lhs_val - rhs_val  # <= constraint
            if viol > tol:
                violations.append(
                    f"Constraint 2 (Plant opening): item j={j}, plant k={k}, "
                    f"U^k_j={lhs_val} > Z^k={rhs_val}"
                )
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": viol,
                    "normalizer": normalizer,
                    "ratio": viol / normalizer,
                })

    # =========================================================================
    # Constraint 3 (Eq. 5): Setup-location linking
    # Y^k_{jt} <= U^k_j, for all j, t, k in K_j
    # =========================================================================
    for j in range(J):
        for t in range(T):
            for k in K_j[j]:
                lhs_val = Y.get((j, k, t), 0)
                rhs_val = U.get((j, k), 0)
                viol = lhs_val - rhs_val
                if viol > tol:
                    violations.append(
                        f"Constraint 3 (Setup-location): item j={j}, plant k={k}, "
                        f"period t={t}, Y^k_{{jt}}={lhs_val} > U^k_j={rhs_val}"
                    )
                    normalizer = max(abs(rhs_val), eps)
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": viol,
                        "normalizer": normalizer,
                        "ratio": viol / normalizer,
                    })

    # =========================================================================
    # Constraints 4-7 involve W variables, which are not stored in solutions.
    # Solve an LP to find W satisfying constraints 4, 5, 6, 7 given Y, U, Z.
    # If infeasible, report violations.
    # =========================================================================
    W, w_infeas_msgs, w_objective = solve_w_feasibility(inst, Y, U, Z)

    if W is None:
        # LP is infeasible — W cannot be found for the given Y, U, Z.
        # We need to determine which constraints are violated.
        # Try relaxing each constraint group to identify the source.

        # Check if constraint 7 (setup forcing) alone makes it infeasible
        # by checking: for each (j,k,t) where Y=0, all W^k_{jt,ell} must be 0.
        # If Y=0 at all (j,k,t) for some j, then no production can happen
        # for item j, violating demand satisfaction (constraint 4).

        # Check demand coverage: for each item j, there must be at least one
        # period t and plant k with Y^k_{jt}=1 to produce item j.
        for j in range(J):
            has_setup = False
            for k in K_j[j]:
                for t in range(T):
                    if Y.get((j, k, t), 0) > 0.5:
                        has_setup = True
                        break
                if has_setup:
                    break
            if not has_setup:
                # No production setup for item j — constraint 4 violated
                lhs_val = 0.0
                rhs_val = 1.0
                viol = abs(lhs_val - rhs_val)
                violations.append(
                    f"Constraint 4 (Demand satisfaction): item j={j}, "
                    f"no setup exists (Y=0 for all k,t), demand cannot be met"
                )
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": viol,
                    "normalizer": normalizer,
                    "ratio": viol / normalizer,
                })

        # Now try LP without capacity to isolate capacity violations
        W_no_cap, _ = solve_w_no_capacity(inst, Y, U, Z)
        if W_no_cap is not None:
            # Capacity is the binding issue. Check capacity with the W from
            # the relaxed LP.
            for k in range(K):
                for t in range(T):
                    lhs_val = 0.0
                    for j in J_k[k]:
                        lhs_val += st[j][k][t] * Y.get((j, k, t), 0)
                        for ell in range(t, T):
                            lhs_val += (pt[j][k][t] * sd[j, t, ell]
                                        * W_no_cap.get((j, k, t, ell), 0))
                    rhs_val = cap[k][t] * Z.get(k, 0)
                    viol = lhs_val - rhs_val
                    if viol > tol:
                        violations.append(
                            f"Constraint 6 (Capacity): plant k={k}, period t={t}, "
                            f"usage={lhs_val:.4f} > capacity={rhs_val:.4f}"
                        )
                        normalizer = max(abs(rhs_val), eps)
                        violation_magnitudes.append({
                            "constraint": 6,
                            "lhs": lhs_val,
                            "rhs": rhs_val,
                            "raw_excess": viol,
                            "normalizer": normalizer,
                            "ratio": viol / normalizer,
                        })
        else:
            # Even without capacity constraint the LP is infeasible.
            # This means flow/demand constraints can't be satisfied with
            # the given Y (setup forcing). Report constraints 4,5,7.
            # We already checked demand coverage above.
            # Report a generic flow infeasibility.
            reported_4 = any(vm["constraint"] == 4 for vm in violation_magnitudes)
            if not reported_4:
                # The setup pattern doesn't allow feasible flow.
                # Constraint 5 (flow balance) or constraint 7 (setup forcing)
                # is the issue.
                violations.append(
                    "Constraint 5/7 (Flow/Setup forcing): "
                    "No feasible W exists for the given Y assignments — "
                    "the setup pattern cannot satisfy demand flow balance"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": 0.0,
                    "rhs": 0.0,
                    "raw_excess": 1.0,
                    "normalizer": eps,
                    "ratio": 1.0 / eps,
                })
    else:
        # W found — verify constraints 4-7 explicitly with computed W values

        # Constraint 4 (Eq. 6): sum_{t} sum_{k in K_j} W^k_{j,0,t} = 1
        for j in range(J):
            lhs_val = sum(W.get((j, k, 0, t), 0)
                          for k in K_j[j] for t in range(T))
            rhs_val = 1.0
            viol = abs(lhs_val - rhs_val)
            if viol > tol:
                violations.append(
                    f"Constraint 4 (Demand satisfaction): item j={j}, "
                    f"sum W^k_{{j,0,t}} = {lhs_val:.6f}, expected 1"
                )
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "raw_excess": viol,
                    "normalizer": normalizer,
                    "ratio": viol / normalizer,
                })

        # Constraint 5 (Eq. 7): Flow balance
        for j in range(J):
            for t in range(1, T):
                for k in K_j[j]:
                    lhs_val = sum(W.get((j, k, ell, t - 1), 0)
                                  for ell in range(t))
                    rhs_val = sum(W.get((j, k, t, ell), 0)
                                  for ell in range(t, T))
                    viol = abs(lhs_val - rhs_val)
                    if viol > tol:
                        violations.append(
                            f"Constraint 5 (Flow balance): item j={j}, "
                            f"plant k={k}, period t={t}, "
                            f"inflow={lhs_val:.6f} != outflow={rhs_val:.6f}"
                        )
                        normalizer = max(abs(rhs_val), eps)
                        violation_magnitudes.append({
                            "constraint": 5,
                            "lhs": lhs_val,
                            "rhs": rhs_val,
                            "raw_excess": viol,
                            "normalizer": normalizer,
                            "ratio": viol / normalizer,
                        })

        # Constraint 6 (Eq. 8): Capacity
        for k in range(K):
            for t in range(T):
                lhs_val = 0.0
                for j in J_k[k]:
                    lhs_val += st[j][k][t] * Y.get((j, k, t), 0)
                    for ell in range(t, T):
                        lhs_val += (pt[j][k][t] * sd[j, t, ell]
                                    * W.get((j, k, t, ell), 0))
                rhs_val = cap[k][t] * Z.get(k, 0)
                viol = lhs_val - rhs_val
                if viol > tol:
                    violations.append(
                        f"Constraint 6 (Capacity): plant k={k}, period t={t}, "
                        f"usage={lhs_val:.4f} > capacity={rhs_val:.4f}"
                    )
                    normalizer = max(abs(rhs_val), eps)
                    violation_magnitudes.append({
                        "constraint": 6,
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "raw_excess": viol,
                        "normalizer": normalizer,
                        "ratio": viol / normalizer,
                    })

        # Constraint 7 (Eq. 9): Setup forcing
        for j in range(J):
            for t in range(T):
                for k in K_j[j]:
                    lhs_val = sum(W.get((j, k, t, ell), 0)
                                  for ell in range(t, T))
                    rhs_val = float(Y.get((j, k, t), 0))
                    viol = lhs_val - rhs_val
                    if viol > tol:
                        violations.append(
                            f"Constraint 7 (Setup forcing): item j={j}, "
                            f"plant k={k}, period t={t}, "
                            f"sum_ell W={lhs_val:.6f} > Y={rhs_val}"
                        )
                        normalizer = max(abs(rhs_val), eps)
                        violation_magnitudes.append({
                            "constraint": 7,
                            "lhs": lhs_val,
                            "rhs": rhs_val,
                            "raw_excess": viol,
                            "normalizer": normalizer,
                            "ratio": viol / normalizer,
                        })

    # =========================================================================
    # Constraint 8 (Eq. 10): Variable domains
    # W >= 0 (already enforced by LP bounds)
    # Y, U in {0, 1}; Z in {0, 1}
    # =========================================================================
    for j in range(J):
        for k in K_j[j]:
            for t in range(T):
                y_val = Y.get((j, k, t), 0)
                viol = abs(y_val - round(y_val))
                if viol > tol:
                    violations.append(
                        f"Constraint 8 (Variable domain): Y^{k}_{{j={j},t={t}}} "
                        f"= {y_val}, not binary"
                    )
                    violation_magnitudes.append({
                        "constraint": 8,
                        "lhs": y_val,
                        "rhs": round(y_val),
                        "raw_excess": viol,
                        "normalizer": max(abs(round(y_val)), eps),
                        "ratio": viol / max(abs(round(y_val)), eps),
                    })

    for j in range(J):
        for k in K_j[j]:
            u_val = U.get((j, k), 0)
            viol = abs(u_val - round(u_val))
            if viol > tol:
                violations.append(
                    f"Constraint 8 (Variable domain): U^{k}_{{j={j}}} "
                    f"= {u_val}, not binary"
                )
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": u_val,
                    "rhs": round(u_val),
                    "raw_excess": viol,
                    "normalizer": max(abs(round(u_val)), eps),
                    "ratio": viol / max(abs(round(u_val)), eps),
                })

    for k in range(K):
        z_val = Z.get(k, 0)
        viol = abs(z_val - round(z_val))
        if viol > tol:
            violations.append(
                f"Constraint 8 (Variable domain): Z^{k} = {z_val}, not binary"
            )
            violation_magnitudes.append({
                "constraint": 8,
                "lhs": z_val,
                "rhs": round(z_val),
                "raw_excess": viol,
                "normalizer": max(abs(round(z_val)), eps),
                "ratio": viol / max(abs(round(z_val)), eps),
            })

    # Check W non-negativity if W was computed
    if W is not None:
        for (j, k, t, ell), w_val in W.items():
            if w_val < -tol:
                viol = abs(w_val)
                violations.append(
                    f"Constraint 8 (Variable domain): W^{k}_{{j={j},t={t},"
                    f"ell={ell}}} = {w_val:.6f} < 0"
                )
                rhs_val = 0.0
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": w_val,
                    "rhs": rhs_val,
                    "raw_excess": viol,
                    "normalizer": max(abs(rhs_val), eps),
                    "ratio": viol / max(abs(rhs_val), eps),
                })

    return violations, violation_magnitudes, w_objective


def solve_w_no_capacity(inst, Y, U, Z):
    """
    Solve LP for W without capacity constraints to isolate capacity violations.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    import numpy as np

    J, T, K = inst['J'], inst['T'], inst['K']
    K_j = inst['K_j']

    w_indices = {}
    idx = 0
    for j in range(J):
        for k in K_j[j]:
            for t in range(T):
                for ell in range(t, T):
                    w_indices[j, k, t, ell] = idx
                    idx += 1
    n_vars = idx

    if n_vars == 0:
        return {}, []

    c = np.zeros(n_vars)

    eq_rows = []
    eq_columns = []
    eq_values = []
    b_eq_rows = []
    ub_rows = []
    ub_columns = []
    ub_values = []
    b_ub_rows = []

    def add_eq(coefficients, rhs):
        row_index = len(b_eq_rows)
        for column, value in coefficients.items():
            eq_rows.append(row_index)
            eq_columns.append(column)
            eq_values.append(value)
        b_eq_rows.append(rhs)

    def add_ub(coefficients, rhs):
        row_index = len(b_ub_rows)
        for column, value in coefficients.items():
            ub_rows.append(row_index)
            ub_columns.append(column)
            ub_values.append(value)
        b_ub_rows.append(rhs)

    # Constraint 4
    for j in range(J):
        row = {}
        for k in K_j[j]:
            for t in range(T):
                key = (j, k, 0, t)
                if key in w_indices:
                    row[w_indices[key]] = 1.0
        add_eq(row, 1.0)

    # Constraint 5
    for j in range(J):
        for t in range(1, T):
            for k in K_j[j]:
                row = {}
                for ell in range(t):
                    key = (j, k, ell, t - 1)
                    if key in w_indices:
                        row[w_indices[key]] = 1.0
                for ell in range(t, T):
                    key = (j, k, t, ell)
                    if key in w_indices:
                        row[w_indices[key]] = -1.0
                add_eq(row, 0.0)

    # Constraint 7 (setup forcing, no capacity)
    for j in range(J):
        for t in range(T):
            for k in K_j[j]:
                row = {}
                for ell in range(t, T):
                    key = (j, k, t, ell)
                    if key in w_indices:
                        row[w_indices[key]] = 1.0
                add_ub(row, float(Y.get((j, k, t), 0)))

    A_eq = coo_matrix(
        (eq_values, (eq_rows, eq_columns)),
        shape=(len(b_eq_rows), n_vars),
    ).tocsr() if b_eq_rows else None
    b_eq = np.array(b_eq_rows) if b_eq_rows else None
    A_ub = coo_matrix(
        (ub_values, (ub_rows, ub_columns)),
        shape=(len(b_ub_rows), n_vars),
    ).tocsr() if b_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(0, None), method='highs', options={"presolve": True},
    )

    W = {}
    if result.success:
        for (j, k, t, ell), i in w_indices.items():
            W[j, k, t, ell] = result.x[i]
        return W, []
    else:
        return None, [result.message]


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CFLPP [NetR] formulation."
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the JSON solution file.')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write the feasibility result JSON.')
    args = parser.parse_args()

    # Load data
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

    # Parse instance
    inst = parse_instance(data)

    # Extract variables
    Y, U, Z = extract_variables(solution, inst)

    # Check feasibility
    violations, violation_magnitudes, w_objective = check_feasibility(
        inst, Y, U, Z
    )

    # Constraint 9 (Tier C, objective consistency): see
    # ``check_objective_consistency`` above for the formula and rationale.
    obj_v, obj_m = check_objective_consistency(
        inst, solution, Y, U, Z, w_objective, new_idx=9
    )
    violations.extend(obj_v)
    violation_magnitudes.extend(obj_m)

    # Build result
    violated_constraints = sorted(set(
        vm["constraint"] for vm in violation_magnitudes
    ))
    feasible = len(violations) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    # Write result
    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {feasible}")
    if violations:
        print(f"Violated constraints: {violated_constraints}")
        for v in violations:
            print(f"  - {v}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
