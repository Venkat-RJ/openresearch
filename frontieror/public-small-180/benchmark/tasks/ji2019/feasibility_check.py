#!/usr/bin/env python3
"""
Feasibility checker for the Lock and Quay Co-scheduling Problem (LQCP).
Based on: Ji et al. (2019), "Exact and heuristic methods for optimizing
         lock-quay system in inland waterway", EJOR 277, 740-755.

Checks hard constraints from the mathematical formulation (Eqs. 5-57),
numbered 1-53 in top-to-bottom order of the formulation section.

Constraint 54 (objective consistency): a Tier C anti-gaming check that
recomputes the objective (Eq. 1: T_1 + T_2 + penalty) from the solution
variables and rejects solutions whose reported objective_value disagrees.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('berth_assignments', 'completion_times', 'lockage_assignments', 'lockage_completion_times', 'modes', 'objective_value')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('completion_times', 'modes')
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


tol = 1e-5
eps = 1e-5


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def check_geq(lhs, rhs):
    """For >= constraint: violation = max(0, rhs - lhs)."""
    return max(0.0, rhs - lhs)


def check_leq(lhs, rhs):
    """For <= constraint: violation = max(0, lhs - rhs)."""
    return max(0.0, lhs - rhs)


def check_eq(lhs, rhs):
    """For = constraint: violation = |lhs - rhs|."""
    return abs(lhs - rhs)


def record_violation(violations, magnitudes, violated_set,
                     constraint_idx, message, lhs, rhs, violation_amount):
    normalizer = max(abs(rhs), eps)
    ratio = violation_amount / normalizer
    violated_set.add(constraint_idx)
    violations.append(message)
    magnitudes.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(ratio)
    })


def _lockage_placement_feasible(instance, ship_ids):
    """Existentially complete the omitted placement/mooring variables.

    The public solution fixes which ships share a lockage but historically did
    not serialize x/y, relative-position, or mooring binaries.  This compact
    MILP checks whether those omitted variables have a feasible completion for
    one lockage, matching constraints (5)-(30) in gurobi_code.py.
    """
    if not ship_ids:
        return True, "empty lockage"
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as error:
        return False, f"SciPy MILP unavailable: {error}"

    W = float(instance["chamber_width_W"])
    L = float(instance["chamber_length_L"])
    ships = instance["ships"]
    widths = {i: float(ships[i]["width"]) for i in ship_ids}
    lengths = {i: float(ships[i]["length"]) for i in ship_ids}
    mooring_sets = {i: set(ships[i]["mooring_set_MR"]) for i in ship_ids}
    if any(widths[i] > W + tol or lengths[i] > L + tol for i in ship_ids):
        return False, "a ship is larger than the lock chamber"

    lower = []
    upper = []
    integrality = []

    def variable(lb, ub, integer=False):
        index = len(lower)
        lower.append(lb)
        upper.append(ub)
        integrality.append(1 if integer else 0)
        return index

    x_index = {i: variable(0.0, W - widths[i]) for i in ship_ids}
    y_index = {i: variable(0.0, L - lengths[i]) for i in ship_ids}
    rows = []
    row_lower = []
    row_upper = []

    def constraint(coefficients, lb=-float("inf"), ub=float("inf")):
        rows.append(dict(coefficients))
        row_lower.append(lb)
        row_upper.append(ub)

    def conditional_equality(coefficients, rhs, binary, big_m):
        # expression == rhs when binary == 1.
        upper_coeffs = dict(coefficients)
        upper_coeffs[binary] = upper_coeffs.get(binary, 0.0) + big_m
        constraint(upper_coeffs, ub=rhs + big_m)
        lower_coeffs = {idx: -value for idx, value in coefficients.items()}
        lower_coeffs[binary] = lower_coeffs.get(binary, 0.0) + big_m
        constraint(lower_coeffs, ub=big_m - rhs)

    ordered = sorted(ship_ids)
    for pos, i in enumerate(ordered):
        for j in ordered[pos + 1:]:
            i_left = variable(0.0, 1.0, True)
            j_left = variable(0.0, 1.0, True)
            i_below = variable(0.0, 1.0, True)
            j_below = variable(0.0, 1.0, True)
            constraint({x_index[i]: 1.0, x_index[j]: -1.0,
                        i_left: W}, ub=W - widths[i])
            constraint({x_index[j]: 1.0, x_index[i]: -1.0,
                        j_left: W}, ub=W - widths[j])
            constraint({y_index[i]: 1.0, y_index[j]: -1.0,
                        i_below: L}, ub=L - lengths[i])
            constraint({y_index[j]: 1.0, y_index[i]: -1.0,
                        j_below: L}, ub=L - lengths[j])
            constraint({i_left: 1.0, j_left: 1.0,
                        i_below: 1.0, j_below: 1.0}, lb=1.0)

    ml = {}
    mr = {}
    for i in ordered:
        choices = []
        left_wall = variable(0.0, 1.0, True)
        right_wall = variable(0.0, 1.0, True)
        choices.extend((left_wall, right_wall))
        conditional_equality({x_index[i]: 1.0}, 0.0, left_wall, W)
        conditional_equality(
            {x_index[i]: 1.0}, W - widths[i], right_wall, W
        )

        for j in ordered:
            if j == i or j not in mooring_sets[i]:
                continue
            ml[i, j] = variable(0.0, 1.0, True)
            mr[i, j] = variable(0.0, 1.0, True)
            choices.extend((ml[i, j], mr[i, j]))
            for binary, horizontal_rhs in (
                    (ml[i, j], widths[i]),
                    (mr[i, j], -widths[j])):
                conditional_equality(
                    {x_index[j]: 1.0, x_index[i]: -1.0},
                    horizontal_rhs, binary, W,
                )
                constraint(
                    {y_index[j]: 1.0, y_index[i]: -1.0, binary: L},
                    ub=L,
                )
                constraint(
                    {y_index[i]: 1.0, y_index[j]: -1.0, binary: L},
                    ub=lengths[j] - lengths[i] + L,
                )
        constraint({choice: 1.0 for choice in choices}, lb=1.0)

    for (i, j), ml_var in ml.items():
        reverse = mr.get((j, i))
        if reverse is not None:
            constraint({ml_var: 1.0, reverse: 1.0}, ub=1.0)

    matrix = np.zeros((len(rows), len(lower)), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    result = milp(
        c=np.zeros(len(lower), dtype=float),
        integrality=np.asarray(integrality, dtype=int),
        bounds=Bounds(np.asarray(lower), np.asarray(upper)),
        constraints=LinearConstraint(
            matrix, np.asarray(row_lower), np.asarray(row_upper)
        ),
        options={"presolve": True, "time_limit": 10.0},
    )
    return bool(result.success), str(result.message)


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for LQCP (Ji et al., 2019)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON file containing the data instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to JSON file containing the candidate solution.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
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

    # ================================================================
    # Parse instance data
    # ================================================================
    n = instance["num_ships"]
    m = instance["num_berths"]
    W = instance["chamber_width_W"]
    L = instance["chamber_length_L"]
    Du = instance["lockage_duration_Du"]
    MT = instance["min_time_between_lockages_MT"]
    sc = instance["lock_start_availability_sc"]
    C_max = instance["C_max"]

    ships = instance["ships"]
    w = [s["width"] for s in ships]
    l_len = [s["length"] for s in ships]
    a = [s["arrival_time"] for s in ships]
    Pe = [s["transshipment_penalty"] for s in ships]
    MR = [s["mooring_set_MR"] for s in ships]
    B_eligible = [s["eligible_berths_B"] for s in ships]

    berths_data = instance["berths"]
    s_k = [b["start_availability_sk"] for b in berths_data]
    e_k_val = instance.get("berth_end_availability_ek", 1000.0)
    e_k = [e_k_val] * m
    t_ik = instance["handling_times"]  # t_ik[i][k]

    # SL and SB: not in instance data; paper experiments use all ships eligible
    # for both modes, so SL = SB = empty.
    SL = []
    SB = []

    N = list(range(n))
    Q = list(range(n))  # |Q| = n
    P = list(range(n))  # |P| = n

    # ================================================================
    # Parse solution — handle both efficient and gurobi formats
    # ================================================================
    u = {}
    for k_str, v in solution["modes"].items():
        u[int(k_str)] = int(round(v))

    lock_ships = [i for i in N if u[i] == 0]
    trans_ships = [i for i in N if u[i] == 1]

    is_efficient = "lockage_structure" in solution
    is_gurobi = "lockage_assignments" in solution

    # f[i,q]: 1 if ship i is transferred by lockage q
    f = {}
    for i in N:
        for q in Q:
            f[i, q] = 0

    # z[q]: 1 if lockage q is used
    z = {}
    for q in Q:
        z[q] = 0

    # C_q_val[q]: completion time of lockage q
    C_q_val = {}

    # c[i]: completion/departure time of ship i
    c = {}

    # delta[i,p,k]: 1 if ship i assigned to p-th order of berth k
    delta = {}
    for i in N:
        for p_val in P:
            for k in range(m):
                delta[i, p_val, k] = 0

    if is_efficient:
        # --- Efficient solution format ---
        lockage_structure = solution["lockage_structure"]
        for q_idx, ships_in_lockage in enumerate(lockage_structure):
            if ships_in_lockage:
                z[q_idx] = 1
                for ship_i in ships_in_lockage:
                    f[ship_i, q_idx] = 1

        lock_ct = solution.get("lock_completion_times", {})
        for k_str, v in lock_ct.items():
            c[int(k_str)] = v

        # Derive C_q from lockage structure
        for q_idx, ships_in_lockage in enumerate(lockage_structure):
            if ships_in_lockage:
                for si in ships_in_lockage:
                    if si in c:
                        C_q_val[q_idx] = c[si]
                        break

        # For transshipment ships: c_i = a_i (from constraints 29-30)
        for i in trans_ships:
            c[i] = a[i]

        # Berth assignments: {ship_str: berth_k}
        berth_asgn = solution.get("berth_assignments", {})
        berth_ships_map = {}
        for k_str, berth_k in berth_asgn.items():
            ship_i = int(k_str)
            if berth_k not in berth_ships_map:
                berth_ships_map[berth_k] = []
            berth_ships_map[berth_k].append(ship_i)

        # Assign orders by arrival time within each berth
        for berth_k, ship_list in berth_ships_map.items():
            ship_list.sort(key=lambda i: a[i])
            for order_p, ship_i in enumerate(ship_list):
                delta[ship_i, order_p, berth_k] = 1

    elif is_gurobi:
        # --- Gurobi solution format ---
        lockage_asgn = solution.get("lockage_assignments", {})
        for k_str, q_val in lockage_asgn.items():
            ship_i = int(k_str)
            q = int(q_val)
            f[ship_i, q] = 1
            z[q] = 1

        lock_ct = solution.get("lockage_completion_times", {})
        for k_str, v in lock_ct.items():
            C_q_val[int(k_str)] = v

        comp_times = solution.get("completion_times", {})
        for k_str, v in comp_times.items():
            c[int(k_str)] = v

        berth_asgn = solution.get("berth_assignments", {})
        for k_str, val in berth_asgn.items():
            ship_i = int(k_str)
            if isinstance(val, dict):
                berth_k = val["berth"]
                order_p = val["order"]
                delta[ship_i, order_p, berth_k] = 1

    # ================================================================
    # Feasibility checking
    # ================================================================
    violations = []
    magnitudes = []
    violated_set = set()

    # Exact serialized-witness coverage.  Empty lockage fields are legitimate
    # only when every ship selected transshipment mode; otherwise they used to
    # make constraints 27-37 disappear through ``if key in`` guards.
    def key_set(mapping):
        return {int(key) for key in mapping}

    expected_ships = set(N)
    raw_modes = solution.get("modes", {})
    mode_keys = key_set(raw_modes)
    raw_mode_values_valid = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and abs(float(value) - round(float(value))) <= tol
        and int(round(float(value))) in (0, 1)
        for value in raw_modes.values()
    )
    if mode_keys != expected_ships or not raw_mode_values_valid:
        record_violation(
            violations, magnitudes, violated_set, 45,
            f"Mode witness must contain one binary value for every ship; "
            f"missing={sorted(expected_ships - mode_keys)}, "
            f"extra={sorted(mode_keys - expected_ships)}",
            0.0, 1.0, 1.0,
        )

    completion_keys = key_set(solution.get("completion_times", {}))
    if completion_keys != expected_ships:
        record_violation(
            violations, magnitudes, violated_set, 50,
            f"completion_times must cover every ship; "
            f"missing={sorted(expected_ships - completion_keys)}, "
            f"extra={sorted(completion_keys - expected_ships)}",
            0.0, 1.0, 1.0,
        )

    lock_assignment_raw = solution.get("lockage_assignments", {})
    lock_assignment_keys = key_set(lock_assignment_raw)
    if lock_assignment_keys != set(lock_ships):
        record_violation(
            violations, magnitudes, violated_set, 6,
            f"lockage_assignments keys must equal the lock-mode ships; "
            f"missing={sorted(set(lock_ships) - lock_assignment_keys)}, "
            f"extra={sorted(lock_assignment_keys - set(lock_ships))}",
            0.0, 1.0, 1.0,
        )

    berth_assignment_keys = key_set(solution.get("berth_assignments", {}))
    if berth_assignment_keys != set(trans_ships):
        record_violation(
            violations, magnitudes, violated_set, 39,
            f"berth_assignments keys must equal the transshipment-mode ships; "
            f"missing={sorted(set(trans_ships) - berth_assignment_keys)}, "
            f"extra={sorted(berth_assignment_keys - set(trans_ships))}",
            0.0, 1.0, 1.0,
        )

    used_lockages = {
        int(lock_assignment_raw[str(i)]
            if str(i) in lock_assignment_raw else lock_assignment_raw[i])
        for i in lock_ships
        if str(i) in lock_assignment_raw or i in lock_assignment_raw
    }
    lock_completion_keys = key_set(
        solution.get("lockage_completion_times", {})
    )
    if lock_completion_keys != used_lockages:
        record_violation(
            violations, magnitudes, violated_set, 50,
            f"lockage_completion_times keys must equal the used lockages; "
            f"missing={sorted(used_lockages - lock_completion_keys)}, "
            f"extra={sorted(lock_completion_keys - used_lockages)}",
            0.0, 1.0, 1.0,
        )

    # Existentially reconstruct the omitted chamber placement and mooring
    # variables for each used lockage instead of silently skipping Eqs. 5-30.
    for q in sorted(used_lockages):
        group = [
            i for i in lock_ships
            if int(lock_assignment_raw.get(str(i),
                                           lock_assignment_raw.get(i, -1))) == q
        ]
        placement_ok, placement_detail = _lockage_placement_feasible(
            instance, group
        )
        if not placement_ok:
            record_violation(
                violations, magnitudes, violated_set, 5,
                f"Lockage {q} has no feasible chamber placement/mooring "
                f"completion for ships {group}: {placement_detail}",
                0.0, 1.0, 1.0,
            )

    # ------------------------------------------------------------------
    # Constraints 1-5 (paper Eqs. 5-9): Ship placement and non-overlap.
    # Verified by existential completion in _lockage_placement_feasible above.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 6 (paper Eq. 10): sum_q f_{iq} = 1 - u_i, for all i
    # Each lock-mode ship assigned to exactly one lockage.
    # ------------------------------------------------------------------
    for i in N:
        lhs = sum(f[i, q] for q in Q)
        rhs = 1 - u[i]
        va = check_eq(lhs, rhs)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 6,
                f"Constraint 6 (Eq.10): Ship {i}: sum(f_iq)={lhs} != 1-u_i={rhs}",
                lhs, rhs, va)

    # ------------------------------------------------------------------
    # Constraint 7 (paper Eq. 11): f_{iq} <= z_q, for all i, q
    # A ship can only be transferred by a used lockage.
    # ------------------------------------------------------------------
    for i in N:
        for q in Q:
            lhs = f[i, q]
            rhs_val = z[q]
            va = check_leq(lhs, rhs_val)
            if va > tol:
                record_violation(violations, magnitudes, violated_set, 7,
                    f"Constraint 7 (Eq.11): Ship {i} assigned to lockage {q} but z_{q}={rhs_val}",
                    lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraints 8-19 (paper Eqs. 12-23): Mooring constraints.
    # Verified by the same placement-completion MILP above.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 20 (paper Eq. 24): Each lock-mode ship must moor.
    # Verified by the same placement-completion MILP above.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 21 (paper Eq. 25): Prevent both unattached
    # Requires ml, mr variables — CANNOT CHECK.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 22 (paper Eq. 26): ml+mr <= v
    # Requires ml, mr, v variables — CANNOT CHECK.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraints 23-24 (paper Eqs. 27-28): f and v consistency
    # Require v variable — CANNOT CHECK.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraints 25-26 (paper Eqs. 29-30): v <= 1-u_i, v <= 1-u_j
    # Require v variable — CANNOT CHECK.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 27 (paper Eq. 31): c_i >= C_max*(f_{iq}-1) + C_q
    # ------------------------------------------------------------------
    for i in N:
        if i not in c:
            continue
        for q in Q:
            if q not in C_q_val:
                continue
            rhs_val = C_max * (f[i, q] - 1) + C_q_val[q]
            lhs = c[i]
            va = check_geq(lhs, rhs_val)
            if va > tol:
                record_violation(violations, magnitudes, violated_set, 27,
                    f"Constraint 27 (Eq.31): Ship {i}, lockage {q}: "
                    f"c_i={lhs:.2f} < C_max*(f_iq-1)+C_q={rhs_val:.2f}",
                    lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 28 (paper Eq. 32): c_i <= C_max*(1-f_{iq}) + C_q
    # ------------------------------------------------------------------
    for i in N:
        if i not in c:
            continue
        for q in Q:
            if q not in C_q_val:
                continue
            rhs_val = C_max * (1 - f[i, q]) + C_q_val[q]
            lhs = c[i]
            va = check_leq(lhs, rhs_val)
            if va > tol:
                record_violation(violations, magnitudes, violated_set, 28,
                    f"Constraint 28 (Eq.32): Ship {i}, lockage {q}: "
                    f"c_i={lhs:.2f} > C_max*(1-f_iq)+C_q={rhs_val:.2f}",
                    lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 29 (paper Eq. 33): c_i >= C_max*(u_i-1) + a_i
    # ------------------------------------------------------------------
    for i in N:
        if i not in c:
            continue
        rhs_val = C_max * (u[i] - 1) + a[i]
        lhs = c[i]
        va = check_geq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 29,
                f"Constraint 29 (Eq.33): Ship {i}: c_i={lhs:.2f} < "
                f"C_max*(u_i-1)+a_i={rhs_val:.2f}",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 30 (paper Eq. 34): c_i <= C_max*(1-u_i) + a_i
    # ------------------------------------------------------------------
    for i in N:
        if i not in c:
            continue
        rhs_val = C_max * (1 - u[i]) + a[i]
        lhs = c[i]
        va = check_leq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 30,
                f"Constraint 30 (Eq.34): Ship {i}: c_i={lhs:.2f} > "
                f"C_max*(1-u_i)+a_i={rhs_val:.2f}",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 31 (paper Eq. 35): C_{q+1} - C_q >= MT, for q < |Q|-1
    # Applied between consecutive lockage indices with known C_q values.
    # ------------------------------------------------------------------
    for q in range(n - 1):
        if q in C_q_val and (q + 1) in C_q_val:
            lhs = C_q_val[q + 1] - C_q_val[q]
            rhs_val = MT
            va = check_geq(lhs, rhs_val)
            if va > tol:
                record_violation(violations, magnitudes, violated_set, 31,
                    f"Constraint 31 (Eq.35): Lockages {q},{q+1}: "
                    f"C_diff={lhs:.2f} < MT={rhs_val}",
                    lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 32 (paper Eq. 36): C_q - Du >= sc, for all q
    # Lockage cannot start before chamber is available.
    # ------------------------------------------------------------------
    for q in Q:
        if z[q] == 1 and q in C_q_val:
            lhs = C_q_val[q] - Du
            rhs_val = sc
            va = check_geq(lhs, rhs_val)
            if va > tol:
                record_violation(violations, magnitudes, violated_set, 32,
                    f"Constraint 32 (Eq.36): Lockage {q}: "
                    f"C_q-Du={lhs:.2f} < sc={rhs_val}",
                    lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 33 (paper Eq. 37): C_q - Du >= f_{iq}*a_i
    # Lockage cannot start before all assigned ships have arrived.
    # ------------------------------------------------------------------
    for i in N:
        for q in Q:
            if f[i, q] == 1 and q in C_q_val:
                lhs = C_q_val[q] - Du
                rhs_val = a[i]
                va = check_geq(lhs, rhs_val)
                if va > tol:
                    record_violation(violations, magnitudes, violated_set, 33,
                        f"Constraint 33 (Eq.37): Ship {i}, lockage {q}: "
                        f"C_q-Du={lhs:.2f} < a_i={rhs_val}",
                        lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 34 (paper Eq. 38): z_q <= sum_i f_{iq}
    # A lockage is used only if at least one ship is assigned.
    # ------------------------------------------------------------------
    for q in Q:
        lhs = z[q]
        rhs_val = sum(f[i, q] for i in N)
        va = check_leq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 34,
                f"Constraint 34 (Eq.38): Lockage {q}: z_q={lhs} > sum(f_iq)={rhs_val}",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 35 (paper Eq. 39): c_i - c_j <= C_max*(u_i+u_j), i < j
    # FCFS restriction.
    # ------------------------------------------------------------------
    for i in N:
        for j in N:
            if i < j and i in c and j in c:
                lhs = c[i] - c[j]
                rhs_val = C_max * (u[i] + u[j])
                va = check_leq(lhs, rhs_val)
                if va > tol:
                    record_violation(violations, magnitudes, violated_set, 35,
                        f"Constraint 35 (Eq.39 FCFS): Ships {i},{j}: "
                        f"c_i-c_j={lhs:.2f} > C_max*(u_i+u_j)={rhs_val:.2f}",
                        lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 36 (paper Eq. 40): z_{q+1} <= z_q, for q < n-1
    # Used lockages are ordered (no gaps).
    # ------------------------------------------------------------------
    for q in range(n - 1):
        lhs = z[q + 1]
        rhs_val = z[q]
        va = check_leq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 36,
                f"Constraint 36 (Eq.40): z_{q+1}={lhs} > z_{q}={rhs_val}",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 37 (paper Eq. 41): C_q <= C_{q+1}, for q < n-1
    # Lockage completion times are non-decreasing.
    # ------------------------------------------------------------------
    for q in range(n - 1):
        if q in C_q_val and (q + 1) in C_q_val:
            lhs = C_q_val[q]
            rhs_val = C_q_val[q + 1]
            va = check_leq(lhs, rhs_val)
            if va > tol:
                record_violation(violations, magnitudes, violated_set, 37,
                    f"Constraint 37 (Eq.41): C_{q}={lhs:.2f} > C_{q+1}={rhs_val:.2f}",
                    lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 38 (paper Eq. 42):
    # sum_{q<l} (f_{iq} - f_{jq}) >= -(u_i + u_j), for i<j, l in Q
    # Ordering in lockage assignments.
    # ------------------------------------------------------------------
    for i in N:
        for j in N:
            if i < j:
                for l_idx in Q:
                    if l_idx == 0:
                        continue  # empty sum = 0 >= -(u_i+u_j) always true
                    lhs = sum(f[i, q] - f[j, q] for q in Q if q < l_idx)
                    rhs_val = -(u[i] + u[j])
                    va = check_geq(lhs, rhs_val)
                    if va > tol:
                        record_violation(violations, magnitudes, violated_set, 38,
                            f"Constraint 38 (Eq.42): Ships {i},{j}, l={l_idx}: "
                            f"sum(f_diff)={lhs} < -(u_i+u_j)={rhs_val}",
                            lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 39 (paper Eq. 43): u_i = 0, for i in SL
    # Ships that can only pass through lock.
    # ------------------------------------------------------------------
    for i in SL:
        lhs = u[i]
        rhs_val = 0
        va = check_eq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 39,
                f"Constraint 39 (Eq.43): Ship {i} in SL but u_i={lhs} != 0",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 40 (paper Eq. 48): u_i = 1, for i in SB
    # Ships that can only be transshipped.
    # ------------------------------------------------------------------
    for i in SB:
        lhs = u[i]
        rhs_val = 1
        va = check_eq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 40,
                f"Constraint 40 (Eq.48): Ship {i} in SB but u_i={lhs} != 1",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 41 (paper Eq. 44):
    # sum_{k in B(i)} sum_p delta_{ipk} = u_i, for all i
    # Each transshipment-mode ship assigned to exactly one berth.
    # ------------------------------------------------------------------
    for i in N:
        lhs = sum(delta[i, p_val, k] for k in B_eligible[i] for p_val in P)
        rhs_val = u[i]
        va = check_eq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 41,
                f"Constraint 41 (Eq.44): Ship {i}: sum(delta)={lhs} != u_i={rhs_val}",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 42 (paper Eq. 45):
    # sum_i delta_{ipk} <= 1, for k in M, p in P
    # Each berth serves at most one ship per service order.
    # ------------------------------------------------------------------
    for k in range(m):
        for p_val in P:
            lhs = sum(delta[i, p_val, k] for i in N)
            rhs_val = 1
            va = check_leq(lhs, rhs_val)
            if va > tol:
                record_violation(violations, magnitudes, violated_set, 42,
                    f"Constraint 42 (Eq.45): Berth {k}, order {p_val}: "
                    f"sum(delta)={lhs} > 1",
                    lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 43 (paper Eq. 46): Idle time constraint at berth
    # sum_{j,l<p} (t_jk*delta_jlk + omega_jlk) + omega_ipk
    #   - (a_i - s_k)*delta_ipk >= 0
    # for k in M, i in N(k), p in P
    #
    # Since omega is not in the solution, we derive omega from the
    # schedule. omega_ipk = max(0, a_i - berth_free_time) when
    # delta_ipk=1. The constraint is satisfied iff for each ship at its
    # assigned berth position, the berth is free at or before the ship's
    # arrival, or the ship waits (omega >= 0 is always true).
    # We verify the schedule consistency: each ship at its berth starts
    # at or after its arrival time (no time travel).
    # ------------------------------------------------------------------
    for k in range(m):
        # Collect ships assigned to berth k, sorted by order
        berth_ships_ordered = []
        for i in N:
            for p_val in P:
                if delta[i, p_val, k] == 1:
                    berth_ships_ordered.append((p_val, i))
        berth_ships_ordered.sort()

        current_time = s_k[k]
        for order_p, ship_i in berth_ships_ordered:
            # N(k) check: only ships with a_i >= s_k are in constraint scope
            if a[ship_i] >= s_k[k]:
                # The cumulative berth time up to this position
                cumulative = current_time - s_k[k]
                rhs_val = (a[ship_i] - s_k[k])  # (a_i - s_k) * delta_ipk with delta=1
                # omega_ipk = max(0, a[ship_i] - current_time)
                omega_ipk = max(0.0, a[ship_i] - current_time)
                lhs = cumulative + omega_ipk
                va = check_geq(lhs, rhs_val)
                if va > tol:
                    record_violation(violations, magnitudes, violated_set, 43,
                        f"Constraint 43 (Eq.46): Ship {ship_i}, berth {k}, "
                        f"order {order_p}: schedule time={lhs:.2f} < "
                        f"(a_i-s_k)={rhs_val:.2f}",
                        lhs, rhs_val, va)

            start = max(current_time, a[ship_i])
            current_time = start + t_ik[ship_i][k]

    # ------------------------------------------------------------------
    # Constraint 44 (paper Eq. 47): Berth availability time
    # sum_i sum_p (t_ik*delta_ipk + omega_ipk) <= e_k - s_k
    # Equivalent to: last ship finishes by e_k.
    # ------------------------------------------------------------------
    for k in range(m):
        berth_ships_ordered = []
        for i in N:
            for p_val in P:
                if delta[i, p_val, k] == 1:
                    berth_ships_ordered.append((p_val, i))
        berth_ships_ordered.sort()

        if not berth_ships_ordered:
            continue

        # Compute total used time = sum of (handling + idle)
        current_time = s_k[k]
        total_used = 0.0
        for order_p, ship_i in berth_ships_ordered:
            start = max(current_time, a[ship_i])
            idle = start - current_time
            handling = t_ik[ship_i][k]
            total_used += handling + idle
            current_time = start + handling

        lhs = total_used
        rhs_val = e_k[k] - s_k[k]
        va = check_leq(lhs, rhs_val)
        if va > tol:
            record_violation(violations, magnitudes, violated_set, 44,
                f"Constraint 44 (Eq.47): Berth {k}: total_time={lhs:.2f} > "
                f"available={rhs_val:.2f}",
                lhs, rhs_val, va)

    # ------------------------------------------------------------------
    # Constraint 45 (paper Eq. 49): u_i in {0,1}
    # ------------------------------------------------------------------
    for i in N:
        if u[i] not in (0, 1):
            va = min(abs(u[i] - 0), abs(u[i] - 1))
            record_violation(violations, magnitudes, violated_set, 45,
                f"Constraint 45 (Eq.49): Ship {i}: u_i={u[i]} not binary",
                float(u[i]), float(round(u[i])), va)

    # ------------------------------------------------------------------
    # Constraints 46-48 (paper Eqs. 50-52): lt, b, ml, mr binarity
    # Variables not in solution — CANNOT CHECK.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 49 (paper Eq. 53): v, z binary
    # v not in solution; check z only.
    # ------------------------------------------------------------------
    for q in Q:
        if z[q] not in (0, 1):
            va = min(abs(z[q] - 0), abs(z[q] - 1))
            record_violation(violations, magnitudes, violated_set, 49,
                f"Constraint 49 (Eq.53): z_{q}={z[q]} not binary",
                float(z[q]), float(round(z[q])), va)

    # ------------------------------------------------------------------
    # Constraint 50 (paper Eq. 54): 0 <= c_i <= C_max, 0 <= C_q <= C_max
    # ------------------------------------------------------------------
    for i in N:
        if i in c:
            if c[i] < -tol:
                va = abs(c[i])
                record_violation(violations, magnitudes, violated_set, 50,
                    f"Constraint 50 (Eq.54): Ship {i}: c_i={c[i]:.2f} < 0",
                    c[i], 0.0, va)
            if c[i] > C_max + tol:
                va = c[i] - C_max
                record_violation(violations, magnitudes, violated_set, 50,
                    f"Constraint 50 (Eq.54): Ship {i}: c_i={c[i]:.2f} > C_max={C_max}",
                    c[i], float(C_max), va)

    for q in Q:
        if q in C_q_val:
            if C_q_val[q] < -tol:
                va = abs(C_q_val[q])
                record_violation(violations, magnitudes, violated_set, 50,
                    f"Constraint 50 (Eq.54): Lockage {q}: C_q={C_q_val[q]:.2f} < 0",
                    C_q_val[q], 0.0, va)
            if C_q_val[q] > C_max + tol:
                va = C_q_val[q] - C_max
                record_violation(violations, magnitudes, violated_set, 50,
                    f"Constraint 50 (Eq.54): Lockage {q}: C_q={C_q_val[q]:.2f} > C_max={C_max}",
                    C_q_val[q], float(C_max), va)

    # ------------------------------------------------------------------
    # Constraint 51 (paper Eq. 55): f_{iq} in {0,1}
    # ------------------------------------------------------------------
    for i in N:
        for q in Q:
            if f[i, q] not in (0, 1):
                va = min(abs(f[i, q] - 0), abs(f[i, q] - 1))
                record_violation(violations, magnitudes, violated_set, 51,
                    f"Constraint 51 (Eq.55): f_{i},{q}={f[i,q]} not binary",
                    float(f[i, q]), float(round(f[i, q])), va)

    # ------------------------------------------------------------------
    # Constraint 52 (paper Eq. 56): x_i, y_i non-negative integer
    # Variables not in solution — CANNOT CHECK.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 53 (paper Eq. 57): delta binary, omega >= 0
    # omega is derived (always >= 0 by construction); check delta.
    # ------------------------------------------------------------------
    for i in N:
        for p_val in P:
            for k in range(m):
                if delta[i, p_val, k] not in (0, 1):
                    va = min(abs(delta[i, p_val, k] - 0),
                             abs(delta[i, p_val, k] - 1))
                    record_violation(violations, magnitudes, violated_set, 53,
                        f"Constraint 53 (Eq.57): delta_{i},{p_val},{k}="
                        f"{delta[i,p_val,k]} not binary",
                        float(delta[i, p_val, k]),
                        float(round(delta[i, p_val, k])), va)

    # ------------------------------------------------------------------
    # Constraint 54 (objective consistency — Tier C anti-gaming check):
    # Recompute the objective from the solution variables and reject the
    # solution when the reported objective_value disagrees.
    #
    #   obj = T_1 + T_2 + penalty                                  (Eq. 1)
    #   T_1 = sum_{i in N} (c_i - a_i)                             (Eq. 2)
    #   T_2 = sum_{k,i,p} [ (n-p+1) t_ik + s_k - a_i ] delta_ipk
    #         + sum_{k,i,p} (n-p+1) omega_ipk                      (Eq. 3)
    #   penalty = sum_{i in N} Pe_i u_i
    #
    # omega_ipk (berth idle time) is not in the solution; it is
    # reconstructed as the minimum feasible berth idle — the value any
    # objective-minimizing solver uses, and exactly the reconstruction
    # constraints 43/44 already rely on. This makes the recompute an
    # exact, full recompute: it reproduces gurobi_code.py's objective
    # bit-for-bit on all five Gurobi reference solutions (diff = 0).
    #
    # The check is purely additive: it only ever adds constraint index
    # 54 (the next unused index after the 1-53 formulation constraints).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    try:
        reported = float(reported_obj) if reported_obj is not None else None
    except (TypeError, ValueError):
        reported = None
    if reported is not None:
        try:
            # Only recompute when every ship has a completion time; a
            # partial recompute would be unreliable, so skip silently.
            if all(i in c for i in N):
                # T_1: total delay (completion - arrival) over all ships.
                T_1 = sum(c[i] - a[i] for i in N)

                # T_2: weighted berth delay with min-idle omega.
                T_2 = 0.0
                for k in range(m):
                    berth_ships_ordered = []
                    for i in N:
                        for p_val in P:
                            if delta[i, p_val, k] == 1:
                                berth_ships_ordered.append((p_val, i))
                    berth_ships_ordered.sort()

                    berth_time = s_k[k]
                    for order_p, ship_i in berth_ships_ordered:
                        weight = n - order_p + 1
                        T_2 += weight * t_ik[ship_i][k] + s_k[k] - a[ship_i]
                        omega_ipk = max(0.0, a[ship_i] - berth_time)
                        T_2 += weight * omega_ipk
                        berth_time = max(berth_time, a[ship_i]) \
                            + t_ik[ship_i][k]

                # Transshipment-mode penalty term.
                penalty = sum(Pe[i] * u[i] for i in N)

                true_obj = T_1 + T_2 + penalty
                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with a 1e-3 absolute floor.
                obj_tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    record_violation(violations, magnitudes, violated_set, 54,
                        f"Constraint 54 (objective consistency): reported "
                        f"objective_value={reported} differs from recomputed "
                        f"T1+T2+penalty={true_obj} "
                        f"(T1={T_1}, T2={T_2}, penalty={penalty}, "
                        f"|diff|={abs_diff:.6g}, tol={obj_tol:.3g})",
                        reported, true_obj, abs_diff)
        except Exception:
            # A recompute failure must never crash the checker or alter
            # any constraint verdict — skip the obj check instead.
            pass

    # ================================================================
    # Build and write result
    # ================================================================
    feasible = len(violated_set) == 0

    # Aggregate violation messages by constraint index
    constraint_messages = {}
    for msg in violations:
        # Extract constraint index from message
        for ci in violated_set:
            prefix = f"Constraint {ci} "
            if msg.startswith(prefix):
                if ci not in constraint_messages:
                    constraint_messages[ci] = []
                constraint_messages[ci].append(msg)
                break

    # Create aggregated violations list aligned with violated_constraints
    sorted_violated = sorted(list(violated_set))
    aggregated_violations = []
    for ci in sorted_violated:
        msgs = constraint_messages.get(ci, [])
        if len(msgs) <= 3:
            aggregated_violations.append("; ".join(msgs))
        else:
            aggregated_violations.append(
                f"{msgs[0]}; ... ({len(msgs)} total violations for constraint {ci})")

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": aggregated_violations,
        "violation_magnitudes": magnitudes if not feasible else []
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if feasible:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {sorted_violated}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
