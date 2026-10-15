#!/usr/bin/env python3
"""
Feasibility checker for Defensive Drill 1 solutions (with objective recompute).

Paper: "Multi-vehicle Cooperative Control Using Mixed Integer Linear Programming"
       Earl & D'Andrea (2005)

Checks the released polygonal MILP semantics exactly. Radii are polygon
apothem values and obstacle checks occur at the N_u control-step boundaries,
matching gurobi_code.py and mathematical_formulation.md.

Constraints:
  1. Defender Dynamics + IC (Eq. 6) - linear discrete-time state transition
  2. Control-Input Polygon Feasibility - exact M_u half-spaces
  3. Auxiliary |u| constraints (Eq. 10) - checked only if zx/zy present
  4. Attacker Dynamics (Eq. 17) - linear motion while active
  5. Attacker Initial Conditions (Eq. 19) - start in attack mode at given pos
  6. Attacker State Machine - mode[k+1]=1 iff mode[k]=1 AND not in-zone AND not intercepted
  7. Defense Zone containment - exact M_dz half-spaces
  8. Interception - exact M_I half-spaces at interpolated defender positions
  9. Defender Zone Avoidance - exact released four-side obstacle polygon
 10. Objective Consistency (Tier C) - reported objective_value must equal
       the recomputed objective J = sum_{j,k>=1} gamma_j[k]
                                  + epsilon * sum_{i,k}(|u_xi[k]| + |u_yi[k]|)
       (when zx/zy are absent, the optimal z = |u| is used as the canonical
       interpretation; when present, the stored z is used directly).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('attackers', 'defenders', 'objective_value')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('attackers', 'defenders')
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



def check_feasibility(inst, sol):
    """Run all feasibility checks; return the result dict (no I/O)."""
    tol = 1e-5
    eps = 1e-5

    # Check if solution has variable data
    if 'defenders' not in sol or 'attackers' not in sol:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution contains no variable data (solver may have failed)."],
            "violation_magnitudes": []
        }

    # Instance parameters
    N_D = inst['N_D']
    N_A = inst['N_A']
    M_u = inst['M_u']
    M_I = inst['M_I']
    M_dz = inst['M_dz']
    N_u = inst['N_u']
    N_a = inst['N_a']
    R_dz = inst['R_dz']
    R_I = inst['R_I']
    T_a_val = inst['T_a']
    T_u_val = inst['T_u']
    defenders_inst = inst['defenders']
    attackers_inst = inst['attackers']

    # Inferred parameters (matching solver code, NOT SPECIFIED IN PAPER)
    H = 100.0
    eps_c = 1e-4
    M_o = 4
    N_o = N_u

    defenders_sol = sol['defenders']
    attackers_sol = sol['attackers']

    # Structural validation: the constraint checks below index defenders_sol[i]
    # for i in range(N_D), attackers_sol[j] for j in range(N_A), and trajectory
    # arrays at [k]/[k+1] up to N_u/N_a. A solution that omits agents or supplies
    # short arrays would otherwise raise IndexError/KeyError and crash the
    # checker (reported upstream as checker_error). Validate up front and return
    # a clean infeasible verdict instead.
    struct_errors = []
    if not isinstance(defenders_sol, list) or len(defenders_sol) != N_D:
        struct_errors.append(
            f"defenders: expected {N_D} entries, got "
            f"{len(defenders_sol) if isinstance(defenders_sol, list) else type(defenders_sol).__name__}")
    if not isinstance(attackers_sol, list) or len(attackers_sol) != N_A:
        struct_errors.append(
            f"attackers: expected {N_A} entries, got "
            f"{len(attackers_sol) if isinstance(attackers_sol, list) else type(attackers_sol).__name__}")

    if not struct_errors:
        # Per-agent key/length checks (only over the indices actually accessed).
        def_arr_full = ['x', 'y', 'xdot', 'ydot']   # indexed up to k+1 => need N_u+1
        def_arr_ctrl = ['ux', 'uy']                 # indexed up to k   => need N_u
        for i in range(N_D):
            ds = defenders_sol[i]
            if not isinstance(ds, dict):
                struct_errors.append(f"Defender {i}: expected object, got {type(ds).__name__}")
                continue
            for key in def_arr_full:
                v = ds.get(key)
                if not isinstance(v, list) or len(v) != N_u + 1:
                    struct_errors.append(
                        f"Defender {i} field '{key}': need >= {N_u + 1} values, "
                        f"got {len(v) if isinstance(v, list) else 'missing'}")
            for key in def_arr_ctrl:
                v = ds.get(key)
                if not isinstance(v, list) or len(v) != N_u:
                    struct_errors.append(
                        f"Defender {i} field '{key}': need >= {N_u} values, "
                        f"got {len(v) if isinstance(v, list) else 'missing'}")
        for j in range(N_A):
            asol = attackers_sol[j]
            if not isinstance(asol, dict):
                struct_errors.append(f"Attacker {j}: expected object, got {type(asol).__name__}")
                continue
            for key in ('p', 'q', 'a', 'gamma'):
                v = asol.get(key)
                if not isinstance(v, list) or len(v) != N_a + 1:
                    struct_errors.append(
                        f"Attacker {j} field '{key}': need >= {N_a + 1} values, "
                        f"got {len(v) if isinstance(v, list) else 'missing'}")
            delta = asol.get('delta')
            if (
                not isinstance(delta, list)
                or len(delta) != N_D
                or any(not isinstance(row, list) or len(row) != N_a + 1
                       for row in delta)
            ):
                struct_errors.append(
                    f"Attacker {j} field 'delta': need shape "
                    f"[{N_D}][{N_a + 1}]")

    if struct_errors:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Malformed solution structure: " + "; ".join(struct_errors[:10])],
            "violation_magnitudes": []
        }

    e_Tu = math.exp(-T_u_val)

    violation_magnitudes = []
    constraint_violations = {}

    def record(cidx, msg, lhs_val, rhs_val, raw_excess):
        normalizer = max(abs(rhs_val), eps)
        ratio = raw_excess / normalizer
        violation_magnitudes.append({
            "constraint": cidx,
            "lhs": float(lhs_val),
            "rhs": float(rhs_val),
            "raw_excess": float(raw_excess),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })
        constraint_violations.setdefault(cidx, []).append(msg)

    def check_eq(cidx, msg, actual, expected):
        diff = abs(actual - expected)
        if diff > tol:
            record(cidx, msg, actual, expected, diff)

    def check_leq(cidx, msg, lhs_val, rhs_val):
        excess = lhs_val - rhs_val
        if excess > tol:
            record(cidx, msg, lhs_val, rhs_val, excess)

    def check_geq(cidx, msg, lhs_val, rhs_val):
        excess = rhs_val - lhs_val
        if excess > tol:
            record(cidx, msg, lhs_val, rhs_val, excess)

    def rb(v):
        return round(float(v))

    # Serialized state indicators are part of the feasibility witness; do not
    # silently coerce fractional values through round().
    for j in range(N_A):
        asol = attackers_sol[j]
        for name in ('a', 'gamma'):
            for k, value in enumerate(asol[name]):
                numeric = float(value)
                nearest = min((0.0, 1.0), key=lambda target: abs(numeric - target))
                check_eq(
                    6,
                    f"Attacker {j} {name}[{k}]={numeric} is not binary",
                    numeric,
                    nearest,
                )
        for i, row in enumerate(asol['delta']):
            for k, value in enumerate(row):
                numeric = float(value)
                nearest = min((0.0, 1.0), key=lambda target: abs(numeric - target))
                check_eq(
                    6,
                    f"Attacker {j} delta[{i}][{k}]={numeric} is not binary",
                    numeric,
                    nearest,
                )

    def def_pos(d_sol, t):
        k_u = int(t / T_u_val)
        if k_u >= N_u:
            k_u = N_u - 1
        tau = t - k_u * T_u_val
        e_tau = math.exp(-tau)
        xa = (d_sol['x'][k_u]
              + (1.0 - e_tau) * d_sol['xdot'][k_u]
              + (tau - 1.0 + e_tau) * d_sol['ux'][k_u])
        ya = (d_sol['y'][k_u]
              + (1.0 - e_tau) * d_sol['ydot'][k_u]
              + (tau - 1.0 + e_tau) * d_sol['uy'][k_u])
        return xa, ya

    # ================================================================
    # Constraint 1: Defender Dynamics (Eq. 6) + Initial Conditions
    # ================================================================
    for i in range(N_D):
        di = defenders_inst[i]
        ds = defenders_sol[i]
        check_eq(1, f"Defender {i} IC x: {ds['x'][0]} != {di['x_s']}",
                 ds['x'][0], di['x_s'])
        check_eq(1, f"Defender {i} IC y: {ds['y'][0]} != {di['y_s']}",
                 ds['y'][0], di['y_s'])
        check_eq(1, f"Defender {i} IC xdot: {ds['xdot'][0]} != {di['xdot_s']}",
                 ds['xdot'][0], di['xdot_s'])
        check_eq(1, f"Defender {i} IC ydot: {ds['ydot'][0]} != {di['ydot_s']}",
                 ds['ydot'][0], di['ydot_s'])
        for k in range(N_u):
            exp_x = (ds['x'][k] + (1.0 - e_Tu) * ds['xdot'][k]
                     + (T_u_val - 1.0 + e_Tu) * ds['ux'][k])
            check_eq(1, f"Defender {i} dynamics x k={k}: {ds['x'][k+1]:.8f} != {exp_x:.8f}",
                     ds['x'][k+1], exp_x)
            exp_y = (ds['y'][k] + (1.0 - e_Tu) * ds['ydot'][k]
                     + (T_u_val - 1.0 + e_Tu) * ds['uy'][k])
            check_eq(1, f"Defender {i} dynamics y k={k}: {ds['y'][k+1]:.8f} != {exp_y:.8f}",
                     ds['y'][k+1], exp_y)
            exp_xd = e_Tu * ds['xdot'][k] + (1.0 - e_Tu) * ds['ux'][k]
            check_eq(1, f"Defender {i} dynamics xdot k={k}: {ds['xdot'][k+1]:.8f} != {exp_xd:.8f}",
                     ds['xdot'][k+1], exp_xd)
            exp_yd = e_Tu * ds['ydot'][k] + (1.0 - e_Tu) * ds['uy'][k]
            check_eq(1, f"Defender {i} dynamics ydot k={k}: {ds['ydot'][k+1]:.8f} != {exp_yd:.8f}",
                     ds['ydot'][k+1], exp_yd)

    # ================================================================
    # Constraint 2: Exact released control-input polygon (Eq. 8)
    # ================================================================
    for i in range(N_D):
        ds = defenders_sol[i]
        for k in range(N_u):
            ux, uy = ds['ux'][k], ds['uy'][k]
            rhs = math.cos(math.pi / M_u)
            for m in range(1, M_u + 1):
                projection = (
                    math.sin(2.0 * math.pi * m / M_u) * ux
                    + math.cos(2.0 * math.pi * m / M_u) * uy
                )
                check_leq(
                    2,
                    f"Defender {i} control polygon k={k}, side={m}: "
                    f"projection={projection:.6f} > {rhs:.6f}",
                    projection,
                    rhs,
                )

    # ================================================================
    # Constraint 3: Auxiliary |u| constraints (Eq. 10)
    # ================================================================
    has_z = len(defenders_sol) > 0 and 'zx' in defenders_sol[0]
    if has_z:
        for i in range(N_D):
            ds = defenders_sol[i]
            for k in range(N_u):
                zx, zy = ds['zx'][k], ds['zy'][k]
                ux, uy = ds['ux'][k], ds['uy'][k]
                check_leq(3, f"Defender {i} ux<=zx k={k}", ux, zx)
                check_leq(3, f"Defender {i} -ux<=zx k={k}", -ux, zx)
                check_leq(3, f"Defender {i} uy<=zy k={k}", uy, zy)
                check_leq(3, f"Defender {i} -uy<=zy k={k}", -uy, zy)
                if zx < -tol:
                    record(3, f"Defender {i} zx<0 k={k}", zx, 0.0, -zx)
                if zy < -tol:
                    record(3, f"Defender {i} zy<0 k={k}", zy, 0.0, -zy)

    # ================================================================
    # Constraint 4: Attacker Dynamics (Eq. 17)
    # ================================================================
    for j in range(N_A):
        ai = attackers_inst[j]
        asol = attackers_sol[j]
        vp, vq = ai['v_p'], ai['v_q']
        for k in range(N_a):
            ak = rb(asol['a'][k])
            exp_p = asol['p'][k] + vp * T_a_val * ak
            check_eq(4, f"Attacker {j} dynamics p k={k}: {asol['p'][k+1]:.8f} != {exp_p:.8f}",
                     asol['p'][k + 1], exp_p)
            exp_q = asol['q'][k] + vq * T_a_val * ak
            check_eq(4, f"Attacker {j} dynamics q k={k}: {asol['q'][k+1]:.8f} != {exp_q:.8f}",
                     asol['q'][k + 1], exp_q)

    # ================================================================
    # Constraint 5: Attacker Initial Conditions (Eq. 19)
    # ================================================================
    for j in range(N_A):
        ai = attackers_inst[j]
        asol = attackers_sol[j]
        check_eq(5, f"Attacker {j} IC p: {asol['p'][0]} != {ai['p_s']}",
                 asol['p'][0], ai['p_s'])
        check_eq(5, f"Attacker {j} IC q: {asol['q'][0]} != {ai['q_s']}",
                 asol['q'][0], ai['q_s'])
        check_eq(5, f"Attacker {j} IC a: {rb(asol['a'][0])} != 1",
                 float(rb(asol['a'][0])), 1.0)

    # ================================================================
    # Constraint 6: Attacker State Machine (self-consistency)
    # ================================================================
    for j in range(N_A):
        asol = attackers_sol[j]
        gamma_arr = asol.get('gamma')
        delta_mat = asol.get('delta')
        for k in range(1, N_a):
            ak = rb(asol['a'][k])
            ak1 = rb(asol['a'][k + 1])
            gm_stored = rb(gamma_arr[k])
            dc_stored = 1 if any(
                rb(delta_mat[i][k]) == 1 for i in range(N_D)
            ) else 0
            expected_ak1 = 1 if (ak == 1 and gm_stored == 0 and dc_stored == 0) else 0
            if ak1 != expected_ak1:
                record(6,
                       f"Attacker {j} state-machine k={k}: a[k]={ak}, "
                       f"gamma={gm_stored}, delta_comb={dc_stored} -> expected a[k+1]={expected_ak1}, got {ak1}",
                       float(ak1), float(expected_ak1), abs(ak1 - expected_ak1))

    # ================================================================
    # Constraint 7: Zone consistency under exact polygon inequalities
    # ================================================================
    for j in range(N_A):
        asol = attackers_sol[j]
        gamma_arr = asol.get('gamma')
        if gamma_arr is None:
            continue
        for k in range(1, N_a + 1):
            pk, qk = asol['p'][k], asol['q'][k]
            gm = rb(gamma_arr[k])
            projections = [
                math.sin(2.0 * math.pi * m / M_dz) * pk
                + math.cos(2.0 * math.pi * m / M_dz) * qk
                for m in range(1, M_dz + 1)
            ]
            if gm == 1:
                for m, projection in enumerate(projections, 1):
                    check_leq(
                        7,
                        f"Attacker {j} zone k={k}, side={m}: gamma=1 but "
                        f"projection={projection:.6f} > R_dz={R_dz}",
                        projection,
                        R_dz,
                    )
            elif max(projections) < R_dz + eps_c - tol:
                record(
                    7,
                    f"Attacker {j} zone k={k}: gamma=0 but no polygon side "
                    f"reaches R_dz+eps_c={R_dz + eps_c}",
                    max(projections),
                    R_dz + eps_c,
                    R_dz + eps_c - max(projections),
                )

    # ================================================================
    # Constraint 8: Interception consistency under exact polygon inequalities
    # ================================================================
    for j in range(N_A):
        asol = attackers_sol[j]
        delta_mat = asol.get('delta')
        if delta_mat is None:
            continue
        for k in range(1, N_a + 1):
            pk, qk = asol['p'][k], asol['q'][k]
            for i in range(N_D):
                ds = defenders_sol[i]
                t_ak = k * T_a_val
                xa_i, ya_i = def_pos(ds, t_ak)
                d = rb(delta_mat[i][k])
                projections = [
                    math.sin(2.0 * math.pi * m / M_I) * (pk - xa_i)
                    + math.cos(2.0 * math.pi * m / M_I) * (qk - ya_i)
                    for m in range(1, M_I + 1)
                ]
                if d == 1:
                    for m, projection in enumerate(projections, 1):
                        check_leq(
                            8,
                            f"Intercept i={i} j={j} k={k}, side={m}: "
                            f"delta=1 but projection={projection:.6f} > R_I={R_I}",
                            projection,
                            R_I,
                        )
                elif max(projections) < R_I + eps_c - tol:
                    record(
                        8,
                        f"Intercept i={i} j={j} k={k}: delta=0 but no "
                        f"polygon side reaches R_I+eps_c={R_I + eps_c}",
                        max(projections),
                        R_I + eps_c,
                        R_I + eps_c - max(projections),
                    )

    # ================================================================
    # Constraint 9: Defender avoidance of the released four-side polygon
    # ================================================================
    for i in range(N_D):
        ds = defenders_sol[i]
        for k in range(1, N_o + 1):
            xk, yk = ds['x'][k], ds['y'][k]
            max_projection = max(
                math.sin(2.0 * math.pi * m / M_o) * xk
                + math.cos(2.0 * math.pi * m / M_o) * yk
                for m in range(1, M_o + 1)
            )
            if max_projection < R_dz + eps_c - tol:
                record(9,
                       f"Defender {i} enters avoidance polygon at k={k}: "
                       f"max projection={max_projection:.6f} < "
                       f"R_dz+eps_c={R_dz + eps_c}",
                       max_projection, R_dz + eps_c,
                       R_dz + eps_c - max_projection)

    # ================================================================
    # Constraint 10: Objective Consistency (Tier C anti-exploit defense)
    # Recompute J = sum_{j,k>=1} gamma_j[k]
    #             + epsilon * sum_{i,k}(z_xi[k] + z_yi[k])
    # When zx/zy are absent from the solution, use z = |u| (the minimum
    # value of z given fixed u, which is what the solver picks at optimum
    # since z appears only in the obj with positive coefficient and the
    # only constraint is z >= |u|). This makes the recomputation a unique
    # deterministic function of the displayed variables, suitable for
    # strict equality checking against the reported objective_value.
    # ================================================================
    reported_raw = sol.get('objective_value')
    if reported_raw is not None:
        try:
            reported = float(reported_raw)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            epsilon = float(inst.get('epsilon', 0.0))
            sum_gamma = 0
            gamma_available = True
            for j in range(N_A):
                asol = attackers_sol[j]
                g = asol.get('gamma')
                if g is None:
                    gamma_available = False
                    break
                for k in range(1, N_a + 1):
                    if k < len(g):
                        sum_gamma += rb(g[k])
            if gamma_available:
                sum_z = 0.0
                z_available = True
                for i in range(N_D):
                    ds = defenders_sol[i]
                    ux_list = ds.get('ux')
                    uy_list = ds.get('uy')
                    if ux_list is None or uy_list is None:
                        z_available = False
                        break
                    zx_list = ds.get('zx')
                    zy_list = ds.get('zy')
                    for k in range(N_u):
                        if zx_list is not None and k < len(zx_list):
                            sum_z += float(zx_list[k])
                        elif k < len(ux_list):
                            sum_z += abs(float(ux_list[k]))
                        if zy_list is not None and k < len(zy_list):
                            sum_z += float(zy_list[k])
                        elif k < len(uy_list):
                            sum_z += abs(float(uy_list[k]))
                if z_available:
                    true_obj = float(sum_gamma) + epsilon * sum_z
                    diff = abs(reported - true_obj)
                    tol_obj = max(1e-6, 5e-6 * abs(true_obj))
                    if diff > tol_obj:
                        record(10,
                               f"Objective consistency violated: reported objective_value="
                               f"{reported} differs from recomputed "
                               f"sum_gamma + epsilon*sum(z|=|u|)={true_obj} "
                               f"(sum_gamma={sum_gamma}, epsilon={epsilon}, sum_z={sum_z}, "
                               f"|diff|={diff:.3g}, tol={tol_obj:.3g})",
                               reported, true_obj, diff)
        elif reported is not None and not math.isfinite(reported):
            # reported is +/-inf or NaN: cannot match any finite recomputed obj.
            # Compute a finite recomputed obj for the message; if all parts
            # are accessible, this still flags constraint 10.
            epsilon = float(inst.get('epsilon', 0.0))
            sum_gamma = 0
            for j in range(N_A):
                asol = attackers_sol[j]
                g = asol.get('gamma')
                if g is None:
                    sum_gamma = None
                    break
                for k in range(1, N_a + 1):
                    if k < len(g):
                        sum_gamma += rb(g[k])
            sum_z = 0.0
            if sum_gamma is not None:
                for i in range(N_D):
                    ds = defenders_sol[i]
                    ux_list = ds.get('ux')
                    uy_list = ds.get('uy')
                    if ux_list is None or uy_list is None:
                        sum_gamma = None
                        break
                    zx_list = ds.get('zx')
                    zy_list = ds.get('zy')
                    for k in range(N_u):
                        if zx_list is not None and k < len(zx_list):
                            sum_z += float(zx_list[k])
                        elif k < len(ux_list):
                            sum_z += abs(float(ux_list[k]))
                        if zy_list is not None and k < len(zy_list):
                            sum_z += float(zy_list[k])
                        elif k < len(uy_list):
                            sum_z += abs(float(uy_list[k]))
            if sum_gamma is not None:
                true_obj = float(sum_gamma) + epsilon * sum_z
                record(10,
                       f"Objective consistency violated: reported objective_value="
                       f"{reported} is not finite; recomputed obj={true_obj}",
                       reported, true_obj, float('inf'))

    # ================================================================
    # Aggregate and output
    # ================================================================
    violated_constraints = sorted(constraint_violations.keys())

    agg_violations = []
    for cidx in violated_constraints:
        msgs = constraint_violations[cidx]
        count = len(msgs)
        if count == 1:
            agg_violations.append(f"Constraint {cidx}: {msgs[0]}")
        else:
            agg_violations.append(
                f"Constraint {cidx}: {count} violations, e.g., {msgs[0]}")

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": agg_violations,
        "violation_magnitudes": violation_magnitudes if not feasible else []
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check feasibility of a candidate solution for Defensive Drill 1.")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the JSON solution file.')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write the JSON feasibility result.')
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)
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

    result = check_feasibility(inst, sol)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    feasible = result["feasible"]
    print(f"Feasibility: {feasible}")
    if not feasible:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
