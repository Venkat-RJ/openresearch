"""
Feasibility checker for the energy systems capacity expansion planning model
from Jacobson et al. (2023) / Rahmaniani et al. (2018).

Checks all hard constraints from the mathematical formulation against a
candidate solution. Constraints are numbered 1-48 counting top to bottom
from the formulation section of math_model.txt.

Constraint mapping:
  1: (12a) New gen capacity limit          2: (12b) Retirement limit
  3: (12c) Non-retirable retirement=0      4: (12d) Total capacity definition
  5: (13a) New storage energy limit        6: (13b) Storage energy ret limit
  7: (13c) Non-retirable stor energy=0     8: (13d) Storage energy definition
  9: (13e) Min duration                   10: (13f) Max duration
 11: (14a) Transmission new cap limit     12: (14b) Transmission cap definition
 13: (15)  Power balance                  14: (16a) Capacity limit (non-UC)
 15: (16b) Storage withdrawal limit       16: (16c) Storage combined limit
 17: (16d) SOC <= energy capacity         18: (16e) Hydro level limit
 19: (16f) Charge headroom                20: (16g) Discharge floor
 21: (17a) Min output (plain)             22: (17b) Min output (hydro)
 23: (18a) Flow upper bound               24: (18b) Flow lower bound
 25: (19)  NSE limit                      26: (20a) SOC balance (interior)
 27: (20b) SOC balance (boundary)         28: (20c) Hydro balance (interior)
 29: (20d) Hydro balance (boundary)       30: (21a) Ramp up (non-UC, interior)
 31: (21b) Ramp down (non-UC, interior)   32: (21c) Ramp up (non-UC, boundary)
 33: (21d) Ramp down (non-UC, boundary)   34: (22a) Commit limit
 35: (22b) Start limit                    36: (22c) Shut limit
 37: (22d) UC min output                  38: (22e) UC max output
 39: (22f) UC transition (interior)       40: (22g) UC transition (boundary)
 41: (23a) UC ramp up (interior)          42: (23b) UC ramp down (interior)
 43: (23c) UC ramp up (boundary)          44: (23d) UC ramp down (boundary)
 45: (24a) Min up time                    46: (24b) Min down time
 47: (26)  RPS constraint                 48: (27a) CO2 constraint
 49: Objective consistency (trusted fixed-investment recourse optimization)
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solverless_witness import require_linked_variables, validate_complete_witness

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('investment_decisions', 'model_variables', 'objective_value', 'transmission_decisions')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('investment_decisions', 'model_variables', 'transmission_decisions')
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


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_violation(lhs, rhs, op):
    """
    Compute violation_amount for a constraint.
    op: '<=', '>=', or '='
    Returns violation_amount (>0 means violated beyond tol).
    """
    if op == '<=':
        return max(0.0, lhs - rhs)
    elif op == '>=':
        return max(0.0, rhs - lhs)
    elif op == '=':
        return abs(lhs - rhs)
    return 0.0


def record_violation(violations_list, magnitudes_list, constraint_idx, msg, lhs, rhs, violation_amount):
    """Record a violation if violation_amount > TOL."""
    if violation_amount > TOL:
        normalizer = max(abs(rhs), EPS)
        ratio = violation_amount / normalizer
        violations_list.append((constraint_idx, msg))
        magnitudes_list.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })


def parse_solution(sol):
    """
    Parse solution into a unified format.
    Returns dict with:
      - 'has_investment': bool
      - 'gen_inv': dict mapping generator_id -> {y_P_new, y_P_ret, y_P_total}
      - 'stor_inv': dict mapping generator_id -> {y_E_new, y_E_ret, y_E_total}
      - 'trans_inv': dict mapping line_id -> {y_F_new, y_F_total}
      - 'has_operational': bool
      - operational data dicts (if available)
    """
    result = {
        'has_investment': False,
        'gen_inv': {},
        'stor_inv': {},
        'trans_inv': {},
        'has_operational': False,
    }

    # Gurobi solution format
    if 'investment_decisions' in sol and sol['investment_decisions'] is not None:
        result['has_investment'] = True
        for entry in sol['investment_decisions']:
            gid = entry['generator_id']
            result['gen_inv'][gid] = {
                'y_P_new': entry.get('y_P_new', 0.0),
                'y_P_ret': entry.get('y_P_ret', 0.0),
                'y_P_total': entry.get('y_P_total_MW', 0.0),
            }
            if 'y_E_total_MWh' in entry:
                result['stor_inv'][gid] = {
                    'y_E_new': entry.get('y_E_new', 0.0),
                    'y_E_ret': entry.get('y_E_ret', 0.0),
                    'y_E_total': entry.get('y_E_total_MWh', 0.0),
                }
        if 'transmission_decisions' in sol and sol['transmission_decisions'] is not None:
            for entry in sol['transmission_decisions']:
                lid = entry['line_id']
                result['trans_inv'][lid] = {
                    'y_F_new': entry.get('y_F_new_MW', 0.0),
                    'y_F_total': entry.get('total_capacity_MW', 0.0),
                }

    # Efficient algorithm solution format
    elif 'investment' in sol and sol['investment'] is not None:
        result['has_investment'] = True
        inv = sol['investment']
        for key, entry in inv.items():
            if str(key).startswith('line_'):
                lid = entry['id']
                result['trans_inv'][lid] = {
                    'y_F_new': entry.get('new_capacity_MW', 0.0),
                    'y_F_total': entry.get('total_capacity_MW', 0.0),
                }
            else:
                gid = entry['id']
                result['gen_inv'][gid] = {
                    'y_P_new': entry.get('new_units', 0.0),
                    'y_P_ret': entry.get('retired_units', 0.0),
                    'y_P_total': entry.get('total_capacity_MW', 0.0),
                }
                if 'total_energy_capacity_MWh' in entry:
                    result['stor_inv'][gid] = {
                        'y_E_new': entry.get('new_energy_units', 0.0),
                        'y_E_ret': entry.get('retired_energy_units', 0.0),
                        'y_E_total': entry.get('total_energy_capacity_MWh', 0.0),
                    }

    # Check for operational data (per-timestep variables)
    if 'operational' in sol and sol['operational'] is not None:
        result['has_operational'] = True
        result['operational'] = sol['operational']

    return result


def check_investment_constraints(instance, parsed_sol):
    """Check constraints 1-12 (investment constraints)."""
    violations = []
    magnitudes = []
    generators = instance['generators']
    gen_by_id = {g['id']: g for g in generators}
    lines = instance['transmission_lines']
    line_by_id = {l['id']: l for l in lines}

    gen_inv = parsed_sol['gen_inv']
    stor_inv = parsed_sol['stor_inv']
    trans_inv = parsed_sol['trans_inv']

    # Identify sets
    G_STOR_ids = [g['id'] for g in generators if g['is_storage']]
    G_NONRET_ids = [g['id'] for g in generators if not g['can_retire']]
    G_NONRET_STOR_ids = [g['id'] for g in generators
                         if not g['can_retire'] and g['is_storage']]

    # --- Constraint 1 (12a): gamma_P * y_P_new <= max_capacity ---
    for g in generators:
        gid = g['id']
        if gid not in gen_inv:
            continue
        lhs = g['capacity_size_MW'] * gen_inv[gid]['y_P_new']
        rhs = g['max_capacity_MW']
        va = compute_violation(lhs, rhs, '<=')
        record_violation(violations, magnitudes, 1,
                         f"Constraint 1 (12a): Generator {gid} ({g['resource_type']}): "
                         f"gamma*y_P_new={lhs:.4f} > max_cap={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 2 (12b): gamma_P * y_P_ret <= existing_capacity ---
    for g in generators:
        gid = g['id']
        if gid not in gen_inv:
            continue
        lhs = g['capacity_size_MW'] * gen_inv[gid]['y_P_ret']
        rhs = g['existing_capacity_MW']
        va = compute_violation(lhs, rhs, '<=')
        record_violation(violations, magnitudes, 2,
                         f"Constraint 2 (12b): Generator {gid} ({g['resource_type']}): "
                         f"gamma*y_P_ret={lhs:.4f} > existing={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 3 (12c): y_P_ret = 0 for non-retirable ---
    for gid in G_NONRET_ids:
        if gid not in gen_inv:
            continue
        g = gen_by_id[gid]
        lhs = gen_inv[gid]['y_P_ret']
        rhs = 0.0
        va = compute_violation(lhs, rhs, '=')
        record_violation(violations, magnitudes, 3,
                         f"Constraint 3 (12c): Generator {gid} ({g['resource_type']}): "
                         f"y_P_ret={lhs:.4f} != 0 (non-retirable)",
                         lhs, rhs, va)

    # --- Constraint 4 (12d): y_P = existing + gamma*(new - ret) ---
    for g in generators:
        gid = g['id']
        if gid not in gen_inv:
            continue
        lhs = gen_inv[gid]['y_P_total']
        rhs = g['existing_capacity_MW'] + g['capacity_size_MW'] * (
            gen_inv[gid]['y_P_new'] - gen_inv[gid]['y_P_ret'])
        va = compute_violation(lhs, rhs, '=')
        record_violation(violations, magnitudes, 4,
                         f"Constraint 4 (12d): Generator {gid} ({g['resource_type']}): "
                         f"y_P_total={lhs:.4f} != computed={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 5 (13a): gamma_E * y_E_new <= max_storage_capacity ---
    for gid in G_STOR_ids:
        if gid not in stor_inv:
            continue
        g = gen_by_id[gid]
        lhs = g['storage_capacity_size_MWh'] * stor_inv[gid]['y_E_new']
        rhs = g['max_storage_capacity_MWh']
        va = compute_violation(lhs, rhs, '<=')
        record_violation(violations, magnitudes, 5,
                         f"Constraint 5 (13a): Storage {gid} ({g['resource_type']}): "
                         f"gamma_E*y_E_new={lhs:.4f} > max_E={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 6 (13b): gamma_E * y_E_ret <= existing_storage ---
    for gid in G_STOR_ids:
        if gid not in stor_inv:
            continue
        g = gen_by_id[gid]
        lhs = g['storage_capacity_size_MWh'] * stor_inv[gid]['y_E_ret']
        rhs = g['existing_storage_capacity_MWh']
        va = compute_violation(lhs, rhs, '<=')
        record_violation(violations, magnitudes, 6,
                         f"Constraint 6 (13b): Storage {gid} ({g['resource_type']}): "
                         f"gamma_E*y_E_ret={lhs:.4f} > existing_E={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 7 (13c): y_E_ret = 0 for non-retirable storage ---
    for gid in G_NONRET_STOR_ids:
        if gid not in stor_inv:
            continue
        g = gen_by_id[gid]
        lhs = stor_inv[gid]['y_E_ret']
        rhs = 0.0
        va = compute_violation(lhs, rhs, '=')
        record_violation(violations, magnitudes, 7,
                         f"Constraint 7 (13c): Storage {gid} ({g['resource_type']}): "
                         f"y_E_ret={lhs:.4f} != 0 (non-retirable)",
                         lhs, rhs, va)

    # --- Constraint 8 (13d): y_E = existing_E + gamma_E*(new_E - ret_E) ---
    for gid in G_STOR_ids:
        if gid not in stor_inv or gid not in gen_inv:
            continue
        g = gen_by_id[gid]
        lhs = stor_inv[gid]['y_E_total']
        rhs = g['existing_storage_capacity_MWh'] + g['storage_capacity_size_MWh'] * (
            stor_inv[gid]['y_E_new'] - stor_inv[gid]['y_E_ret'])
        va = compute_violation(lhs, rhs, '=')
        record_violation(violations, magnitudes, 8,
                         f"Constraint 8 (13d): Storage {gid} ({g['resource_type']}): "
                         f"y_E_total={lhs:.4f} != computed={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 9 (13e): min_duration * y_P <= y_E ---
    for gid in G_STOR_ids:
        if gid not in stor_inv or gid not in gen_inv:
            continue
        g = gen_by_id[gid]
        lhs = g['min_duration_MWh_per_MW'] * gen_inv[gid]['y_P_total']
        rhs = stor_inv[gid]['y_E_total']
        va = compute_violation(lhs, rhs, '<=')
        record_violation(violations, magnitudes, 9,
                         f"Constraint 9 (13e): Storage {gid} ({g['resource_type']}): "
                         f"min_dur*y_P={lhs:.4f} > y_E={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 10 (13f): max_duration * y_P >= y_E ---
    for gid in G_STOR_ids:
        if gid not in stor_inv or gid not in gen_inv:
            continue
        g = gen_by_id[gid]
        lhs = g['max_duration_MWh_per_MW'] * gen_inv[gid]['y_P_total']
        rhs = stor_inv[gid]['y_E_total']
        va = compute_violation(lhs, rhs, '>=')
        record_violation(violations, magnitudes, 10,
                         f"Constraint 10 (13f): Storage {gid} ({g['resource_type']}): "
                         f"max_dur*y_P={lhs:.4f} < y_E={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 11 (14a): y_F_new <= max_new_capacity ---
    for l in lines:
        lid = l['id']
        if lid not in trans_inv:
            continue
        lhs = trans_inv[lid]['y_F_new']
        rhs = l['max_new_capacity_MW']
        va = compute_violation(lhs, rhs, '<=')
        record_violation(violations, magnitudes, 11,
                         f"Constraint 11 (14a): Line {lid}: "
                         f"y_F_new={lhs:.4f} > max_new={rhs:.4f}",
                         lhs, rhs, va)

    # --- Constraint 12 (14b): y_F = existing + y_F_new ---
    for l in lines:
        lid = l['id']
        if lid not in trans_inv:
            continue
        lhs = trans_inv[lid]['y_F_total']
        rhs = l['existing_capacity_MW'] + trans_inv[lid]['y_F_new']
        va = compute_violation(lhs, rhs, '=')
        record_violation(violations, magnitudes, 12,
                         f"Constraint 12 (14b): Line {lid}: "
                         f"y_F_total={lhs:.4f} != computed={rhs:.4f}",
                         lhs, rhs, va)

    return violations, magnitudes


def check_non_negativity(instance, parsed_sol):
    """Check non-negativity of investment variables."""
    violations = []
    magnitudes = []
    generators = instance['generators']
    gen_inv = parsed_sol['gen_inv']
    stor_inv = parsed_sol['stor_inv']
    trans_inv = parsed_sol['trans_inv']

    # y_P_new, y_P_ret, y_P >= 0
    for g in generators:
        gid = g['id']
        if gid not in gen_inv:
            continue
        for var_name, var_val in [('y_P_new', gen_inv[gid]['y_P_new']),
                                   ('y_P_ret', gen_inv[gid]['y_P_ret']),
                                   ('y_P_total', gen_inv[gid]['y_P_total'])]:
            if var_val < -TOL:
                # Map to constraint 4 (12d) as the capacity definition
                lhs = 0.0
                rhs = var_val
                va = abs(var_val)
                record_violation(violations, magnitudes, 4,
                                 f"Non-negativity: Generator {gid} {var_name}={var_val:.6f} < 0",
                                 var_val, 0.0, va)

    for gid in stor_inv:
        for var_name, var_val in [('y_E_new', stor_inv[gid]['y_E_new']),
                                   ('y_E_ret', stor_inv[gid]['y_E_ret']),
                                   ('y_E_total', stor_inv[gid]['y_E_total'])]:
            if var_val < -TOL:
                lhs = var_val
                rhs = 0.0
                va = abs(var_val)
                record_violation(violations, magnitudes, 8,
                                 f"Non-negativity: Storage {gid} {var_name}={var_val:.6f} < 0",
                                 var_val, 0.0, va)

    for lid in trans_inv:
        for var_name, var_val in [('y_F_new', trans_inv[lid]['y_F_new']),
                                   ('y_F_total', trans_inv[lid]['y_F_total'])]:
            if var_val < -TOL:
                lhs = var_val
                rhs = 0.0
                va = abs(var_val)
                record_violation(violations, magnitudes, 12,
                                 f"Non-negativity: Line {lid} {var_name}={var_val:.6f} < 0",
                                 var_val, 0.0, va)

    return violations, magnitudes


def check_operational_constraints(instance, parsed_sol):
    """
    Check operational constraints 13-48 if per-timestep data is available.
    Returns (violations, magnitudes).
    """
    violations = []
    magnitudes = []

    if not parsed_sol['has_operational']:
        return violations, magnitudes

    op = parsed_sol['operational']
    generators = instance['generators']
    gen_by_id = {g['id']: g for g in generators}
    lines = instance['transmission_lines']
    line_by_id = {l['id']: l for l in lines}
    nse_segments = instance['nse_segments']
    dims = instance['problem_dimensions']
    subperiods_info = instance['subperiods']
    num_zones = dims['num_zones']
    total_T = dims['total_timesteps']
    W = subperiods_info['num_subperiods']
    delta_w = subperiods_info['hours_per_subperiod']
    weights = subperiods_info['subperiod_weights']
    avail_data = instance['availability_profiles']
    demand_data = instance['demand']
    policy = instance['policy']

    gen_inv = parsed_sol['gen_inv']
    stor_inv = parsed_sol['stor_inv']
    trans_inv = parsed_sol['trans_inv']

    # Build sigma
    sigma = {}
    for g in generators:
        gid_str = str(g['id'])
        if gid_str in avail_data:
            prof = avail_data[gid_str]
            if prof['type'] == 'timeseries':
                sigma[g['id']] = prof['values']
            else:
                sigma[g['id']] = [prof['value']] * total_T
        else:
            sigma[g['id']] = [1.0] * total_T

    # Build alpha
    alpha = [0.0] * total_T
    for w in range(W):
        for t in range(w * delta_w, (w + 1) * delta_w):
            alpha[t] = weights[w] / delta_w

    # Extract operational variables
    x_inj = op.get('x_inj', {})
    x_wdw = op.get('x_wdw', {})
    x_nse = op.get('x_nse', {})
    x_soc = op.get('x_soc', {})
    x_lvl = op.get('x_lvl', {})
    x_spill = op.get('x_spill', {})
    x_flow = op.get('x_flow', {})
    x_commit = op.get('x_commit', {})
    x_start = op.get('x_start', {})
    x_shut = op.get('x_shut', {})
    x_rps = op.get('x_rps', {})
    x_co2 = op.get('x_co2', {})

    # Helper to get operational value
    def get_val(d, key, default=0.0):
        return d.get(str(key), d.get(key, default))

    def get_val2(d, key1, key2, default=0.0):
        k = f"{key1},{key2}"
        return d.get(k, d.get((key1, key2), default))

    def get_val3(d, key1, key2, key3, default=0.0):
        k = f"{key1},{key2},{key3}"
        return d.get(k, d.get((key1, key2, key3), default))

    # Generator sets
    G_UC_ids = [g['id'] for g in generators if g['is_UC']]
    G_STOR_ids = [g['id'] for g in generators if g['is_storage']]
    G_HYDRO_ids = [g['id'] for g in generators if g['is_hydro']]
    G_nonUC_ids = [g['id'] for g in generators if not g['is_UC']]
    G_PLAIN_ids = [g['id'] for g in generators
                   if not g['is_UC'] and not g['is_storage'] and not g['is_hydro']]
    G_RPS_ids = [g['id'] for g in generators if g['is_RPS_qualifying']]

    gens_in_zone = {z: [] for z in range(1, num_zones + 1)}
    stor_in_zone = {z: [] for z in range(1, num_zones + 1)}
    for g in generators:
        gens_in_zone[g['zone']].append(g['id'])
        if g['is_storage']:
            stor_in_zone[g['zone']].append(g['id'])

    I_in = {z: [] for z in range(1, num_zones + 1)}
    I_out = {z: [] for z in range(1, num_zones + 1)}
    for l in lines:
        I_out[l['from_zone']].append(l['id'])
        I_in[l['to_zone']].append(l['id'])

    def circ_prev(t, w):
        t_start = w * delta_w
        if t == t_start:
            return t_start + delta_w - 1
        return t - 1

    # --- Constraint 13 (15): Power balance ---
    if x_inj:
        for z in range(1, num_zones + 1):
            for t in range(total_T):
                lhs_val = 0.0
                for gid in gens_in_zone[z]:
                    lhs_val += get_val2(x_inj, gid, t)
                for gid in stor_in_zone[z]:
                    lhs_val -= get_val2(x_wdw, gid, t)
                for lid in I_out[z]:
                    lhs_val -= get_val2(x_flow, lid, t)
                for lid in I_in[z]:
                    lhs_val += get_val2(x_flow, lid, t)
                for seg in nse_segments:
                    s = seg['segment']
                    lhs_val += get_val3(x_nse, s, z, t)
                rhs_val = demand_data[str(z)][t]
                va = compute_violation(lhs_val, rhs_val, '=')
                record_violation(violations, magnitudes, 13,
                                 f"Constraint 13 (15): Power balance z={z} t={t}: "
                                 f"LHS={lhs_val:.4f} != demand={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 14 (16a): Capacity limit for non-UC generators ---
    if x_inj:
        for gid in G_PLAIN_ids + G_HYDRO_ids:
            g = gen_by_id[gid]
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for t in range(total_T):
                lhs_val = get_val2(x_inj, gid, t)
                rhs_val = sigma[gid][t] * yP
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 14,
                                 f"Constraint 14 (16a): Gen {gid} t={t}: "
                                 f"inj={lhs_val:.4f} > sigma*yP={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 15 (16b): Storage withdrawal limit ---
    if x_wdw:
        for gid in G_STOR_ids:
            g = gen_by_id[gid]
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for t in range(total_T):
                lhs_val = get_val2(x_wdw, gid, t)
                rhs_val = sigma[gid][t] * yP
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 15,
                                 f"Constraint 15 (16b): Storage {gid} t={t}: "
                                 f"wdw={lhs_val:.4f} > sigma*yP={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 16 (16c): Storage combined limit ---
    if x_inj and x_wdw:
        for gid in G_STOR_ids:
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for t in range(total_T):
                lhs_val = get_val2(x_inj, gid, t) + get_val2(x_wdw, gid, t)
                rhs_val = yP
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 16,
                                 f"Constraint 16 (16c): Storage {gid} t={t}: "
                                 f"inj+wdw={lhs_val:.4f} > yP={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 17 (16d): SOC <= energy capacity ---
    if x_soc:
        for gid in G_STOR_ids:
            yE = stor_inv.get(gid, {}).get('y_E_total', 0.0)
            for t in range(total_T):
                lhs_val = get_val2(x_soc, gid, t)
                rhs_val = yE
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 17,
                                 f"Constraint 17 (16d): Storage {gid} t={t}: "
                                 f"soc={lhs_val:.4f} > yE={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 18 (16e): Hydro level limit ---
    if x_lvl:
        for gid in G_HYDRO_ids:
            g = gen_by_id[gid]
            rho = g['duration_MWh_per_MW']
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for t in range(total_T):
                lhs_val = get_val2(x_lvl, gid, t)
                rhs_val = rho * yP
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 18,
                                 f"Constraint 18 (16e): Hydro {gid} t={t}: "
                                 f"lvl={lhs_val:.4f} > rho*yP={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 19 (16f): Charge headroom ---
    if x_wdw and x_soc:
        for gid in G_STOR_ids:
            g = gen_by_id[gid]
            eta_plus = g['charge_efficiency']
            yE = stor_inv.get(gid, {}).get('y_E_total', 0.0)
            for t in range(total_T):
                lhs_val = eta_plus * get_val2(x_wdw, gid, t)
                rhs_val = yE - get_val2(x_soc, gid, t)
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 19,
                                 f"Constraint 19 (16f): Storage {gid} t={t}: "
                                 f"eta+*wdw={lhs_val:.4f} > yE-soc={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 20 (16g): Discharge floor ---
    if x_inj and x_soc:
        for gid in G_STOR_ids:
            g = gen_by_id[gid]
            eta_minus = g['discharge_efficiency']
            for t in range(total_T):
                lhs_val = get_val2(x_inj, gid, t) / eta_minus
                rhs_val = get_val2(x_soc, gid, t)
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 20,
                                 f"Constraint 20 (16g): Storage {gid} t={t}: "
                                 f"inj/eta-={lhs_val:.4f} > soc={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 21 (17a): Min output (plain generators) ---
    if x_inj:
        for gid in G_PLAIN_ids:
            g = gen_by_id[gid]
            min_out = g.get('min_output_frac', 0.0)
            if min_out <= 0:
                continue
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for t in range(total_T):
                lhs_val = get_val2(x_inj, gid, t)
                rhs_val = min_out * yP
                va = compute_violation(lhs_val, rhs_val, '>=')
                record_violation(violations, magnitudes, 21,
                                 f"Constraint 21 (17a): Gen {gid} t={t}: "
                                 f"inj={lhs_val:.4f} < min_out*yP={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 22 (17b): Min output (hydro) ---
    if x_inj:
        for gid in G_HYDRO_ids:
            g = gen_by_id[gid]
            min_out = g.get('min_output_frac', 0.0)
            if min_out <= 0:
                continue
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for t in range(total_T):
                inj = get_val2(x_inj, gid, t)
                spill = get_val2(x_spill, gid, t) if x_spill else 0.0
                lhs_val = inj + spill
                rhs_val = min_out * yP
                va = compute_violation(lhs_val, rhs_val, '>=')
                record_violation(violations, magnitudes, 22,
                                 f"Constraint 22 (17b): Hydro {gid} t={t}: "
                                 f"inj+spill={lhs_val:.4f} < min_out*yP={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 23 (18a): Flow upper bound ---
    if x_flow:
        for l in lines:
            lid = l['id']
            yF = trans_inv.get(lid, {}).get('y_F_total', l['existing_capacity_MW'])
            for t in range(total_T):
                lhs_val = get_val2(x_flow, lid, t)
                rhs_val = yF
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 23,
                                 f"Constraint 23 (18a): Line {lid} t={t}: "
                                 f"flow={lhs_val:.4f} > yF={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 24 (18b): Flow lower bound ---
    if x_flow:
        for l in lines:
            lid = l['id']
            yF = trans_inv.get(lid, {}).get('y_F_total', l['existing_capacity_MW'])
            for t in range(total_T):
                lhs_val = -get_val2(x_flow, lid, t)
                rhs_val = yF
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 24,
                                 f"Constraint 24 (18b): Line {lid} t={t}: "
                                 f"-flow={lhs_val:.4f} > yF={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 25 (19): NSE limit ---
    if x_nse:
        for seg in nse_segments:
            s = seg['segment']
            max_frac = seg['max_frac']
            for z in range(1, num_zones + 1):
                for t in range(total_T):
                    lhs_val = get_val3(x_nse, s, z, t)
                    rhs_val = max_frac * demand_data[str(z)][t]
                    va = compute_violation(lhs_val, rhs_val, '<=')
                    record_violation(violations, magnitudes, 25,
                                     f"Constraint 25 (19): NSE s={s} z={z} t={t}: "
                                     f"nse={lhs_val:.4f} > limit={rhs_val:.4f}",
                                     lhs_val, rhs_val, va)

    # --- Constraints 26-27 (20a-20b): Storage SOC balance ---
    if x_soc and x_inj and x_wdw:
        for gid in G_STOR_ids:
            g = gen_by_id[gid]
            eta_plus = g['charge_efficiency']
            eta_minus = g['discharge_efficiency']
            eta_disch = g['self_discharge_rate']
            for w_idx in range(W):
                t_start = w_idx * delta_w
                for t in range(t_start, t_start + delta_w):
                    t_prev = circ_prev(t, w_idx)
                    soc_t = get_val2(x_soc, gid, t)
                    soc_prev = get_val2(x_soc, gid, t_prev)
                    wdw_t = get_val2(x_wdw, gid, t)
                    inj_t = get_val2(x_inj, gid, t)
                    lhs_val = soc_t - soc_prev
                    rhs_val = eta_plus * wdw_t - inj_t / eta_minus - eta_disch * soc_prev
                    va = compute_violation(lhs_val, rhs_val, '=')
                    cstr_idx = 27 if t == t_start else 26
                    record_violation(violations, magnitudes, cstr_idx,
                                     f"Constraint {cstr_idx} ({'20b' if t == t_start else '20a'}): "
                                     f"SOC balance storage {gid} t={t}: "
                                     f"LHS={lhs_val:.6f} != RHS={rhs_val:.6f}",
                                     lhs_val, rhs_val, va)

    # --- Constraints 28-29 (20c-20d): Hydro reservoir balance ---
    if x_lvl and x_inj:
        for gid in G_HYDRO_ids:
            g = gen_by_id[gid]
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for w_idx in range(W):
                t_start = w_idx * delta_w
                for t in range(t_start, t_start + delta_w):
                    t_prev = circ_prev(t, w_idx)
                    lvl_t = get_val2(x_lvl, gid, t)
                    lvl_prev = get_val2(x_lvl, gid, t_prev)
                    inj_t = get_val2(x_inj, gid, t)
                    spill_t = get_val2(x_spill, gid, t) if x_spill else 0.0
                    lhs_val = lvl_t - lvl_prev
                    rhs_val = sigma[gid][t] * yP - inj_t - spill_t
                    va = compute_violation(lhs_val, rhs_val, '=')
                    cstr_idx = 29 if t == t_start else 28
                    record_violation(violations, magnitudes, cstr_idx,
                                     f"Constraint {cstr_idx} ({'20d' if t == t_start else '20c'}): "
                                     f"Hydro balance {gid} t={t}: "
                                     f"LHS={lhs_val:.6f} != RHS={rhs_val:.6f}",
                                     lhs_val, rhs_val, va)

    # --- Constraints 30-33 (21a-21d): Ramping for non-UC ---
    if x_inj:
        for gid in G_PLAIN_ids + G_HYDRO_ids:
            g = gen_by_id[gid]
            mu_up = g.get('ramp_up_frac_per_hr', 1.0)
            mu_dn = g.get('ramp_dn_frac_per_hr', 1.0)
            if mu_up >= 1.0 and mu_dn >= 1.0:
                continue
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for w_idx in range(W):
                t_start = w_idx * delta_w
                for t in range(t_start, t_start + delta_w):
                    t_prev = circ_prev(t, w_idx)
                    inj_t = get_val2(x_inj, gid, t)
                    inj_prev = get_val2(x_inj, gid, t_prev)
                    is_boundary = (t == t_start)
                    # Ramp up
                    if mu_up < 1.0:
                        lhs_val = inj_t - inj_prev
                        rhs_val = mu_up * yP
                        va = compute_violation(lhs_val, rhs_val, '<=')
                        cstr_idx = 32 if is_boundary else 30
                        record_violation(violations, magnitudes, cstr_idx,
                                         f"Constraint {cstr_idx} ({'21c' if is_boundary else '21a'}): "
                                         f"Ramp up gen {gid} t={t}: "
                                         f"delta={lhs_val:.4f} > limit={rhs_val:.4f}",
                                         lhs_val, rhs_val, va)
                    # Ramp down
                    if mu_dn < 1.0:
                        lhs_val = inj_prev - inj_t
                        rhs_val = mu_dn * yP
                        va = compute_violation(lhs_val, rhs_val, '<=')
                        cstr_idx = 33 if is_boundary else 31
                        record_violation(violations, magnitudes, cstr_idx,
                                         f"Constraint {cstr_idx} ({'21d' if is_boundary else '21b'}): "
                                         f"Ramp dn gen {gid} t={t}: "
                                         f"delta={lhs_val:.4f} > limit={rhs_val:.4f}",
                                         lhs_val, rhs_val, va)

    # --- Constraints 34-36 (22a-22c): UC commit/start/shut limits ---
    if x_commit:
        for gid in G_UC_ids:
            g = gen_by_id[gid]
            size_P = g['capacity_size_MW']
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            for t in range(total_T):
                # (22a)
                lhs_val = size_P * get_val2(x_commit, gid, t)
                rhs_val = yP
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 34,
                                 f"Constraint 34 (22a): UC {gid} t={t}: "
                                 f"gamma*commit={lhs_val:.4f} > yP={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)
                # (22b)
                lhs_val = size_P * get_val2(x_start, gid, t)
                va = compute_violation(lhs_val, yP, '<=')
                record_violation(violations, magnitudes, 35,
                                 f"Constraint 35 (22b): UC {gid} t={t}: "
                                 f"gamma*start={lhs_val:.4f} > yP={yP:.4f}",
                                 lhs_val, yP, va)
                # (22c)
                lhs_val = size_P * get_val2(x_shut, gid, t)
                va = compute_violation(lhs_val, yP, '<=')
                record_violation(violations, magnitudes, 36,
                                 f"Constraint 36 (22c): UC {gid} t={t}: "
                                 f"gamma*shut={lhs_val:.4f} > yP={yP:.4f}",
                                 lhs_val, yP, va)

    # --- Constraint 37 (22d): UC min output ---
    if x_inj and x_commit:
        for gid in G_UC_ids:
            g = gen_by_id[gid]
            min_out = g.get('min_output_frac', 0.0)
            size_P = g['capacity_size_MW']
            for t in range(total_T):
                lhs_val = get_val2(x_inj, gid, t)
                rhs_val = min_out * size_P * get_val2(x_commit, gid, t)
                va = compute_violation(lhs_val, rhs_val, '>=')
                record_violation(violations, magnitudes, 37,
                                 f"Constraint 37 (22d): UC {gid} t={t}: "
                                 f"inj={lhs_val:.4f} < min*gamma*commit={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraint 38 (22e): UC max output ---
    if x_inj and x_commit:
        for gid in G_UC_ids:
            g = gen_by_id[gid]
            size_P = g['capacity_size_MW']
            for t in range(total_T):
                lhs_val = get_val2(x_inj, gid, t)
                rhs_val = sigma[gid][t] * size_P * get_val2(x_commit, gid, t)
                va = compute_violation(lhs_val, rhs_val, '<=')
                record_violation(violations, magnitudes, 38,
                                 f"Constraint 38 (22e): UC {gid} t={t}: "
                                 f"inj={lhs_val:.4f} > sigma*gamma*commit={rhs_val:.4f}",
                                 lhs_val, rhs_val, va)

    # --- Constraints 39-40 (22f-22g): UC state transition ---
    if x_commit and x_start and x_shut:
        for gid in G_UC_ids:
            for w_idx in range(W):
                t_start_w = w_idx * delta_w
                for t in range(t_start_w, t_start_w + delta_w):
                    t_prev = circ_prev(t, w_idx)
                    lhs_val = (get_val2(x_commit, gid, t)
                               - get_val2(x_commit, gid, t_prev))
                    rhs_val = (get_val2(x_start, gid, t)
                               - get_val2(x_shut, gid, t))
                    va = compute_violation(lhs_val, rhs_val, '=')
                    is_boundary = (t == t_start_w)
                    cstr_idx = 40 if is_boundary else 39
                    record_violation(violations, magnitudes, cstr_idx,
                                     f"Constraint {cstr_idx} ({'22g' if is_boundary else '22f'}): "
                                     f"UC transition {gid} t={t}: "
                                     f"LHS={lhs_val:.6f} != RHS={rhs_val:.6f}",
                                     lhs_val, rhs_val, va)

    # --- Constraints 41-44 (23a-23d): UC ramping ---
    if x_inj and x_commit and x_start and x_shut:
        for gid in G_UC_ids:
            g = gen_by_id[gid]
            size_P = g['capacity_size_MW']
            min_out = g.get('min_output_frac', 0.0)
            mu_up = g.get('ramp_up_frac_per_hr', 1.0)
            mu_dn = g.get('ramp_dn_frac_per_hr', 1.0)
            for w_idx in range(W):
                t_start_w = w_idx * delta_w
                for t in range(t_start_w, t_start_w + delta_w):
                    t_prev = circ_prev(t, w_idx)
                    inj_t = get_val2(x_inj, gid, t)
                    inj_prev = get_val2(x_inj, gid, t_prev)
                    commit_t = get_val2(x_commit, gid, t)
                    start_t = get_val2(x_start, gid, t)
                    shut_t = get_val2(x_shut, gid, t)
                    sig_t = sigma[gid][t]
                    is_boundary = (t == t_start_w)
                    # (23a/23c) Ramp up
                    lhs_val = inj_t - inj_prev
                    rhs_val = (size_P * mu_up * (commit_t - start_t)
                               + size_P * min(sig_t, max(min_out, mu_up)) * start_t
                               - size_P * min_out * shut_t)
                    va = compute_violation(lhs_val, rhs_val, '<=')
                    cstr_idx = 43 if is_boundary else 41
                    record_violation(violations, magnitudes, cstr_idx,
                                     f"Constraint {cstr_idx} ({'23c' if is_boundary else '23a'}): "
                                     f"UC ramp up {gid} t={t}: "
                                     f"delta={lhs_val:.4f} > limit={rhs_val:.4f}",
                                     lhs_val, rhs_val, va)
                    # (23b/23d) Ramp down
                    lhs_val = inj_prev - inj_t
                    rhs_val = (size_P * mu_dn * (commit_t - start_t)
                               + size_P * min(sig_t, max(min_out, mu_dn)) * shut_t
                               - size_P * min_out * start_t)
                    va = compute_violation(lhs_val, rhs_val, '<=')
                    cstr_idx = 44 if is_boundary else 42
                    record_violation(violations, magnitudes, cstr_idx,
                                     f"Constraint {cstr_idx} ({'23d' if is_boundary else '23b'}): "
                                     f"UC ramp dn {gid} t={t}: "
                                     f"delta={lhs_val:.4f} > limit={rhs_val:.4f}",
                                     lhs_val, rhs_val, va)

    # --- Constraint 45 (24a): Min up time ---
    if x_commit and x_start:
        for gid in G_UC_ids:
            g = gen_by_id[gid]
            tau_up = g.get('min_up_time_hr', 0)
            if tau_up <= 0:
                continue
            for w_idx in range(W):
                t_start_w = w_idx * delta_w
                for t in range(t_start_w, t_start_w + delta_w):
                    idx_in_sub = t - t_start_w
                    starts_sum = 0.0
                    for k in range(tau_up):
                        tk = t_start_w + ((idx_in_sub - k) % delta_w)
                        starts_sum += get_val2(x_start, gid, tk)
                    lhs_val = get_val2(x_commit, gid, t)
                    rhs_val = starts_sum
                    va = compute_violation(lhs_val, rhs_val, '>=')
                    record_violation(violations, magnitudes, 45,
                                     f"Constraint 45 (24a): Min up time UC {gid} t={t}: "
                                     f"commit={lhs_val:.4f} < sum_starts={rhs_val:.4f}",
                                     lhs_val, rhs_val, va)

    # --- Constraint 46 (24b): Min down time ---
    if x_commit and x_shut:
        for gid in G_UC_ids:
            g = gen_by_id[gid]
            size_P = g['capacity_size_MW']
            tau_dn = g.get('min_down_time_hr', 0)
            if tau_dn <= 0:
                continue
            yP = gen_inv.get(gid, {}).get('y_P_total', 0.0)
            max_units = yP / size_P if size_P > 0 else 0.0
            for w_idx in range(W):
                t_start_w = w_idx * delta_w
                for t in range(t_start_w, t_start_w + delta_w):
                    idx_in_sub = t - t_start_w
                    shuts_sum = 0.0
                    for k in range(tau_dn):
                        tk = t_start_w + ((idx_in_sub - k) % delta_w)
                        shuts_sum += get_val2(x_shut, gid, tk)
                    lhs_val = get_val2(x_commit, gid, t) + shuts_sum
                    rhs_val = max_units
                    va = compute_violation(lhs_val, rhs_val, '<=')
                    record_violation(violations, magnitudes, 46,
                                     f"Constraint 46 (24b): Min down time UC {gid} t={t}: "
                                     f"commit+shuts={lhs_val:.4f} > yP/gamma={rhs_val:.4f}",
                                     lhs_val, rhs_val, va)

    # --- Constraint 47 (26): RPS constraint ---
    if x_inj and policy.get('RPS_enabled', False):
        rps_share = policy['RPS_share']
        total_rps_gen = 0.0
        total_rps_slack = 0.0
        total_demand_val = 0.0
        for w_idx in range(W):
            t_start_w = w_idx * delta_w
            for t in range(t_start_w, t_start_w + delta_w):
                for gid in G_RPS_ids:
                    total_rps_gen += alpha[t] * get_val2(x_inj, gid, t)
                for z in range(1, num_zones + 1):
                    total_demand_val += alpha[t] * demand_data[str(z)][t]
            total_rps_slack += get_val(x_rps, w_idx) if x_rps else 0.0
        lhs_val = total_rps_gen + total_rps_slack
        rhs_val = rps_share * total_demand_val
        va = compute_violation(lhs_val, rhs_val, '>=')
        record_violation(violations, magnitudes, 47,
                         f"Constraint 47 (26): RPS: "
                         f"rps_gen+slack={lhs_val:.4f} < required={rhs_val:.4f}",
                         lhs_val, rhs_val, va)

    # --- Constraint 48 (27a): CO2 constraint ---
    if x_inj and policy.get('CO2_cap_enabled', False):
        co2_cap_rate = policy['CO2_cap_tons_per_MWh']
        total_demand_val = 0.0
        for w_idx in range(W):
            t_start_w = w_idx * delta_w
            for t in range(t_start_w, t_start_w + delta_w):
                for z in range(1, num_zones + 1):
                    total_demand_val += alpha[t] * demand_data[str(z)][t]
        co2_cap = co2_cap_rate * total_demand_val

        total_co2 = 0.0
        total_co2_slack = 0.0
        for w_idx in range(W):
            t_start_w = w_idx * delta_w
            for t in range(t_start_w, t_start_w + delta_w):
                for g in generators:
                    gid = g['id']
                    co2_rate = g['co2_tons_per_MWh']
                    if co2_rate > 0:
                        total_co2 += alpha[t] * co2_rate * get_val2(x_inj, gid, t)
                for gid in G_STOR_ids:
                    g2 = gen_by_id[gid]
                    co2_rate = g2['co2_tons_per_MWh']
                    if co2_rate > 0:
                        total_co2 += alpha[t] * co2_rate * get_val2(x_wdw, gid, t)
            total_co2_slack += get_val(x_co2, w_idx) if x_co2 else 0.0
        lhs_val = total_co2 - total_co2_slack
        rhs_val = co2_cap
        va = compute_violation(lhs_val, rhs_val, '<=')
        record_violation(violations, magnitudes, 48,
                         f"Constraint 48 (27a): CO2: "
                         f"emissions-slack={lhs_val:.4f} > cap={rhs_val:.4f}",
                         lhs_val, rhs_val, va)

    return violations, magnitudes


def check_objective_consistency_legacy_envelope(instance, parsed_sol, solution):
    """
    Constraint 49 (Tier C anti-exploit, envelope check on objective_value).

    The solution carries only first-stage investment variables (y_P, y_E, y_F
    and the new/ret pieces). Operational variables (x_inj, x_wdw, x_nse,
    x_start, x_commit, x_rps, x_co2) are NOT in the solution schema, so a
    full obj recompute is impossible. Instead we bracket the reported
    objective with an envelope derivable from the first-stage variables and
    the instance data.

    Lower bound (rejects obj=0 / "do-nothing" / "claim-zero-cost" exploits):
        obj_lower = c^FIXED       (= eq. 29 of the paper, computed exactly)
      The other components (c^VAR, c^NSE, c^START, c^RPS, c^CO2) are sums
      of non-negative variables times non-negative coefficients, so all >= 0.

    Upper bound (rejects obj=MAX_FLOAT / arbitrarily-huge exploits):
        c^VAR   <= sum_t alpha_t * sum_g c_g^VAR * y_g^P
                   (since x_inj, x_wdw <= sigma * y_P <= y_P (16a/16b), and
                    for storage x_inj + x_wdw <= y_P (16c), so the c_g^VAR
                    cost on the storage pair sums to at most c_g^VAR * y_P)
        c^NSE   <= sum_t alpha_t * sum_{s,z} c_{s,z}^NSE * max_frac_s * d_{z,t}
                   (eq. 19 bounds x_nse per segment)
        c^START <= sum_t alpha_t * sum_{g in UC} c_g^START * y_g^P / gamma_g
                   (x_start <= y_P/gamma per (22b))
        c^RPS   <= c^RPS_cost * RPS_share * total_weighted_demand
                   (slack at most equals the RPS requirement)
        c^CO2   <= c^CO2_cost * sum_t alpha_t * sum_g co2_rate * y_g^P
                   (slack absorbs emissions at most equal to gross emissions)

    Tolerance: max(1e-3 absolute, 1e-3 relative). For numbers on the order
    of 1e9, this is roughly 1e6 tolerance --- tight enough to catch any
    LLM-style fabrication, loose enough not to trip on Gurobi's MIP gap or
    floating-point noise.
    """
    violations = []
    magnitudes = []

    if not parsed_sol['has_investment']:
        return violations, magnitudes

    reported_obj = solution.get("objective_value")
    if reported_obj is None:
        return violations, magnitudes
    try:
        reported = float(reported_obj)
    except (TypeError, ValueError):
        return violations, magnitudes
    if not math.isfinite(reported):
        # Non-finite reported obj is unambiguously bogus -> reject as
        # exceeding any sensible envelope.
        violations.append((49, f"Constraint 49 (obj consistency): reported "
                               f"objective_value={reported_obj} is not finite."))
        magnitudes.append({
            "constraint": 49, "lhs": float('inf'), "rhs": 0.0,
            "raw_excess": float('inf'), "normalizer": EPS, "ratio": float('inf'),
        })
        return violations, magnitudes

    generators = instance['generators']
    lines = instance['transmission_lines']
    nse_segments = instance['nse_segments']
    dims = instance['problem_dimensions']
    subperiods_info = instance['subperiods']
    policy = instance['policy']
    demand_data = instance['demand']
    num_zones = dims['num_zones']
    total_T = dims['total_timesteps']
    W = subperiods_info['num_subperiods']
    delta_w = subperiods_info['hours_per_subperiod']
    weights = subperiods_info['subperiod_weights']

    gen_inv = parsed_sol['gen_inv']
    stor_inv = parsed_sol['stor_inv']
    trans_inv = parsed_sol['trans_inv']

    # alpha_t weights for annualisation
    alpha = [0.0] * total_T
    for w_idx in range(W):
        for h in range(delta_w):
            alpha[w_idx * delta_w + h] = weights[w_idx] / delta_w
    sum_alpha = sum(alpha)

    def _f(x, default=0.0):
        try:
            v = float(x)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    # ----- Lower bound: c^FIXED exactly (eq. 29) --------------------------
    c_fixed = 0.0
    for g in generators:
        gid = g['id']
        inv_p = gen_inv.get(gid, {})
        yPn = _f(inv_p.get('y_P_new', 0.0))
        yPt = _f(inv_p.get('y_P_total', 0.0))
        # c_g^{P,INV} * gamma * y_P_NEW
        c_fixed += _f(g.get('inv_cost_per_MW_yr', 0.0)) \
                   * _f(g.get('capacity_size_MW', 0.0)) * yPn
        # c_g^{P,FOM} * y_P
        c_fixed += _f(g.get('fom_cost_per_MW_yr', 0.0)) * yPt
        if g.get('is_storage'):
            inv_e = stor_inv.get(gid, {})
            yEn = _f(inv_e.get('y_E_new', 0.0))
            yEt = _f(inv_e.get('y_E_total', 0.0))
            # c_g^{E,INV} * gamma_E * y_E_NEW
            c_fixed += _f(g.get('inv_cost_energy_per_MWh_yr', 0.0)) \
                       * _f(g.get('storage_capacity_size_MWh', 0.0)) * yEn
            # c_g^{E,FOM} * y_E
            c_fixed += _f(g.get('fom_cost_energy_per_MWh_yr', 0.0)) * yEt
        if g.get('is_hydro'):
            rho = _f(g.get('duration_MWh_per_MW', 0.0))
            # c_g^{E,INV} * rho * y_P_NEW (hydro energy-side investment)
            c_fixed += _f(g.get('inv_cost_energy_per_MWh_yr', 0.0)) * rho * yPn
            # c_g^{E,FOM} * rho * y_P  (hydro energy-side FOM)
            c_fixed += _f(g.get('fom_cost_energy_per_MWh_yr', 0.0)) * rho * yPt
    for l in lines:
        lid = l['id']
        t_inv = trans_inv.get(lid, {})
        yFn = _f(t_inv.get('y_F_new', 0.0))
        c_fixed += _f(l.get('inv_cost_per_MW_yr', 0.0)) * yFn

    obj_lower = c_fixed

    # ----- Upper bound: c^FIXED + per-component upper bounds --------------
    # c^VAR upper bound: alpha_t * c_g^VAR * y_P per (g,t). For storage, c^VAR
    # is charged on both x_inj and x_wdw but x_inj + x_wdw <= y_P (16c) so the
    # bound is still c_g^VAR * y_P per timestep.
    c_var_upper = 0.0
    for g in generators:
        gid = g['id']
        yPt = _f(gen_inv.get(gid, {}).get('y_P_total', 0.0))
        cv = _f(g.get('var_cost_per_MWh', 0.0))
        c_var_upper += sum_alpha * cv * yPt

    # c^NSE upper bound from (19): x_nse_{s,z,t} <= max_frac_s * d_{z,t}.
    c_nse_upper = 0.0
    for seg in nse_segments:
        cs = _f(seg.get('cost_per_MWh', 0.0))
        mfs = _f(seg.get('max_frac', 0.0))
        for z in range(1, num_zones + 1):
            d_list = demand_data.get(str(z), [])
            for t in range(total_T):
                d_zt = _f(d_list[t]) if t < len(d_list) else 0.0
                c_nse_upper += alpha[t] * cs * mfs * d_zt

    # c^START upper bound: x_start <= y_P / gamma per (22b).
    c_start_upper = 0.0
    for g in generators:
        if not g.get('is_UC'):
            continue
        gid = g['id']
        yPt = _f(gen_inv.get(gid, {}).get('y_P_total', 0.0))
        cs = _f(g.get('start_cost_per_unit', 0.0))
        gamma = _f(g.get('capacity_size_MW', 1.0), default=1.0)
        if gamma > 0.0:
            c_start_upper += sum_alpha * cs * (yPt / gamma)

    # c^RPS upper bound: x_w^RPS slack at most equals the RPS requirement.
    c_rps_upper = 0.0
    if policy.get('RPS_enabled', False):
        rps_cost = _f(policy.get('RPS_noncompliance_cost_per_MWh', 0.0))
        rps_share = _f(policy.get('RPS_share', 0.0))
        twd = 0.0
        for z in range(1, num_zones + 1):
            d_list = demand_data.get(str(z), [])
            for t in range(total_T):
                d_zt = _f(d_list[t]) if t < len(d_list) else 0.0
                twd += alpha[t] * d_zt
        c_rps_upper = rps_cost * rps_share * twd

    # c^CO2 upper bound: x_w^CO2 slack at most equals gross emissions.
    c_co2_upper = 0.0
    if policy.get('CO2_cap_enabled', False):
        co2_cost = _f(policy.get('CO2_noncompliance_cost_per_ton', 0.0))
        for g in generators:
            gid = g['id']
            yPt = _f(gen_inv.get(gid, {}).get('y_P_total', 0.0))
            co2r = _f(g.get('co2_tons_per_MWh', 0.0))
            c_co2_upper += sum_alpha * co2_cost * co2r * yPt

    obj_upper = c_fixed + c_var_upper + c_nse_upper + c_start_upper + c_rps_upper + c_co2_upper

    # Tolerances: 0.1% relative with a small absolute floor.
    tol_lo = max(1e-3, 1e-3 * abs(obj_lower))
    tol_hi = max(1e-3, 1e-3 * abs(obj_upper))

    shortfall = obj_lower - reported
    overshoot = reported - obj_upper

    if shortfall > tol_lo:
        record_violation(
            violations, magnitudes, 49,
            f"Constraint 49 (Objective consistency): reported objective_value="
            f"{reported:.6f} is below lower bound c^FIXED={obj_lower:.6f} "
            f"(shortfall={shortfall:.6g}, tol={tol_lo:.6g})",
            obj_lower, reported, shortfall,
        )

    if overshoot > tol_hi:
        record_violation(
            violations, magnitudes, 49,
            f"Constraint 49 (Objective consistency): reported objective_value="
            f"{reported:.6f} exceeds upper bound={obj_upper:.6f} "
            f"(c^FIXED={c_fixed:.4f}, c^VAR_max={c_var_upper:.4f}, "
            f"c^NSE_max={c_nse_upper:.4f}, c^START_max={c_start_upper:.4f}, "
            f"c^RPS_max={c_rps_upper:.4f}, c^CO2_max={c_co2_upper:.4f}); "
            f"overshoot={overshoot:.6g}, tol={tol_hi:.6g}",
            reported, obj_upper, overshoot,
        )

    return violations, magnitudes


def check_first_stage_coverage(instance, solution):
    """Require one complete, finite record for every first-stage entity."""
    messages = []
    generator_entries = solution.get("investment_decisions")
    line_entries = solution.get("transmission_decisions")
    if not isinstance(generator_entries, list) or not isinstance(line_entries, list):
        return [(0, "First-stage decisions must be JSON lists")], []

    expected_generators = {g["id"]: g for g in instance["generators"]}
    expected_lines = {line["id"]: line for line in instance["transmission_lines"]}

    generator_ids = [
        entry.get("generator_id") for entry in generator_entries
        if isinstance(entry, dict)
    ]
    line_ids = [
        entry.get("line_id") for entry in line_entries
        if isinstance(entry, dict)
    ]
    if len(generator_ids) != len(generator_entries):
        messages.append("Every investment_decisions entry must be an object")
    if len(line_ids) != len(line_entries):
        messages.append("Every transmission_decisions entry must be an object")
    if len(generator_ids) != len(set(generator_ids)):
        messages.append("Generator investment records contain duplicate IDs")
    if len(line_ids) != len(set(line_ids)):
        messages.append("Transmission investment records contain duplicate IDs")
    if set(generator_ids) != set(expected_generators):
        messages.append(
            "Generator investment records do not exactly cover the instance IDs"
        )
    if set(line_ids) != set(expected_lines):
        messages.append(
            "Transmission investment records do not exactly cover the instance IDs"
        )

    for entry in generator_entries:
        if not isinstance(entry, dict) or entry.get("generator_id") not in expected_generators:
            continue
        generator = expected_generators[entry["generator_id"]]
        fields = ["y_P_new", "y_P_ret", "y_P_total_MW"]
        if generator["is_storage"]:
            fields.extend(["y_E_new", "y_E_ret", "y_E_total_MWh"])
        for field in fields:
            value = entry.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                messages.append(
                    f"Generator {entry['generator_id']} has invalid or missing {field}"
                )

    for entry in line_entries:
        if not isinstance(entry, dict) or entry.get("line_id") not in expected_lines:
            continue
        for field in ("y_F_new_MW", "total_capacity_MW"):
            value = entry.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                messages.append(
                    f"Line {entry['line_id']} has invalid or missing {field}"
                )

    return [(0, message) for message in messages], []


def check_objective_consistency(instance, parsed_sol, solution, instance_path):
    """Validate complete operations and bind the scalar to their exact cost."""
    reported = float(solution["objective_value"])
    expected, errors, _ = validate_complete_witness(
        instance_path, solution, Path(__file__).resolve().parent
    )
    links = {}
    entries = {entry["generator_id"]: entry for entry in solution["investment_decisions"]}
    for index, generator in enumerate(instance["generators"]):
        entry = entries[generator["id"]]
        links[f"yPnew_{index}"] = float(entry["y_P_new"])
        links[f"yPret_{index}"] = float(entry["y_P_ret"])
        if generator["is_storage"]:
            links[f"yEnew_{index}"] = float(entry["y_E_new"])
            links[f"yEret_{index}"] = float(entry["y_E_ret"])
    line_entries = {entry["line_id"]: entry for entry in solution["transmission_decisions"]}
    for index, line in enumerate(instance["transmission_lines"]):
        links[f"yFnew_{index}"] = float(line_entries[line["id"]]["y_F_new_MW"])
    errors.extend(require_linked_variables(solution, links))
    if errors or expected is None:
        return [(49, "Constraint 49 (complete witness): " + "; ".join(errors[:20]))], []
    # Published MIP incumbents can retain a few parts per million of recourse
    # suboptimality even after the first-stage integers are fixed.  Five ppm
    # covers that numerical residue while remaining far below eval mutations.
    tolerance = max(1e-3, 5e-6 * max(1.0, abs(expected)))
    difference = abs(reported - expected)
    if difference <= tolerance:
        return [], []
    return [(
        49,
        f"Constraint 49 (objective consistency): reported objective_value="
        f"{reported:.6f} does not equal trusted fixed-investment cost="
        f"{expected:.6f} (difference={difference:.6g}, tol={tolerance:.6g})",
    )], [{
        "constraint": 49,
        "lhs": reported,
        "rhs": expected,
        "raw_excess": difference,
        "normalizer": max(abs(expected), EPS),
        "ratio": difference / max(abs(expected), EPS),
    }]


def check_feasibility(instance, solution, instance_path=None):
    """In-memory feasibility check entrypoint (used by the diff-test harness).
    Returns the same dict shape that main() writes to disk."""
    parsed = parse_solution(solution)

    if not parsed['has_investment']:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["No investment solution provided (null or missing)"],
            "violation_magnitudes": [],
        }

    all_violations = []
    all_magnitudes = []

    v, m = check_first_stage_coverage(instance, solution)
    all_violations.extend(v); all_magnitudes.extend(m)
    if all_violations:
        return {
            "feasible": False,
            "violated_constraints": [0],
            "violations": [message for _, message in all_violations],
            "violation_magnitudes": all_magnitudes,
        }

    v, m = check_investment_constraints(instance, parsed)
    all_violations.extend(v); all_magnitudes.extend(m)
    v, m = check_non_negativity(instance, parsed)
    all_violations.extend(v); all_magnitudes.extend(m)
    v, m = check_operational_constraints(instance, parsed)
    all_violations.extend(v); all_magnitudes.extend(m)
    if instance_path is None:
        v, m = [(49, "instance_path is required for solverless validation")], []
    else:
        v, m = check_objective_consistency(instance, parsed, solution, instance_path)
    all_violations.extend(v); all_magnitudes.extend(m)

    violated_indices = sorted(set(idx for idx, _ in all_violations))
    violation_messages = []
    for idx in violated_indices:
        msgs = [msg for i, msg in all_violations if i == idx]
        if len(msgs) <= 3:
            violation_messages.extend(msgs)
        else:
            violation_messages.append(msgs[0])
            violation_messages.append(f"... and {len(msgs) - 1} more violations of constraint {idx}")

    return {
        "feasible": len(violated_indices) == 0,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": all_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for energy systems planning model")
    parser.add_argument('--instance_path', required=True,
                        help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True,
                        help='Path to candidate solution JSON file')
    parser.add_argument('--result_path', required=True,
                        help='Path to write feasibility result JSON file')
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

    result = check_feasibility(instance, solution, args.instance_path)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"Solution is FEASIBLE.")
    else:
        print(
            "Solution is INFEASIBLE. Violated constraints: "
            f"{result['violated_constraints']}"
        )


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
