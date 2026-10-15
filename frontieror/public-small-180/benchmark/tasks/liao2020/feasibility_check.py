#!/usr/bin/env python3
"""
Feasibility checker for the Distributionally Robust Outpatient Colonoscopy
Scheduling (DROCS) problem.

Reference: Shehadeh, Cohn, and Jiang (2019/2020), EJOR.

Checks the hard constraints of the DR-bimodal MILP (Proposition 4, Eq. 15):
  Constraint 1: Assignment constraints (Eq. 1 / 15b)
  Constraint 2: Scheduling constraints (Eq. 2 / 15b)
  Constraints 3-12: complete auxiliary/dual witness satisfies the trusted
                     offline coefficient system.
  Constraint 13: objective_value equals the submitted complete witness cost.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solverless_witness import require_linked_variables, validate_complete_witness

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'c_g', 'c_o', 'c_w', 'cost_structure', 'model_variables', 'objective_value', 'patient_start_times', 'schedule')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('assignment', 'model_variables', 'patient_start_times', 'schedule')
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
    with open(path, "r") as f:
        return json.load(f)


def record_violation(violations_list, magnitudes_list, constraint_idx,
                     message, lhs, rhs, violation_amount):
    """Record a constraint violation."""
    violations_list.append((constraint_idx, message))
    normalizer = max(abs(rhs), EPS)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": lhs,
        "rhs": rhs,
        "raw_excess": violation_amount,
        "normalizer": normalizer,
        "ratio": violation_amount / normalizer,
    })


def check_feasibility(instance, solution, instance_path):
    P = instance["num_patients_P"]
    L = instance["provider_service_hours_L_minutes"]

    violations = []  # list of (constraint_idx, message)
    magnitudes = []  # list of violation magnitude dicts

    cost_key = next(iter(instance["cost_structures"]))
    expected_cost = instance["cost_structures"][cost_key]
    if solution["cost_structure"] != cost_key:
        record_violation(
            violations, magnitudes, 13,
            f"cost_structure must be the benchmark selector {cost_key!r}",
            1.0, 0.0, 1.0,
        )
    for field in ("c_w", "c_g", "c_o"):
        difference = abs(float(solution[field]) - float(expected_cost[field]))
        if difference > TOL:
            record_violation(
                violations, magnitudes, 13,
                f"{field}={solution[field]} does not match "
                f"{cost_key}.{field}={expected_cost[field]}",
                float(solution[field]), float(expected_cost[field]), difference,
            )

    # ------------------------------------------------------------------
    # Reconstruct x_{p,i} (binary assignment) and t_i (schedule)
    # ------------------------------------------------------------------
    # Detect solution format:
    #   efficient_solution: assignment maps position(str) -> patient(int)
    #                       also has "sequence" key
    #   gurobi_solution:   assignment maps patient(str) -> position(int)
    #                       also has "patient_start_times" key
    raw_assignment = solution["assignment"]
    schedule_raw = solution["schedule"]

    if "sequence" in solution:
        # efficient format: key=position, value=patient
        pos_to_pat = {int(k): int(v) for k, v in raw_assignment.items()}
        pat_to_pos = {v: k for k, v in pos_to_pat.items()}
    elif "patient_start_times" in solution:
        # gurobi format: key=patient, value=position
        pat_to_pos = {int(k): int(v) for k, v in raw_assignment.items()}
        pos_to_pat = {v: k for k, v in pat_to_pos.items()}
    else:
        # Fallback: try to detect by checking if values form valid positions
        # Assume position->patient (efficient format)
        pos_to_pat = {int(k): int(v) for k, v in raw_assignment.items()}
        pat_to_pos = {v: k for k, v in pos_to_pat.items()}

    # Build x[p][i] matrix
    x = {}
    for p in range(1, P + 1):
        for i in range(1, P + 1):
            x[(p, i)] = 0
    for pos, pat in pos_to_pat.items():
        if 1 <= pos <= P and 1 <= pat <= P:
            x[(pat, pos)] = 1

    # Build t[i] schedule
    t = {}
    for i in range(1, P + 1):
        key = str(i)
        if key in schedule_raw:
            t[i] = float(schedule_raw[key])
        else:
            t[i] = 0.0

    expected_indices = set(range(1, P + 1))
    if set(pos_to_pat) != expected_indices or set(pat_to_pos) != expected_indices:
        record_violation(
            violations, magnitudes, 1,
            "assignment must contain every patient and position exactly once",
            float(len(pos_to_pat)), float(P), 1.0,
        )
    parsed_schedule_keys = {int(key) for key in schedule_raw}
    if parsed_schedule_keys != expected_indices:
        record_violation(
            violations, magnitudes, 2,
            f"schedule keys must exactly cover positions 1..{P}",
            float(len(parsed_schedule_keys)), float(P), 1.0,
        )

    patient_start_times = solution["patient_start_times"]
    parsed_patient_times = {
        int(patient): float(start)
        for patient, start in patient_start_times.items()
    }
    if set(parsed_patient_times) != expected_indices:
        record_violation(
            violations, magnitudes, 2,
            f"patient_start_times keys must exactly cover patients 1..{P}",
            float(len(parsed_patient_times)), float(P), 1.0,
        )
    for patient in expected_indices & set(parsed_patient_times) & set(pat_to_pos):
        expected_start = t[pat_to_pos[patient]]
        difference = abs(parsed_patient_times[patient] - expected_start)
        if difference > TOL:
            record_violation(
                violations, magnitudes, 2,
                f"patient_start_times[{patient}]={parsed_patient_times[patient]} "
                f"does not match schedule position {pat_to_pos[patient]}="
                f"{expected_start}",
                parsed_patient_times[patient], expected_start, difference,
            )

    # ==================================================================
    # Constraint 1: Assignment constraints (Eq. 1)
    #   sum_p x_{p,i} = 1   for all i = 1..P  (each position one patient)
    #   sum_i x_{p,i} = 1   for all p = 1..P  (each patient one position)
    #   x_{p,i} in {0,1}
    # ==================================================================

    # Check binary: x values should be 0 or 1
    for p in range(1, P + 1):
        for i in range(1, P + 1):
            val = x[(p, i)]
            if val not in (0, 1):
                viol = abs(val - round(val))
                if viol > TOL:
                    record_violation(violations, magnitudes, 1,
                                     f"x[{p},{i}]={val} is not binary",
                                     val, round(val), viol)

    # Each position gets exactly one patient
    for i in range(1, P + 1):
        lhs_val = sum(x[(p, i)] for p in range(1, P + 1))
        rhs_val = 1.0
        viol_amount = abs(lhs_val - rhs_val)
        if viol_amount > TOL:
            record_violation(violations, magnitudes, 1,
                             f"Position {i}: sum of assignments = {lhs_val}, expected 1",
                             lhs_val, rhs_val, viol_amount)

    # Each patient assigned to exactly one position
    for p in range(1, P + 1):
        lhs_val = sum(x[(p, i)] for i in range(1, P + 1))
        rhs_val = 1.0
        viol_amount = abs(lhs_val - rhs_val)
        if viol_amount > TOL:
            record_violation(violations, magnitudes, 1,
                             f"Patient {p}: sum of assignments = {lhs_val}, expected 1",
                             lhs_val, rhs_val, viol_amount)

    # Check that assignment is a valid permutation (no duplicate patients/positions)
    assigned_patients = list(pos_to_pat.values())
    assigned_positions = list(pos_to_pat.keys())
    if len(set(assigned_patients)) != len(assigned_patients):
        dup_pats = [p for p in assigned_patients if assigned_patients.count(p) > 1]
        record_violation(violations, magnitudes, 1,
                         f"Duplicate patient assignments: patients {set(dup_pats)} assigned multiple times",
                         len(set(assigned_patients)), P, P - len(set(assigned_patients)))

    if len(set(assigned_positions)) != len(assigned_positions):
        dup_pos = [i for i in assigned_positions if assigned_positions.count(i) > 1]
        record_violation(violations, magnitudes, 1,
                         f"Duplicate position assignments: positions {set(dup_pos)} used multiple times",
                         len(set(assigned_positions)), P, P - len(set(assigned_positions)))

    # Check all patients 1..P are assigned
    expected_patients = set(range(1, P + 1))
    actual_patients = set(assigned_patients)
    if actual_patients != expected_patients:
        missing = expected_patients - actual_patients
        extra = actual_patients - expected_patients
        msg_parts = []
        if missing:
            msg_parts.append(f"missing patients {missing}")
        if extra:
            msg_parts.append(f"unexpected patients {extra}")
        record_violation(violations, magnitudes, 1,
                         f"Assignment coverage error: {'; '.join(msg_parts)}",
                         len(actual_patients & expected_patients), P,
                         len(missing | extra))

    # Check all positions 1..P are filled
    expected_positions = set(range(1, P + 1))
    actual_positions = set(assigned_positions)
    if actual_positions != expected_positions:
        missing = expected_positions - actual_positions
        extra = actual_positions - expected_positions
        msg_parts = []
        if missing:
            msg_parts.append(f"unfilled positions {missing}")
        if extra:
            msg_parts.append(f"unexpected positions {extra}")
        record_violation(violations, magnitudes, 1,
                         f"Position coverage error: {'; '.join(msg_parts)}",
                         len(actual_positions & expected_positions), P,
                         len(missing | extra))

    # ==================================================================
    # Constraint 2: Scheduling constraints (Eq. 2)
    #   t_1 = 0
    #   0 <= t_i <= L       for all i = 1..P
    #   t_i >= t_{i-1}      for all i = 2..P
    # ==================================================================

    # t_1 = 0
    viol_amount = abs(t[1] - 0.0)
    if viol_amount > TOL:
        record_violation(violations, magnitudes, 2,
                         f"t_1 = {t[1]}, expected 0",
                         t[1], 0.0, viol_amount)

    # 0 <= t_i <= L
    for i in range(1, P + 1):
        # t_i >= 0 (i.e., 0 <= t_i; equivalently, -t_i <= 0)
        if t[i] < -TOL:
            viol_amount = -t[i]
            record_violation(violations, magnitudes, 2,
                             f"t_{i} = {t[i]} < 0 (lower bound violated)",
                             t[i], 0.0, viol_amount)

        # t_i <= L
        if t[i] > L + TOL:
            viol_amount = t[i] - L
            record_violation(violations, magnitudes, 2,
                             f"t_{i} = {t[i]} > L = {L} (upper bound violated)",
                             t[i], L, viol_amount)

    # t_i >= t_{i-1} for i = 2..P
    for i in range(2, P + 1):
        # t_i >= t_{i-1}  <==>  t_{i-1} - t_i <= 0 (lhs=t_{i-1}, rhs=t_i as >= constraint)
        if t[i] < t[i - 1] - TOL:
            viol_amount = t[i - 1] - t[i]
            record_violation(violations, magnitudes, 2,
                             f"t_{i} = {t[i]} < t_{i-1} = {t[i-1]} (non-decreasing order violated)",
                             t[i], t[i - 1], viol_amount)

    # ==================================================================
    # Constraints 3-13: validate every auxiliary/dual variable and the
    # objective by direct substitution into the trusted model coefficients.
    # ==================================================================
    valid_perm = (
        set(pos_to_pat) == set(range(1, P + 1))
        and set(pat_to_pos) == set(range(1, P + 1))
    )
    valid_schedule = set(int(key) for key in schedule_raw) == set(
        range(1, P + 1)
    )
    if valid_perm and valid_schedule:
        expected_links = {
            f"x[{patient},{position}]": float(pat_to_pos[patient] == position)
            for patient in range(1, P + 1)
            for position in range(1, P + 1)
        }
        expected_links.update({f"t[{position}]": t[position] for position in range(1, P + 1)})
        expected_objective, witness_errors, _ = validate_complete_witness(
            instance_path, solution, Path(__file__).resolve().parent
        )
        witness_errors.extend(require_linked_variables(solution, expected_links))
        for message in witness_errors:
            record_violation(violations, magnitudes, 13, message, 1.0, 0.0, 1.0)
        reported_objective = float(solution["objective_value"])
        difference = abs(reported_objective - float(expected_objective or 0.0))
        objective_tolerance = max(
            1e-3, 2e-6 * abs(float(expected_objective or 0.0))
        )
        if expected_objective is not None and difference > objective_tolerance:
            record_violation(
                violations, magnitudes, 13,
                f"Objective consistency: objective_value={reported_objective} "
                f"does not match complete-witness DRO value "
                f"{expected_objective} (|diff|={difference:.4g}, "
                f"tol={objective_tolerance:.4g}).",
                reported_objective, expected_objective, difference,
            )

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    violated_indices = sorted(set(idx for idx, _ in violations))
    violation_messages = []
    for idx in violated_indices:
        msgs = [msg for c_idx, msg in violations if c_idx == idx]
        violation_messages.append(f"Constraint {idx}: " + "; ".join(msgs))

    feasible = len(violated_indices) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": magnitudes if not feasible else [],
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for DROCS (DR-bimodal MILP)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
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

    result = check_feasibility(instance, solution, args.instance_path)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: All checkable constraints satisfied.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} constraint(s) violated.")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
