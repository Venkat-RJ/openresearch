"""
Feasibility checker for the 0-1 Knapsack Problem with Setups (KPS).

Checks all hard constraints from the KPS_1 formulation:
  Constraint 1 (capacity):  sum w_{ij} x_{ij} + sum d_i y_i <= b
  Constraint 2 (linking):   x_{ij} <= y_i  for all i, j
  Constraint 3 (binary x):  x_{ij} in {0, 1}
  Constraint 4 (binary y):  y_i in {0, 1}
  Constraint 5 (objective): reported objective_value must equal
                            sum_{i,j} p_{ij} x_{ij} - sum_i f_i y_i
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('families_selected', 'items_selected', 'objective_value')
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



def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    N = instance["N"]
    b = instance["knapsack_capacity"]
    families = instance["families"]

    # Build lookup: family_id -> family data
    fam_lookup = {}
    for fam in families:
        fam_lookup[fam["family_id"]] = fam
        # Build item lookup within family
        item_lookup = {}
        for item in fam["items"]:
            item_lookup[item["item_id"]] = item
        fam["_item_lookup"] = item_lookup

    families_selected = set(solution.get("families_selected", []))
    items_selected = solution.get("items_selected", [])

    # Build x values: (family_id, item_id) -> 1
    x_vals = {}
    for entry in items_selected:
        key = (entry["family"], entry["item"])
        x_vals[key] = 1

    # Build y values
    y_vals = {}
    for fam in families:
        fid = fam["family_id"]
        y_vals[fid] = 1 if fid in families_selected else 0

    # --- Constraint 4: y_i in {0, 1} ---
    for fid in families_selected:
        if fid not in fam_lookup:
            violation_amount = 1.0
            rhs = 0.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(4)
            violations.append(f"Family {fid} in families_selected is not a valid family ID")
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(fid),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # --- Constraint 3: x_{ij} in {0, 1} ---
    # Check that all selected items reference valid families and item IDs
    for entry in items_selected:
        fid = entry["family"]
        iid = entry["item"]
        if fid not in fam_lookup:
            violation_amount = 1.0
            rhs = 0.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(3)
            violations.append(f"Item ({fid},{iid}) references invalid family {fid}")
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": 1.0,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })
        elif iid not in fam_lookup[fid]["_item_lookup"]:
            violation_amount = 1.0
            rhs = 0.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(3)
            violations.append(f"Item ({fid},{iid}) has invalid item_id {iid} in family {fid}")
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": 1.0,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # Check for duplicate items (x_{ij} > 1 would violate binary domain)
    from collections import Counter
    item_counts = Counter((e["family"], e["item"]) for e in items_selected)
    for (fid, iid), count in item_counts.items():
        if count > 1:
            violation_amount = float(count - 1)
            rhs = 1.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(3)
            violations.append(f"Item ({fid},{iid}) selected {count} times, violates binary domain")
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(count),
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # --- Constraint 2: x_{ij} <= y_i (linking) ---
    for entry in items_selected:
        fid = entry["family"]
        iid = entry["item"]
        x_val = 1.0
        y_val = float(y_vals.get(fid, 0))
        # x_{ij} <= y_i  =>  violation if x_{ij} - y_i > tol
        violation_amount = x_val - y_val
        if violation_amount > tol:
            rhs = y_val
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(2)
            violations.append(
                f"Item ({fid},{iid}) selected but family {fid} not activated (x=1, y=0)"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": x_val,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # --- Constraint 1: capacity ---
    # LHS = sum w_{ij} x_{ij} + sum d_i y_i
    total_weight = 0.0
    for entry in items_selected:
        fid = entry["family"]
        iid = entry["item"]
        if fid in fam_lookup and iid in fam_lookup[fid]["_item_lookup"]:
            total_weight += fam_lookup[fid]["_item_lookup"][iid]["weight"]

    total_setup_cap = 0.0
    for fid in families_selected:
        if fid in fam_lookup:
            total_setup_cap += fam_lookup[fid]["setup_capacity"]

    lhs_capacity = total_weight + total_setup_cap
    rhs_capacity = float(b)
    violation_amount = lhs_capacity - rhs_capacity
    if violation_amount > tol:
        normalizer = max(abs(rhs_capacity), eps)
        ratio = violation_amount / normalizer
        violated_constraints_set.add(1)
        violations.append(
            f"Capacity exceeded: total weight {total_weight} + setup capacity "
            f"{total_setup_cap} = {lhs_capacity} > {rhs_capacity}"
        )
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs_capacity,
            "rhs": rhs_capacity,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio,
        })

    # --- Constraint 5: objective consistency (Tier C defense) ---
    # Recompute  obj = sum p_{ij} x_{ij} - sum f_i y_i  from the solution
    # variables and reject when the reported value disagrees. Only valid
    # (family, item) entries contribute; invalid ones are already flagged
    # by constraints 3/4 and their would-be contributions are undefined.
    reported_obj_raw = solution.get("objective_value")
    reported_obj = None
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
    if reported_obj is not None:
        true_obj = 0.0
        for (fid, iid) in x_vals:
            if fid in fam_lookup and iid in fam_lookup[fid]["_item_lookup"]:
                true_obj += float(fam_lookup[fid]["_item_lookup"][iid]["profit"])
        for fid in families_selected:
            if fid in fam_lookup:
                true_obj -= float(fam_lookup[fid]["setup_cost"])

        abs_diff = abs(reported_obj - true_obj)
        # KPS profits and setup costs are integers per the formulation, so
        # the true objective is integer-valued. Use 0.5 absolute tolerance
        # so any integer mismatch fires; keep a tiny relative floor for
        # robustness on very large magnitudes (e.g. sys.float_info.max).
        obj_tol = max(0.5, 1e-6 * abs(true_obj))
        if abs_diff > obj_tol:
            normalizer = max(abs(true_obj), eps)
            ratio = abs_diff / normalizer
            violated_constraints_set.add(5)
            violations.append(
                f"Objective consistency violated: reported objective_value="
                f"{reported_obj} differs from recomputed "
                f"sum p_{{ij}} x_{{ij}} - sum f_i y_i = {true_obj} "
                f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(reported_obj),
                "rhs": float(true_obj),
                "raw_excess": float(abs_diff),
                "normalizer": normalizer,
                "ratio": ratio,
            })

    feasible = len(violated_constraints_set) == 0
    violated_constraints = sorted(violated_constraints_set)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for KPS (0-1 Knapsack Problem with Setups)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)
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

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
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
