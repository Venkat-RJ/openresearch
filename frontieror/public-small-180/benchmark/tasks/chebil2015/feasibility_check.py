"""
Feasibility checker for the Knapsack Problem with Setup (KPS).
Based on: Chebil & Khemakhem (2015), Computers & Operations Research.

Constraints (numbered top-to-bottom from the formulation):
  Constraint 1: sum_i sum_j a_{ij} x_{ij} + sum_i d_i y_i <= b  (capacity)
  Constraint 2: x_{ij} <= y_i  for all i, j                     (linking)
  Constraint 3: x_{ij}, y_i in {0, 1}                           (binary)
  Constraint 4: reported objective_value must equal
                sum_i sum_j c_{ij} x_{ij} + sum_i f_i y_i        (obj consistency)
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('classes', 'objective_value')
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


TOL = 1e-5
EPS = 1e-5


def check_feasibility(instance, solution):
    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    b = instance["b"]
    inst_classes = {cls["class_id"]: cls for cls in instance["classes"]}
    sol_classes = {cls["class_id"]: cls for cls in solution["classes"]}

    # --- Constraint 3: Binary constraints on x_ij and y_i ---
    for cid, sol_cls in sol_classes.items():
        y_i = sol_cls["y_i"]
        if y_i not in (0, 1) and abs(y_i - round(y_i)) > TOL:
            violated_constraints.add(3)
            violation_amount = min(abs(y_i - 0), abs(y_i - 1))
            rhs = round(y_i)
            normalizer = max(abs(rhs), EPS)
            violations.append(f"y_{cid} = {y_i} is not binary")
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(y_i),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

        for item_sol in sol_cls["items"]:
            x_ij = item_sol["x_ij"]
            jid = item_sol["item_id"]
            if x_ij not in (0, 1) and abs(x_ij - round(x_ij)) > TOL:
                violated_constraints.add(3)
                violation_amount = min(abs(x_ij - 0), abs(x_ij - 1))
                rhs = round(x_ij)
                normalizer = max(abs(rhs), EPS)
                violations.append(
                    f"x_{cid}_{jid} = {x_ij} is not binary"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(x_ij),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

    # --- Constraint 1: Capacity constraint ---
    # LHS = sum_i sum_j a_ij * x_ij + sum_i d_i * y_i
    # RHS = b
    # Constraint: LHS <= RHS
    lhs_cap = 0.0
    for cid, sol_cls in sol_classes.items():
        inst_cls = inst_classes[cid]
        y_i = sol_cls["y_i"]
        d_i = inst_cls["d_i"]
        lhs_cap += d_i * y_i

        inst_items = {it["item_id"]: it for it in inst_cls["items"]}
        for item_sol in sol_cls["items"]:
            jid = item_sol["item_id"]
            x_ij = item_sol["x_ij"]
            a_ij = inst_items[jid]["a_ij"]
            lhs_cap += a_ij * x_ij

    violation_amount = max(lhs_cap - b, 0.0)
    if violation_amount > TOL:
        violated_constraints.add(1)
        normalizer = max(abs(b), EPS)
        violations.append(
            f"Capacity exceeded: total weight {lhs_cap} > capacity {b}"
        )
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": float(lhs_cap),
            "rhs": float(b),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    # --- Constraint 2: Linking constraints x_ij <= y_i ---
    for cid, sol_cls in sol_classes.items():
        y_i = sol_cls["y_i"]
        for item_sol in sol_cls["items"]:
            x_ij = item_sol["x_ij"]
            jid = item_sol["item_id"]
            # LHS = x_ij, RHS = y_i, constraint: LHS <= RHS
            violation_amount = max(x_ij - y_i, 0.0)
            if violation_amount > TOL:
                violated_constraints.add(2)
                normalizer = max(abs(y_i), EPS)
                violations.append(
                    f"Linking violated: x_{cid}_{jid} = {x_ij} > y_{cid} = {y_i}"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(x_ij),
                    "rhs": float(y_i),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer),
                })

    # --- Constraint 4: Objective consistency ---
    # True obj = sum_i sum_j c_ij * x_ij + sum_i f_i * y_i
    # Must equal reported objective_value within tolerance.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for cid, sol_cls in sol_classes.items():
                inst_cls = inst_classes[cid]
                y_i = sol_cls["y_i"]
                f_i = inst_cls["f_i"]
                true_obj += f_i * y_i

                inst_items = {it["item_id"]: it for it in inst_cls["items"]}
                for item_sol in sol_cls["items"]:
                    jid = item_sol["item_id"]
                    x_ij = item_sol["x_ij"]
                    c_ij = inst_items[jid]["c_ij"]
                    true_obj += c_ij * x_ij
            true_obj = float(true_obj)

            abs_diff = abs(reported - true_obj)
            # All coefficients (c_ij, f_i) are integers and variables are binary,
            # so the true objective is an integer. A 0.5 absolute floor catches
            # any integer mismatch >= 1, plus a 0.1% relative band for safety.
            tol = max(0.5, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(4)
                normalizer = max(abs(true_obj), EPS)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_ij(c_ij*x_ij)+sum_i(f_i*y_i)="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer),
                })

    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for KPS (Chebil & Khemakhem 2015)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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
