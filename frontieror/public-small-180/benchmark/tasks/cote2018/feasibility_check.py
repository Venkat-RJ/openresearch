"""
Feasibility checker for the Cutting Stock Problem (CSP) arc-flow formulation
from Côté & Iori (2018).

Checks the candidate solution (objective value z = number of bins) against
the hard constraints of the arc-flow formulation (eqs 11-14):

  Constraint 1 (eq 12): Flow conservation — z must be non-negative and
      sufficient for a valid flow to exist (z >= continuous lower bound).
  Constraint 2 (eq 13): Demand satisfaction — total bin capacity z*W must
      accommodate all items; each item must fit in a bin (w_i <= W).
  Constraint 3 (eq 14): Non-negativity and integrality — z must be a
      non-negative integer.

The candidate contains the complete sparse arc-flow vector.  Missing model
variables are interpreted as zero.  The checker substitutes that vector into
trusted offline model coefficients; it does not invoke an optimizer.
"""

import json
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solverless_witness import require_linked_variables, validate_complete_witness

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('model_variables', 'objective_value')
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



def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_solution(path):
    with open(path, 'r') as f:
        return json.load(f)


def check_feasibility(instance, solution, instance_path=None):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violated_constraints = set()
    violation_magnitudes = []

    W = instance["bin_capacity"]
    items = instance["items"]

    z = solution.get("objective_value")

    # --- Constraint 3 (eq 14): Non-negativity and integrality ---
    # x_{pq} >= 0, integer; z >= 0, integer
    # Check z is a valid non-negative integer.

    if z is None:
        violated_constraints.add(3)
        violations.append(
            "Constraint 3: objective_value is None (no feasible solution reported)."
        )
        # Cannot check further constraints without a valid z
        result = {
            "feasible": False,
            "violated_constraints": sorted(violated_constraints),
            "violations": violations,
            "violation_magnitudes": violation_magnitudes,
        }
        return result

    # Check z >= 0
    if z < -tol:
        violation_amount = abs(z)
        rhs = 0.0
        normalizer = max(abs(rhs), eps)
        violated_constraints.add(3)
        violations.append(
            f"Constraint 3: z = {z} is negative (must be >= 0)."
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": float(z),
            "rhs": rhs,
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    # Check z is integer
    if abs(z - round(z)) > tol:
        violation_amount = abs(z - round(z))
        rhs = round(z)
        normalizer = max(abs(rhs), eps)
        violated_constraints.add(3)
        violations.append(
            f"Constraint 3: z = {z} is not integer (nearest integer is {round(z)})."
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": float(z),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    z_int = z if isinstance(z, int) else (round(z) if abs(z - round(z)) <= tol else z)

    # --- Constraint 2 (eq 13): Demand satisfaction ---
    # sum_{(q, q+w_i) in A} x_{q,q+w_i} >= d_i for each item type i.
    # Necessary conditions:
    #   (a) Each item must fit in a bin: w_i <= W
    #   (b) Total capacity must suffice: z * W >= sum(w_i * d_i)
    #       Equivalently: z >= ceil(sum(w_i * d_i) / W)

    # Check (a): each item fits
    for item in items:
        w_i = item["width"]
        d_i = item["demand"]
        if w_i > W:
            violation_amount = w_i - W
            rhs = float(W)
            normalizer = max(abs(rhs), eps)
            violated_constraints.add(2)
            violations.append(
                f"Constraint 2: Item type {item['type_id']} has width {w_i} "
                f"exceeding bin capacity {W}; demand {d_i} cannot be satisfied."
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(w_i),
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # Check (b): z >= ceil(sum(w_i * d_i) / W)
    total_width = sum(item["width"] * item["demand"] for item in items)
    lb_continuous = total_width / W
    lb = math.ceil(lb_continuous - tol)  # ceil with tolerance

    if z_int < lb - tol:
        # LHS is z (number of bins), RHS is lb (minimum needed)
        # This is a >= constraint: z >= lb, so violation = lb - z
        violation_amount = lb - z_int
        rhs = float(lb)
        normalizer = max(abs(rhs), eps)
        violated_constraints.add(2)
        violations.append(
            f"Constraint 2: z = {z_int} bins insufficient; need at least {lb} "
            f"bins to satisfy total demand (total item width = {total_width}, "
            f"bin capacity = {W})."
        )
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": float(z_int),
            "rhs": rhs,
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    # Tighter lower bound: items with w_i > W/2 cannot share a bin
    large_item_bins = sum(
        item["demand"] for item in items if item["width"] > W / 2
    )
    if z_int < large_item_bins - tol:
        violation_amount = large_item_bins - z_int
        rhs = float(large_item_bins)
        normalizer = max(abs(rhs), eps)
        if 2 not in violated_constraints or True:
            # Only add if this gives a tighter bound
            violated_constraints.add(2)
            violations.append(
                f"Constraint 2: z = {z_int} bins insufficient; at least "
                f"{large_item_bins} bins needed for items with width > W/2."
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(z_int),
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # --- Constraint 1 (eq 12): Flow conservation ---
    # At vertex 0: outflow - inflow = z  (z >= 0)
    # At vertex W: outflow - inflow = -z
    # At other vertices: outflow - inflow = 0
    # Necessary condition: z >= 0 (already checked under constraint 3)
    # Additional necessary condition from flow structure:
    #   A valid integral flow of value z through the network from 0 to W must
    #   exist that respects demand constraints. The continuous relaxation lower
    #   bound is ceil(sum(w_i * d_i) / W), already checked above.
    #   We also check that z does not exceed the trivial upper bound
    #   (one item per bin).

    total_items = sum(item["demand"] for item in items)
    if z_int > total_items + tol:
        # z exceeds the number of items — more bins than items
        # This means the flow is sending more units 0->W than there are items,
        # which violates flow conservation (arcs would need negative values
        # or extra loss-only bins beyond what's needed).
        # LHS = z, RHS = total_items, constraint type: <=
        violation_amount = z_int - total_items
        rhs = float(total_items)
        normalizer = max(abs(rhs), eps)
        violated_constraints.add(1)
        violations.append(
            f"Constraint 1: z = {z_int} exceeds total number of items "
            f"{total_items}; flow conservation cannot hold with more bins "
            f"than items."
        )
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": float(z_int),
            "rhs": rhs,
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer),
        })

    # Validate the complete arc-flow witness without running a solver.
    if not violated_constraints:
        if instance_path is None:
            witness_objective, witness_errors = None, ["instance_path is required for solverless validation"]
        else:
            witness_objective, witness_errors, _ = validate_complete_witness(
                instance_path, solution, Path(__file__).resolve().parent
            )
            witness_errors.extend(require_linked_variables(solution, {"z": float(z_int)}))
        if witness_errors:
            violated_constraints.add(1)
            violations.extend(f"Constraint 1: {message}" for message in witness_errors)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(z_int),
                "rhs": float(z_int),
                "raw_excess": 1.0,
                "normalizer": max(abs(float(z_int)), eps),
                "ratio": 1.0 / max(abs(float(z_int)), eps),
            })
        elif witness_objective is None or abs(float(z_int) - witness_objective) > tol:
            violated_constraints.add(1)
            difference = abs(float(z_int) - float(witness_objective or 0.0))
            violations.append(
                f"Constraint 1: objective_value={z_int} does not equal the "
                f"complete witness objective={witness_objective}."
            )
            violation_magnitudes.append({
                "constraint": 1, "lhs": float(z_int), "rhs": witness_objective,
                "raw_excess": difference, "normalizer": max(abs(float(witness_objective or 0.0)), eps),
                "ratio": difference / max(abs(float(witness_objective or 0.0)), eps),
            })

    # Build result
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
        description="Feasibility checker for CSP arc-flow (Côté & Iori 2018)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
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


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
