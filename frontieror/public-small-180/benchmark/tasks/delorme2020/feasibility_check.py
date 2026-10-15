#!/usr/bin/env python3
"""
Feasibility checker for Bin Packing / Cutting Stock from Delorme & Iori (2019).

Verifies the ORIGINAL BPP/CSP constraints against the original solution
structure (per-bin item lists), independent of the reflect-graph reformulation
used by gurobi_code.py. A solution produced by any other method (e.g. a
heuristic that does not use arc flows) can therefore be checked against the
same rules.

Constraints checked:
  Constraint 1: Demand satisfaction
                For each item type j, the number of times type j appears
                across all bins must be at least d_j.
  Constraint 2: Bin capacity
                Each bin's total weight must not exceed c.
  Constraint 3: Bin count consistency
                len(bin_assignments) must equal the reported number of bins
                (objective value / num_bins).
  Constraint 4: Domain
                bin_assignments must be a list of lists of valid item-type
                indices in {0, ..., m-1}.

When bin_assignments is missing the checker reports indeterminate
(feasible=None) and only enforces the necessary lower bounds on z that any
BPP solution must satisfy (total-weight / capacity and large-item count).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('bin_assignments', 'num_bins', 'objective_value')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('num_bins',)
_FRONTIEROR_DERIVED_FIELD_RULES = (('num_bins', 'length', 'bin_assignments', None),)
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



def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    c = data["c"]
    weights = data["weights"]
    demands = data.get("demands", [1] * len(weights))
    return c, weights, demands, data


def load_solution(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_violation_amount(lhs, rhs, op, tol):
    if op == ">=":
        amount = rhs - lhs
    elif op == "<=":
        amount = lhs - rhs
    elif op == "==":
        amount = abs(lhs - rhs)
    else:
        amount = 0.0
    return amount if amount > tol else None


def add_violation(violated_set, violations_list, magnitudes_list,
                  constraint_idx, message, lhs, rhs, op, tol, eps):
    violation_amount = compute_violation_amount(lhs, rhs, op, tol)
    if violation_amount is None:
        return
    normalizer = max(abs(rhs), eps)
    violated_set.add(constraint_idx)
    violations_list.append(message)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(violation_amount / normalizer),
    })


def check_bpp_packing(c, weights, demands, z_int, bin_assignments, tol, eps):
    """Verify a BPP packing against the original problem constraints.

    bin_assignments is a list of bins; each bin is a list of integer item
    *widths* (weights). The per-weight demand is computed by grouping the
    instance's raw weights/demands by width.
    """
    violated_set = set()
    violations_list = []
    magnitudes_list = []

    # Aggregate the instance's per-width demand from raw (weights, demands).
    width_demand = {}
    for w, d in zip(weights, demands):
        width_demand[w] = width_demand.get(w, 0) + d
    valid_widths = set(width_demand.keys())

    if not isinstance(bin_assignments, list):
        violated_set.add(4)
        violations_list.append(
            f"bin_assignments must be a list, got {type(bin_assignments).__name__}"
        )
        magnitudes_list.append({
            "constraint": 4, "lhs": 0.0, "rhs": 1.0,
            "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
        })
        return violated_set, violations_list, magnitudes_list

    sanitized_bins = []
    for b_idx, b in enumerate(bin_assignments):
        if not isinstance(b, list):
            violated_set.add(4)
            violations_list.append(
                f"bin_assignments[{b_idx}] is not a list (got {type(b).__name__})"
            )
            magnitudes_list.append({
                "constraint": 4, "lhs": 0.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
            sanitized_bins.append([])
            continue
        clean = []
        for w in b:
            if not isinstance(w, int) or w not in valid_widths:
                violated_set.add(4)
                violations_list.append(
                    f"bin_assignments[{b_idx}] contains invalid item width {w!r} "
                    f"(must be an integer width that appears in the instance)"
                )
                magnitudes_list.append({
                    "constraint": 4,
                    "lhs": float(w) if isinstance(w, (int, float)) else -1.0,
                    "rhs": float(c),
                    "raw_excess": 1.0,
                    "normalizer": max(float(c), eps),
                    "ratio": 1.0 / max(float(c), eps),
                })
            else:
                clean.append(w)
        sanitized_bins.append(clean)

    # Constraint 2: bin capacity
    for b_idx, b in enumerate(sanitized_bins):
        total_w = sum(b)
        add_violation(
            violated_set, violations_list, magnitudes_list,
            2,
            f"Bin {b_idx} total weight {total_w} exceeds capacity {c}",
            lhs=total_w, rhs=c, op="<=", tol=tol, eps=eps,
        )

    # Constraint 1: demand satisfaction (per width)
    width_count = {}
    for b in sanitized_bins:
        for w in b:
            width_count[w] = width_count.get(w, 0) + 1
    for w, d_required in width_demand.items():
        d_packed = width_count.get(w, 0)
        add_violation(
            violated_set, violations_list, magnitudes_list,
            1,
            f"Width {w} packed {d_packed} time(s) but demand is {d_required}",
            lhs=d_packed, rhs=d_required, op=">=", tol=tol, eps=eps,
        )

    # Constraint 3: bin count consistency
    add_violation(
        violated_set, violations_list, magnitudes_list,
        3,
        f"Number of bins in assignment ({len(bin_assignments)}) does not match "
        f"reported z ({z_int})",
        lhs=len(bin_assignments), rhs=z_int, op="==", tol=tol, eps=eps,
    )

    return violated_set, violations_list, magnitudes_list


def check_z_lower_bounds(c, weights, demands, z, tol, eps):
    """Necessary lower bounds on z when no packing is provided."""
    violated_set = set()
    violations_list = []
    magnitudes_list = []

    total_weight = sum(w * d for w, d in zip(weights, demands))
    lb_capacity = math.ceil(total_weight / c)
    add_violation(
        violated_set, violations_list, magnitudes_list,
        3,
        f"Reported z ({z}) below capacity lower bound "
        f"ceil({total_weight}/{c})={lb_capacity}",
        lhs=z, rhs=lb_capacity, op=">=", tol=tol, eps=eps,
    )

    half_c = c / 2.0
    large_item_count = sum(d for w, d in zip(weights, demands) if w > half_c)
    add_violation(
        violated_set, violations_list, magnitudes_list,
        3,
        f"Reported z ({z}) below large-item lower bound ({large_item_count} items "
        f"with weight > c/2 each need their own bin)",
        lhs=z, rhs=large_item_count, op=">=", tol=tol, eps=eps,
    )

    return violated_set, violations_list, magnitudes_list


def main():
    parser = argparse.ArgumentParser(
        description="BPP-native feasibility checker (Delorme & Iori 2019)"
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    tol = 1e-5
    eps = 1e-5

    c, weights, demands, _ = load_instance(args.instance_path)
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

    z = solution.get("objective_value")
    if z is None:
        z = solution.get("num_bins")

    if z is None:
        status = solution.get("status", "unknown")
        result = {
            "feasible": None,
            "violated_constraints": [],
            "violations": [f"No solution to check (status={status})"],
            "violation_magnitudes": [],
        }
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
        return

    z_float = float(z)
    z_int = int(round(z_float))

    bin_assignments = solution.get("bin_assignments")

    if bin_assignments is not None:
        violated_set, violations_list, magnitudes_list = check_bpp_packing(
            c, weights, demands, z_int, bin_assignments, tol, eps
        )
        feasible = len(violated_set) == 0
    else:
        violated_set, violations_list, magnitudes_list = check_z_lower_bounds(
            c, weights, demands, z_float, tol, eps
        )
        if violated_set:
            feasible = False
        else:
            feasible = None
            violations_list.append(
                "bin_assignments missing — only necessary lower bounds on z were "
                "checked; full feasibility cannot be verified."
            )

    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations_list,
        "violation_magnitudes": magnitudes_list,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
