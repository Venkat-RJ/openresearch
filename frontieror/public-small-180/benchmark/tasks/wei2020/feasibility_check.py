#!/usr/bin/env python3
"""
Feasibility checker for the 1D Bin-Packing Problem (1D-BPP / 1D-BPPC)
based on the set-partitioning formulation from Wei et al. (2019).

Checks hard constraints numbered top-to-bottom from the formulation:
  Constraint 1 (eq 2): Each item assigned to exactly one bin.
  Constraint 2 (eq 3): Subset row (SR) inequalities for item triplets.
  Constraint 3 (eq 4): Binary/integrality of pattern selection variables.
  Constraint 4 (eq 5): Bin capacity — total weight per bin <= c.
  Constraint 8 (eq 9): Conflict constraints (BPPC only) — no two
                        conflicting items share a bin.
  Constraint 9       : Objective consistency — reported objective_value
                        must equal the number of non-empty bins in the
                        solution (z = sum_b x_b). Tier C defense against
                        score-gaming exploits.

Constraints 5-7 (eqs 6-8) are internal to the pricing sub-problem and
do not apply to the master solution.
"""

import argparse
import json
from itertools import combinations
from collections import Counter

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('bins', 'objective_value')
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
    n = instance["n"]
    c = instance["c"]
    weights = instance["weights"]
    conflicts = instance.get("conflicts", [])

    bins = solution["bins"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 1 (eq 2): Each item assigned to exactly one bin.
    #   sum_{p in P} a_{ip} x_p = 1,  for all i in N
    # LHS = number of bins containing item i, RHS = 1.
    # ------------------------------------------------------------------
    item_count = Counter()
    for bin_items in bins:
        for item in bin_items:
            item_count[item] += 1

    # Check for items appearing more than once
    for item in range(n):
        count = item_count.get(item, 0)
        if abs(count - 1) > TOL:
            violation_amount = abs(count - 1)
            rhs = 1.0
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer

            if count == 0:
                msg = f"Item {item} is not assigned to any bin"
            else:
                msg = f"Item {item} is assigned to {count} bins instead of exactly 1"

            if 1 not in violated_constraints:
                violations.append(msg)
                violated_constraints.add(1)
            else:
                violations.append(msg)

            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(count),
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # Check for invalid item indices
    all_items_in_bins = set()
    for bin_items in bins:
        for item in bin_items:
            all_items_in_bins.add(item)
            if item < 0 or item >= n:
                violation_amount = 1.0
                rhs = 1.0
                normalizer = max(abs(rhs), EPS)
                ratio = violation_amount / normalizer

                msg = f"Item {item} is out of valid range [0, {n-1}]"
                if 1 not in violated_constraints:
                    violated_constraints.add(1)
                violations.append(msg)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": 0.0,
                    "rhs": rhs,
                    "raw_excess": float(violation_amount),
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # ------------------------------------------------------------------
    # NOTE: Subset-row (SR) inequalities (paper Eq. 3) are rank-1 valid
    # inequalities of the set-partitioning reformulation, NOT constraints
    # of the original 1D-BPP/BPPC. They are intentionally NOT enforced
    # here: any solver that does not use the SR cuts (e.g. compact
    # assignment MIP, FFD heuristic) must not be falsely flagged as
    # infeasible.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 3 (eq 4): Binary/integrality x_p in {0, 1}.
    # The solution format gives bins as lists of items, which implicitly
    # means each pattern is either used (appears in bins list) or not.
    # This is always satisfied by the solution format.
    # ------------------------------------------------------------------
    # No violation possible from the solution format.

    # ------------------------------------------------------------------
    # Constraint 4 (eq 5): Bin capacity.
    #   sum_{i in N_p} w_i <= c,  for each used bin/pattern p
    # LHS = total weight of items in bin, RHS = c.
    # ------------------------------------------------------------------
    for b_idx, bin_items in enumerate(bins):
        total_weight = sum(weights[item] for item in bin_items
                          if 0 <= item < n)
        rhs = float(c)
        violation_amount = max(0.0, total_weight - rhs)
        if violation_amount > TOL:
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer

            msg = (f"Capacity exceeded on bin {b_idx}: total weight {total_weight} "
                   f"> capacity {c}")
            if 4 not in violated_constraints:
                violated_constraints.add(4)
            violations.append(msg)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(total_weight),
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # ------------------------------------------------------------------
    # Constraint 8 (eq 9): Conflict constraints (BPPC only).
    #   y_i + y_j <= 1,  for all {i, j} in E
    #   i.e., no two conflicting items share a bin.
    # LHS = 1 if both i,j in same bin, 0 otherwise. RHS = 1.
    # ------------------------------------------------------------------
    if conflicts:
        conflict_set = set()
        for edge in conflicts:
            i, j = edge
            conflict_set.add((min(i, j), max(i, j)))

        for b_idx, bin_items in enumerate(bins):
            bin_list = list(bin_items)
            for a in range(len(bin_list)):
                for b in range(a + 1, len(bin_list)):
                    pair = (min(bin_list[a], bin_list[b]),
                            max(bin_list[a], bin_list[b]))
                    if pair in conflict_set:
                        lhs = 2.0
                        rhs = 1.0
                        violation_amount = lhs - rhs
                        normalizer = max(abs(rhs), EPS)
                        ratio = violation_amount / normalizer

                        msg = (f"Conflict violated in bin {b_idx}: "
                               f"items {pair[0]} and {pair[1]} are in conflict")
                        if 8 not in violated_constraints:
                            violated_constraints.add(8)
                        violations.append(msg)
                        violation_magnitudes.append({
                            "constraint": 8,
                            "lhs": lhs,
                            "rhs": rhs,
                            "raw_excess": float(violation_amount),
                            "normalizer": normalizer,
                            "ratio": ratio,
                        })

    # ------------------------------------------------------------------
    # Constraint 9: Objective consistency (Tier C anti-exploit defense).
    #   z(F) = sum_{p in P} x_p = number of patterns (bins) used.
    # The solution writes `bins` as the list of used patterns, so the
    # true objective is exactly the count of non-empty bins. Reject if
    # the reported objective_value disagrees by more than 0.5 (the obj
    # is integer-valued, so any mismatch >=1 is a real lie).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(1 for bin_items in bins if len(bin_items) > 0))
            abs_diff = abs(reported - true_obj)
            tol = 0.5  # objective is integer count of bins used
            if abs_diff > tol:
                normalizer = max(abs(true_obj), EPS)
                ratio = abs_diff / normalizer
                msg = (f"Objective consistency violated: reported objective_value="
                       f"{reported} differs from recomputed number-of-used-bins="
                       f"{true_obj} (|diff|={abs_diff:.3g}, tol={tol})")
                violated_constraints.add(9)
                violations.append(msg)
                violation_magnitudes.append({
                    "constraint": 9,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": normalizer,
                    "ratio": ratio,
                })

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
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
        description="Feasibility checker for 1D-BPP/BPPC (Wei et al. 2019)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    with open(args.solution_path) as f:
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

    if result["feasible"]:
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
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
