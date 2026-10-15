"""
Feasibility checker for the Multidimensional Multiple Choice Knapsack Problem (MMKP).

Reference: Mansini and Zanotti (2020), "A Core-Based Exact Algorithm for the
           Multidimensional Multiple Choice Knapsack Problem",
           INFORMS Journal on Computing.

Constraints (numbered top-to-bottom from the formulation):
  Constraint 1 (eq 2): Resource capacity constraints
      sum_{i,j} w_{ij}^k x_{ij} <= c_k,  for each k = 1,...,m
  Constraint 2 (eq 3): Multiple choice constraints
      sum_j x_{ij} = 1,  for each i = 1,...,n
  Constraint 3 (eq 4): Binary domain
      x_{ij} in {0, 1}
  Constraint 4 (obj consistency): Reported objective_value must equal the
      profit sum recomputed from the instance's true p_{ij} for each
      selected (group_id, item_id) pair.  Defends against LLM score-gaming
      where the candidate solution returns a fabricated objective_value
      while keeping the constraint variables feasible.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'selected_items')
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

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    n = instance["parameters"]["n_groups"]
    m = instance["parameters"]["m_resources"]
    capacities = instance["capacities"]
    groups = instance["groups"]

    selected = solution["selected_items"]

    # Build lookup: group_id -> list of selected items
    selected_by_group = {}
    for item in selected:
        gid = item["group_id"]
        selected_by_group.setdefault(gid, []).append(item)

    # Build instance item lookup for binary domain check
    instance_items = {}
    for group in groups:
        gid = group["group_id"]
        for it in group["items"]:
            instance_items[(gid, it["item_id"])] = it

    # ------------------------------------------------------------------
    # Constraint 1 (eq 2): Resource capacity constraints
    #   sum_{i,j} w_{ij}^k x_{ij} <= c_k, for k = 1,...,m
    # ------------------------------------------------------------------
    constraint_1_msgs = []
    for k in range(m):
        lhs = sum(item["weights"][k] for item in selected)
        rhs = capacities[k]
        violation_amount = max(0.0, lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(1)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            constraint_1_msgs.append(
                f"Resource {k}: total weight {lhs} exceeds capacity {rhs}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    if constraint_1_msgs:
        violations.append(
            "Resource capacity violated: " + "; ".join(constraint_1_msgs)
        )

    # ------------------------------------------------------------------
    # Constraint 2 (eq 3): Multiple choice constraints
    #   sum_j x_{ij} = 1, for each i = 1,...,n
    # ------------------------------------------------------------------
    constraint_2_msgs = []
    for i in range(n):
        count = len(selected_by_group.get(i, []))
        lhs = float(count)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            violated_constraints.add(2)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            if count == 0:
                constraint_2_msgs.append(f"Group {i}: no item selected")
            else:
                constraint_2_msgs.append(
                    f"Group {i}: {count} items selected instead of 1"
                )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    if constraint_2_msgs:
        violations.append(
            "Multiple choice violated: " + "; ".join(constraint_2_msgs)
        )

    # ------------------------------------------------------------------
    # Constraint 3 (eq 4): Binary domain
    #   x_{ij} in {0, 1}
    #   Check that every selected item references a valid (group_id, item_id)
    #   in the instance.  Since the solution format lists selected items
    #   explicitly (x=1), x values are inherently 0 or 1; we verify that
    #   the referenced items actually exist.
    # ------------------------------------------------------------------
    constraint_3_msgs = []
    for item in selected:
        gid = item["group_id"]
        jid = item["item_id"]
        if (gid, jid) not in instance_items:
            violated_constraints.add(3)
            lhs = 1.0  # solution sets x=1 for this item
            rhs = 0.0  # but item doesn't exist, so valid domain is empty
            violation_amount = 1.0
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            constraint_3_msgs.append(
                f"Item ({gid}, {jid}) does not exist in the instance"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    if constraint_3_msgs:
        violations.append(
            "Binary domain violated: " + "; ".join(constraint_3_msgs)
        )

    # ------------------------------------------------------------------
    # Constraint 4 (objective consistency, Tier C defense)
    #   reported objective_value must equal sum_{(i,j) selected} p_{ij}
    #   recomputed from the instance's true profits.  Only fires when the
    #   prior constraints don't already render the comparison meaningless
    #   (skip if any selected item references an unknown (gid,jid), since
    #   we can't look up its true profit -- that case is already flagged
    #   by constraint 3).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Recompute using only items that exist in the instance.  If any
            # selected (gid,jid) is unknown, constraint 3 already fires; we
            # still recompute over the known ones so the obj check remains
            # useful for the common case where all items exist.
            true_obj = 0.0
            unknown_in_selected = False
            for item in selected:
                key = (item["group_id"], item["item_id"])
                inst_item = instance_items.get(key)
                if inst_item is None:
                    unknown_in_selected = True
                    continue
                true_obj += float(inst_item["profit"])
            if not unknown_in_selected:
                abs_diff = abs(reported - true_obj)
                # Profits are integers; tighten tolerance so an integer
                # mismatch by >=1 fires.  Use absolute floor of 0.5 plus
                # a 0.1% relative band for very large objectives.
                tol_obj = max(0.5, 1e-3 * abs(true_obj))
                if abs_diff > tol_obj:
                    violated_constraints.add(4)
                    normalizer = max(abs(true_obj), eps)
                    ratio = abs_diff / normalizer
                    violations.append(
                        f"Objective consistency violated: reported "
                        f"objective_value={reported} differs from recomputed "
                        f"sum p_ij over selected items={true_obj} "
                        f"(|diff|={abs_diff:.3g}, tol={tol_obj:.3g})"
                    )
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": float(reported),
                        "rhs": float(true_obj),
                        "raw_excess": float(abs_diff),
                        "normalizer": float(normalizer),
                        "ratio": float(ratio),
                    })

    feasible = len(violated_constraints) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MMKP (Mansini & Zanotti, 2020)"
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
