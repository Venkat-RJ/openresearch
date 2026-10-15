#!/usr/bin/env python3
"""
Feasibility checker for the released quadratic Uncapacitated Facility
Location (qUFL) benchmark.

Verifies candidate solutions against the mathematical formulation from:
Fischetti, Ljubic, Sinnl (2016) "Redesigning Benders Decomposition for
Large-Scale Facility Location", Management Science.

Constraints are numbered top-to-bottom as they appear in the formulation:

  Section 2.2 (qUFL, original quadratic form — objective uses c_ij * x_ij^2):
    Constraint 5:  sum_i x_ij = 1,          for all j in J         [eq 6]
    Constraint 6:  x_ij <= y_i,             for all i in I, j in J [eq 7]
    Constraint 8:  x_ij >= 0,               for all i in I, j in J [eq 9]
    Constraint 10: y_i in {0,1},            for all i in I         [eq 11]

  Objective-consistency (Tier C anti-gaming defense, this `_new` variant):
    Constraint 14: reported objective_value must match the cost recomputed
                   from the solution variables (y, x). For qUFL it is
                   sum_i f_i*y_i + sum_ij c_ij*x_ij^2             [eq 4].
                   This is fully determined by the dimension-complete (y, x)
                   witness. The required variant discriminator is exactly
                   "quadratic"; linear UFL is not accepted by this case.

The perspective reformulation auxiliary `z_ij` and its constraint
`x_ij^2 <= z_ij * y_i` (eq 8) is NOT checked: it is a solver-ready
linearization aux that a generated algorithm operating on (x, y) need not
produce. Same for the Benders-master valid inequality `sum y_i >= 2` (eq 13),
which is a cutting-plane device, not an original UFL/qUFL requirement.
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'variant', 'x', 'y')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('x', 'y')
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


def check_feasibility(instance, solution):
    """Check all hard constraints and return violations."""
    n = instance["num_facilities"]
    m = instance["num_customers"]

    y = solution.get("y")
    x = solution.get("x")
    variant = solution.get("variant")

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs, rhs, operator):
        """Record a constraint violation with normalized magnitude."""
        if operator == "eq":
            violation_amount = abs(lhs - rhs)
        elif operator == "leq":
            violation_amount = lhs - rhs
        elif operator == "geq":
            violation_amount = rhs - lhs
        else:
            violation_amount = 0.0

        if violation_amount > TOL:
            violated_constraints.add(constraint_idx)
            if msg not in violations:
                violations.append(msg)
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # The released benchmark target is the qUFL model.  The coefficients can
    # also support a linear UFL experiment, but accepting either objective in
    # one evaluator makes the target ambiguous and lets a candidate choose its
    # preferred interpretation.
    if variant != "quadratic":
        record_violation(
            14,
            f"Released fischetti2017 cases require variant='quadratic', got {variant!r}",
            0.0,
            1.0,
            "eq",
        )

    valid_shape = (
        isinstance(y, list)
        and len(y) == n
        and isinstance(x, list)
        and len(x) == n
        and all(isinstance(row, list) and len(row) == m for row in x)
    )
    if not valid_shape:
        record_violation(
            14,
            f"Expected y shape [{n}] and x shape [{n}][{m}]",
            0.0,
            1.0,
            "eq",
        )
    else:
        values = list(y) + [value for row in x for value in row]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value != value
            or value in (float("inf"), float("-inf"))
            for value in values
        ):
            record_violation(
                14,
                "Every x/y entry must be a finite JSON number",
                0.0,
                1.0,
                "eq",
            )

    if variant != "quadratic" or not valid_shape or violated_constraints:
        return {
            "feasible": False,
            "violated_constraints": sorted(violated_constraints),
            "violations": violations,
            "violation_magnitudes": violation_magnitudes,
        }

    if variant == "quadratic":
        # ----- Check constraints 5, 6, 8, 10 for the original qUFL -----
        # Perspective aux z and its SOC constraint (7) are intentionally NOT
        # checked; `sum y_i >= 2` (Benders-VI, eq 13) is also NOT checked.

        # Constraint 5 (eq 6): sum_i x_ij = 1 for all j
        for j in range(m):
            lhs = sum(x[i][j] for i in range(n))
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                record_violation(5,
                    f"Constraint 5 (eq 6): Assignment violated for customer {j}: "
                    f"sum_i x_ij = {lhs:.6f} != 1",
                    lhs, rhs, "eq")

        # Constraint 6 (eq 7): x_ij <= y_i for all i, j
        for i in range(n):
            for j in range(m):
                lhs = x[i][j]
                rhs = y[i]
                if lhs - rhs > TOL:
                    record_violation(6,
                        f"Constraint 6 (eq 7): Linking violated for facility {i}, "
                        f"customer {j}: x[{i}][{j}]={lhs:.6f} > y[{i}]={rhs}",
                        lhs, rhs, "leq")

        # Constraint 8 (eq 9): x_ij >= 0 for all i, j
        for i in range(n):
            for j in range(m):
                lhs = x[i][j]
                rhs = 0.0
                if rhs - lhs > TOL:
                    record_violation(8,
                        f"Constraint 8 (eq 9): Non-negativity violated for "
                        f"x[{i}][{j}]={lhs:.6f} < 0",
                        lhs, rhs, "geq")

        # Constraint 10 (eq 11): y_i in {0, 1} for all i
        for i in range(n):
            val = y[i]
            dist = min(abs(val - 0.0), abs(val - 1.0))
            if dist > TOL:
                record_violation(10,
                    f"Constraint 10 (eq 11): Integrality violated for "
                    f"y[{i}]={val:.6f}",
                    val, round(val), "eq")

    elif variant == "linear" and x is not None:
        # ----- Check constraints 1-4 (disaggregated linear MILP) -----

        # Constraint 1 (eq 2): sum_i x_ij = 1 for all j
        for j in range(m):
            lhs = sum(x[i][j] for i in range(n))
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                record_violation(1,
                    f"Constraint 1 (eq 2): Assignment violated for customer {j}: "
                    f"sum_i x_ij = {lhs:.6f} != 1",
                    lhs, rhs, "eq")

        # Constraint 2 (eq 3): x_ij <= y_i for all i, j
        for i in range(n):
            for j in range(m):
                lhs = x[i][j]
                rhs = y[i]
                if lhs - rhs > TOL:
                    record_violation(2,
                        f"Constraint 2 (eq 3): Linking violated for facility {i}, "
                        f"customer {j}: x[{i}][{j}]={lhs:.6f} > y[{i}]={rhs}",
                        lhs, rhs, "leq")

        # Constraint 3: x_ij >= 0 for all i, j
        for i in range(n):
            for j in range(m):
                lhs = x[i][j]
                rhs = 0.0
                if rhs - lhs > TOL:
                    record_violation(3,
                        f"Constraint 3: Non-negativity violated for "
                        f"x[{i}][{j}]={lhs:.6f} < 0",
                        lhs, rhs, "geq")

        # Constraint 4: y_i in {0, 1} for all i
        for i in range(n):
            val = y[i]
            dist = min(abs(val - 0.0), abs(val - 1.0))
            if dist > TOL:
                record_violation(4,
                    f"Constraint 4: Integrality violated for y[{i}]={val:.6f}",
                    val, round(val), "eq")

    else:
        # Solution has only y (no x). We cannot verify assignment/linking
        # without x, so only check y integrality. The Benders-master valid
        # inequality `sum y_i >= 2` (eq 13) is NOT checked: it is a
        # cutting-plane device, not an original UFL/qUFL requirement.
        if y is not None:
            for i in range(n):
                val = y[i]
                dist = min(abs(val - 0.0), abs(val - 1.0))
                if dist > TOL:
                    record_violation(13,
                        f"Constraint 13 (eq 15): Integrality violated for "
                        f"y[{i}]={val:.6f}",
                        val, round(val), "eq")

    # ----- Constraint 14: objective-value consistency (Tier C defense) -----
    # Recompute the objective from the solution variables and reject when the
    # self-reported `objective_value` disagrees with what (y, x) actually
    # costs. The UFL objective  sum_i f_i*y_i + sum_ij c_ij*x_ij  [eq 1] and
    # the qUFL objective  sum_i f_i*y_i + sum_ij c_ij*x_ij^2  [eq 4] are both
    # fully determined by (y, x), so this is an exact full recompute.  The
    # released interface fixes the variant to quadratic.
    reported_obj = solution.get("objective_value")
    opening_costs = instance.get("opening_costs")
    allocation_costs = instance.get("allocation_costs")
    if reported_obj is not None and opening_costs is not None and y is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if (reported is not None and len(y) >= n and len(opening_costs) >= n):
            open_cost = sum(float(opening_costs[i]) * float(y[i])
                            for i in range(n))
            x_ok = (
                allocation_costs is not None
                and len(x) == n and all(len(x[i]) == m for i in range(n))
                and len(allocation_costs) >= n
                and all(len(allocation_costs[i]) >= m for i in range(n))
            )
            if x_ok:
                # Full recompute of the released quadratic allocation cost.
                quad_obj = open_cost + sum(
                    float(allocation_costs[i][j]) * float(x[i][j]) ** 2
                    for i in range(n) for j in range(m))
                abs_diff = abs(reported - quad_obj)
                # Serialized continuous x values can differ from Gurobi's
                # internal incumbent at the low-ppm level.
                tol = max(1e-6, 5e-6 * abs(quad_obj))
                if abs_diff > tol:
                    record_violation(14,
                        f"Constraint 14: Objective consistency violated: "
                        f"reported objective_value={reported} differs from "
                        f"the released quadratic recompute "
                        f"sum_i f_i*y_i + sum_ij c_ij*x_ij^2={quad_obj:.6f} "
                        f"(opening cost sum_i f_i*y_i={open_cost:.6f}; "
                        f"|diff|={abs_diff:.6g}, "
                        f"tol={tol:.6g})",
                        reported, quad_obj, "eq")
            else:
                # No x available: degrade to the opening-cost lower bound,
                # valid because allocation costs are non-negative.
                tol = max(1e-3, 1e-3 * abs(open_cost))
                if open_cost - reported > tol:
                    record_violation(14,
                        f"Constraint 14: Objective consistency violated: "
                        f"reported objective_value={reported} is below the "
                        f"opening-cost lower bound sum_i f_i*y_i="
                        f"{open_cost:.6f} (allocation costs are non-negative; "
                        f"x not provided, so only a lower bound is checked, "
                        f"tol={tol:.6g})",
                        reported, open_cost, "geq")

    # Build result
    sorted_violated = sorted(violated_constraints)
    feasible = len(sorted_violated) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for UFL/qUFL solutions"
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


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
