"""
Feasibility checker for the Capacitated Facility Location Problem (CFLP).

Reference: Klose & Drexl (2005), "Lower Bounds for the Capacitated Facility
Location Problem Based on Column Generation", Management Science 51(11):1689-1705.

Constraints (numbered top-to-bottom from the formulation):
  (1) Demand:      sum_j x_{ij} = 1            for all i in I
  (2) Capacity:    sum_i d_i x_{ij} <= s_j y_j  for all j in J
  (3) Var bounds:  0 <= x_{ij} <= 1             for all i in I, j in J
  (4) Var bounds:  0 <= y_j <= 1                for all j in J
  (5) Integrality: y_j in {0, 1}               for all j in J
  (6) Objective consistency: the reported objective_value must equal the
      objective recomputed from the solution variables,
      Z = sum_{i,j} c_{ij} x_{ij} + sum_j f_j y_j.
      This is a Tier C defense against score-gaming exploits where a
      candidate returns a fabricated objective_value while the routes /
      assignments remain technically feasible. Both x_{ij} and y_j are
      fully present in the solution, so a full recompute is possible.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignments', 'objective_value', 'open_plants')
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    n_cust = instance["num_customers"]
    n_plant = instance["num_plants"]
    demands = instance["customers"]["demands"]
    capacities = instance["plants"]["capacities"]

    open_plants = set(solution["open_plants"])
    assignments = solution["assignments"]

    # Build y vector: y_j = 1 if j in open_plants, else 0
    y = [1.0 if j in open_plants else 0.0 for j in range(n_plant)]

    # Build x matrix from assignments (sparse dict of dicts)
    # assignments is keyed by customer index (string) -> {plant_index (string): fraction}
    x = [[0.0] * n_plant for _ in range(n_cust)]
    for i_str, alloc in assignments.items():
        i = int(i_str)
        for j_str, val in alloc.items():
            j = int(j_str)
            x[i][j] = val

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint (1): Demand  --  sum_j x_{ij} = 1  for all i in I
    # =========================================================================
    for i in range(n_cust):
        lhs = sum(x[i][j] for j in range(n_plant))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            violated_constraints.add(1)
            violations.append(
                f"Constraint (1) violated: customer {i} demand fraction sum = {lhs:.8f}, expected 1.0"
            )
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (2): Capacity  --  sum_i d_i x_{ij} <= s_j y_j  for all j in J
    # =========================================================================
    for j in range(n_plant):
        lhs = sum(demands[i] * x[i][j] for i in range(n_cust))
        rhs = capacities[j] * y[j]
        violation_amount = lhs - rhs  # positive when LHS > RHS
        if violation_amount > TOL:
            violated_constraints.add(2)
            violations.append(
                f"Constraint (2) violated: plant {j} load = {lhs:.8f} exceeds capacity {rhs:.8f}"
            )
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (3): Variable bounds on x  --  0 <= x_{ij} <= 1  for all i, j
    # =========================================================================
    for i in range(n_cust):
        for j in range(n_plant):
            val = x[i][j]
            # Check lower bound: 0 <= x_{ij}  =>  x_{ij} >= 0
            if val < -TOL:
                violation_amount = -val  # how much below 0
                violated_constraints.add(3)
                violations.append(
                    f"Constraint (3) violated: x[{i}][{j}] = {val:.8f} < 0"
                )
                rhs = 0.0
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": val,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })
            # Check upper bound: x_{ij} <= 1
            if val > 1.0 + TOL:
                violation_amount = val - 1.0  # how much above 1
                violated_constraints.add(3)
                violations.append(
                    f"Constraint (3) violated: x[{i}][{j}] = {val:.8f} > 1"
                )
                rhs = 1.0
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": val,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # =========================================================================
    # Constraint (4): Variable bounds on y  --  0 <= y_j <= 1  for all j in J
    # =========================================================================
    for j in range(n_plant):
        val = y[j]
        if val < -TOL:
            violation_amount = -val
            violated_constraints.add(4)
            violations.append(
                f"Constraint (4) violated: y[{j}] = {val:.8f} < 0"
            )
            rhs = 0.0
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": val,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })
        if val > 1.0 + TOL:
            violation_amount = val - 1.0
            violated_constraints.add(4)
            violations.append(
                f"Constraint (4) violated: y[{j}] = {val:.8f} > 1"
            )
            rhs = 1.0
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": val,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (5): Integrality  --  y_j in {0, 1}  for all j in J
    # =========================================================================
    for j in range(n_plant):
        val = y[j]
        dist_to_int = min(abs(val - 0.0), abs(val - 1.0))
        if dist_to_int > TOL:
            violated_constraints.add(5)
            violations.append(
                f"Constraint (5) violated: y[{j}] = {val:.8f} is not binary"
            )
            # For integrality, LHS is the value, RHS is nearest integer
            nearest_int = round(val)
            rhs = float(nearest_int)
            violation_amount = abs(val - rhs)
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": val,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # =========================================================================
    # Constraint (6): Objective consistency  --  the reported objective_value
    # must equal the objective recomputed from the solution variables:
    #     Z = sum_{i,j} c_{ij} x_{ij} + sum_j f_j y_j
    # where c_{ij} = shipping_costs[i][j] (cost of supplying ALL of customer
    # i's demand from plant j, so c_{ij} x_{ij} already accounts for demand)
    # and f_j = plants.fixed_costs[j]. Both x_{ij} and y_j are present in the
    # solution, so this is an exact full recompute. A 0.1% relative tolerance
    # (with a 1e-3 absolute floor) absorbs floating-point accumulation while
    # still rejecting fabricated objective values (e.g. 0 or float-max).
    # =========================================================================
    fixed_costs = instance["plants"]["fixed_costs"]
    shipping_costs = instance["shipping_costs"]
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            shipping_total = 0.0
            for i in range(n_cust):
                row = shipping_costs[i]
                xrow = x[i]
                for j in range(n_plant):
                    xv = xrow[j]
                    if xv != 0.0:
                        shipping_total += row[j] * xv
            fixed_total = sum(fixed_costs[j] * y[j] for j in range(n_plant))
            true_obj = shipping_total + fixed_total
            violation_amount = abs(reported - true_obj)
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if violation_amount > tol:
                violated_constraints.add(6)
                violations.append(
                    f"Constraint (6) violated: reported objective_value = {reported:.8f} "
                    f"differs from recomputed total cost {true_obj:.8f} "
                    f"(|diff| = {violation_amount:.6g}, tol = {tol:.6g})"
                )
                rhs = true_obj
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": reported,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })
        elif reported is not None:
            # Non-finite reported objective (inf / nan) can never match a
            # finite recomputed cost -- treat as an objective inconsistency.
            shipping_total = 0.0
            for i in range(n_cust):
                row = shipping_costs[i]
                xrow = x[i]
                for j in range(n_plant):
                    xv = xrow[j]
                    if xv != 0.0:
                        shipping_total += row[j] * xv
            fixed_total = sum(fixed_costs[j] * y[j] for j in range(n_plant))
            true_obj = shipping_total + fixed_total
            violated_constraints.add(6)
            violations.append(
                f"Constraint (6) violated: reported objective_value = {reported} "
                f"is not finite; recomputed total cost = {true_obj:.8f}"
            )
            normalizer = max(abs(true_obj), EPS)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": reported,
                "rhs": true_obj,
                "raw_excess": float("inf"),
                "normalizer": normalizer,
                "ratio": float("inf"),
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
        description="Feasibility checker for the Capacitated Facility Location Problem (CFLP)."
    )
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", required=True, help="Path to the solution JSON file.")
    parser.add_argument("--result_path", required=True, help="Path to write the feasibility result JSON file.")
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

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
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
