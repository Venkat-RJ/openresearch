"""
Feasibility checker for the 3LSPD-U (Three-Level Lot-Sizing Problem with
Distribution and Uncapacitated production).

Checks the STD formulation constraints from Cunha & Melo (2021):
  Constraint (1): Objective function (not a hard constraint)
  Constraint (2): Inventory balance for plant and warehouses (equality)
  Constraint (3): Inventory balance for retailers (equality)
  Constraint (4): Setup forcing constraints (<=)
  Constraint (5): Nonnegativity of x and s (>=)
  Constraint (6): Integrality of y (binary)
  Constraint (7): Objective consistency (Tier C anti-gaming defense) -- the
                  reported objective_value is recomputed from the solution's
                  setup variables y and inventory variables s and must agree
                  with z_STD within tolerance.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value',)
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



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    NR = instance["num_retailers"]
    NW = instance["num_warehouses"]
    NT = instance["num_periods"]
    demands = instance["demands"]  # [NR][NT]
    assignment = instance["retailer_warehouse_assignment"]  # retailer r -> warehouse index
    warehouse_retailers = {}
    for w_key, r_list in instance["warehouse_retailer_sets"].items():
        warehouse_retailers[int(w_key)] = r_list

    # Compute aggregated demands
    # Warehouse demand: sum of assigned retailers' demands
    wh_demand = [[0.0] * NT for _ in range(NW)]
    for w in range(NW):
        for t in range(NT):
            wh_demand[w][t] = sum(demands[r][t] for r in warehouse_retailers[w])

    # Plant demand: total demand
    plant_demand = [0.0] * NT
    for t in range(NT):
        plant_demand[t] = sum(demands[r][t] for r in range(NR))

    # Cumulative demand from period t to NT-1 (0-indexed)
    # d^i_{t|T|} = sum_{l=t}^{NT-1} d^i_l
    retailer_cum = [[0.0] * NT for _ in range(NR)]
    for r in range(NR):
        retailer_cum[r][NT - 1] = demands[r][NT - 1]
        for t in range(NT - 2, -1, -1):
            retailer_cum[r][t] = retailer_cum[r][t + 1] + demands[r][t]

    wh_cum = [[0.0] * NT for _ in range(NW)]
    for w in range(NW):
        wh_cum[w][NT - 1] = wh_demand[w][NT - 1]
        for t in range(NT - 2, -1, -1):
            wh_cum[w][t] = wh_cum[w][t + 1] + wh_demand[w][t]

    plant_cum = [0.0] * NT
    plant_cum[NT - 1] = plant_demand[NT - 1]
    for t in range(NT - 2, -1, -1):
        plant_cum[t] = plant_cum[t + 1] + plant_demand[t]

    # Extract decision variables from solution
    # x_plant[t], s_plant[t], y_plant[t]
    # x_warehouses[w][t], s_warehouses[w][t], y_warehouses[w][t]
    # x_retailers[r][t], s_retailers[r][t], y_retailers[r][t]
    x_plant = solution.get("x_plant")
    s_plant = solution.get("s_plant")
    y_plant = solution.get("y_plant")
    x_warehouses = solution.get("x_warehouses")
    s_warehouses = solution.get("s_warehouses")
    y_warehouses = solution.get("y_warehouses")
    x_retailers = solution.get("x_retailers")
    s_retailers = solution.get("s_retailers")
    y_retailers = solution.get("y_retailers")

    missing_vars = [
        name for name, v in [
            ("x_plant", x_plant), ("s_plant", s_plant), ("y_plant", y_plant),
            ("x_warehouses", x_warehouses), ("s_warehouses", s_warehouses),
            ("y_warehouses", y_warehouses),
            ("x_retailers", x_retailers), ("s_retailers", s_retailers),
            ("y_retailers", y_retailers),
        ] if v is None
    ]

    if missing_vars:
        # Reviewer note Fix_7: solutions that only export objective_value used
        # to pass vacuously. Required decision variables MUST be present.
        return {
            "feasible": False,
            "violated_constraints": [0],
            "violations": [
                "Solution is missing required decision variable(s): "
                + ", ".join(missing_vars)
            ],
            "violation_magnitudes": []
        }

    def record_violation(constraint_idx, message, lhs, rhs, violation_amount):
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violated_constraints_set.add(constraint_idx)
        violations.append(message)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio
        })

    # =========================================================================
    # Constraint (2): Inventory balance for plant and warehouses
    # s^i_{t-1} + x^i_t = sum_{j in delta(i)} x^j_t + s^i_t
    # For plant: s_plant[t-1] + x_plant[t] = sum_w x_warehouses[w][t] + s_plant[t]
    # For warehouse w: s_warehouses[w][t-1] + x_warehouses[w][t] = sum_r x_retailers[r][t] + s_warehouses[w][t]
    # =========================================================================

    # Plant balance
    for t in range(NT):
        s_prev = 0.0 if t == 0 else s_plant[t - 1]
        lhs = s_prev + x_plant[t]
        rhs = sum(x_warehouses[w][t] for w in range(NW)) + s_plant[t]
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            record_violation(
                2,
                f"Plant inventory balance violated in period {t+1}: "
                f"LHS={lhs:.6f}, RHS={rhs:.6f}",
                lhs, rhs, violation_amount
            )

    # Warehouse balance
    for w in range(NW):
        for t in range(NT):
            s_prev = 0.0 if t == 0 else s_warehouses[w][t - 1]
            lhs = s_prev + x_warehouses[w][t]
            rhs_val = sum(x_retailers[r][t] for r in warehouse_retailers[w]) + s_warehouses[w][t]
            violation_amount = abs(lhs - rhs_val)
            if violation_amount > tol:
                record_violation(
                    2,
                    f"Warehouse {w} inventory balance violated in period {t+1}: "
                    f"LHS={lhs:.6f}, RHS={rhs_val:.6f}",
                    lhs, rhs_val, violation_amount
                )

    # =========================================================================
    # Constraint (3): Inventory balance for retailers
    # s^r_{t-1} + x^r_t = d^r_t + s^r_t
    # =========================================================================
    for r in range(NR):
        for t in range(NT):
            s_prev = 0.0 if t == 0 else s_retailers[r][t - 1]
            lhs = s_prev + x_retailers[r][t]
            rhs_val = demands[r][t] + s_retailers[r][t]
            violation_amount = abs(lhs - rhs_val)
            if violation_amount > tol:
                record_violation(
                    3,
                    f"Retailer {r} inventory balance violated in period {t+1}: "
                    f"LHS={lhs:.6f}, RHS={rhs_val:.6f}",
                    lhs, rhs_val, violation_amount
                )

    # =========================================================================
    # Constraint (4): Setup forcing constraints
    # x^i_t <= d^i_{t|T|} * y^i_t
    # =========================================================================

    # Plant
    for t in range(NT):
        lhs = x_plant[t]
        rhs_val = plant_cum[t] * y_plant[t]
        violation_amount = max(lhs - rhs_val, 0.0)
        if violation_amount > tol:
            record_violation(
                4,
                f"Plant setup forcing violated in period {t+1}: "
                f"x={lhs:.6f} > cum_demand*y={rhs_val:.6f}",
                lhs, rhs_val, violation_amount
            )

    # Warehouses
    for w in range(NW):
        for t in range(NT):
            lhs = x_warehouses[w][t]
            rhs_val = wh_cum[w][t] * y_warehouses[w][t]
            violation_amount = max(lhs - rhs_val, 0.0)
            if violation_amount > tol:
                record_violation(
                    4,
                    f"Warehouse {w} setup forcing violated in period {t+1}: "
                    f"x={lhs:.6f} > cum_demand*y={rhs_val:.6f}",
                    lhs, rhs_val, violation_amount
                )

    # Retailers
    for r in range(NR):
        for t in range(NT):
            lhs = x_retailers[r][t]
            rhs_val = retailer_cum[r][t] * y_retailers[r][t]
            violation_amount = max(lhs - rhs_val, 0.0)
            if violation_amount > tol:
                record_violation(
                    4,
                    f"Retailer {r} setup forcing violated in period {t+1}: "
                    f"x={lhs:.6f} > cum_demand*y={rhs_val:.6f}",
                    lhs, rhs_val, violation_amount
                )

    # =========================================================================
    # Constraint (5): Nonnegativity: x^i_t, s^i_t >= 0
    # =========================================================================

    # Plant
    for t in range(NT):
        for var_name, val in [("x_plant", x_plant[t]), ("s_plant", s_plant[t])]:
            if val < -tol:
                violation_amount = abs(val)
                record_violation(
                    5,
                    f"{var_name}[{t+1}] = {val:.6f} < 0",
                    val, 0.0, violation_amount
                )

    # Warehouses
    for w in range(NW):
        for t in range(NT):
            for var_name, val in [("x_warehouses", x_warehouses[w][t]),
                                   ("s_warehouses", s_warehouses[w][t])]:
                if val < -tol:
                    violation_amount = abs(val)
                    record_violation(
                        5,
                        f"{var_name}[{w}][{t+1}] = {val:.6f} < 0",
                        val, 0.0, violation_amount
                    )

    # Retailers
    for r in range(NR):
        for t in range(NT):
            for var_name, val in [("x_retailers", x_retailers[r][t]),
                                   ("s_retailers", s_retailers[r][t])]:
                if val < -tol:
                    violation_amount = abs(val)
                    record_violation(
                        5,
                        f"{var_name}[{r}][{t+1}] = {val:.6f} < 0",
                        val, 0.0, violation_amount
                    )

    # =========================================================================
    # Constraint (6): Integrality: y^i_t in {0, 1}
    # =========================================================================

    # Plant
    for t in range(NT):
        val = y_plant[t]
        dist = min(abs(val - 0.0), abs(val - 1.0))
        if dist > tol:
            record_violation(
                6,
                f"y_plant[{t+1}] = {val:.6f} is not binary",
                val, round(val), dist
            )

    # Warehouses
    for w in range(NW):
        for t in range(NT):
            val = y_warehouses[w][t]
            dist = min(abs(val - 0.0), abs(val - 1.0))
            if dist > tol:
                record_violation(
                    6,
                    f"y_warehouses[{w}][{t+1}] = {val:.6f} is not binary",
                    val, round(val), dist
                )

    # Retailers
    for r in range(NR):
        for t in range(NT):
            val = y_retailers[r][t]
            dist = min(abs(val - 0.0), abs(val - 1.0))
            if dist > tol:
                record_violation(
                    6,
                    f"y_retailers[{r}][{t+1}] = {val:.6f} is not binary",
                    val, round(val), dist
                )

    # =========================================================================
    # Constraint (7): Objective consistency (Tier C anti-gaming defense)
    # Full recompute of z_STD (eq. 1) from the solution variables:
    #   z_STD = sum_t ( sum_i sc^i_t y^i_t + sum_i hc^i_t s^i_t )
    # The STD objective depends ONLY on the setup variables y and the
    # inventory variables s, both of which are fully present in the
    # solution -- so an exact recompute is possible. Reject when the
    # self-reported objective_value disagrees beyond tolerance (this is the
    # path LLM-evolved candidates exploit by reporting obj=0 / obj=MAX_FLOAT
    # while the routes themselves pass constraints 2-6).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            sc_plant = instance["fixed_costs"]["plant"]        # [NT]
            sc_warehouses = instance["fixed_costs"]["warehouses"]  # [NW][NT]
            sc_retailers = instance["fixed_costs"]["retailers"]    # [NR][NT]
            hc_plant = float(instance["holding_costs"]["plant"])      # scalar
            hc_warehouse = float(instance["holding_costs"]["warehouses"])  # scalar
            hc_retailers = instance["holding_costs"]["retailers"]     # [NR]

            # Setup costs: sum_t sum_i sc^i_t * y^i_t
            setup_cost = sum(sc_plant[t] * y_plant[t] for t in range(NT))
            setup_cost += sum(sc_warehouses[w][t] * y_warehouses[w][t]
                              for w in range(NW) for t in range(NT))
            setup_cost += sum(sc_retailers[r][t] * y_retailers[r][t]
                              for r in range(NR) for t in range(NT))

            # Holding costs: sum_t sum_i hc^i_t * s^i_t
            holding_cost = hc_plant * sum(s_plant[t] for t in range(NT))
            holding_cost += hc_warehouse * sum(s_warehouses[w][t]
                                               for w in range(NW)
                                               for t in range(NT))
            holding_cost += sum(float(hc_retailers[r]) * s_retailers[r][t]
                                for r in range(NR) for t in range(NT))

            true_obj = float(setup_cost + holding_cost)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                record_violation(
                    7,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed z_STD="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol_obj:.6g})",
                    reported, true_obj, abs_diff
                )
        elif reported is not None:
            # Non-finite reported objective (inf / nan) can never equal a
            # finite recomputed z_STD -- treat as an obj-consistency failure.
            record_violation(
                7,
                f"Objective consistency violated: reported objective_value="
                f"{reported_obj} is not a finite number",
                reported, 0.0, float("inf")
            )

    # Build result
    violated_constraints = sorted(violated_constraints_set)
    _domain_check_vars_binary = [("y_plant", y_plant), ("y_warehouses", y_warehouses)]
    _domain_check_vars_integer = []

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 1: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > tol or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > tol:
                        violated_constraints.add(1)
                        violations.append(
                            f"Constraint 1 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        violation_magnitudes.append({
                            "constraint": 1,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    feasible = len(violated_constraints) == 0

    # Aggregate violation messages by constraint index
    aggregated_violations = []
    for c_idx in violated_constraints:
        msgs = [v for v in violations if v.startswith(
            {2: ("Plant inventory", "Warehouse"),
             3: ("Retailer",),
             4: ("Plant setup", "Warehouse", "Retailer"),
             5: ("x_", "s_"),
             6: ("y_",)}.get(c_idx, ("",))[0]
        ) or True]
        # Simpler: count violations per constraint
        count = sum(1 for vm in violation_magnitudes if vm["constraint"] == c_idx)
        aggregated_violations.append(
            f"Constraint ({c_idx}) violated {count} time(s)"
        )

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": aggregated_violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for 3LSPD-U (Cunha & Melo, 2021)"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON file containing the feasibility result")
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
