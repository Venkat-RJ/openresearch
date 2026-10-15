#!/usr/bin/env python3
"""
Feasibility checker for QSAP/AQSAP (Quadratic Semi-Assignment Problem).
Paper: Bayani, Rostami, Adulyasak, Rousseau (2021)
"A dual bounding framework for binary quadratic combinatorial optimization"

Checks the compact formulation (Equations 12-14):
  Constraint 1 (Eq 13): Each client assigned to exactly one server
  Constraint 2 (Eq 14): Assignment variables are binary (0 or 1)
  Constraint 3: Server indices are valid (0 <= j < h)
  Constraint 4: All clients are assigned
  Constraint 5: Objective value matches the computed cost
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'objective_value')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('assignment',)
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



def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for QSAP/AQSAP (Bayani et al. 2021)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    tol = 1e-4

    # ---- Load instance ----
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    # ---- Load solution ----
    with open(args.solution_path, "r") as f:
        sol = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
        if _frontieror_contract_result is not None:
            with open(args.result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return

    n = data["n_clients"]
    h = data["n_servers"]
    linear_costs = data["linear_costs"]
    quadratic_costs = data["quadratic_costs"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Check if solution has a valid objective value
    obj_val = sol.get("objective_value")
    assignment = sol.get("assignment", {})

    if obj_val is None or not assignment:
        result = {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No feasible solution found (objective_value is null or assignment is empty)"],
            "violation_magnitudes": []
        }
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("Feasibility check: No solution to check (null objective or empty assignment)")
        return

    # Convert assignment keys to int
    assign = {}
    for k, v in assignment.items():
        assign[int(k)] = int(v)

    # ================================================================
    # Constraint 1 (Eq 13): Each client assigned to exactly one server
    #   sum_{e in delta(i)} x_e = 1  for all i in N
    # ================================================================
    for i in range(n):
        if i not in assign:
            violated_constraints.add(1)
            violations.append(
                f"Constraint 1 (Assignment): Client {i} is not assigned to any server"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": 0,
                "rhs": 1,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0
            })

    # Check no client is assigned multiple times (shouldn't happen with dict)
    # but verify all clients are covered
    assigned_clients = set(assign.keys())
    expected_clients = set(range(n))
    missing = expected_clients - assigned_clients
    if missing:
        violated_constraints.add(1)
        violations.append(
            f"Constraint 1 (Assignment): Missing clients: {sorted(missing)}"
        )
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": len(assigned_clients),
            "rhs": n,
            "raw_excess": float(len(missing)),
            "normalizer": float(n),
            "ratio": float(len(missing)) / n
        })

    # ================================================================
    # Constraint 2 (Eq 14): Valid server indices
    #   x_e in {0,1}, server j in {0,...,h-1}
    # ================================================================
    for i, j in assign.items():
        if j < 0 or j >= h:
            violated_constraints.add(2)
            violations.append(
                f"Constraint 2 (Server range): Client {i} assigned to server {j}, "
                f"but valid range is [0, {h-1}]"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": j,
                "rhs": h - 1,
                "raw_excess": float(max(j - (h-1), -j)),
                "normalizer": float(h),
                "ratio": float(max(j - (h-1), -j)) / h
            })

    # ================================================================
    # Constraint 3: Objective value check
    #   obj = sum_{e} c_e x_e + sum_{(e,f)} q_{ef} x_e x_f
    # ================================================================
    if not violated_constraints:
        # Compute linear cost
        computed_linear = 0.0
        for i in range(n):
            j = assign[i]
            computed_linear += linear_costs[i][j]

        # Build quadratic cost lookup
        quad_dict = {}
        for entry in quadratic_costs:
            i1, j1, i2, j2 = int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3])
            q_val = entry[4]
            e = (i1, j1)
            f = (i2, j2)
            if (e, f) not in quad_dict and (f, e) not in quad_dict:
                quad_dict[(e, f)] = q_val

        # Compute quadratic cost
        computed_quad = 0.0
        for (e, f), q_val in quad_dict.items():
            i1, j1 = e
            i2, j2 = f
            # x_e = 1 iff client i1 assigned to server j1
            # x_f = 1 iff client i2 assigned to server j2
            if assign.get(i1) == j1 and assign.get(i2) == j2:
                computed_quad += q_val

        computed_obj = computed_linear + computed_quad

        obj_diff = abs(computed_obj - obj_val)
        normalizer = max(abs(obj_val), abs(computed_obj), 1e-5)
        relative_diff = obj_diff / normalizer

        if relative_diff > tol:
            violated_constraints.add(3)
            violations.append(
                f"Constraint 3 (Objective): Reported objective = {obj_val:.6f}, "
                f"computed objective = {computed_obj:.6f}, "
                f"relative diff = {relative_diff:.6e}"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": round(obj_val, 10),
                "rhs": round(computed_obj, 10),
                "raw_excess": round(obj_diff, 10),
                "normalizer": round(normalizer, 10),
                "ratio": round(relative_diff, 10)
            })

    # ---- Build result ----
    _domain_check_vars_binary = []  # assignment is facility index, not binary
    _domain_check_vars_integer = []

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 4: Binary domain — variables must be 0 or 1
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
                        violated_constraints.add(4)
                        violations.append(
                            f"Constraint 4 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        violation_magnitudes.append({
                            "constraint": 4,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else []
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility check complete. Result: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    if not feasible:
        print(f"Violated constraints: {sorted(violated_constraints)}")
        print(f"Number of violations: {len(violations)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
