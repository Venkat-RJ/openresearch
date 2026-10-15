"""
Feasibility checker for the Fixed Charge Transportation Problem (FCTP).

Checks the candidate solution against Formulation F0 from Roberti, Bartolini,
Mingozzi (2014).

Hard constraints (numbered per the paper):
  (2) Supply:       sum_j x_{ij} = a_i,           for all i in S
  (3) Demand:       sum_i x_{ij} = b_j,           for all j in T
  (4) Linking:      x_{ij} <= m_{ij} * y_{ij},    for all (i,j) in A
  (5) Non-neg:      x_{ij} >= 0,                  for all (i,j) in A
  (6) Binary:       y_{ij} in {0, 1},             for all (i,j) in A
  (7) Objective consistency (Tier C): reported objective_value must equal
      sum_{i,j} (c_{ij} * x_{ij} + f_{ij} * y_{ij}) within tolerance.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('arcs_used', 'flow', 'objective_value')
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
    with open(path, "r") as fh:
        return json.load(fh)


def check_feasibility(instance, solution):
    m = instance["num_sources"]
    n = instance["num_sinks"]
    a = instance["supply"]
    b = instance["demand"]
    cap = instance["capacity"]  # m_{ij} = min(a_i, b_j)

    flow_raw = solution["flow"]
    swapped = solution.get("swapped", False)

    # Domain pre-check: flow and arcs_used must be 2D arrays of the right
    # shape (m x n in non-swapped orientation, n x m if swapped). Fail
    # cleanly here rather than crashing with IndexError downstream.
    expected_outer = n if swapped else m
    expected_inner = m if swapped else n
    _shape_violations = []
    if not isinstance(flow_raw, list) or len(flow_raw) != expected_outer:
        _shape_violations.append(
            f"flow has outer length {len(flow_raw) if isinstance(flow_raw, list) else 'non-list'}, "
            f"expected {expected_outer}"
        )
    else:
        for i, row in enumerate(flow_raw):
            if not isinstance(row, list) or len(row) != expected_inner:
                _shape_violations.append(
                    f"flow row {i} length {len(row) if isinstance(row, list) else 'non-list'}, "
                    f"expected {expected_inner}"
                )
                break
    if "arcs_used" in solution:
        au = solution["arcs_used"]
        if not isinstance(au, list) or len(au) != expected_outer:
            _shape_violations.append(
                f"arcs_used has outer length {len(au) if isinstance(au, list) else 'non-list'}, "
                f"expected {expected_outer}"
            )
        else:
            for i, row in enumerate(au):
                if not isinstance(row, list) or len(row) != expected_inner:
                    _shape_violations.append(
                        f"arcs_used row {i} length {len(row) if isinstance(row, list) else 'non-list'}, "
                        f"expected {expected_inner}"
                    )
                    break
    if _shape_violations:
        return {
            "feasible": False,
            "violated_constraints": [0],
            "violations": [f"Domain: {m_msg}" for m_msg in _shape_violations],
            "violation_magnitudes": [{
                "constraint": 0, "lhs": 0.0, "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            } for _ in _shape_violations],
        }

    # If the solution was produced in swapped orientation, transpose the flow
    # matrix back to the original (sources x sinks) layout.
    if swapped:
        # flow_raw is (n x m) when swapped; transpose to (m x n)
        x = [[float(flow_raw[j][i]) for j in range(n)] for i in range(m)]
    else:
        x = [[float(flow_raw[i][j]) for j in range(n)] for i in range(m)]

    # Derive y_{ij}: 1 if x_{ij} > 0, else 0
    # If arcs_used is provided, use it; otherwise infer from flow.
    if "arcs_used" in solution and not swapped:
        y = solution["arcs_used"]
    elif "arcs_used" in solution and swapped:
        arcs_raw = solution["arcs_used"]
        y = [[arcs_raw[j][i] for j in range(n)] for i in range(m)]
    else:
        y = [[1 if x[i][j] > TOL else 0 for j in range(n)] for i in range(m)]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint (2): Supply — sum_j x_{ij} = a_i, for all i
    # ------------------------------------------------------------------
    for i in range(m):
        lhs = sum(x[i][j] for j in range(n))
        rhs = float(a[i])
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            violated_constraints.add(2)
            violations.append(
                f"Constraint (2): Supply violation at source {i}: "
                f"sum_j x[{i}][j] = {lhs}, but a[{i}] = {rhs}"
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

    # ------------------------------------------------------------------
    # Constraint (3): Demand — sum_i x_{ij} = b_j, for all j
    # ------------------------------------------------------------------
    for j in range(n):
        lhs = sum(x[i][j] for i in range(m))
        rhs = float(b[j])
        violation_amount = abs(lhs - rhs)
        if violation_amount > TOL:
            violated_constraints.add(3)
            violations.append(
                f"Constraint (3): Demand violation at sink {j}: "
                f"sum_i x[i][{j}] = {lhs}, but b[{j}] = {rhs}"
            )
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint (4): Linking — x_{ij} <= m_{ij} * y_{ij}, for all (i,j)
    # ------------------------------------------------------------------
    for i in range(m):
        for j in range(n):
            m_ij = float(cap[i][j])
            lhs = x[i][j]
            rhs = m_ij * float(y[i][j])
            violation_amount = lhs - rhs  # LHS - RHS for <= constraint
            if violation_amount > TOL:
                violated_constraints.add(4)
                violations.append(
                    f"Constraint (4): Linking violation on arc ({i},{j}): "
                    f"x[{i}][{j}] = {lhs} > m[{i}][{j}]*y[{i}][{j}] = {rhs}"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint (5): Non-negativity — x_{ij} >= 0, for all (i,j)
    # ------------------------------------------------------------------
    for i in range(m):
        for j in range(n):
            lhs = x[i][j]
            rhs = 0.0
            violation_amount = rhs - lhs  # RHS - LHS for >= constraint
            if violation_amount > TOL:
                violated_constraints.add(5)
                violations.append(
                    f"Constraint (5): Non-negativity violation on arc ({i},{j}): "
                    f"x[{i}][{j}] = {lhs} < 0"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint (6): Binary domain — y_{ij} in {0, 1}, for all (i,j)
    # ------------------------------------------------------------------
    for i in range(m):
        for j in range(n):
            y_val = float(y[i][j])
            # Check if y_val is 0 or 1 within tolerance
            dist_0 = abs(y_val - 0.0)
            dist_1 = abs(y_val - 1.0)
            violation_amount = min(dist_0, dist_1)
            if violation_amount > TOL:
                violated_constraints.add(6)
                # For binary domain, treat as equality to nearest binary value
                nearest = 0.0 if dist_0 <= dist_1 else 1.0
                lhs = y_val
                rhs = nearest
                violations.append(
                    f"Constraint (6): Binary violation on arc ({i},{j}): "
                    f"y[{i}][{j}] = {y_val} not in {{0, 1}}"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint (7): Objective consistency (Tier C defense) — the reported
    # objective_value must equal sum_{i,j} (c_{ij} x_{ij} + f_{ij} y_{ij})
    # within tolerance. All variables in this formula are present in the
    # solution (flow -> x, arcs_used -> y) and the instance (variable_cost ->
    # c, fixed_cost -> f), so this is a FULL recompute.
    # ------------------------------------------------------------------
    c = instance.get("variable_cost")
    f = instance.get("fixed_cost")
    reported_obj = solution.get("objective_value")
    if c is not None and f is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            true_obj = 0.0
            for i in range(m):
                for j in range(n):
                    true_obj += float(c[i][j]) * x[i][j] + float(f[i][j]) * float(y[i][j])
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(7)
                violations.append(
                    f"Constraint (7): Objective consistency violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_ij(c_ij*x_ij + f_ij*y_ij)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })
        elif reported is not None and not math.isfinite(reported):
            # Non-finite reported obj (inf / nan / sys.float_info.max-style) is
            # never a valid FCTP cost — reject as obj-consistency violation.
            violated_constraints.add(7)
            violations.append(
                f"Constraint (7): Objective consistency violated: reported "
                f"objective_value={reported} is non-finite or out of range"
            )
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": reported,
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": 1.0,
                "ratio": float("inf"),
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
        description="Feasibility checker for FCTP (Formulation F0)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON candidate solution file"
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON feasibility result"
    )
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

    with open(args.result_path, "w") as fh:
        json.dump(result, fh, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE — no constraint violations found.")
    else:
        print(f"INFEASIBLE — violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
