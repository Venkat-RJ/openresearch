"""
Feasibility checker for RCPSP/max-π (Resource-Constrained Project Scheduling
Problem with Partially Renewable Resources and General Temporal Constraints).

From: Watermeyer & Zimmermann (2022), OR Spectrum 44:575-602

Constraints (numbered top-to-bottom from the formulation):
  (1) S_j - S_i >= delta_ij  for all (i,j) in E        [temporal constraints]
  (2) S_0 = 0                                            [project start]
  (3) r^c_k(S) <= R_k  for all k in R                   [resource constraints]
  (4) S_i in Z_{>=0}  for all i in V                     [non-negative integers]
  (5) objective_value == S_{n+1}                         [obj consistency / Tier C]
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'schedule')
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


TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    schedule = {int(k): v for k, v in solution["schedule"].items()}

    # Parse instance
    activities = {a["id"]: a for a in instance["activities"]}
    activity_ids = sorted(activities.keys())
    p = {a_id: activities[a_id]["processing_time"] for a_id in activity_ids}
    temporal_constraints = instance["temporal_constraints"]
    resources = instance["resources"]

    # === Constraint (4): S_i in Z_{>=0} for all i in V ===
    for i in activity_ids:
        if i not in schedule:
            violated_constraints.add(4)
            violations.append(f"Activity {i} has no start time in the schedule")
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float("nan"),
                "rhs": 0.0,
                "raw_excess": float("inf"),
                "normalizer": EPS,
                "ratio": float("inf"),
            })
            continue
        s_i = schedule[i]
        # Check non-negative
        if s_i < -TOL:
            va = abs(s_i)
            rhs = 0.0
            if va > TOL:
                violated_constraints.add(4)
                violations.append(f"Activity {i} has negative start time S_{i}={s_i}")
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(s_i),
                    "rhs": rhs,
                    "raw_excess": float(va),
                    "normalizer": max(abs(rhs), EPS),
                    "ratio": float(va) / max(abs(rhs), EPS),
                })
        # Check integer
        if abs(s_i - round(s_i)) > TOL:
            va = abs(s_i - round(s_i))
            rhs = round(s_i)
            if va > TOL:
                violated_constraints.add(4)
                violations.append(
                    f"Activity {i} has non-integer start time S_{i}={s_i}"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(s_i),
                    "rhs": float(rhs),
                    "raw_excess": float(va),
                    "normalizer": max(abs(float(rhs)), EPS),
                    "ratio": float(va) / max(abs(float(rhs)), EPS),
                })

    # If missing activities, can't check further constraints reliably
    if any(i not in schedule for i in activity_ids):
        return _build_result(violated_constraints, violations, violation_magnitudes)

    # === Constraint (2): S_0 = 0 ===
    s_0 = schedule[0]
    va = abs(s_0 - 0.0)
    if va > TOL:
        violated_constraints.add(2)
        violations.append(f"Source activity 0 must start at time 0, got S_0={s_0}")
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": float(s_0),
            "rhs": 0.0,
            "raw_excess": float(va),
            "normalizer": max(abs(0.0), EPS),
            "ratio": float(va) / max(abs(0.0), EPS),
        })

    # === Constraint (1): S_j - S_i >= delta_ij for all (i,j) in E ===
    tc_violations = []
    for tc in temporal_constraints:
        i = tc["from_activity"]
        j = tc["to_activity"]
        d = tc["time_lag"]
        lhs = schedule[j] - schedule[i]
        rhs = d
        # This is a >= constraint: lhs >= rhs, violation = max(rhs - lhs, 0)
        va = max(rhs - lhs, 0.0)
        if va > TOL:
            tc_violations.append((i, j, d, lhs, rhs, va))

    if tc_violations:
        violated_constraints.add(1)
        for i, j, d, lhs, rhs, va in tc_violations:
            violations.append(
                f"Temporal constraint ({i},{j}): S_{j}-S_{i}={lhs} < delta={d}"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(va),
                "normalizer": max(abs(float(rhs)), EPS),
                "ratio": float(va) / max(abs(float(rhs)), EPS),
            })

    # === Constraint (3): r^c_k(S) <= R_k for all k in R ===
    for res in resources:
        kid = res["resource_id"]
        pi_k = set(res["Pi_k"])
        capacity = res["capacity"]
        demands = {int(k): v for k, v in res["demands"].items()}

        total_consumption = 0
        for i in activity_ids:
            demand = demands.get(i, 0)
            if demand == 0:
                continue
            s_i = schedule[i]
            pi = p[i]
            if pi == 0:
                continue
            # r^u_{ik}(S_i) = |{t' in Pi_k : S_i < t' <= S_i + p_i}|
            usage = sum(1 for t in pi_k if s_i < t <= s_i + pi)
            total_consumption += usage * demand

        lhs = total_consumption
        rhs = capacity
        # <= constraint: violation = max(lhs - rhs, 0)
        va = max(lhs - rhs, 0.0)
        if va > TOL:
            violated_constraints.add(3)
            violations.append(
                f"Resource {kid}: total consumption {lhs} exceeds capacity {rhs}"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(va),
                "normalizer": max(abs(float(rhs)), EPS),
                "ratio": float(va) / max(abs(float(rhs)), EPS),
            })

    # === Constraint (5): objective_value == S_{n+1} (Tier C obj consistency) ===
    # The objective is the start time of the project-end activity, which is
    # the activity with the largest id (== num_real_activities + 1). Every
    # variable the obj depends on is in the solution, so a full recompute is
    # exact. Reject when the reported objective_value disagrees.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # Determine end-activity id robustly: prefer num_real_activities+1
            # from the instance when available, else fall back to max id.
            n_real = instance.get("num_real_activities")
            if isinstance(n_real, int):
                end_id = n_real + 1
            else:
                end_id = max(activity_ids)
            if end_id in schedule:
                true_obj = float(schedule[end_id])
                # Objective is an integer count of time periods; use a tight
                # tolerance (0.1% relative with a 0.5 absolute floor so any
                # integer mismatch >=1 fires). NaN/inf reported values are
                # treated as violating regardless.
                tol = max(0.5, 1e-3 * abs(true_obj))
                bad = (not math.isfinite(reported)) or (abs(reported - true_obj) > tol)
                if bad:
                    abs_diff = (abs(reported - true_obj)
                                if math.isfinite(reported) else float("inf"))
                    violated_constraints.add(5)
                    violations.append(
                        f"Objective consistency violated: reported objective_value="
                        f"{reported} differs from S_{{{end_id}}}={true_obj} "
                        f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                    )
                    violation_magnitudes.append({
                        "constraint": 5,
                        "lhs": float(reported) if math.isfinite(reported) else float("inf"),
                        "rhs": float(true_obj),
                        "raw_excess": float(abs_diff) if math.isfinite(abs_diff) else float("inf"),
                        "normalizer": max(abs(float(true_obj)), EPS),
                        "ratio": (float(abs_diff) / max(abs(float(true_obj)), EPS))
                                  if math.isfinite(abs_diff) else float("inf"),
                    })

    return _build_result(violated_constraints, violations, violation_magnitudes)


def _build_result(violated_constraints, violations, violation_magnitudes):
    vc_sorted = sorted(violated_constraints)
    return {
        "feasible": len(vc_sorted) == 0,
        "violated_constraints": vc_sorted,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for RCPSP/max-pi"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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
