#!/usr/bin/env python3
"""
Feasibility checker for RACP solutions based on Kreter et al. (2018):
"Mixed-Integer Linear Programming and Constraint Programming Formulations
 for Solving Resource Availability Cost Problems"

Checks hard constraints from the abstract RACP formulation (Section 2.1):
  Constraint (1): objective_value == sum_{k in R} c_k * R_k  (objective)
  Constraint (2): S_j - S_i >= p_i           for <i,j> in A      (precedence)
  Constraint (3): r_k(S,t) <= R_k            for k in R, t in T  (resource capacity)
  Constraint (4): S_i in {0,1,...,d_bar}      for i in V          (domain)
  Constraint (5): R_k >= 0                    for k in R          (non-negativity)

For RACP/max, constraint (6) replaces (2): S_j - S_i >= delta_{ij}.

This is the obj-recompute ("Tier C") variant of feasibility_check.py: it adds
Constraint (1), an objective-consistency check that recomputes the total
resource provision cost from the solution's resource_amounts and the
instance's resource_costs, and rejects solutions whose reported
objective_value disagrees. All other constraint logic is byte-for-byte
identical to feasibility_check.py.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'resource_amounts', 'start_times')
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



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    activities = instance['activities']
    precedence_relations = instance['precedence_relations']
    n_resources = instance['num_resource_types']
    problem_type = instance.get('problem_type', 'RACP')

    if 'max_project_duration' in instance:
        d_bar = instance['max_project_duration']
    else:
        d_bar = int(instance.get('deadline_factor', 1.0) *
                    instance.get('earliest_project_completion', 0))

    durations = {a['id']: a['duration'] for a in activities}
    requirements = {a['id']: a['resource_requirements'] for a in activities}
    act_ids = [a['id'] for a in activities]

    # Parse solution
    start_times = {int(k): v for k, v in solution['start_times'].items()}
    resource_amounts = {int(k): v for k, v in solution['resource_amounts'].items()}

    # Domain check on solution dict keys: every start_times key must be a
    # valid activity id; every resource_amounts key must be a valid
    # resource-type index in [0, n_resources).
    _domain_violations = []
    _act_id_set = set(act_ids)
    for sid in start_times.keys():
        if sid not in _act_id_set:
            _domain_violations.append(
                f"start_times key {sid!r} is not a valid activity id"
            )
    for rk in resource_amounts.keys():
        if not (0 <= rk < n_resources):
            _domain_violations.append(
                f"resource_amounts key {rk!r} is not a valid resource index "
                f"in [0, {n_resources})"
            )

    # Build arc weights
    arc_weights = {}
    if problem_type == 'RACP/max' and 'arc_weights' in instance:
        for idx, (i, j) in enumerate(precedence_relations):
            arc_weights[(i, j)] = instance['arc_weights'][idx]
    else:
        for (i, j) in precedence_relations:
            arc_weights[(i, j)] = durations[i]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    for _msg in _domain_violations:
        violated_constraints.add(0)
        violations.append(f"Domain: {_msg}")
        violation_magnitudes.append({
            "constraint": 0, "lhs": 0.0, "rhs": 0.0,
            "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
        })

    # Determine which constraint index to use for precedence
    # For RACP: constraint (2); for RACP/max: constraint (6)
    if problem_type == 'RACP/max':
        prec_constraint_idx = 6
    else:
        prec_constraint_idx = 2

    # =========================================================================
    # Constraint (2) or (6): Precedence / Temporal constraints
    # S_j - S_i >= p_i (RACP) or S_j - S_i >= delta_{ij} (RACP/max)
    # This is a >= constraint: violation_amount = max(0, RHS - LHS)
    # =========================================================================
    for (i, j) in precedence_relations:
        delta_ij = arc_weights[(i, j)]
        lhs = start_times[j] - start_times[i]
        rhs = delta_ij
        violation_amount = max(0, rhs - lhs)
        if violation_amount > tol:
            violated_constraints.add(prec_constraint_idx)
            violations.append(
                f"Precedence violated: S_{j} - S_{i} = {lhs} < {rhs} "
                f"(arc <{i},{j}>, required lag {delta_ij})"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": prec_constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

    # =========================================================================
    # Constraint (3): Resource capacity
    # r_k(S,t) <= R_k for each k in R, t in {0,...,d_bar-1}
    # r_k(S,t) = sum_{i active at t} r_{ik}
    # Active set: A(S,t) = {i in V^r | S_i <= t < S_i + p_i}
    # This is a <= constraint: violation_amount = max(0, LHS - RHS)
    # =========================================================================
    for k in range(n_resources):
        R_k = resource_amounts[k]
        for t in range(d_bar):
            usage = 0
            for a in activities:
                i = a['id']
                p_i = durations[i]
                if p_i > 0 and start_times[i] <= t < start_times[i] + p_i:
                    usage += requirements[i][k]
            lhs = usage
            rhs = R_k
            violation_amount = max(0, lhs - rhs)
            if violation_amount > tol:
                violated_constraints.add(3)
                violations.append(
                    f"Resource capacity violated: resource {k} at time {t}, "
                    f"usage {lhs} > capacity {rhs}"
                )
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer)
                })

    # =========================================================================
    # Constraint (4): Domain constraint
    # S_i in {0, 1, ..., d_bar} for all i in V
    # Check: 0 <= S_i <= d_bar and S_i is integer
    # Treat as two-sided: violation if S_i < 0 or S_i > d_bar or not integer
    # =========================================================================
    for i in act_ids:
        s_i = start_times[i]
        # Check integrality
        if abs(s_i - round(s_i)) > tol:
            violated_constraints.add(4)
            violation_amount = abs(s_i - round(s_i))
            violations.append(
                f"Domain violated: S_{i} = {s_i} is not integer"
            )
            normalizer = max(abs(d_bar), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(s_i),
                "rhs": float(round(s_i)),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })
        # Check lower bound: S_i >= 0 (i.e., 0 <= S_i, or S_i >= 0)
        if s_i < -tol:
            violated_constraints.add(4)
            violation_amount = abs(s_i)
            violations.append(
                f"Domain violated: S_{i} = {s_i} < 0"
            )
            normalizer = max(abs(0), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(s_i),
                "rhs": 0.0,
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })
        # Check upper bound: S_i <= d_bar
        if s_i > d_bar + tol:
            violated_constraints.add(4)
            violation_amount = s_i - d_bar
            violations.append(
                f"Domain violated: S_{i} = {s_i} > d_bar = {d_bar}"
            )
            normalizer = max(abs(d_bar), eps)
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(s_i),
                "rhs": float(d_bar),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

    # =========================================================================
    # Constraint (5): Non-negativity of resource amounts
    # R_k >= 0 for all k in R
    # This is a >= constraint: violation_amount = max(0, 0 - R_k) = max(0, -R_k)
    # =========================================================================
    for k in range(n_resources):
        R_k = resource_amounts[k]
        violation_amount = max(0, -R_k)
        if violation_amount > tol:
            violated_constraints.add(5)
            violations.append(
                f"Non-negativity violated: R_{k} = {R_k} < 0"
            )
            normalizer = eps  # RHS = 0, so normalizer = max(|0|, eps) = eps
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(R_k),
                "rhs": 0.0,
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

    # =========================================================================
    # Constraint (1): Objective consistency  (Tier C obj-recompute defense)
    # Formulation (1):  Minimize  sum_{k in R} c_k * R_k
    # The reported objective_value must equal the total resource provision
    # cost recomputed from the solution's resource_amounts (R_k) and the
    # instance's resource_costs (c_k). Both are fully present, so this is an
    # exact full recompute. Rejects fabricated objective_value reports (e.g.
    # obj=0 or obj=sys.float_info.max) that would otherwise pass the
    # constraint-only check above.
    # Comparison tolerance: tol_obj = max(1e-3, 1e-3 * |true_obj|) (0.1% rel).
    # =========================================================================
    resource_costs = instance.get('resource_costs')
    reported_obj = solution.get('objective_value')
    if resource_costs is not None and reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        true_obj = None
        if reported is not None:
            try:
                true_obj = float(sum(
                    float(resource_costs[k]) * float(resource_amounts[k])
                    for k in range(n_resources)
                ))
            except (KeyError, IndexError, TypeError, ValueError):
                true_obj = None
        if reported is not None and true_obj is not None:
            abs_diff = abs(reported - true_obj)
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                violated_constraints.add(1)
                violations.append(
                    f"Objective consistency violated: reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_k(c_k*R_k)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer)
                })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for RACP solutions (Kreter et al., 2018)"
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument('--solution_path', type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument('--result_path', type=str, required=True,
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result['feasible']:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
