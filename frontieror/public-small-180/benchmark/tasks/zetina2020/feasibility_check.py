#!/usr/bin/env python3
"""
Feasibility checker for TSP and MaxCut solutions from:
"Efficient Internal Strategies in Quantum Relaxation based Branch-and-Bound"
Matsuyama, Huang, Nishimura, Yamashiro (2024)

Checks hard constraints from the mathematical formulation (math_model.txt):
  TSP (Eq. 8):
    Constraint 1 (C1): sum_{i=0}^{N-1} x_{i,t} = 1  for all t  (assignment)
    Constraint 2 (C2): sum_{t=0}^{N-1} x_{i,t} = 1  for all i  (visit)
  MaxCut (Eq. 7):
    No hard constraints (unconstrained).
  Both:
    Constraint 3 (C3): variable domain x in {0,1}.
    Constraint 4 (C4): objective consistency -- the reported objective_value
        must equal the value recomputed from the binary solution variables:
          TSP:    sum_t sum_{i,j} d_{ij} x_{i,t} x_{j,(t+1) mod N}
          MaxCut: sum_{(i,j) in E} w_{ij} * (x_i + x_j - 2 x_i x_j)
                  (equivalent to (1/2) * sum (1 - s_i s_j) with s = 1 - 2x)
        This catches LLM-evolved exploits that lie about objective_value.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'solution')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('solution',)
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



def parse_solution_vector(solution_dict, problem_type, n):
    """Parse solution dict into a flat binary vector of length n.

    Handles both efficient (flat "0","1",...) and gurobi ("x_i_t" or "x_i") formats.
    """
    x = [0] * n

    first_key = next(iter(solution_dict))
    if first_key.startswith("x_"):
        for key, val in solution_dict.items():
            parts = key.split("_")
            if problem_type == "TSP":
                i, t = int(parts[1]), int(parts[2])
                N = int(n ** 0.5)
                idx = i * N + t
            else:
                idx = int(parts[1])
            x[idx] = float(val)
    else:
        for key, val in solution_dict.items():
            x[int(key)] = float(val)

    return x


def check_tsp_feasibility(x, N, instance):
    """Check TSP hard constraints and return violations."""
    tol = 1e-5
    eps = 1e-5
    violations = []
    violation_messages = []
    violation_magnitudes = []

    for t in range(N):
        lhs = sum(x[i * N + t] for i in range(N))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })
            violation_messages.append(
                f"C1 violated at time step t={t}: sum_i x_{{i,{t}}} = {lhs}, expected 1"
            )
            if 1 not in violations:
                violations.append(1)

    for i in range(N):
        lhs = sum(x[i * N + t] for t in range(N))
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })
            violation_messages.append(
                f"C2 violated at city i={i}: sum_t x_{{{i},t}} = {lhs}, expected 1"
            )
            if 2 not in violations:
                violations.append(2)

    return violations, violation_messages, violation_magnitudes


def check_maxcut_feasibility(x, instance):
    """MaxCut is unconstrained -- always feasible."""
    return [], [], []


def _compute_tsp_objective(x, N, distance_matrix):
    """Recompute TSP obj = sum_t sum_{i,j} d_{ij} x_{i,t} x_{j,(t+1) mod N}."""
    obj = 0.0
    for t in range(N):
        tn = (t + 1) % N
        for i in range(N):
            xi = x[i * N + t]
            if xi == 0:
                continue
            for j in range(N):
                xj = x[j * N + tn]
                if xj == 0:
                    continue
                obj += float(distance_matrix[i][j]) * xi * xj
    return float(obj)


def _compute_maxcut_objective(x, edges, edge_weights):
    """Recompute MaxCut obj = sum_{(i,j) in E} w_{ij} * (x_i + x_j - 2 x_i x_j).

    This equals (1/2) * sum (1 - s_i s_j) with s = 1 - 2x and gives the
    weighted count of cut edges. If edge_weights is None, every edge weighs 1.
    """
    obj = 0.0
    for k, (i, j) in enumerate(edges):
        w = float(edge_weights[k]) if edge_weights is not None else 1.0
        xi = x[i]
        xj = x[j]
        obj += w * (xi + xj - 2.0 * xi * xj)
    return float(obj)


def check_feasibility(instance, solution):
    """Run all constraint checks and return a dict result (same shape as main writes)."""
    problem_type = instance["problem_type"]
    solution_dict = solution["solution"]

    tol = 1e-5
    eps = 1e-5

    if problem_type == "TSP":
        N = instance["parameters"]["n_cities"]
        n = N * N
        x = parse_solution_vector(solution_dict, problem_type, n)
        violated_constraints, violation_msgs, violation_mags = check_tsp_feasibility(
            x, N, instance
        )
        keys = set(solution_dict)
        dense_named = {f"x_{i}_{t}" for i in range(N) for t in range(N)}
        dense_flat = {str(index) for index in range(n)}
        if keys not in (dense_named, dense_flat):
            violated_constraints.append(3)
            missing = sorted(dense_named - keys)[:12]
            extra = sorted(keys - dense_named)[:12]
            violation_msgs.append(
                f"C3 violated: TSP solution must contain one value for every "
                f"city/time pair; missing={missing}, extra={extra}"
            )
            violation_mags.append({
                "constraint": 3, "lhs": float(len(keys)), "rhs": float(n),
                "raw_excess": 1.0, "normalizer": float(max(n, 1)),
                "ratio": 1.0 / float(max(n, 1)),
            })

        tour = solution.get("tour")
        derived_tour = [
            max(range(N), key=lambda city: x[city * N + step])
            for step in range(N)
        ]
        if (not isinstance(tour, list)
                or len(tour) != N
                or set(tour) != set(range(N))
                or tour != derived_tour):
            if 3 not in violated_constraints:
                violated_constraints.append(3)
            violation_msgs.append(
                f"C3 violated: TSP tour={tour!r} must be the permutation "
                f"derived from solution variables={derived_tour}"
            )
            violation_mags.append({
                "constraint": 3, "lhs": 0.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
    elif problem_type == "MaxCut":
        n_nodes = instance["parameters"]["n_nodes"]
        x = parse_solution_vector(solution_dict, problem_type, n_nodes)
        violated_constraints, violation_msgs, violation_mags = check_maxcut_feasibility(
            x, instance
        )
        keys = set(solution_dict)
        dense_named = {f"x_{i}" for i in range(n_nodes)}
        dense_flat = {str(i) for i in range(n_nodes)}
        if keys not in (dense_named, dense_flat):
            violated_constraints.append(3)
            missing = sorted(dense_named - keys)[:12]
            extra = sorted(keys - dense_named)[:12]
            violation_msgs.append(
                f"C3 violated: MaxCut solution must contain exactly one binary "
                f"value per node; missing={missing}, extra={extra}"
            )
            violation_mags.append({
                "constraint": 3, "lhs": float(len(keys)),
                "rhs": float(n_nodes), "raw_excess": 1.0,
                "normalizer": float(max(n_nodes, 1)),
                "ratio": 1.0 / float(max(n_nodes, 1)),
            })
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Constraint 3 (C3): Variable domain -- x in {0,1}
    for key, val in solution_dict.items():
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        viol = min(abs(v - 0.0), abs(v - 1.0))
        if viol > tol:
            if 3 not in violated_constraints:
                violated_constraints.append(3)
            violation_msgs.append(
                f"C3 violated at {key}: value = {v} not in {{0, 1}}"
            )
            violation_mags.append({
                "constraint": 3,
                "lhs": float(v),
                "rhs": 1.0,
                "raw_excess": float(viol),
                "normalizer": 1.0,
                "ratio": float(viol),
            })

    # Constraint 4 (C4): Objective consistency -- recompute from x and compare.
    # Defends against LLM exploits that pass constraint checks but lie about
    # objective_value. We snap x to {0,1} before recomputing so the obj check
    # is independent of the C3 domain check (each constraint failure should
    # surface independently rather than masking one another).
    reported_obj = solution.get("objective_value")
    try:
        reported = float(reported_obj) if reported_obj is not None else None
    except (TypeError, ValueError):
        reported = None

    if reported is not None:
        xb = [int(round(min(1.0, max(0.0, float(v))))) for v in x]
        if problem_type == "TSP":
            N = instance["parameters"]["n_cities"]
            dm = instance["data"]["distance_matrix"]
            true_obj = _compute_tsp_objective(xb, N, dm)
            # TSP obj is a sum of euclidean distances -> 0.1% relative tol
            tol_obj = max(1e-3, 1e-3 * abs(true_obj))
        else:
            edges = instance["data"]["edges"]
            edge_weights = instance["data"].get("edge_weights")
            true_obj = _compute_maxcut_objective(xb, edges, edge_weights)
            # MaxCut obj is a weighted edge count -> tighten so an off-by-one fires
            tol_obj = max(0.5, 1e-3 * abs(true_obj))
        diff = abs(reported - true_obj)
        if diff > tol_obj:
            if 4 not in violated_constraints:
                violated_constraints.append(4)
            violation_msgs.append(
                f"C4 violated: reported objective_value={reported} differs from "
                f"recomputed objective={true_obj} (|diff|={diff:.6g}, tol={tol_obj:.3g})"
            )
            normalizer = max(abs(true_obj), eps)
            violation_mags.append({
                "constraint": 4,
                "lhs": float(reported),
                "rhs": float(true_obj),
                "raw_excess": float(diff),
                "normalizer": float(normalizer),
                "ratio": float(diff / normalizer),
            })

    violated_constraints = sorted(set(violated_constraints))
    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violation_msgs,
        "violation_magnitudes": violation_mags,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for TSP/MaxCut solutions (obj-recompute variant)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
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
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
