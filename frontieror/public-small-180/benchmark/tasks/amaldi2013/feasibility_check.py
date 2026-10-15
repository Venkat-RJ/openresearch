"""
Feasibility checker for the Minimum Hyperplanes Clustering Problem (Min-HCP).

Based on: Amaldi, Dhyani, and Ceselli (2013),
"Column Generation for the Minimum Hyperplanes Clustering Problem",
INFORMS Journal on Computing.

Checks the ORIGINAL geometric problem (circle containment, i.e., point must
lie within epsilon of its assigned hyperplane in Euclidean distance).
Does NOT enforce the big-M linearization (Eq. 11-12) or unit-norm constraint
(Eq. 13) — those are MILP-formulation artifacts, not original-problem rules.

This patched version additionally enforces:
  Constraint 2 (Eq. 2, objective consistency): the reported objective_value
  must equal sum_j y_j == number of hyperplanes present in the solution
  (since every hyperplane listed in `hyperplanes` has y_j = 1). Protects
  against LLM score-gaming exploits that fabricate the objective.
"""

import argparse
import json
import math
import sys
from decimal import Decimal, localcontext
import numpy as np

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('hyperplanes', 'objective_value', 'point_assignments')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('point_assignments',)
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
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _finite_decimal(value):
    # JSON numbers only; do not let strings, booleans or nonfinite values
    # participate in geometry. Decimal avoids float norm/dot overflow.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("hyperplane coefficients must be finite JSON numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("hyperplane coefficients must be finite JSON numbers")
    return Decimal(str(value))


def _geometric_distances(points, w, w0, dimension):
    if not isinstance(w, list) or len(w) != dimension:
        raise ValueError("hyperplane normal dimension does not match instance")
    with localcontext() as ctx:
        # Covers the exponent span of products of finite binary64 values,
        # including subnormals and cancellation, without changing tolerance.
        ctx.prec = 1600
        normal = [_finite_decimal(v) for v in w]
        offset = _finite_decimal(w0)
        norm = sum((v * v for v in normal), Decimal(0)).sqrt()
        if norm == 0:
            return None
        cap = Decimal(str(sys.float_info.max))
        distances = []
        for point in points:
            residual = sum((_finite_decimal(v) * a for v, a in zip(point, normal)), Decimal(0)) - offset
            distance = abs(residual) / norm
            # Finite diagnostic values even for geometrically enormous distances.
            distances.append(float(min(distance, cap)))
        return distances


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    # ---- Check if solution contains actual solution data ----
    if "hyperplanes" not in solution or solution.get("hyperplanes") is None:
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No solution data in solution file"],
            "violation_magnitudes": []
        }

    # ---- Extract instance data ----
    n = instance["n"]
    d = instance["d"]
    epsilon = instance["epsilon"]
    points = np.array(instance["points"])  # n x d
    K = math.ceil(n / d)  # upper bound on number of hyperplanes

    # ---- Extract solution data ----
    hyperplanes = solution["hyperplanes"]
    num_hyperplanes = len(hyperplanes)

    # Reconstruct D_ij and y_j from solution.
    # j ranges over [0, num_hyperplanes-1] (the hyperplanes present in solution).
    # The model uses K potential hyperplane slots; unused slots have y_j=0.
    # We check constraints only for the hyperplanes in the solution (y_j=1)
    # plus verify that any hyperplane slot beyond those has D_ij=0 (trivially true).

    # Build D matrix: D[i][j] = 1 if point i is assigned to hyperplane j
    D = np.zeros((n, num_hyperplanes), dtype=int)
    for j, hp in enumerate(hyperplanes):
        for i in hp["assigned_points"]:
            if 0 <= i < n:
                D[i, j] = 1

    # y_j = 1 for all hyperplanes in the solution
    y = np.ones(num_hyperplanes, dtype=int)

    # ----------------------------------------------------------------
    # Constraint 1 (original geometric rule): for each assigned point i,
    # Euclidean distance from point_i to its assigned hyperplane j must be
    # <= epsilon.  Distance = |a_i . w_j - w_j^0| / ||w_j||_2.
    # This replaces the big-M formulation constraints (Eq. 11-12), which
    # are MILP linearization artifacts, not original-problem rules.
    # ----------------------------------------------------------------
    constraint_1_violated = False
    for j, hp in enumerate(hyperplanes):
        try:
            distances = _geometric_distances(instance["points"], hp["w"], hp["w0"], d)
        except (ValueError, ArithmeticError) as exc:
            return {"feasible": False, "violated_constraints": [5],
                    "violations": [f"Invalid hyperplane {j}: {exc}"],
                    "violation_magnitudes": []}
        if distances is None:
            # Exactly zero is degenerate; tiny nonzero rescalings are valid.
            if int(D[:, j].sum()) > 0:
                constraint_1_violated = True
                violated_constraints_set.add(1)
                violations.append(f"Constraint 1 violated: hyperplane {j} has zero normal vector but has assigned points")
            continue
        for i in range(n):
            if D[i, j] == 1:
                dist = distances[i]
                if dist > epsilon + tol:
                    violation_amount = dist - epsilon
                    normalizer = max(abs(epsilon), eps)
                    ratio = violation_amount / normalizer
                    if not constraint_1_violated:
                        constraint_1_violated = True
                        violated_constraints_set.add(1)
                    violation_magnitudes.append({
                        "constraint": 1,
                        "lhs": float(dist),
                        "rhs": float(epsilon),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(min(ratio, sys.float_info.max))
                    })

    if constraint_1_violated:
        count = sum(1 for vm in violation_magnitudes if vm["constraint"] == 1)
        violations.append(
            f"Constraint 1 (original distance rule) violated: {count} (point, "
            f"hyperplane) pair(s) where distance > epsilon"
        )

    # ----------------------------------------------------------------
    # Constraint 2 (Eq. 2, objective consistency): the reported
    # objective_value must equal sum_j y_j == len(hyperplanes), since
    # every hyperplane listed in the solution has y_j = 1.
    # This rejects exploits that fabricate the reported objective.
    # ----------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(num_hyperplanes)
            abs_diff = abs(reported - true_obj)
            # obj is an integer count of hyperplanes; mismatch of >= 1 fires
            obj_tol = 0.5
            if abs_diff > obj_tol:
                normalizer = max(abs(true_obj), eps, 1.0)
                ratio = abs_diff / normalizer
                violated_constraints_set.add(2)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(min(ratio, sys.float_info.max))
                })
                violations.append(
                    f"Constraint 2 (Eq.2, objective consistency) violated: "
                    f"reported objective_value={reported} differs from recomputed "
                    f"sum_j(y_j)=len(hyperplanes)={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )

    # ----------------------------------------------------------------
    # Constraint 3 (Eq. 5): sum_j D_ij >= 1 for 1 <= i <= n
    #   Each point must be assigned to at least one hyperplane.
    # ----------------------------------------------------------------
    constraint_3_violated = False
    for i in range(n):
        lhs_val = float(np.sum(D[i, :]))
        rhs_val = 1.0
        violation_amount = rhs_val - lhs_val  # >= constraint: RHS - LHS
        if violation_amount > tol:
            normalizer = max(abs(rhs_val), eps)
            ratio = violation_amount / normalizer
            if not constraint_3_violated:
                constraint_3_violated = True
                violated_constraints_set.add(3)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(min(ratio, sys.float_info.max))
            })

    if constraint_3_violated:
        uncovered = [i for i in range(n) if np.sum(D[i, :]) < 1]
        violations.append(
            f"Constraint 3 (Eq.5) violated: {len(uncovered)} point(s) not assigned "
            f"to any hyperplane: {uncovered[:10]}{'...' if len(uncovered) > 10 else ''}"
        )

    # ----------------------------------------------------------------
    # Constraint 4 (Eq. 6): D_ij <= y_j for 1 <= i <= n, 1 <= j <= K
    #   A point can only be assigned to a hyperplane that is in use.
    #   Since all hyperplanes in the solution have y_j=1, this is automatically
    #   satisfied for all j in the solution. We still check explicitly.
    # ----------------------------------------------------------------
    constraint_4_violated = False
    for j in range(num_hyperplanes):
        for i in range(n):
            lhs_val = float(D[i, j])
            rhs_val = float(y[j])
            violation_amount = lhs_val - rhs_val  # <= constraint: LHS - RHS
            if violation_amount > tol:
                normalizer = max(abs(rhs_val), eps)
                ratio = violation_amount / normalizer
                if not constraint_4_violated:
                    constraint_4_violated = True
                    violated_constraints_set.add(4)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(lhs_val),
                    "rhs": float(rhs_val),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(min(ratio, sys.float_info.max))
                })

    if constraint_4_violated:
        violations.append(
            "Constraint 4 (Eq.6) violated: point(s) assigned to unused hyperplane(s)"
        )

    # ----------------------------------------------------------------
    # Constraint 5 (Eq. 7): w_j in R^d, w_j^0 in R for 1 <= j <= K
    #   Variable domain constraint - check that dimensions match.
    # ----------------------------------------------------------------
    constraint_5_violated = False
    for j, hp in enumerate(hyperplanes):
        w = hp["w"]
        w0 = hp["w0"]
        lhs_val = float(len(w))
        rhs_val = float(d)
        violation_amount = abs(lhs_val - rhs_val)
        if violation_amount > tol:
            normalizer = max(abs(rhs_val), eps)
            ratio = violation_amount / normalizer
            if not constraint_5_violated:
                constraint_5_violated = True
                violated_constraints_set.add(5)
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": float(lhs_val),
                "rhs": float(rhs_val),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(min(ratio, sys.float_info.max))
            })

    if constraint_5_violated:
        violations.append(
            f"Constraint 5 (Eq.7) violated: hyperplane normal vector dimension "
            f"does not match d={d}"
        )

    # ----------------------------------------------------------------
    # Constraint 6 (Eq. 8): D_ij in {0,1} for 1 <= i <= n, 1 <= j <= K
    #   Binary constraint on assignment variables.
    # ----------------------------------------------------------------
    constraint_6_violated = False
    for j in range(num_hyperplanes):
        for i in range(n):
            val = D[i, j]
            if val not in (0, 1):
                violation_amount = min(abs(val - 0), abs(val - 1))
                if violation_amount > tol:
                    normalizer = max(1.0, eps)
                    ratio = violation_amount / normalizer
                    if not constraint_6_violated:
                        constraint_6_violated = True
                        violated_constraints_set.add(6)
                    violation_magnitudes.append({
                        "constraint": 6,
                        "lhs": float(val),
                        "rhs": float(round(val)),
                        "raw_excess": float(violation_amount),
                        "normalizer": float(normalizer),
                        "ratio": float(min(ratio, sys.float_info.max))
                    })

    if constraint_6_violated:
        violations.append(
            "Constraint 6 (Eq.8) violated: D_ij values are not binary"
        )

    # Check for invalid point indices in assigned_points
    for j, hp in enumerate(hyperplanes):
        invalid_pts = [i for i in hp["assigned_points"] if i < 0 or i >= n]
        if invalid_pts:
            if 6 not in violated_constraints_set:
                violated_constraints_set.add(6)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": float(len(invalid_pts)),
                "rhs": 0.0,
                "raw_excess": float(len(invalid_pts)),
                "normalizer": max(0.0, eps),
                "ratio": float(len(invalid_pts)) / eps
            })
            violations.append(
                f"Constraint 6 (Eq.8) violated: hyperplane {j} has {len(invalid_pts)} "
                f"invalid point indices out of range [0, {n-1}]"
            )

    # ----------------------------------------------------------------
    # Constraint 7 (Eq. 9): y_j in {0,1} for 1 <= j <= K
    #   Binary constraint on hyperplane usage variables.
    #   Since all hyperplanes in solution have y_j=1, check trivially satisfied.
    # ----------------------------------------------------------------
    constraint_7_violated = False
    for j in range(num_hyperplanes):
        val = y[j]
        if val not in (0, 1):
            violation_amount = min(abs(val - 0), abs(val - 1))
            if violation_amount > tol:
                normalizer = max(1.0, eps)
                ratio = violation_amount / normalizer
                if not constraint_7_violated:
                    constraint_7_violated = True
                    violated_constraints_set.add(7)
                violation_magnitudes.append({
                    "constraint": 7,
                    "lhs": float(val),
                    "rhs": float(round(val)),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(min(ratio, sys.float_info.max))
                })

    if constraint_7_violated:
        violations.append(
            "Constraint 7 (Eq.9) violated: y_j values are not binary"
        )

    # ----------------------------------------------------------------
    # Constraint 8 SKIPPED: ||w_j||_2 = 1 (Eq. 13) is a formulation-specific
    # normalization used to make the big-M linearization (Eqs. 11-12) exact.
    # Since we use direct Euclidean distance (|a_i·w - w_0| / ||w||_2) for the
    # containment check, w may have any non-zero norm; the distance is invariant
    # to the scale of (w, w_0). Not checked.
    # ----------------------------------------------------------------

    # ---- Assemble result ----
    feasible = len(violated_constraints_set) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Min-HCP (Amaldi et al. 2013)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
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

    write_json(args.result_path, result)

    if result["feasible"] is None:
        print("NO SOLUTION - Cannot check feasibility.")
        for v in result["violations"]:
            print(f"  - {v}")
    elif result["feasible"]:
        print("FEASIBLE - No constraint violations found.")
    else:
        print(f"INFEASIBLE - Violated constraints: {result['violated_constraints']}")
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
