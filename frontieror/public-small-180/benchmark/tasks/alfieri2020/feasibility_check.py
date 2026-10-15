"""
feasibility_check.py
====================
Feasibility checker for Graph Coloring Problem (GCP) solutions.

Checks hard constraints from the mathematical model (math_model.txt):
  Constraint (2): sum_{c in C} x_{i,c} = 1,  for all i in V  (assignment)
  Constraint (3): x_{i,c} + x_{j,c} <= z_c,  for all (i,j) in E, c in C  (conflict)
  Constraint (4): x_{i,c} in {0,1},  for all i in V, c in C  (binary x)
  Constraint (5): z_c in {0,1},  for all c in C  (binary z)
  Constraint (15): objective consistency -- reported objective_value
                   must equal the number of distinct colors used in
                   `coloring` (full recompute), or fall within
                   [1, greedy_upper_bound] for solutions without
                   explicit `coloring` (lower/upper bound check).

Constraints (7)-(8) belong to GCP-SC, (10)-(11) to RMP, (13)-(14) to MWISP.
These are checked only when the solution provides the corresponding variables.

Gurobi solutions provide a `coloring` dict -> constraints 2-5 are checked.
Efficient (CG-MLPH) solutions provide only an LP bound with no decision
variable values -> no variable-level constraints can be verified.
"""

import argparse
import json
import sys

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('coloring', 'objective_value')
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

# Constraint index used for objective-value consistency. Chosen as the
# next unused index after the highest model-constraint number (14, MWISP).
OBJ_CONSISTENCY_IDX = 15


def greedy_coloring(n: int, edges: list) -> int:
    """Compute greedy coloring upper bound. Returns number of colors used."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    colors = [-1] * n
    for v in range(n):
        used = {colors[u] for u in adj[v] if colors[u] != -1}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return max(colors) + 1 if n > 0 else 0


def check_gurobi_solution(instance: dict, solution: dict) -> dict:
    """Check constraints 2-5 of GCP-compact for a solution with coloring."""
    n = instance["num_nodes"]
    edges = instance["edges"]
    coloring = solution["coloring"]  # dict: str(vertex) -> color

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # Reconstruct x_{i,c} and z_c from coloring
    # coloring maps vertex (as string) -> color index
    color_assignment = {}
    for key, val in coloring.items():
        color_assignment[int(key)] = int(val)

    # ------------------------------------------------------------------
    # Vertex index validation: indices must be in {0, ..., n-1}
    # ------------------------------------------------------------------
    for i in color_assignment:
        if i < 0 or i >= n:
            violated_set.add(2)
            violations.append(
                f"Constraint (2): Vertex index {i} is out of range [0, {n-1}]"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": 0.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })

    used_colors = set(color_assignment.values())
    num_colors = max(used_colors) + 1 if used_colors else 0

    # Color set size as determined by greedy_coloring (same as gurobi_code.py)
    model_num_colors = greedy_coloring(n, edges)

    # ------------------------------------------------------------------
    # Color label validation: colors may take any non-negative integer
    # labels — the labels themselves are arbitrary (e.g., colors
    # {5, 10, 20} uses 3 distinct colors, same as {0, 1, 2}). What matters
    # is the NUMBER of distinct colors used (= the objective value), which
    # the paper's greedy upper bound caps. We skip the previous check that
    # forced labels into [0, model_num_colors - 1], because that's a
    # Gurobi formulation restriction, not a business rule. Negative or
    # non-integer labels are still flagged.
    # ------------------------------------------------------------------
    for i, c in color_assignment.items():
        if not isinstance(c, int) or c < 0:
            violated_set.add(4)
            violations.append(
                f"Constraint (4): Vertex {i} assigned invalid color {c} "
                f"(must be a non-negative integer label)"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(c) if isinstance(c, (int, float)) else 0.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": max(1.0, EPS),
                "ratio": 1.0,
            })

    # Also verify that the number of distinct colors used does not exceed
    # the greedy upper bound (a valid instance-level sanity check).
    num_distinct_used = len(set(color_assignment.values())) if color_assignment else 0
    if num_distinct_used > model_num_colors:
        violated_set.add(4)
        violations.append(
            f"Constraint (4): number of distinct colors used ({num_distinct_used}) "
            f"exceeds greedy upper bound ({model_num_colors})"
        )
        violation_magnitudes.append({
            "constraint": 4,
            "lhs": float(num_distinct_used),
            "rhs": float(model_num_colors),
            "raw_excess": float(num_distinct_used - model_num_colors),
            "normalizer": max(float(model_num_colors), EPS),
            "ratio": float(num_distinct_used - model_num_colors) / max(float(model_num_colors), EPS),
        })

    # z_c = 1 if color c is used, 0 otherwise
    z = {c: 1 for c in used_colors}

    # ------------------------------------------------------------------
    # Constraint (2): sum_{c in C} x_{i,c} = 1 for all i in V
    # Each vertex must be assigned exactly one color.
    # ------------------------------------------------------------------
    for i in range(n):
        if i not in color_assignment:
            # Vertex not assigned any color: LHS=0, RHS=1
            lhs = 0.0
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                violated_set.add(2)
                violations.append(
                    f"Constraint (2): Vertex {i} is not assigned any color "
                    f"(sum x_{{i,c}} = {lhs}, expected 1)"
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

    # Check for vertices assigned multiple colors (not possible with dict,
    # but check that values are valid non-negative integers)
    for i in range(n):
        if i in color_assignment:
            c = color_assignment[i]
            if not isinstance(c, int) or c < 0:
                lhs = 0.0
                rhs = 1.0
                violation_amount = abs(lhs - rhs)
                violated_set.add(2)
                violations.append(
                    f"Constraint (2): Vertex {i} has invalid color {c}"
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
    # Constraint (3): x_{i,c} + x_{j,c} <= z_c for all (i,j) in E, c in C
    # For a coloring, this means: if adjacent vertices i,j share color c,
    # then LHS = 2, RHS = z_c = 1, so violated (2 > 1).
    # ------------------------------------------------------------------
    for u, v in edges:
        if u in color_assignment and v in color_assignment:
            if color_assignment[u] == color_assignment[v]:
                c = color_assignment[u]
                lhs = 2.0  # x_{u,c} + x_{v,c} = 1 + 1
                rhs = float(z.get(c, 0))  # z_c (1 if color used)
                violation_amount = max(lhs - rhs, 0.0)
                if violation_amount > TOL:
                    violated_set.add(3)
                    violations.append(
                        f"Constraint (3): Adjacent vertices {u} and {v} "
                        f"share color {c} (x_{{{u},{c}}} + x_{{{v},{c}}} = "
                        f"{lhs} > z_{{{c}}} = {rhs})"
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
    # Constraint (4): x_{i,c} in {0,1} for all i in V, c in C
    # Automatically satisfied by the coloring representation (each vertex
    # maps to exactly one integer color, so x_{i,c} is 0 or 1).
    # We verify integrality of the color values.
    # ------------------------------------------------------------------
    for i in range(n):
        if i in color_assignment:
            c = color_assignment[i]
            if not isinstance(c, (int, float)):
                violated_set.add(4)
                violations.append(
                    f"Constraint (4): Vertex {i} color value {c} is not binary"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(c) if isinstance(c, (int, float)) else 0.0,
                    "rhs": 1.0,
                    "raw_excess": 1.0,
                    "normalizer": 1.0,
                    "ratio": 1.0,
                })
            elif isinstance(c, float) and abs(c - round(c)) > TOL:
                lhs = c
                rhs = round(c)
                violation_amount = abs(lhs - rhs)
                violated_set.add(4)
                violations.append(
                    f"Constraint (4): Vertex {i} color {c} is not integer"
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
    # Constraint (5): z_c in {0,1} for all c in C
    # Automatically satisfied: z_c is derived as 1 if color c is used,
    # 0 otherwise. Always integral by construction.
    # ------------------------------------------------------------------
    # No violations possible from the coloring representation.

    # ------------------------------------------------------------------
    # Symmetry-breaking constraint (z_c >= z_{c+1}) SKIPPED: this is a
    # Gurobi-specific tie-breaking device to reduce the search tree when
    # color labels are interchangeable. It is NOT a business rule — a
    # valid coloring that uses colors {0, 2, 5} is just as legitimate as
    # one that uses {0, 1, 2}. An LLM algorithm may output colors with
    # arbitrary labels (e.g., from a heuristic that never renumbers);
    # the adjacency + cluster-assignment constraints already guarantee
    # a proper coloring, and the objective counts distinct colors used.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint (15): objective-value consistency (Tier C defense).
    # The GCP objective is sum_{c in C} z_c = number of distinct colors
    # used. With the full coloring in hand we recompute this exactly and
    # reject solutions whose reported `objective_value` disagrees.
    # Tolerance: 0.5 (integer count — a mismatch of >= 1 fires).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(len(set(color_assignment.values()))) if color_assignment else 0.0
            abs_diff = abs(reported - true_obj)
            tol = 0.5  # objective is an integer count
            if abs_diff > tol:
                violated_set.add(OBJ_CONSISTENCY_IDX)
                violations.append(
                    f"Constraint ({OBJ_CONSISTENCY_IDX}): Objective consistency "
                    f"violated: reported objective_value={reported} differs from "
                    f"recomputed |{{c : exists i with x_{{i,c}}=1}}|={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={tol:.3g})"
                )
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": OBJ_CONSISTENCY_IDX,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def check_efficient_solution(instance: dict, solution: dict) -> dict:
    """
    Check feasibility of an efficient (CG-MLPH) solution.

    The efficient algorithm outputs only an LP lower bound and metadata
    (objective_value, lp_lower_bound, num_cg_iterations, etc.) without
    explicit decision variable values. Without x_s values, constraints
    (7)-(8) of GCP-SC and (10)-(11) of RMP cannot be verified at the
    variable level.

    We verify basic consistency: the objective value must be non-negative
    (chromatic number >= 1 for non-empty graphs), and must not exceed
    the greedy upper bound (a valid LP / IP solution can never exceed it).
    """
    violations = []
    violation_magnitudes = []
    violated_set = set()

    n = instance["num_nodes"]
    edges = instance["edges"]
    obj = solution.get("objective_value")

    if obj is not None and n > 0:
        # The LP lower bound should be >= 1 for a non-empty graph
        # (at least one color is needed). This is implied by constraint (7)/(10):
        # sum x_s >= 1 for each vertex, and objective = sum x_s >= 1.
        # However, the LP relaxation can yield fractional values, so
        # obj >= 1 is not strictly required. We only flag if obj < 0.
        if obj < -TOL:
            lhs = obj
            rhs = 0.0
            violation_amount = abs(lhs)
            violated_set.add(7)
            violations.append(
                f"Constraint (7): Objective value {obj} is negative, "
                f"violating non-negativity implied by coverage constraints"
            )
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 7,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ------------------------------------------------------------------
    # Constraint (15): objective-value consistency (Tier C defense).
    # Without decision variables we cannot fully recompute the LP /
    # chromatic-number objective. Instead we apply the tightest
    # data-derived envelope on the LP relaxation value:
    #   1.0  <=  obj  <=  greedy_coloring(V, E)
    # The lower bound 1 holds for any non-empty graph: every vertex must
    # be covered (sum_{s ∋ i} x_s >= 1), so the LP minimum is >= 1.
    # The upper bound holds because the greedy coloring constructs an
    # integer-feasible solution, which dominates the LP / IP optimum.
    # ------------------------------------------------------------------
    if obj is not None and n > 0:
        try:
            reported = float(obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            lower_bound = 1.0
            upper_bound = float(greedy_coloring(n, edges))
            # 0.1% relative slack on each side to absorb numerical noise.
            lo_tol = max(1e-3, 1e-3 * abs(lower_bound))
            hi_tol = max(1e-3, 1e-3 * abs(upper_bound))
            if reported < lower_bound - lo_tol or reported > upper_bound + hi_tol:
                if reported < lower_bound - lo_tol:
                    violation_amount = lower_bound - reported
                    bound_str = (
                        f"below LP lower bound 1.0 "
                        f"(any non-empty graph requires >=1 color)"
                    )
                else:
                    violation_amount = reported - upper_bound
                    bound_str = (
                        f"above greedy upper bound {upper_bound} "
                        f"(LP / IP optimum cannot exceed the constructive greedy)"
                    )
                violated_set.add(OBJ_CONSISTENCY_IDX)
                violations.append(
                    f"Constraint ({OBJ_CONSISTENCY_IDX}): Objective consistency "
                    f"violated: reported objective_value={reported} is {bound_str} "
                    f"(envelope=[{lower_bound}, {upper_bound}])"
                )
                normalizer = max(abs(upper_bound), EPS)
                violation_magnitudes.append({
                    "constraint": OBJ_CONSISTENCY_IDX,
                    "lhs": reported,
                    "rhs": upper_bound if reported > upper_bound else lower_bound,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def check_feasibility(instance: dict, solution: dict) -> dict:
    """Dispatch the same way as main(): coloring -> gurobi check, else efficient check."""
    if "coloring" in solution and solution["coloring"] is not None:
        return check_gurobi_solution(instance, solution)
    return check_efficient_solution(instance, solution)


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Graph Coloring Problem solutions"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the data instance",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the candidate solution",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to write the JSON file containing the feasibility result",
    )
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

    # Detect solution type based on presence of coloring data
    if "coloring" in solution and solution["coloring"] is not None:
        result = check_gurobi_solution(instance, solution)
    else:
        result = check_efficient_solution(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        for v in result["violations"]:
            print(f"  - {v}")
    print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
