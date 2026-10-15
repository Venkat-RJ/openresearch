"""
Feasibility checker for the Capacitated Hub Location Problem with Single Assignment (CHLPSA).

Mathematical model from:
  Contreras, Diaz, and Fernandez (2011),
  "Branch and Price for Large-Scale Capacitated Hub Location Problems with Single Assignment",
  INFORMS Journal on Computing, 23(1), pp. 41-55.

Constraints are numbered 1-8 corresponding to equations (2)-(9) in the paper,
counting strictly from top to bottom in the formulation section.

Constraint 9 (Tier-C, added in this _new variant): objective-value consistency.
  The reported `objective_value` must match the recomputed objective from the
  solution variables (hubs + assignment), since hubs+assignment fully determine
  z_{kk}, z_{ik} and (via constraints 4-5) x_{ijkm}, and therefore the full
  objective in Eq. (1).
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignment', 'hubs', 'objective_value')
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



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # Handle missing or empty solution data
    if (not solution.get("hubs") and not solution.get("assignment")) or \
       (not solution.get("assignment")):
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No solution data in solution file"],
            "violation_magnitudes": [],
        }

    n = instance["n"]
    N = range(n)
    O = instance["outgoing_flow"]
    D = instance["total_flow_D"]
    b = instance["capacities"]

    hubs = solution["hubs"]
    assignment = solution["assignment"]

    hub_set = set(hubs)

    # z_{ik} = 1 iff assignment[i] == k
    # x_{ijkm} = 1 iff assignment[i] == k and assignment[j] == m

    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs, rhs, operator):
        """Record a violation with normalized magnitude."""
        if operator == "eq":
            violation_amount = abs(lhs - rhs)
        elif operator in ("leq", "lt"):
            violation_amount = max(lhs - rhs, 0.0)
        elif operator in ("geq", "gt"):
            violation_amount = max(rhs - lhs, 0.0)
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append((constraint_idx, msg))
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # =========================================================================
    # Constraint 1 (Eq. 2): sum_{k,m} x_{ijkm} = 1, for all i,j in N
    # =========================================================================
    # x_{ijkm} = 1 iff assignment[i]==k and assignment[j]==m.
    # For each (i,j), exactly one (k,m) pair satisfies this (k=assignment[i],
    # m=assignment[j]), so sum = 1 iff assignment[i] and assignment[j] are
    # valid node indices in {0,...,n-1}. Check this for all i,j.
    for i in N:
        ai = assignment[i]
        if ai < 0 or ai >= n:
            # sum would be 0 for any j
            for j in N:
                record_violation(1,
                    f"Constraint 1 (Eq.2) violated: sum_km x[{i},{j},k,m] = 0 != 1 "
                    f"(node {i} has invalid assignment {ai})",
                    0.0, 1.0, "eq")
            continue
        for j in N:
            aj = assignment[j]
            if aj < 0 or aj >= n:
                record_violation(1,
                    f"Constraint 1 (Eq.2) violated: sum_km x[{i},{j},k,m] = 0 != 1 "
                    f"(node {j} has invalid assignment {aj})",
                    0.0, 1.0, "eq")

    # =========================================================================
    # Constraint 2 (Eq. 3): z_{ik} <= z_{kk}, for all i,k in N
    # =========================================================================
    # z_{ik}=1 only when assignment[i]==k. z_{kk}=1 only when assignment[k]==k.
    # Violation: assignment[i]==k but assignment[k]!=k (i assigned to a non-hub).
    for i in N:
        k = assignment[i]
        if 0 <= k < n and assignment[k] != k:
            record_violation(2,
                f"Constraint 2 (Eq.3) violated: z[{i},{k}]=1 > z[{k},{k}]=0 "
                f"(node {i} assigned to non-hub {k})",
                1.0, 0.0, "leq")

    # =========================================================================
    # Constraint 3 (Eq. 4): sum_{m} x_{ijkm} = z_{ik}, for all i,j,k in N
    # =========================================================================
    # For fixed i,j,k:
    #   LHS = sum_m x_{ijkm}. x_{ijkm}=1 requires assignment[i]==k AND
    #         assignment[j]==m. If assignment[i]==k, exactly one m works
    #         (m=assignment[j]), so LHS=1. If assignment[i]!=k, LHS=0.
    #   RHS = z_{ik} = 1 if assignment[i]==k, else 0.
    # So LHS always equals RHS as long as assignments are valid indices.
    # Violations only occur if assignment[i] is out of range (already caught
    # in Constraint 1) or assignment[j] is out of range. In the latter case,
    # when assignment[i]==k, LHS=0 but RHS=1.
    for i in N:
        ai = assignment[i]
        if ai < 0 or ai >= n:
            continue  # already reported in constraint 1
        k = ai  # only k=assignment[i] gives z_{ik}=1
        for j in N:
            aj = assignment[j]
            if aj < 0 or aj >= n:
                # LHS = 0 (no valid m), RHS = z_{ik} = 1
                record_violation(3,
                    f"Constraint 3 (Eq.4) violated: sum_m x[{i},{j},{k},m]=0 != z[{i},{k}]=1 "
                    f"(node {j} has invalid assignment {aj})",
                    0.0, 1.0, "eq")

    # =========================================================================
    # Constraint 4 (Eq. 5): sum_{k} x_{ijkm} = z_{jm}, for all i,j,m in N
    # =========================================================================
    # Symmetric to Constraint 3. For fixed i,j,m:
    #   LHS = sum_k x_{ijkm}. x_{ijkm}=1 requires assignment[j]==m AND
    #         assignment[i]==k. If assignment[j]==m, LHS=1 (k=assignment[i]).
    #         If assignment[j]!=m, LHS=0.
    #   RHS = z_{jm} = 1 if assignment[j]==m, else 0.
    # Again LHS==RHS unless assignments are invalid.
    for j in N:
        aj = assignment[j]
        if aj < 0 or aj >= n:
            continue  # already reported
        m = aj
        for i in N:
            ai = assignment[i]
            if ai < 0 or ai >= n:
                record_violation(4,
                    f"Constraint 4 (Eq.5) violated: sum_k x[{i},{j},k,{m}]=0 != z[{j},{m}]=1 "
                    f"(node {i} has invalid assignment {ai})",
                    0.0, 1.0, "eq")

    # =========================================================================
    # Constraint 5 (Eq. 6): sum_{i} O_i * z_{ik} <= b_k * z_{kk}, for all k in N
    # =========================================================================
    for k in N:
        lhs = sum(O[i] for i in N if assignment[i] == k)
        z_kk = 1.0 if assignment[k] == k else 0.0
        rhs = b[k] * z_kk
        if lhs - rhs > tol:
            record_violation(5,
                f"Constraint 5 (Eq.6) violated: hub {k} capacity exceeded: "
                f"incoming flow {lhs:.4f} > capacity {rhs:.4f}",
                lhs, rhs, "leq")

    # =========================================================================
    # Constraint 6 (Eq. 7): sum_{k} b_k * z_{kk} >= D
    # =========================================================================
    lhs = sum(b[k] for k in N if assignment[k] == k)
    rhs = D
    if rhs - lhs > tol:
        record_violation(6,
            f"Constraint 6 (Eq.7) violated: total hub capacity {lhs:.4f} < total demand {rhs:.4f}",
            lhs, rhs, "geq")

    # =========================================================================
    # Constraint 7 (Eq. 8): x_{ijkm} >= 0, for all i,j,k,m in N
    # =========================================================================
    # All x values are 0 or 1 by construction (from integer assignment), so
    # non-negativity is always satisfied. We verify this holds.
    # The only non-zero x values are x[i,j,assignment[i],assignment[j]] = 1.
    # All others are 0. Both 0 and 1 are >= 0.
    # No violation possible with valid binary construction.
    for i in N:
        for j in N:
            k = assignment[i]
            m = assignment[j]
            x_val = 1.0  # by construction
            if x_val < -tol:
                record_violation(7,
                    f"Constraint 7 (Eq.8) violated: x[{i},{j},{k},{m}]={x_val} < 0",
                    x_val, 0.0, "geq")

    # =========================================================================
    # Constraint 8 (Eq. 9): z_{ik} in {0, 1}, for all i,k in N
    # =========================================================================
    # z values are exactly 0 or 1 by construction from integer assignment.
    # Verify assignments produce valid binary values.
    for i in N:
        ai = assignment[i]
        if not isinstance(ai, int) or ai < 0 or ai >= n:
            # z values cannot be properly defined
            record_violation(8,
                f"Constraint 8 (Eq.9) violated: node {i} has invalid assignment {ai}, "
                f"z variables not properly binary",
                float('nan'), 0.0, "eq")

    # =========================================================================
    # Constraint 9 (Tier-C, objective consistency): the reported
    # objective_value must equal the recomputed total cost of the solution.
    #
    # Recompute (Eq. 1):
    #     obj = sum_{k in hubs} f_k
    #         + sum_{i,j in N} W_{ij}
    #               * (chi * d[i][a[i]] + alpha * d[a[i]][a[j]] + delta * d[a[j]][j])
    # where a[i] = assignment[i].
    #
    # This is only meaningful if the assignment indices are valid (otherwise
    # the existing constraints 1/3/4/8 already capture infeasibility).
    # Tolerance: max(1e-3 absolute, 1e-3 * |recomputed|) -> 0.1% relative.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None

        if reported is not None:
            try:
                flow = instance["flow_matrix"]
                dist = instance["distance_matrix"]
                f_setup = instance["setup_costs"]
                cp = instance["cost_parameters"]
                chi = float(cp["collection_cost_chi"])
                alpha = float(cp["transfer_cost_alpha"])
                delta = float(cp["distribution_cost_delta"])

                # Verify assignment indices are valid before recomputing.
                indices_valid = all(
                    isinstance(assignment[i], int) and 0 <= assignment[i] < n
                    for i in N
                )

                if indices_valid:
                    setup_total = 0.0
                    for k in hubs:
                        if isinstance(k, int) and 0 <= k < n:
                            setup_total += float(f_setup[k])

                    routing_total = 0.0
                    for i in N:
                        ai = assignment[i]
                        d_i_ai = float(dist[i][ai])
                        for j in N:
                            aj = assignment[j]
                            wij = float(flow[i][j])
                            if wij == 0.0:
                                continue
                            cost_per_unit = (
                                chi * d_i_ai
                                + alpha * float(dist[ai][aj])
                                + delta * float(dist[aj][j])
                            )
                            routing_total += wij * cost_per_unit

                    true_obj = setup_total + routing_total
                    abs_diff = abs(reported - true_obj)
                    # 0.1% relative tolerance with 1e-3 absolute floor.
                    obj_tol = max(1e-3, 1e-3 * abs(true_obj))
                    if abs_diff > obj_tol:
                        record_violation(9,
                            f"Constraint 9 (obj consistency) violated: reported "
                            f"objective_value={reported} differs from recomputed "
                            f"sum_k f_k z_kk + sum_ij W_ij*(chi*d_i_ai + alpha*d_ai_aj + delta*d_aj_j)"
                            f"={true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                            reported, true_obj, "eq")
            except (KeyError, IndexError, TypeError, ValueError):
                # Instance data missing or malformed; skip obj check.
                pass

    # Aggregate results
    violated_constraint_indices = sorted(set(v[0] for v in violations))
    violation_messages = []
    for idx in violated_constraint_indices:
        msgs = [v[1] for v in violations if v[0] == idx]
        if len(msgs) == 1:
            violation_messages.append(msgs[0])
        else:
            violation_messages.append(f"{msgs[0]} (and {len(msgs)-1} more similar violations)")

    feasible = len(violated_constraint_indices) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraint_indices,
        "violations": violation_messages,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for CHLPSA (Contreras et al. 2011)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path for the output feasibility result JSON file")
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

    if result["feasible"] is None:
        print(f"Solution could not be checked.")
        for msg in result["violations"]:
            print(f"  - {msg}")
    elif result["feasible"]:
        print(f"Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE.")
        print(f"Violated constraints: {result['violated_constraints']}")
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
