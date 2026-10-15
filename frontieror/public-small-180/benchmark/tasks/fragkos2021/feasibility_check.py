"""
Feasibility checker for the Stochastic Reliable Uncapacitated Facility Location
Problem (RUFLP) from Li, Li, Shu, Song, Zhang (2021).

Checks hard constraints from the mathematical formulation (math_model.txt),
numbered strictly top to bottom:

  Constraint 1:  Ambiguity set probability interval (Eq 1)
                 sum_{matching xi} p_xi in [q_k^-, q_k^+] for all k
  Constraint 2:  Ambiguity set normalization (Eq 1)
                 sum_{xi} p_xi = 1
  Constraint 3:  Binary domain (Decision Variables section)
                 x_j in {0,1} for all j in J
  Constraint 4:  Epigraph constraint (Eq 4)
                 eta >= E_p[Q(x, xi)] for all p in P
  Constraint 5:  RP cut constraint
                 (implied by C4; algorithmic — not separately checked)
  Constraint 6:  Valid cut (Eq 5)
                 (implied by C4; algorithmic — not separately checked)
  Constraint 7:  Separation normalization (Eq 6, sum p_xi = 1)
                 Internal to separation sub-problem — not checkable from solution
  Constraint 8:  Separation upper bound (Eq 6, <= q_k^+)
                 Internal to separation sub-problem — not checkable from solution
  Constraint 9:  Separation lower bound (Eq 6, >= q_k^-)
                 Internal to separation sub-problem — not checkable from solution
  Constraint 10: Dual feasibility (Eq 7, alpha + sum >= Q)
                 Internal to dual sub-problem — not checkable from solution
  Constraint 11: Dual non-negativity overline_beta (Eq 7)
                 Internal to dual sub-problem — not checkable from solution
  Constraint 12: Dual non-negativity underline_beta (Eq 7)
                 Internal to dual sub-problem — not checkable from solution
  Constraint 13: Pricing pi bound (Eq 9)
                 Internal to pricing sub-problem — not checkable from solution
  Constraint 14: Pricing lambda <= xi_j (Eq 9)
                 Internal to pricing sub-problem — not checkable from solution
  Constraint 15: Pricing lambda <= 1 - xi_j (Eq 9)
                 Internal to pricing sub-problem — not checkable from solution
  Constraint 16: Pricing lambda >= sum expression (Eq 9)
                 Internal to pricing sub-problem — not checkable from solution
  Constraint 17: Pricing lambda >= 0 (Eq 9)
                 Internal to pricing sub-problem — not checkable from solution
  Constraint 18: Pricing xi binary (Eq 9)
                 Internal to pricing sub-problem — not checkable from solution
  Constraint 19: Objective-value consistency (Tier C objective recompute)
                 objective_value == sum_j f_j x_j + E_{p^dagger}[Q(x, xi)]
                 The reported objective_value must equal the true objective of
                 the reported facility-location vector x. For the stochastic
                 case P is a singleton and x is the only obj-determining
                 variable, so this is an EXACT full recompute. Constraint 4
                 only rejects an objective reported too LOW (eta < E[Q]);
                 constraint 19 additionally rejects an objective reported too
                 HIGH and tightens the check to a two-sided equality, defeating
                 fabricated objective_value exploits.

Constraints 1-2 define the ambiguity set P (a property of the problem, not the
solution). For the stochastic case, P is a singleton — automatically satisfied.

Constraints 5-18 involve variables internal to the cutting plane, separation,
dual, or pricing sub-problems (p, alpha, beta, pi, lambda, xi). These variables
are not part of the candidate solution and cannot be checked.

Checkable hard constraints on the candidate solution (x, objective_value):
  - Constraint 3: x_j in {0,1} for all j
  - Constraint 4: eta >= E[Q(x, xi)]  (where eta = objective_value - setup_cost)
  - Constraint 19: objective_value == setup_cost + E[Q(x, xi)]  (full recompute)
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('facility_locations', 'objective_value')
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



# -----------------------------------------------------------------------
# Algorithm 1 (from paper): Evaluate E_{p^dagger}[Q(x, xi)]
# for independent facility disruptions.  O(|I||J| log |J|)
# -----------------------------------------------------------------------
def compute_expected_cost(x_vals, q, c, penalty, demands, n_fac, n_cus):
    """
    Compute E_{p^dagger}[Q(x, xi)] using Algorithm 1 from the paper.

    For independent disruptions, the expected transportation cost for customer i
    is computed by sorting open facilities by cost and accumulating:
        cost_i = sum_t gamma * q_{j_t} * d_i * c_{i,j_t}
    where gamma = Prob(all cheaper facilities disrupted).

    Dummy facility 0 (always available at penalty cost) is appended last.
    """
    DUMMY = n_fac
    open_facs = [j for j in range(n_fac) if x_vals[j] > 0.5]

    total = 0.0
    for i in range(n_cus):
        # Sort open facilities by transportation cost ascending; dummy last
        L_i = sorted(open_facs, key=lambda j: c[i][j]) + [DUMMY]

        gamma = 1.0  # Prob(all previous facilities disrupted)
        for j_t in L_i:
            if j_t == DUMMY:
                q_t = 1.0        # dummy always online
                c_t = penalty[i]
            else:
                q_t = q[j_t]     # survival probability
                c_t = c[i][j_t]

            total += gamma * q_t * demands[i] * c_t
            gamma *= (1.0 - q_t)

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Stochastic RUFLP (Li et al. 2021)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON problem instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON candidate solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path for the output JSON feasibility result")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load instance
    # ------------------------------------------------------------------
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n_fac = instance["num_facilities"]
    n_cus = instance["num_customers"]
    nodes = instance["nodes"]

    demands = [nodes[i]["demand"] for i in range(n_cus)]
    f_costs = [nodes[j]["fixed_setup_cost"] for j in range(n_fac)]
    q = [nodes[j]["survival_probability"] for j in range(n_fac)]
    c = instance["transportation_cost_matrix"]
    penalty = instance["penalty_cost"]

    # ------------------------------------------------------------------
    # Load candidate solution
    # ------------------------------------------------------------------
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

    x_vals = solution["facility_locations"]
    obj_reported = solution["objective_value"]

    # ------------------------------------------------------------------
    # Tolerance parameters
    # ------------------------------------------------------------------
    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ==================================================================
    # Constraint 3: x_j in {0,1} for all j in J
    # (Binary domain from Decision Variables section)
    # ==================================================================

    # 3a: Check dimensionality — |x| must equal |J|
    if len(x_vals) != n_fac:
        violated_constraints.add(3)
        lhs = float(len(x_vals))
        rhs = float(n_fac)
        violation_amount = abs(lhs - rhs)
        normalizer = max(abs(rhs), eps)
        violations.append(
            f"Dimension mismatch: facility_locations has length {len(x_vals)}, "
            f"expected {n_fac}"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer,
        })

    # 3b: Check each x_j is binary (0 or 1)
    for j in range(min(len(x_vals), n_fac)):
        val = x_vals[j]
        # x_j should be 0 or 1; measure distance to nearest binary value
        dist_to_binary = min(abs(val - 0.0), abs(val - 1.0))
        if dist_to_binary > tol:
            violated_constraints.add(3)
            lhs = float(val)
            rhs = round(val)  # nearest binary
            violation_amount = dist_to_binary
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Facility {j}: x_{j} = {val} is not binary "
                f"(distance to nearest binary value = {dist_to_binary:.6e})"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": float(rhs),
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ==================================================================
    # Constraint 4: eta >= E_p[Q(x, xi)] for all p in P
    # (Epigraph constraint from Eq 4)
    #
    # For the stochastic case, P is a singleton, so this reduces to:
    #   eta >= E_{p^dagger}[Q(x, xi)]
    #
    # Since objective_value = sum_j f_j x_j + eta, we derive:
    #   eta = objective_value - setup_cost
    # and check eta >= E[Q(x, xi)].
    # ==================================================================
    if len(x_vals) == n_fac:
        # Compute setup cost
        setup_cost = sum(
            f_costs[j] * (1.0 if x_vals[j] > 0.5 else 0.0)
            for j in range(n_fac)
        )

        # Compute E[Q(x, xi)] via Algorithm 1
        expected_transport = compute_expected_cost(
            x_vals, q, c, penalty, demands, n_fac, n_cus
        )

        # Derive eta from reported objective
        eta = obj_reported - setup_cost

        # Check: eta >= E[Q(x, xi)]
        # Equivalently: objective_value >= setup_cost + E[Q(x, xi)]
        # LHS = eta, RHS = E[Q(x, xi)]
        # For >= constraint: violation_amount = max(0, RHS - LHS)
        lhs_val = eta
        rhs_val = expected_transport
        violation_amount = max(0.0, rhs_val - lhs_val)

        if violation_amount > tol:
            violated_constraints.add(4)
            normalizer = max(abs(rhs_val), eps)
            violations.append(
                f"Epigraph violated: eta = {eta:.6f} < E[Q(x, xi)] = "
                f"{expected_transport:.6f} "
                f"(objective_value = {obj_reported:.6f}, "
                f"setup_cost = {setup_cost:.6f}, "
                f"true_objective = {setup_cost + expected_transport:.6f})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

        # ==============================================================
        # Constraint 19: Objective-value consistency (Tier C obj recompute)
        #
        # objective_value must equal the true objective of the reported
        # facility-location vector x:
        #   true_obj = sum_j f_j x_j  +  E_{p^dagger}[Q(x, xi)]
        #            = setup_cost     +  expected_transport
        #
        # For the stochastic RUFLP, P is a singleton and x (facility_locations)
        # is the ONLY objective-determining variable in the solution, so this
        # is an EXACT full recompute (all five Gurobi reference solutions
        # recompute to abs_diff == 0).
        #
        # Constraint 4 above only rejects an objective reported too LOW
        # (eta < E[Q]); constraint 19 additionally rejects an objective
        # reported too HIGH and tightens the check to a two-sided equality,
        # defeating fabricated objective_value exploits (obj = 0,
        # obj = sys.float_info.max, obj = huge constant, ...).
        # ==============================================================
        true_obj = setup_cost + expected_transport
        obj_diff = abs(float(obj_reported) - true_obj)
        # 0.1% relative tolerance with a 1e-3 absolute floor
        obj_tol = max(1e-3, 1e-3 * abs(true_obj))

        if obj_diff > obj_tol:
            violated_constraints.add(19)
            normalizer = max(abs(true_obj), eps)
            violations.append(
                f"Objective consistency violated: reported objective_value = "
                f"{obj_reported} differs from recomputed "
                f"setup_cost + E[Q(x, xi)] = {true_obj:.6f} "
                f"(setup_cost = {setup_cost:.6f}, "
                f"E[Q(x, xi)] = {expected_transport:.6f}, "
                f"|diff| = {obj_diff:.6e}, tol = {obj_tol:.6e})"
            )
            violation_magnitudes.append({
                "constraint": 19,
                "lhs": float(obj_reported),
                "rhs": float(true_obj),
                "raw_excess": obj_diff,
                "normalizer": normalizer,
                "ratio": obj_diff / normalizer,
            })

    # ==================================================================
    # Constraints 1-2: Ambiguity set constraints on probability
    #   distribution p. For stochastic RUFLP, P is a known singleton —
    #   these are part of the problem definition, not the solution.
    #   Not checkable from candidate solution.
    #
    # Constraints 5-6: RP cut and Valid Cut — algorithmic constraints
    #   implied by Constraint 4. Not separately checked.
    #
    # Constraints 7-18: Internal to separation, dual, and pricing
    #   sub-problems. Variables (p, alpha, beta, pi, lambda, xi) are
    #   not part of the candidate solution. Not checkable.
    # ==================================================================

    # ------------------------------------------------------------------
    # Compile results
    # ------------------------------------------------------------------
    feasible = len(violated_constraints) == 0
    sorted_violated = sorted(violated_constraints)

    result = {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    if feasible:
        print("FEASIBLE: All checkable hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {sorted_violated}")
        for v in violations:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
