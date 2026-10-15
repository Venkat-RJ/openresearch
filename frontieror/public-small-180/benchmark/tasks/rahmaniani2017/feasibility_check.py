#!/usr/bin/env python3
"""
Feasibility checker for the Uncapacitated Hub Location Problem
with Multiple Assignments (UHLPMA).

Checks all hard constraints from both the original formulation (Eqs 2-5)
and the path-based formulation (Eqs 7-10) as listed in math_model.txt.

Constraint numbering (top to bottom from the formulation sections):
  Constraint 1 (Eq 2):  sum_{i,j in H} x_{ijk} = 1,  forall k in K
  Constraint 2 (Eq 3):  sum_j x_{ijk} + sum_{j!=i} x_{jik} <= z_i,  forall i in H, k in K
  Constraint 3 (Eq 4):  x_{ijk} >= 0,  forall i,j in H, k in K
  Constraint 4 (Eq 5):  z_i in {0,1},  forall i in H
  Constraint 5 (Eq 7):  sum_{e in E_k} x_{ek} = 1,  forall k in K
  Constraint 6 (Eq 8):  sum_{e in E_k: i in e} x_{ek} <= z_i,  forall i in H, k in K
  Constraint 7 (Eq 9):  x_{ek} >= 0,  forall k in K, e in E_k
  Constraint 8 (Eq 10): z_i in {0,1},  forall i in H
  Constraint 9 (NEW):   Objective-value envelope check (Tier C anti-exploit).
                        reported objective_value must satisfy
                        obj_lower <= reported <= obj_upper, where
                          obj_lower = sum_k W_k * min(chi, tau, delta) * d[o(k), d(k)]
                                    + min_i f_i
                          obj_upper = 1e15 (loose absolute ceiling to reject
                                            sys.float_info.max-style sentinels;
                                            the per-instance tight upper bound
                                            cannot accommodate the reference
                                            Gurobi solutions, so this acts as a
                                            sanity cap rather than a tight envelope).
                        The lower bound uses triangle inequality:
                          F_ek = W_k * (chi*d[o,i] + tau*d[i,j] + delta*d[j,d])
                               >= W_k * min(chi,tau,delta) * d[o,d]
                        so it is a valid LB regardless of which hubs are open.
                        Plus min_i f_i because >=1 hub must be open for feasibility.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('assignments', 'objective_value', 'open_hubs')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ('assignments', 'objective_value', 'open_hubs')
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



def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    # ----------------------------------------------------------------
    # Extract instance data
    # ----------------------------------------------------------------
    nodes = instance["nodes"]
    n = len(nodes)
    H = list(range(n))
    params = instance["parameters"]
    chi = params["chi"]
    delta = params["delta"]
    tau = params["tau"]

    commodities_raw = instance["commodities"]

    # Merge opposite commodity pairs when chi == delta (Property 3)
    merge = abs(chi - delta) < 1e-9
    if merge:
        merged = {}
        for comm in commodities_raw:
            o, d = comm["origin"], comm["destination"]
            key = (min(o, d), max(o, d))
            if key in merged:
                merged[key]["flow"] += comm["flow"]
            else:
                merged[key] = {
                    "origin": key[0],
                    "destination": key[1],
                    "flow": comm["flow"],
                }
        commodities = list(merged.values())
    else:
        commodities = [
            {"origin": c["origin"], "destination": c["destination"], "flow": c["flow"]}
            for c in commodities_raw
        ]

    num_K = len(commodities)

    # ----------------------------------------------------------------
    # Extract solution data
    # ----------------------------------------------------------------
    open_hubs = set(solution["open_hubs"])
    z = {i: (1.0 if i in open_hubs else 0.0) for i in H}

    assignments = solution.get("assignments", [])

    # Build lookup: (origin, destination) -> hub_edge
    # and commodity index -> hub_edge
    assign_by_od = {}
    assign_by_idx = {}
    for a in assignments:
        od_key = (a["origin"], a["destination"])
        assign_by_od[od_key] = a["hub_edge"]
        assign_by_idx[a["commodity"]] = a["hub_edge"]

    # Build expected commodity set: (origin, destination)
    expected_od = set()
    for c in commodities:
        expected_od.add((c["origin"], c["destination"]))

    # ----------------------------------------------------------------
    # Check constraints
    # ----------------------------------------------------------------
    violations = []
    violation_magnitudes = []
    violated_set = set()

    def record_violation(constraint_idx, msg, lhs, rhs, raw_excess):
        violated_set.add(constraint_idx)
        normalizer = max(abs(rhs), eps)
        violations.append(msg)
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": raw_excess,
            "normalizer": normalizer,
            "ratio": raw_excess / normalizer,
        })

    raw_open_hubs = solution["open_hubs"]
    if (
        not raw_open_hubs
        or len(set(raw_open_hubs)) != len(raw_open_hubs)
        or any(isinstance(hub, bool) or not isinstance(hub, int) or hub not in H
               for hub in raw_open_hubs)
    ):
        record_violation(
            4,
            "open_hubs must be a nonempty list of unique in-range node ids",
            float(len(set(raw_open_hubs))), float(len(raw_open_hubs)), 1.0,
        )

    expected_indices = set(range(num_K))
    observed_indices = []
    observed_od = []
    assignment_costs = {}
    distances = instance["distances"]
    for position, assignment in enumerate(assignments):
        try:
            commodity_index = int(assignment["commodity"])
            origin = int(assignment["origin"])
            destination = int(assignment["destination"])
            hub_edge = assignment["hub_edge"]
            reported_cost = float(assignment["cost"])
        except (KeyError, TypeError, ValueError) as error:
            record_violation(
                1, f"Assignment {position} is malformed: {error}",
                1.0, 0.0, 1.0,
            )
            continue
        observed_indices.append(commodity_index)
        observed_od.append((origin, destination))
        if commodity_index not in expected_indices:
            record_violation(
                1, f"Assignment uses unknown commodity {commodity_index}",
                float(commodity_index), float(num_K - 1), 1.0,
            )
            continue
        commodity = commodities[commodity_index]
        expected_pair = (commodity["origin"], commodity["destination"])
        if (origin, destination) != expected_pair:
            record_violation(
                1,
                f"Commodity {commodity_index} reports OD {(origin, destination)}, "
                f"expected {expected_pair}",
                1.0, 0.0, 1.0,
            )
        valid_edge = (
            isinstance(hub_edge, list)
            and len(hub_edge) in (1, 2)
            and all(isinstance(hub, int) and not isinstance(hub, bool)
                    and hub in H for hub in hub_edge)
            and (len(hub_edge) == 1 or hub_edge[0] < hub_edge[1])
        )
        if not valid_edge:
            record_violation(
                2,
                f"Commodity {commodity_index} hub_edge must be one hub or a "
                "strictly increasing pair of in-range hubs",
                float(len(hub_edge)) if isinstance(hub_edge, list) else 0.0,
                1.0, 1.0,
            )
            continue
        if any(hub not in open_hubs for hub in hub_edge):
            record_violation(
                2,
                f"Commodity {commodity_index} uses a hub that is not open",
                1.0, 0.0, 1.0,
            )

        flow = float(commodity["flow"])
        if len(hub_edge) == 1:
            first = second = hub_edge[0]
            expected_cost = flow * (
                chi * distances[origin][first]
                + delta * distances[second][destination]
            )
        else:
            first, second = hub_edge
            forward = flow * (
                chi * distances[origin][first]
                + tau * distances[first][second]
                + delta * distances[second][destination]
            )
            reverse = flow * (
                chi * distances[origin][second]
                + tau * distances[second][first]
                + delta * distances[first][destination]
            )
            expected_cost = min(forward, reverse)
        assignment_costs[commodity_index] = expected_cost
        cost_difference = abs(reported_cost - expected_cost)
        if not math.isfinite(reported_cost) or cost_difference > max(
                1e-5, 1e-9 * abs(expected_cost)):
            record_violation(
                9,
                f"Commodity {commodity_index} cost={reported_cost} does not "
                f"match decision-derived cost {expected_cost}",
                reported_cost, expected_cost, cost_difference,
            )

    if (
        len(assignments) != num_K
        or set(observed_indices) != expected_indices
        or len(observed_indices) != len(set(observed_indices))
        or len(observed_od) != len(set(observed_od))
    ):
        record_violation(
            1,
            "assignments must contain exactly one unique record for every "
            "merged commodity",
            float(len(set(observed_indices))), float(num_K),
            float(abs(len(set(observed_indices)) - num_K) or 1),
        )

    # ==================================================================
    # Constraint 1 (Eq 2): sum_{i,j in H} x_{ijk} = 1, forall k in K
    # Each commodity must be assigned exactly once.
    # ==================================================================
    for k_idx, comm in enumerate(commodities):
        od = (comm["origin"], comm["destination"])
        count = 1.0 if od in assign_by_od else 0.0
        lhs = count
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            record_violation(
                1,
                f"Constraint 1 (Eq 2): Commodity ({od[0]},{od[1]}) "
                f"has {int(count)} assignment(s), expected 1",
                lhs, rhs, violation_amount,
            )

    # ==================================================================
    # Constraint 2 (Eq 3): sum_j x_{ijk} + sum_{j!=i} x_{jik} <= z_i
    # If hub i is used in the route for commodity k, z_i must be 1.
    # ==================================================================
    for comm in commodities:
        od = (comm["origin"], comm["destination"])
        if od not in assign_by_od:
            continue
        hub_edge = assign_by_od[od]
        for i in hub_edge:
            lhs = 1.0  # hub i is used by this commodity
            rhs = z.get(i, 0.0)
            violation_amount = max(0.0, lhs - rhs)
            if violation_amount > tol:
                record_violation(
                    2,
                    f"Constraint 2 (Eq 3): Hub {i} used by commodity "
                    f"({od[0]},{od[1]}) but hub {i} is not open (z_{i}={rhs})",
                    lhs, rhs, violation_amount,
                )

    # ==================================================================
    # Constraint 3 (Eq 4): x_{ijk} >= 0, forall i,j in H, k in K
    # Non-negativity of routing variables. With the solution format
    # (assignments are binary 0/1), this is trivially satisfied.
    # We verify no negative implicit values exist.
    # ==================================================================
    # (No violation possible: x values are implicitly 0 or 1)

    # ==================================================================
    # Constraint 4 (Eq 5): z_i in {0, 1}, forall i in H
    # Binary hub-opening variables.
    # ==================================================================
    for i in H:
        z_val = z[i]
        violation_amount = min(abs(z_val - 0.0), abs(z_val - 1.0))
        if violation_amount > tol:
            record_violation(
                4,
                f"Constraint 4 (Eq 5): z_{i} = {z_val} is not binary",
                z_val, round(z_val), violation_amount,
            )

    # ==================================================================
    # Constraint 5 (Eq 7): sum_{e in E_k} x_{ek} = 1, forall k in K
    # Each commodity must be assigned to exactly one hub edge.
    # (Path-based formulation, equivalent to Constraint 1.)
    # ==================================================================
    for k_idx, comm in enumerate(commodities):
        od = (comm["origin"], comm["destination"])
        count = 1.0 if od in assign_by_od else 0.0
        lhs = count
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            record_violation(
                5,
                f"Constraint 5 (Eq 7): Commodity ({od[0]},{od[1]}) "
                f"has {int(count)} hub-edge assignment(s), expected 1",
                lhs, rhs, violation_amount,
            )

    # ==================================================================
    # Constraint 6 (Eq 8): sum_{e in E_k: i in e} x_{ek} <= z_i
    # Routing through hub i requires z_i = 1.
    # (Path-based formulation, equivalent to Constraint 2.)
    # ==================================================================
    for comm in commodities:
        od = (comm["origin"], comm["destination"])
        if od not in assign_by_od:
            continue
        hub_edge = assign_by_od[od]
        for i in hub_edge:
            lhs = 1.0
            rhs = z.get(i, 0.0)
            violation_amount = max(0.0, lhs - rhs)
            if violation_amount > tol:
                record_violation(
                    6,
                    f"Constraint 6 (Eq 8): Hub {i} in edge for commodity "
                    f"({od[0]},{od[1]}) but hub {i} is not open (z_{i}={rhs})",
                    lhs, rhs, violation_amount,
                )

    # ==================================================================
    # Constraint 7 (Eq 9): x_{ek} >= 0, forall k in K, e in E_k
    # Non-negativity of path-based routing variables.
    # ==================================================================
    # (No violation possible: x values are implicitly 0 or 1)

    # ==================================================================
    # Constraint 8 (Eq 10): z_i in {0, 1}, forall i in H
    # Binary hub-opening variables (path-based, same as Constraint 4).
    # ==================================================================
    for i in H:
        z_val = z[i]
        violation_amount = min(abs(z_val - 0.0), abs(z_val - 1.0))
        if violation_amount > tol:
            record_violation(
                8,
                f"Constraint 8 (Eq 10): z_{i} = {z_val} is not binary",
                z_val, round(z_val), violation_amount,
            )

    # ==================================================================
    # Constraint 9: exact fixed + transportation objective from the complete
    # hub/commodity witness and the already-TC-scaled distance matrix.
    # ==================================================================
    if set(assignment_costs) == expected_indices:
        fixed_cost = sum(float(nodes[hub]["fixed_cost"]) for hub in open_hubs)
        transport_cost = sum(assignment_costs.values())
        expected_objective = fixed_cost + transport_cost
        reported = float(solution["objective_value"])
        difference = abs(reported - expected_objective)
        objective_tolerance = max(1e-4, 1e-8 * abs(expected_objective))
        if difference > objective_tolerance:
            record_violation(
                9,
                f"Constraint 9 (Objective consistency): objective_value="
                f"{reported} does not match fixed + transportation cost "
                f"{expected_objective} (fixed={fixed_cost}, "
                f"transport={transport_cost}, |diff|={difference:.6g})",
                reported, expected_objective, difference,
            )

        for field, expected_value in (
            ("num_open_hubs", float(len(open_hubs))),
            ("total_fixed_cost", fixed_cost),
            ("total_transport_cost", transport_cost),
        ):
            if field not in solution:
                continue
            field_value = float(solution[field])
            field_difference = abs(field_value - expected_value)
            if field_difference > max(1e-4, 1e-8 * abs(expected_value)):
                record_violation(
                    9,
                    f"{field}={field_value} does not match "
                    f"decision-derived value {expected_value}",
                    field_value, expected_value, field_difference,
                )

    # ----------------------------------------------------------------
    # Build result
    # ----------------------------------------------------------------
    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for UHLPMA solutions."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
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
