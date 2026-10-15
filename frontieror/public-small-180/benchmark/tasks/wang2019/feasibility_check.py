#!/usr/bin/env python3
"""
Feasibility checker for Service Network Design with Heterogeneous Fleet.
Paper: Wang, Qi, Cheng, Zhang (2019) - EJOR 276, 483-494

Checks the arc-based formulation (Equations 1-6):
  Constraint 1 (Eq 2): Flow conservation for each commodity at each node
  Constraint 2 (Eq 3): Design-balance for each fleet type at each node
  Constraint 3 (Eq 4): Capacity linking on each arc
  Constraint 4 (Eq 5): Non-negativity of flow variables
  Constraint 5 (Eq 6): Integrality and non-negativity of vehicle variables
  Constraint 6 (Eq 1): Objective-value consistency -- the reported
      objective_value must match the recomputed cost
        sum_{k,(i,j)} c_{ij}^k * x_{ij}^k + sum_{f,(i,j)} h_{ij}^f * y_{ij}^f
      from the solution variables.  Tier C defense against LLM
      score-gaming exploits that pass constraints but lie about cost.
"""

import json
import argparse
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('flows', 'objective_value', 'vehicles')
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



def _flow_cost(arc, k):
    fc = arc.get("flow_costs")
    if isinstance(fc, dict):
        if str(k) in fc:
            return fc[str(k)]
        if k in fc:
            return fc[k]
    elif isinstance(fc, list) and 0 <= k < len(fc):
        return fc[k]
    return arc["flow_cost"]


def _fixed_cost(fleet, arc_id):
    ac = fleet.get("arc_fixed_costs")
    if isinstance(ac, dict):
        if str(arc_id) in ac:
            return ac[str(arc_id)]
        if arc_id in ac:
            return ac[arc_id]
    elif isinstance(ac, list) and 0 <= arc_id < len(ac):
        return ac[arc_id]
    return fleet["fixed_cost"]


def check_feasibility(data, sol):
    tol = 1e-5
    eps = 1e-5

    nodes = data["nodes"]
    arcs = data["arcs"]
    commodities = data["commodities"]
    fleet_types = data["fleet_types"]

    arc_set = set()
    for a in arcs:
        arc_set.add((a["from"], a["to"]))

    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for a in arcs:
        successors[a["from"]].append(a["to"])
        predecessors[a["to"]].append(a["from"])

    x = defaultdict(float)
    for key, val in sol.get("flows", {}).items():
        parts = key.split("_")
        k = int(parts[1])
        i = int(parts[2])
        j = int(parts[3])
        x[(k, i, j)] = float(val)

    y = defaultdict(float)
    for key, val in sol.get("vehicles", {}).items():
        parts = key.split("_")
        f_id = int(parts[1])
        i = int(parts[2])
        j = int(parts[3])
        y[(f_id, i, j)] = float(val)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ================================================================
    # Constraint 1 (Eq 2): Flow conservation
    # ================================================================
    for kd in commodities:
        k = kd["id"]
        origin = kd["origin"]
        dest = kd["destination"]
        demand = kd["demand"]
        for i in nodes:
            if i == origin:
                w = demand
            elif i == dest:
                w = -demand
            else:
                w = 0

            outflow = sum(x[(k, i, j)] for j in successors[i])
            inflow = sum(x[(k, j, i)] for j in predecessors[i])
            lhs = outflow - inflow
            rhs = float(w)
            violation_amount = abs(lhs - rhs)

            if violation_amount > tol:
                violated_constraints.add(1)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 1 (Flow conservation): commodity {k} at node {i}: "
                    f"net flow = {lhs:.6f}, required = {rhs:.6f}"
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": round(lhs, 10),
                    "rhs": round(rhs, 10),
                    "raw_excess": round(violation_amount, 10),
                    "normalizer": round(normalizer, 10),
                    "ratio": round(ratio, 10)
                })

    # ================================================================
    # Constraint 2 (Eq 3): Design-balance
    # ================================================================
    for fd in fleet_types:
        f_id = fd["id"]
        for i in nodes:
            outflow = sum(y[(f_id, i, j)] for j in successors[i])
            inflow = sum(y[(f_id, j, i)] for j in predecessors[i])
            lhs = outflow - inflow
            rhs = 0.0
            violation_amount = abs(lhs - rhs)

            if violation_amount > tol:
                violated_constraints.add(2)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 2 (Design-balance): fleet type {f_id} at node {i}: "
                    f"net vehicles = {lhs:.6f}, required = 0"
                )
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": round(lhs, 10),
                    "rhs": round(rhs, 10),
                    "raw_excess": round(violation_amount, 10),
                    "normalizer": round(normalizer, 10),
                    "ratio": round(ratio, 10)
                })

    # ================================================================
    # Constraint 3 (Eq 4): Capacity linking
    # ================================================================
    for a in arcs:
        i, j = a["from"], a["to"]
        total_flow = sum(x[(kd["id"], i, j)] for kd in commodities)
        total_capacity = sum(fd["capacity"] * y[(fd["id"], i, j)] for fd in fleet_types)
        lhs = total_flow
        rhs = total_capacity
        violation_amount = max(lhs - rhs, 0.0)

        if violation_amount > tol:
            violated_constraints.add(3)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append(
                f"Constraint 3 (Capacity): arc ({i},{j}): "
                f"total flow = {lhs:.6f}, capacity = {rhs:.6f}"
            )
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": round(lhs, 10),
                "rhs": round(rhs, 10),
                "raw_excess": round(violation_amount, 10),
                "normalizer": round(normalizer, 10),
                "ratio": round(ratio, 10)
            })

    # ================================================================
    # Constraint 4 (Eq 5): Non-negativity of flow variables
    # ================================================================
    for kd in commodities:
        k = kd["id"]
        for a in arcs:
            i, j = a["from"], a["to"]
            val = x[(k, i, j)]
            lhs = val
            rhs = 0.0
            violation_amount = max(-lhs, 0.0)

            if violation_amount > tol:
                violated_constraints.add(4)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 4 (Non-negativity flow): x_{k}_{i}_{j} = {val:.6f} < 0"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": round(lhs, 10),
                    "rhs": round(rhs, 10),
                    "raw_excess": round(violation_amount, 10),
                    "normalizer": round(normalizer, 10),
                    "ratio": round(ratio, 10)
                })

    for (k, i, j), val in x.items():
        if (i, j) not in arc_set and val > tol:
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4 (Non-negativity flow): flow x_{k}_{i}_{j} = {val:.6f} "
                f"on non-existent arc ({i},{j})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": round(val, 10),
                "rhs": 0.0,
                "raw_excess": round(val, 10),
                "normalizer": eps,
                "ratio": round(val / eps, 10)
            })

    # ================================================================
    # Constraint 5 (Eq 6): Integrality and non-negativity of vehicles
    # ================================================================
    for fd in fleet_types:
        f_id = fd["id"]
        for a in arcs:
            i, j = a["from"], a["to"]
            val = y[(f_id, i, j)]

            if val < -tol:
                lhs = val
                rhs = 0.0
                violation_amount = -val
                violated_constraints.add(5)
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violations.append(
                    f"Constraint 5 (Non-neg vehicle): y_{f_id}_{i}_{j} = {val:.6f} < 0"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": round(lhs, 10),
                    "rhs": round(rhs, 10),
                    "raw_excess": round(violation_amount, 10),
                    "normalizer": round(normalizer, 10),
                    "ratio": round(ratio, 10)
                })

            if val >= -tol:
                rounded = round(val)
                int_violation = abs(val - rounded)
                if int_violation > tol:
                    lhs = val
                    rhs = float(rounded)
                    violation_amount = int_violation
                    violated_constraints.add(5)
                    normalizer = max(abs(rhs), eps)
                    ratio = violation_amount / normalizer
                    violations.append(
                        f"Constraint 5 (Integrality vehicle): y_{f_id}_{i}_{j} = {val:.6f} "
                        f"is not integer (nearest int = {rounded})"
                    )
                    violation_magnitudes.append({
                        "constraint": 5,
                        "lhs": round(lhs, 10),
                        "rhs": round(rhs, 10),
                        "raw_excess": round(violation_amount, 10),
                        "normalizer": round(normalizer, 10),
                        "ratio": round(ratio, 10)
                    })

    for (f_id, i, j), val in y.items():
        if (i, j) not in arc_set and val > tol:
            violated_constraints.add(5)
            violations.append(
                f"Constraint 5 (Vehicle domain): y_{f_id}_{i}_{j} = {val:.6f} "
                f"on non-existent arc ({i},{j})"
            )
            violation_magnitudes.append({
                "constraint": 5,
                "lhs": round(val, 10),
                "rhs": 0.0,
                "raw_excess": round(val, 10),
                "normalizer": eps,
                "ratio": round(val / eps, 10)
            })

    # ================================================================
    # Constraint 6 (Eq 1): Objective-value consistency (Tier C)
    #   reported_obj must match
    #     sum_{k,(i,j)} c_{ij}^k * x_{ij}^k
    #   + sum_{f,(i,j)} h_{ij}^f * y_{ij}^f
    # recomputed from the solution variables (legal arcs only --
    # flows/vehicles on non-existent arcs are already caught by 4/5).
    # ================================================================
    reported_raw = sol.get("objective_value")
    if reported_raw is not None:
        try:
            reported = float(reported_raw)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            commodity_ids = [kd["id"] for kd in commodities]
            true_obj = 0.0
            for a in arcs:
                i, j = a["from"], a["to"]
                for k in commodity_ids:
                    xv = x[(k, i, j)]
                    if xv:
                        true_obj += _flow_cost(a, k) * xv
            for fd in fleet_types:
                f_id = fd["id"]
                for a in arcs:
                    i, j = a["from"], a["to"]
                    yv = y[(f_id, i, j)]
                    if yv:
                        true_obj += _fixed_cost(fd, a["id"]) * yv

            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints.add(6)
                normalizer = max(abs(true_obj), eps)
                ratio = abs_diff / normalizer
                violations.append(
                    f"Constraint 6 (Objective consistency): reported objective_value="
                    f"{reported} differs from recomputed total cost="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
                )
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": round(reported, 10),
                    "rhs": round(true_obj, 10),
                    "raw_excess": round(abs_diff, 10),
                    "normalizer": round(normalizer, 10),
                    "ratio": round(ratio, 10)
                })

    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else []
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Service Network Design (Wang et al. 2019)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        data = json.load(f)
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

    result = check_feasibility(data, sol)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    feasible = result["feasible"]
    print(f"Feasibility check complete. Result: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    if not feasible:
        print(f"Violated constraints: {result['violated_constraints']}")
        print(f"Number of violations: {len(result['violations'])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
