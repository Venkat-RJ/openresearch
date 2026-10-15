#!/usr/bin/env python3
"""
Feasibility checker for the Alternative-Fuel Station Location Problem (ASLP)
as formulated in Tran, Nagy, Nguyen, Wassan (2018).

Checks the three hard constraints from the [ASLP] formulation:
  Constraint 1 (Eq 15): Arc coverage — for each refueled OD pair q (y_q=1),
      every directed arc on the round-trip path must be covered by at least
      one open station in K^q_{jk}.
  Constraint 2 (Eq 16): Station count — exactly p stations are located.
  Constraint 3 (Eq 17): Binary integrality — x_i, y_q in {0, 1}.
  Constraint 4 (Eq 14, obj consistency): Reported objective_value must equal
      the recomputed sum_q f_q * y_q (Tier C defense against score-gaming).
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'refueled_od_pairs', 'stations')
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
    with open(path, 'r') as f:
        return json.load(f)


def build_arc_distance_lookup(arcs):
    """Build bidirectional distance lookup from arc list."""
    dist = {}
    for arc in arcs:
        i, j, d = arc['from'], arc['to'], arc['distance']
        dist[(i, j)] = d
        dist[(j, i)] = d
    return dist


def compute_K_sets(od_pairs, arc_dist, R):
    """
    For each OD pair, compute K^q_{jk} for each directed arc on the round trip.

    Returns a list (one per OD pair) of lists of (K_set, covered_by_origin) tuples.
    K_set is the set of candidate node IDs; covered_by_origin indicates whether
    the arc is automatically covered by the vehicle's initial full tank.
    """
    all_arc_info = []

    for q_idx, od in enumerate(od_pairs):
        path = od['shortest_path']
        L = len(path)

        if L < 2:
            all_arc_info.append([])
            continue

        # Edge distances along the path
        edge_dists = []
        for i in range(L - 1):
            d = arc_dist.get((path[i], path[i + 1]))
            if d is None:
                raise ValueError(
                    f"No arc distance for ({path[i]}, {path[i+1]}) "
                    f"on OD pair {q_idx}"
                )
            edge_dists.append(d)

        # Round-trip: 2L-1 positions, 2(L-1) arcs
        num_positions = 2 * L - 1
        num_arcs = 2 * (L - 1)

        # Node at each round-trip position
        rt_nodes = list(path) + list(reversed(path[:-1]))

        # Cumulative distances along the round trip
        rt_cum = [0.0] * num_positions
        for p_pos in range(1, num_positions):
            if p_pos < L:
                rt_cum[p_pos] = rt_cum[p_pos - 1] + edge_dists[p_pos - 1]
            else:
                ret_edge_idx = 2 * (L - 1) - p_pos
                rt_cum[p_pos] = rt_cum[p_pos - 1] + edge_dists[ret_edge_idx]

        arc_info = []
        for p in range(num_arcs):
            arc_end_cum = rt_cum[p + 1]

            # Origin's initial full tank covers this arc?
            if arc_end_cum <= R + 1e-9:
                arc_info.append((set(), True))
                continue

            # Find all positions c <= p where refueling covers this arc
            K = set()
            for c in range(p + 1):
                if arc_end_cum - rt_cum[c] <= R + 1e-9:
                    K.add(rt_nodes[c])

            arc_info.append((K, False))

        all_arc_info.append(arc_info)

    return all_arc_info


def determine_y_values(solution, od_pairs, all_arc_info, open_stations_set):
    """
    Determine y_q values from the solution.

    If the solution has 'refueled_od_pairs', use those (y_q=1 for listed indices).
    Otherwise, compute y_q: y_q=1 iff every non-origin-covered arc on the
    round trip of OD pair q has at least one open station in its K set.
    """
    num_q = len(od_pairs)
    y = [0] * num_q

    if 'refueled_od_pairs' in solution:
        for q_idx in solution['refueled_od_pairs']:
            if 0 <= q_idx < num_q:
                y[q_idx] = 1
    else:
        # Infer y_q from station placement
        for q_idx in range(num_q):
            arc_info = all_arc_info[q_idx]
            refueled = True
            for K_set, covered_by_origin in arc_info:
                if covered_by_origin:
                    continue
                if len(K_set.intersection(open_stations_set)) == 0:
                    refueled = False
                    break
            y[q_idx] = 1 if refueled else 0

    return y


def check_feasibility(instance, solution):
    """
    Check all hard constraints and return feasibility result.
    """
    n = instance['parameters']['n']
    p = instance['parameters']['p']
    R = instance['parameters']['R']
    nodes = instance['nodes']
    arcs = instance['arcs']
    od_pairs = instance['od_pairs']

    node_ids = set(node['id'] for node in nodes)
    arc_dist = build_arc_distance_lookup(arcs)

    # Extract solution
    stations = solution.get('stations', [])
    open_stations_set = set(stations)

    # Compute K sets
    all_arc_info = compute_K_sets(od_pairs, arc_dist, R)

    # Determine y_q values
    y = determine_y_values(solution, od_pairs, all_arc_info, open_stations_set)

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint 1 (Eq 15): Arc coverage
    #   sum_{i in K^q_{jk}} x_i >= y_q
    #   For each refueled OD pair (y_q=1), every arc must have at least one
    #   open station in K^q_{jk}.
    # =========================================================================
    for q_idx in range(len(od_pairs)):
        if y[q_idx] != 1:
            continue

        arc_info = all_arc_info[q_idx]
        od = od_pairs[q_idx]

        for arc_idx, (K_set, covered_by_origin) in enumerate(arc_info):
            if covered_by_origin:
                continue

            # LHS = number of open stations in K^q_{jk}
            lhs = len(K_set.intersection(open_stations_set))
            rhs = 1.0  # y_q = 1

            # >= constraint: violation if rhs > lhs
            violation_amount = rhs - lhs
            if violation_amount > TOL:
                violated_constraints.add(1)
                origin = od['origin']
                dest = od['destination']
                violations.append(
                    f"Constraint 1 (Eq 15): OD pair {q_idx} "
                    f"(origin={origin}, dest={dest}) arc index {arc_idx} "
                    f"has no open station in K set to cover it "
                    f"(|K ∩ open| = {lhs}, need >= {rhs})"
                )
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": float(lhs),
                    "rhs": float(rhs),
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer)
                })

    # =========================================================================
    # Constraint 2 (Eq 16): Station count
    #   sum_{i in N} x_i = p
    # =========================================================================
    lhs = len(stations)
    rhs = p
    violation_amount = abs(lhs - rhs)
    if violation_amount > TOL:
        violated_constraints.add(2)
        violations.append(
            f"Constraint 2 (Eq 16): Number of stations is {lhs}, "
            f"expected exactly {rhs}"
        )
        normalizer = max(abs(rhs), EPS)
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(violation_amount / normalizer)
        })

    # =========================================================================
    # Constraint 3 (Eq 17): Binary integrality
    #   x_i in {0, 1} for all i in N
    #   y_q in {0, 1} for all q in Q
    # =========================================================================
    # Check x_i: stations must be valid node IDs (each appearing at most once)
    # and all values are implicitly 0 or 1 by the list representation.
    # Check for duplicate stations
    if len(stations) != len(set(stations)):
        violated_constraints.add(3)
        from collections import Counter
        counts = Counter(stations)
        dups = {s: c for s, c in counts.items() if c > 1}
        violations.append(
            f"Constraint 3 (Eq 17): Duplicate station IDs found: {dups}"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": float(len(stations)),
            "rhs": float(len(set(stations))),
            "raw_excess": float(len(stations) - len(set(stations))),
            "normalizer": float(max(abs(len(set(stations))), EPS)),
            "ratio": float((len(stations) - len(set(stations))) /
                           max(abs(len(set(stations))), EPS))
        })

    # Check that all station IDs are valid nodes
    invalid_stations = open_stations_set - node_ids
    if invalid_stations:
        violated_constraints.add(3)
        violations.append(
            f"Constraint 3 (Eq 17): Station IDs not in node set N: "
            f"{sorted(invalid_stations)}"
        )
        for inv_id in sorted(invalid_stations):
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": float(inv_id),
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": float(EPS),
                "ratio": float(1.0 / EPS)
            })

    # Check y_q integrality (if provided explicitly)
    if 'refueled_od_pairs' in solution:
        for q_idx in solution['refueled_od_pairs']:
            if q_idx < 0 or q_idx >= len(od_pairs):
                violated_constraints.add(3)
                violations.append(
                    f"Constraint 3 (Eq 17): refueled_od_pairs index {q_idx} "
                    f"is out of range [0, {len(od_pairs)-1}]"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(q_idx),
                    "rhs": float(len(od_pairs) - 1),
                    "raw_excess": float(q_idx - (len(od_pairs) - 1)),
                    "normalizer": float(max(abs(len(od_pairs) - 1), EPS)),
                    "ratio": float((q_idx - (len(od_pairs) - 1)) /
                                   max(abs(len(od_pairs) - 1), EPS))
                })

    # =========================================================================
    # Constraint 4 (Eq 14, obj consistency): Reported objective_value must
    # match recomputed z_0 = sum_q f_q * y_q. Tier C defense against
    # LLM score-gaming exploits (e.g. obj=sys.float_info.max).
    # =========================================================================
    reported_obj = solution.get('objective_value')
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and not (math.isnan(reported) or math.isinf(reported)):
            true_obj = 0.0
            for q_idx in range(len(od_pairs)):
                if y[q_idx] == 1:
                    true_obj += float(od_pairs[q_idx].get('flow', 0.0))
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(4)
                violations.append(
                    f"Constraint 4 (Eq 14, obj consistency): reported "
                    f"objective_value={reported} differs from recomputed "
                    f"sum_q f_q * y_q={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )
                normalizer = max(abs(true_obj), EPS)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer)
                })
        elif reported is not None:
            # NaN/Inf reported obj: reject as obj-consistency violation.
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4 (Eq 14, obj consistency): reported "
                f"objective_value={reported_obj} is not a finite number"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float('inf') if math.isinf(reported) else float('nan'),
                "rhs": 0.0,
                "raw_excess": float('inf'),
                "normalizer": float(EPS),
                "ratio": float('inf')
            })

    # Build result
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for ASLP (Tran et al., 2018)"
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument('--solution_path', type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument('--result_path', type=str, required=True,
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result['feasible']:
        print(f"FEASIBLE — no constraint violations detected.")
    else:
        print(f"INFEASIBLE — violated constraints: {result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
