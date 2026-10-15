"""
Feasibility checker for the Time-Dependent Travelling Salesman Problem (TDTSP).

Verifies a candidate solution against the mathematical formulation in
Cordeau, Ghiani, and Guerriero (2014), Transportation Science 48(1), pp. 46-58.

Constraints checked (original TDTSP core only):
  Constraint  2  (eq 14): In-degree = 1 for every vertex
  Constraint  3  (eq 15): Out-degree = 1 for every vertex
  Constraint  4  (eq 16): Subtour elimination (Hamiltonian tour)
  Constraint  5  (eq 17): Binary domain x_{ij} in {0,1}
  Constraint 15  (obj):   Objective consistency: reported objective_value must
                          equal the TDTSP tour duration recomputed from the
                          tour and instance data (full recompute, eq 12).

Skipped (valid inequalities / period-decomposition reformulation):
  Constraint  1  (eq 13): Path-based lower-bound linking (VI)
  Constraints 6-11 (eqs 20-25): Period-decomposition VIs on z_h, y_h
  Constraint 12  (eq 26): sum y_h = 1 (reformulation-only)
  Constraint 13  (eq 27): z_h >= 0, y_h in {0,1} (reformulation vars)
  Constraint 14  (eq 31): z >= z_bar(c*) (initial LB, not an original constraint)
These involve auxiliary variables z_h, y_h and path-based lower bounds that
a generated algorithm operating on the original (x_ij, z) space need not
produce; checking them would only verify the paper's cutting-plane bounds,
not the underlying TDTSP feasibility.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'tour')
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



# ============================================================================
# Tolerance constants
# ============================================================================
TOL = 1e-5
EPS = 1e-5


# ============================================================================
# Algorithm 1: Travel Time Computation (from the paper)
# ============================================================================

def compute_travel_time_on_arc(L_ij, speeds_for_periods, departure_time,
                               period_boundaries):
    """Compute travel time on arc (i,j) departing at departure_time."""
    if L_ij <= 1e-12:
        return 0.0

    H = len(speeds_for_periods)
    t = departure_time

    # Find starting period k such that T_k <= t <= T_{k+1}.
    k = H - 1
    for h in range(H):
        if period_boundaries[h] <= t <= period_boundaries[h + 1] + 1e-12:
            k = h
            break

    d = L_ij
    speed = speeds_for_periods[k]
    if speed <= 1e-15:
        return float('inf')

    t_prime = t + d / speed

    while t_prime > period_boundaries[k + 1] + 1e-12 and k < H - 1:
        d -= speed * (period_boundaries[k + 1] - t)
        t = period_boundaries[k + 1]
        k += 1
        speed = speeds_for_periods[k]
        if speed <= 1e-15:
            return float('inf')
        t_prime = t + d / speed

    return t_prime - departure_time


# ============================================================================
# Tour / Path Evaluation
# ============================================================================

def evaluate_tour_td(tour, distances, speeds_v_ijh, period_boundaries):
    """Evaluate TDTSP tour duration starting at time 0."""
    current_time = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        L_ij = distances[i][j]
        speeds = speeds_v_ijh[i][j]
        tt = compute_travel_time_on_arc(L_ij, speeds, current_time,
                                        period_boundaries)
        if tt == float('inf'):
            return float('inf')
        current_time += tt
    return current_time


def evaluate_tour_common_congestion(tour, distances, max_speeds, b_h,
                                     period_boundaries):
    """Evaluate tour under common congestion: v_{ijh} = b_h * u_{ij}."""
    current_time = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        L_ij = distances[i][j]
        u_ij = max_speeds[i][j]
        speeds = [bh * u_ij for bh in b_h]
        tt = compute_travel_time_on_arc(L_ij, speeds, current_time,
                                        period_boundaries)
        if tt == float('inf'):
            return float('inf')
        current_time += tt
    return current_time


def evaluate_path_td(path, distances, speeds_v_ijh, period_boundaries):
    """Evaluate TDTSP traversal time of a path starting at time 0."""
    current_time = 0.0
    for idx in range(len(path) - 1):
        a = path[idx]
        b = path[idx + 1]
        L_ab = distances[a][b]
        speeds = speeds_v_ijh[a][b]
        tt = compute_travel_time_on_arc(L_ab, speeds, current_time,
                                        period_boundaries)
        if tt == float('inf'):
            return float('inf')
        current_time += tt
    return current_time


def compute_atsp_cost_of_path(path, distances, max_speeds):
    """Compute z_bar_bar(p) = sum of L_{ab}/u_{ab} along the path."""
    cost = 0.0
    for idx in range(len(path) - 1):
        a = path[idx]
        b = path[idx + 1]
        if max_speeds[a][b] <= 1e-15:
            return float('inf')
        cost += distances[a][b] / max_speeds[a][b]
    return cost


def compute_lb_p(path, distances, max_speeds, speeds_v_ijh, b_h,
                 period_boundaries, z_bar_bar_c_star):
    """Compute LB_p (Algorithm 2 from the paper)."""
    z_p = evaluate_path_td(path, distances, speeds_v_ijh, period_boundaries)
    if z_p == float('inf'):
        return float('inf')

    z_bar_bar_p = compute_atsp_cost_of_path(path, distances, max_speeds)
    if z_bar_bar_p == float('inf'):
        return float('inf')

    if z_bar_bar_c_star < z_bar_bar_p:
        return z_p
    else:
        remaining_length = z_bar_bar_c_star - z_bar_bar_p
        if remaining_length <= 1e-12:
            return z_p
        tau = compute_travel_time_on_arc(
            remaining_length, b_h, z_p, period_boundaries)
        if tau == float('inf'):
            return float('inf')
        return z_p + tau


# ============================================================================
# Compute z_h and y_h from the tour under common congestion
# ============================================================================

def compute_zh_yh_from_tour(tour, distances, max_speeds, b_h,
                             period_boundaries):
    """
    Compute the z_h and y_h auxiliary variable values from a tour.

    z_h represents the portion of tour duration attributable to time period h
    under common congestion (v_{ijh} = b_h * u_{ij}).

    y_h = 1 iff the tour completion time under common congestion falls in
    period h, i.e., T_h <= z_bar(c) <= T_{h+1}.
    """
    H = len(b_h)

    # First compute the common-congestion tour time z_bar(c) and track
    # how much time is spent in each period.
    z_h_vals = [0.0] * H
    current_time = 0.0

    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        L_ij = distances[i][j]
        u_ij = max_speeds[i][j]

        if L_ij <= 1e-12:
            continue

        speeds = [bh * u_ij for bh in b_h]

        # Simulate Algorithm 1 tracking time per period
        t = current_time
        k = H - 1
        for h in range(H):
            if period_boundaries[h] <= t <= period_boundaries[h + 1] + 1e-12:
                k = h
                break

        d = L_ij
        speed = speeds[k]
        if speed <= 1e-15:
            break

        t_prime = t + d / speed

        while t_prime > period_boundaries[k + 1] + 1e-12 and k < H - 1:
            time_in_period = period_boundaries[k + 1] - t
            z_h_vals[k] += speed * time_in_period / u_ij if u_ij > 1e-15 else 0.0
            d -= speed * time_in_period
            t = period_boundaries[k + 1]
            k += 1
            speed = speeds[k]
            if speed <= 1e-15:
                break
            t_prime = t + d / speed

        if speed > 1e-15:
            time_in_final = t_prime - t
            z_h_vals[k] += speed * time_in_final / u_ij if u_ij > 1e-15 else 0.0

        current_time_new = current_time + compute_travel_time_on_arc(
            L_ij, speeds, current_time, period_boundaries)
        current_time = current_time_new

    # z_bar(c) = common congestion tour time
    z_bar_c = current_time

    # Determine y_h: which period does z_bar(c) fall in?
    y_h_vals = [0] * H
    for h in range(H):
        if period_boundaries[h] - TOL <= z_bar_c <= period_boundaries[h + 1] + TOL:
            y_h_vals[h] = 1
            break
    else:
        # If beyond all periods, assign to last
        if z_bar_c > period_boundaries[H]:
            y_h_vals[H - 1] = 1

    # z_h values: under common congestion, the traversal time per period h
    # is sum of L_ij/u_ij for time spent in period h, scaled by b_h.
    # Actually, z_h represents the sum of (L_ij/u_ij) contributions that fall
    # within period h. Let's recompute properly.
    #
    # The z_h auxiliary variables satisfy:
    #   z_bar(c) = sum_h z_h / b_h  (from constraint 20)
    # where z_h = b_h * (time spent in period h under common congestion)
    # equivalently z_h / b_h = time in period h, so z_h = b_h * (time in period h).
    #
    # We track the actual time spent in each period under common congestion.
    z_h_time = [0.0] * H
    current_time = 0.0

    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        L_ij = distances[i][j]
        u_ij = max_speeds[i][j]

        if L_ij <= 1e-12:
            continue

        speeds = [bh * u_ij for bh in b_h]

        t = current_time
        k = H - 1
        for h in range(H):
            if period_boundaries[h] <= t <= period_boundaries[h + 1] + 1e-12:
                k = h
                break

        d = L_ij
        speed = speeds[k]
        if speed <= 1e-15:
            break

        t_prime = t + d / speed

        while t_prime > period_boundaries[k + 1] + 1e-12 and k < H - 1:
            time_in_period = period_boundaries[k + 1] - t
            z_h_time[k] += time_in_period
            d -= speed * time_in_period
            t = period_boundaries[k + 1]
            k += 1
            speed = speeds[k]
            if speed <= 1e-15:
                break
            t_prime = t + d / speed

        if speed > 1e-15:
            time_in_final = t_prime - t
            z_h_time[k] += time_in_final

        current_time += compute_travel_time_on_arc(
            L_ij, speeds, current_time, period_boundaries)

    # z_h = b_h * (time spent in period h)
    z_h_final = [b_h[h] * z_h_time[h] for h in range(H)]

    return z_h_final, y_h_vals, z_bar_c


# ============================================================================
# ATSP helpers
# ============================================================================

def compute_atsp_tour_cost(tour, distances, max_speeds):
    """Compute z_bar_bar(c) = sum L_ij/u_ij along the tour."""
    cost = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        if max_speeds[i][j] <= 1e-15:
            return float('inf')
        cost += distances[i][j] / max_speeds[i][j]
    return cost


# ============================================================================
# Violation recording helper
# ============================================================================

def record_violation(constraint_idx, lhs, rhs, op, violations_list,
                     violated_set, messages_list, message):
    """Check if a constraint is violated and record it."""
    if op == '>=':
        violation_amount = rhs - lhs
    elif op == '<=':
        violation_amount = lhs - rhs
    elif op == '=':
        violation_amount = abs(lhs - rhs)
    else:
        violation_amount = 0.0

    if violation_amount > TOL:
        normalizer = max(abs(rhs), EPS)
        ratio = violation_amount / normalizer
        violations_list.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio,
        })
        violated_set.add(constraint_idx)
        messages_list.append(message)
        return True
    return False


# ============================================================================
# Main feasibility check
# ============================================================================

def check_feasibility(instance_data, solution_data):
    """
    Check feasibility of a candidate TDTSP solution.

    Returns a dict with feasibility results.
    """
    violations_list = []     # list of violation magnitude dicts
    violated_set = set()     # set of violated constraint indices
    messages_list = []       # human-readable violation messages

    # Parse instance
    params = instance_data["parameters"]
    n = params["total_vertices"]
    H = params["H"]
    b_h = params["b_h"]
    period_boundaries = params["period_boundaries"]
    distances = instance_data["arcs"]["distances"]
    max_speeds = instance_data["arcs"]["max_speeds_u_ij"]
    speeds_v_ijh = instance_data["arcs"]["speeds_v_ijh"]

    # Parse solution
    tour = solution_data["tour"]
    z_reported = solution_data["objective_value"]

    if z_reported is None or not tour:
        return {
            "feasible": False,
            "violated_constraints": [2, 3, 4],
            "violations": ["Solution is empty or has no objective value"],
            "violation_magnitudes": [],
        }

    # Derive x_{ij} from tour
    x = {}
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        x[i, j] = 1

    # -----------------------------------------------------------------------
    # Constraint 2 (eq 14): In-degree = 1 for each vertex in V u {0}
    # sum_{i in V\{j}} x_{ij} = 1 for all j in V u {0}
    # -----------------------------------------------------------------------
    in_degree = [0] * n
    for (i, j) in x:
        if 0 <= j < n:
            in_degree[j] += 1

    for j in range(n):
        lhs = in_degree[j]
        rhs = 1.0
        record_violation(2, float(lhs), rhs, '=', violations_list,
                         violated_set, messages_list,
                         f"In-degree of vertex {j} is {lhs}, expected 1")

    # -----------------------------------------------------------------------
    # Constraint 3 (eq 15): Out-degree = 1 for each vertex in V u {0}
    # sum_{j in V\{i}} x_{ij} = 1 for all i in V u {0}
    # -----------------------------------------------------------------------
    out_degree = [0] * n
    for (i, j) in x:
        if 0 <= i < n:
            out_degree[i] += 1

    for i in range(n):
        lhs = out_degree[i]
        rhs = 1.0
        record_violation(3, float(lhs), rhs, '=', violations_list,
                         violated_set, messages_list,
                         f"Out-degree of vertex {i} is {lhs}, expected 1")

    # -----------------------------------------------------------------------
    # Constraint 4 (eq 16): Subtour elimination — tour must be Hamiltonian
    # The tour must visit all n vertices exactly once and form a single cycle
    # starting and ending at depot 0.
    # -----------------------------------------------------------------------
    # Check tour starts and ends at depot
    tour_valid = True
    if tour[0] != 0 or tour[-1] != 0:
        record_violation(4, 0.0, 1.0, '>=', violations_list,
                         violated_set, messages_list,
                         f"Tour does not start and end at depot 0: "
                         f"starts at {tour[0]}, ends at {tour[-1]}")
        tour_valid = False

    # Check all vertices visited
    visited = set(tour[:-1])  # exclude final depot duplicate
    if len(visited) != n:
        missing = set(range(n)) - visited
        extra = visited - set(range(n))
        # Compute a violation amount: how many vertices are missing
        lhs = float(len(visited))
        rhs = float(n)
        record_violation(4, lhs, rhs, '=', violations_list,
                         violated_set, messages_list,
                         f"Tour visits {len(visited)} vertices but should visit {n}. "
                         f"Missing: {sorted(missing)}, Extra: {sorted(extra)}")
        tour_valid = False

    # Check for repeated vertices (subtour)
    interior = tour[1:-1]
    if len(interior) != len(set(interior)):
        from collections import Counter
        counts = Counter(interior)
        duplicates = {v: c for v, c in counts.items() if c > 1}
        record_violation(4, 0.0, 1.0, '>=', violations_list,
                         violated_set, messages_list,
                         f"Tour has repeated vertices (subtour): {duplicates}")
        tour_valid = False

    # -----------------------------------------------------------------------
    # Constraint 5 (eq 17): Binary domain x_{ij} in {0,1}
    # Since x is derived from tour arcs (always 0 or 1), this is
    # automatically satisfied. We verify arc indices are valid.
    # -----------------------------------------------------------------------
    for (i, j) in x:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            record_violation(5, 0.0, 1.0, '>=', violations_list,
                             violated_set, messages_list,
                             f"Invalid arc ({i},{j}) in tour: vertex index "
                             f"out of range [0,{n-1}] or self-loop")

    # If the tour is not structurally valid, skip objective-related checks
    if not tour_valid:
        return _build_result(violated_set, messages_list, violations_list)

    # -----------------------------------------------------------------------
    # Compute derived quantities needed for remaining constraints
    # -----------------------------------------------------------------------

    # Actual TDTSP tour duration
    z_actual = evaluate_tour_td(tour, distances, speeds_v_ijh,
                                period_boundaries)

    # -----------------------------------------------------------------------
    # Constraint 15 (obj, eq 12): Objective consistency.
    # The TDTSP objective is z = total tour duration. Since the tour is fully
    # in the solution and travel times are deterministic given the speed law,
    # we can recompute z exactly via evaluate_tour_td and reject solutions
    # whose reported objective_value disagrees beyond tolerance.
    # -----------------------------------------------------------------------
    try:
        reported = float(z_reported)
    except (TypeError, ValueError):
        reported = None
    if reported is not None and math.isfinite(z_actual):
        abs_diff = abs(reported - z_actual)
        tol = max(1e-3, 1e-3 * abs(z_actual))  # 0.1% relative, 1e-3 absolute floor
        if abs_diff > tol:
            record_violation(15, reported, z_actual, '=', violations_list,
                             violated_set, messages_list,
                             f"Objective consistency violated: reported "
                             f"objective_value={reported} differs from "
                             f"recomputed tour duration={z_actual} "
                             f"(|diff|={abs_diff:.3g}, tol={tol:.3g})")

    # Constraints 1, 6-14 SKIPPED: path-based LB (VI) and period-decomposition
    # reformulation constraints on z_h, y_h, plus initial z_bar(c*) LB (eq 31).
    # A generated algorithm operating on the original (x_ij, z) space does not
    # produce z_h, y_h and need not satisfy the paper's cutting-plane bounds —
    # they only tighten the LP relaxation, not TDTSP integer feasibility.

    return _build_result(violated_set, messages_list, violations_list)


def _build_result(violated_set, messages_list, violations_list):
    """Build the output result dictionary."""
    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": messages_list,
        "violation_magnitudes": violations_list,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for TDTSP solutions "
                    "(Cordeau, Ghiani, Guerriero 2014).")
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance.")
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution.")
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)

    with open(args.solution_path, "r") as f:
        solution_data = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(solution_data)
        if _frontieror_contract_result is not None:
            with open(args.result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return

    result = check_feasibility(instance_data, solution_data)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
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
