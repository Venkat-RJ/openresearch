"""
Feasibility checker for the Multiple Depot Vehicle Scheduling Problem (MDVSP).

Based on: Hadjar, Marcotte, and Soumis (2006),
"A Branch-and-Cut Algorithm for the Multiple Depot Vehicle Scheduling Problem",
Operations Research 54(1), pp. 130-149.

Checks the hard constraints from the multicommodity flow formulation (P):
  Constraint 1 (eq 2): Assignment — each trip assigned to exactly one vehicle
  Constraint 2 (eq 3): Depot capacity — vehicles from depot k <= v_k
  Constraint 3 (eq 4): Flow conservation — valid arcs between consecutive
                        tasks and between depots and tasks in each route
  Constraint 4 (eq 1): Objective consistency — the reported objective_value
                        must equal the cost recomputed from the route arcs
                        (sum of depot-to-task, task-to-task, and
                        task-to-depot arc costs). This is a Tier-C defense
                        against fabricated objective values.

NOTE: this is the obj-recompute-augmented variant of feasibility_check.py.
The original file is kept untouched; constraints 0-3 are byte-identical in
logic to the original.
"""

import json
import argparse
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'routes')
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



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    # ---------------------------------------------------------------
    # Extract instance data
    # ---------------------------------------------------------------
    num_tasks = instance['num_tasks']
    depots = {d['id']: d for d in instance['depots']}
    tasks = {t['id']: t for t in instance['tasks']}
    all_task_ids = set(tasks.keys())

    # Build compatible pairs lookup
    compatible_pairs = set()
    compatible_travel_time = {}
    for cp in instance['compatible_pairs']:
        ti, tj = cp['task_i'], cp['task_j']
        compatible_pairs.add((ti, tj))
        compatible_travel_time[(ti, tj)] = cp['travel_time']

    # Build depot-to-task arc lookup
    depot_to_task_arcs = set()
    for dt in instance['depot_to_task_costs']:
        depot_to_task_arcs.add((dt['depot_id'], dt['task_id']))

    # Build task-to-depot arc lookup
    task_to_depot_arcs = set()
    for td in instance['task_to_depot_costs']:
        task_to_depot_arcs.add((td['task_id'], td['depot_id']))

    routes = solution['routes']

    # ---------------------------------------------------------------
    # Domain check: depot id in each route must be a valid depot;
    # every task id appearing in a route's task list must be a valid
    # task id in the instance.
    # ---------------------------------------------------------------
    for r_idx, route in enumerate(routes):
        depot_id = route.get('depot')
        if depot_id not in depots:
            violated_constraints_set.add(0)
            violations.append(
                f"Route {r_idx} depot {depot_id!r} not in instance depots"
            )
            violation_magnitudes.append({
                "constraint": 0,
                "lhs": float(depot_id) if isinstance(depot_id, (int, float)) else 0.0,
                "rhs": 0.0, "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })
        for tid in route.get('tasks', []):
            if tid not in all_task_ids:
                violated_constraints_set.add(0)
                violations.append(
                    f"Route {r_idx} contains invalid task id {tid!r}"
                )
                violation_magnitudes.append({
                    "constraint": 0,
                    "lhs": float(tid) if isinstance(tid, (int, float)) else 0.0,
                    "rhs": 0.0, "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
                })

    # ---------------------------------------------------------------
    # Constraint 1 (eq 2): Assignment constraints
    # Each trip must be assigned to exactly one vehicle (route).
    # sum_{k in K} sum_{j in delta+(i)} X^k_{ij} = 1, for i = 1..n
    # ---------------------------------------------------------------
    task_assignment_count = defaultdict(int)
    for route in routes:
        for task_id in route['tasks']:
            task_assignment_count[task_id] += 1

    # Check for tasks assigned more than once
    for task_id in sorted(task_assignment_count.keys()):
        count = task_assignment_count[task_id]
        if count != 1:
            lhs = float(count)
            rhs = 1.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints_set.add(1)
                violations.append(
                    f"Task {task_id} is assigned to {count} routes (expected exactly 1)"
                )
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio
                })

    # Check for unassigned tasks
    unassigned = all_task_ids - set(task_assignment_count.keys())
    for task_id in sorted(unassigned):
        lhs = 0.0
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(1)
            violations.append(
                f"Task {task_id} is not assigned to any route (expected exactly 1)"
            )
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # ---------------------------------------------------------------
    # Constraint 2 (eq 3): Depot capacity constraints
    # sum_{j=1}^{n} X^k_{n+k, j} <= v_k, for k in K
    # i.e., number of routes starting from depot k <= capacity of depot k
    # ---------------------------------------------------------------
    depot_route_count = defaultdict(int)
    for route in routes:
        depot_route_count[route['depot']] += 1

    for depot_id in sorted(depots.keys()):
        v_k = depots[depot_id]['capacity']
        count = depot_route_count.get(depot_id, 0)
        lhs = float(count)
        rhs = float(v_k)
        violation_amount = max(lhs - rhs, 0.0)  # <= constraint
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(2)
            violations.append(
                f"Depot {depot_id} uses {count} vehicles but capacity is {v_k}"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # ---------------------------------------------------------------
    # Constraint 3 (eq 4): Flow conservation constraints
    # For each colour k and each node i (trips and depot):
    #   sum_{j in delta-(i)} X^k_{ji} - sum_{j in delta+(i)} X^k_{ij} = 0
    #
    # In the route representation, this means:
    #   (a) Each route must use valid arcs: depot->task, task->task, task->depot
    #   (b) Consecutive tasks must be compatible (arc must exist in the graph)
    #   (c) The depot pull-out and pull-in arcs must exist
    #
    # If an arc doesn't exist, the corresponding X variable cannot be 1,
    # which would break flow conservation (inflow != outflow at the node).
    # ---------------------------------------------------------------
    for r_idx, route in enumerate(routes):
        depot_id = route['depot']
        task_list = route['tasks']

        if len(task_list) == 0:
            continue

        # Check depot -> first task arc exists
        first_task = task_list[0]
        if (depot_id, first_task) not in depot_to_task_arcs:
            # Flow conservation violated at depot node and first task node
            # LHS = inflow - outflow. The arc (depot, first_task) with colour k
            # doesn't exist, so this X variable cannot be set to 1.
            # We report this as a flow imbalance of magnitude 1 at the first task.
            lhs = 1.0  # flow that would need to exist but doesn't
            rhs = 0.0  # flow conservation requires balance = 0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints_set.add(3)
                violations.append(
                    f"Route {r_idx + 1}: No arc from depot {depot_id} to "
                    f"first task {first_task}"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio
                })

        # Check consecutive task-to-task arcs
        for i in range(len(task_list) - 1):
            ti = task_list[i]
            tj = task_list[i + 1]
            if (ti, tj) not in compatible_pairs:
                # The arc (ti, tj) doesn't exist — flow conservation violated
                # at node tj (no valid incoming arc of this colour)
                # Check compatibility: b_i + t_{ij} <= a_j
                task_i = tasks.get(ti)
                task_j = tasks.get(tj)
                if task_i and task_j:
                    msg = (
                        f"Route {r_idx + 1} (depot {depot_id}): "
                        f"No compatible arc from task {ti} "
                        f"(end_time={task_i['end_time']}) to task {tj} "
                        f"(start_time={task_j['start_time']})"
                    )
                else:
                    msg = (
                        f"Route {r_idx + 1} (depot {depot_id}): "
                        f"No compatible arc from task {ti} to task {tj}"
                    )
                lhs = 1.0
                rhs = 0.0
                violation_amount = abs(lhs - rhs)
                if violation_amount > tol:
                    normalizer = max(abs(rhs), eps)
                    ratio = violation_amount / normalizer
                    violated_constraints_set.add(3)
                    violations.append(msg)
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": violation_amount,
                        "normalizer": normalizer,
                        "ratio": ratio
                    })

        # Check last task -> depot arc exists
        last_task = task_list[-1]
        if (last_task, depot_id) not in task_to_depot_arcs:
            lhs = 1.0
            rhs = 0.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                ratio = violation_amount / normalizer
                violated_constraints_set.add(3)
                violations.append(
                    f"Route {r_idx + 1}: No arc from last task {last_task} "
                    f"to depot {depot_id}"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": ratio
                })

    # ---------------------------------------------------------------
    # Constraint 4 (eq 1): Objective consistency  [Tier-C obj recompute]
    #
    #   min sum_{(i,j,k) in A} c_{ij} X^k_{ij}
    #
    # Every variable the objective depends on (the arcs traversed by each
    # route) is fully present in the solution, so the objective can be
    # recomputed EXACTLY by summing the per-arc costs:
    #   - depot -> first task : depot_to_task_costs[(depot, t1)]['cost']
    #                           (already includes the depot dispatch penalty)
    #   - task_i -> task_{i+1}: compatible_pairs[(ti, tj)]['cost']
    #   - last task -> depot  : task_to_depot_costs[(tm, depot)]['cost']
    #
    # If any traversed arc is missing from the instance lookups, the route
    # is itself arc-infeasible (already flagged by constraint 3) and the
    # objective cannot be recomputed reliably; in that case we SKIP the
    # obj check so this checker never adds constraint 4 spuriously.
    # ---------------------------------------------------------------
    compatible_cost = {}
    for cp in instance['compatible_pairs']:
        compatible_cost[(cp['task_i'], cp['task_j'])] = cp['cost']
    depot_to_task_cost = {}
    for dt in instance['depot_to_task_costs']:
        depot_to_task_cost[(dt['depot_id'], dt['task_id'])] = dt['cost']
    task_to_depot_cost = {}
    for td in instance['task_to_depot_costs']:
        task_to_depot_cost[(td['task_id'], td['depot_id'])] = td['cost']

    reported_obj = solution.get('objective_value')
    try:
        reported = float(reported_obj)
    except (TypeError, ValueError):
        reported = None

    obj_recomputable = True
    true_obj = 0.0
    for route in routes:
        depot_id = route.get('depot')
        task_list = route.get('tasks', [])
        if len(task_list) == 0:
            continue
        arc = depot_to_task_cost.get((depot_id, task_list[0]))
        if arc is None:
            obj_recomputable = False
            break
        true_obj += arc
        for i in range(len(task_list) - 1):
            arc = compatible_cost.get((task_list[i], task_list[i + 1]))
            if arc is None:
                obj_recomputable = False
                break
            true_obj += arc
        if not obj_recomputable:
            break
        arc = task_to_depot_cost.get((task_list[-1], depot_id))
        if arc is None:
            obj_recomputable = False
            break
        true_obj += arc

    if obj_recomputable and reported is not None:
        abs_diff = abs(reported - true_obj)
        # 0.1% relative tolerance with a 1e-3 absolute floor.
        obj_tol = max(1e-3, 1e-3 * abs(true_obj))
        if abs_diff > obj_tol:
            normalizer = max(abs(true_obj), eps)
            ratio = abs_diff / normalizer
            violated_constraints_set.add(4)
            violations.append(
                f"Objective consistency violated: reported objective_value="
                f"{reported} differs from recomputed total arc cost="
                f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(reported),
                "rhs": float(true_obj),
                "raw_excess": abs_diff,
                "normalizer": normalizer,
                "ratio": ratio
            })

    # ---------------------------------------------------------------
    # Build result
    # ---------------------------------------------------------------
    feasible = len(violated_constraints_set) == 0
    violated_constraints = sorted(violated_constraints_set)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MDVSP (Hadjar et al. 2006)"
    )
    parser.add_argument(
        '--instance_path', required=True,
        help='Path to the JSON file containing the data instance'
    )
    parser.add_argument(
        '--solution_path', required=True,
        help='Path to the JSON file containing the candidate solution'
    )
    parser.add_argument(
        '--result_path', required=True,
        help='Path to write the JSON file containing the feasibility result'
    )
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
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} constraint type(s) violated.")
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
