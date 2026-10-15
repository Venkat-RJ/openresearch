"""Shared executable contract for the released Potthoff et al. instances."""

from collections import defaultdict


RELEASED_VARIANT = "relief_point_completion_v1"


def parse_time(value):
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


class ReleasedInstance:
    def __init__(self, data):
        if data.get("released_problem_variant") != RELEASED_VARIANT:
            raise ValueError(
                "instance must declare released_problem_variant="
                + RELEASED_VARIANT
            )
        self.data = data
        self.params = data["parameters"]
        self.rescheduling_time = data["disruption"]["rescheduling_time_minutes"]
        disruption_canceled = set(data["disruption"].get("canceled_tasks", []))
        self.tasks = {
            task["task_id"]: task
            for task in data["tasks"]
            if task["departure_time_minutes"] >= self.rescheduling_time
            and task["task_id"] not in disruption_canceled
        }
        self.duties = {
            duty["duty_id"]: duty
            for duty in data.get("active_duties", []) + data.get("reserve_duties", [])
        }
        self.stations = {station["id"] for station in data["network"]["stations"]}
        self.taxis = {}
        for connection in data["network"].get("taxi_connections", []):
            pair = (connection["from_station"], connection["to_station"])
            self.taxis[pair] = connection["travel_time_minutes"]
            if connection.get("bidirectional", False):
                self.taxis[(pair[1], pair[0])] = connection["travel_time_minutes"]
        self.tasks_by_departure = defaultdict(list)
        for task in self.tasks.values():
            self.tasks_by_departure[task["departure_station"]].append(task)

    def duty_state(self, duty):
        if duty["duty_type"] == "reserve":
            return (
                duty.get("standby_station", duty["crew_base"]),
                max(self.rescheduling_time, duty["start_time_minutes"]),
            )
        station = duty["crew_base"]
        available = duty["start_time_minutes"]
        for task in duty.get("task_details", []):
            departure = parse_time(task["departure_time"])
            if departure >= self.rescheduling_time:
                break
            station = task["arrival_station"]
            available = max(available, parse_time(task["arrival_time"]))
        return station, available

    @staticmethod
    def can_drive(duty, task):
        route = set(duty.get("route_knowledge", []))
        licenses = set(duty.get("rolling_stock_licenses", []))
        stock = task.get("rolling_stock_type", "")
        return (
            task["departure_station"] in route
            and task["arrival_station"] in route
            and (not stock or stock in licenses)
        )


def validate_completion(inst, duty, covered_tasks, declared_taxi, declared_release):
    """Validate one submitted ordered completion and derive its return mode."""
    if not isinstance(covered_tasks, list) or any(
        not isinstance(task_id, str) for task_id in covered_tasks
    ):
        return None, "covered_tasks must be a list of task identifiers"
    if len(covered_tasks) != len(set(covered_tasks)):
        return None, "covered_tasks contains duplicate task identifiers"
    if not isinstance(declared_taxi, bool) or not isinstance(declared_release, bool):
        return None, "uses_taxi and release_at_relief_point must be booleans"

    station, clock = inst.duty_state(duty)
    max_end = duty["end_time_minutes"] + inst.params["max_overtime_minutes"]
    min_connection = inst.params["min_connection_time_minutes"]
    taxi_used = False

    for position, task_id in enumerate(covered_tasks):
        task = inst.tasks.get(task_id)
        if task is None:
            return None, f"task {task_id} is not in the reschedulable set N"
        if not inst.can_drive(duty, task):
            return None, f"duty is not licensed to operate task {task_id}"
        required_gap = 0 if position == 0 else min_connection
        if task["departure_station"] == station:
            if task["departure_time_minutes"] < clock + required_gap:
                return None, f"task {task_id} starts before the duty can reach it"
        else:
            travel = inst.taxis.get((station, task["departure_station"]))
            if travel is None or clock + travel > task["departure_time_minutes"]:
                return None, f"no timely connection to task {task_id}"
            taxi_used = True
        if task["arrival_time_minutes"] > max_end:
            return None, f"task {task_id} ends after the duty horizon"
        station = task["arrival_station"]
        clock = task["arrival_time_minutes"]

    base = duty["crew_base"]
    release_used = False
    if station != base:
        travel = inst.taxis.get((station, base))
        if travel is not None and clock + travel <= max_end:
            taxi_used = True
            clock += travel
            station = base
        elif declared_release and station in inst.stations:
            # Several generated instances contain duties for which no path to
            # the original base exists inside the frozen horizon.  The public
            # release makes the generator's historical fallback explicit:
            # the driver may be released at the current relief point.
            release_used = True
        else:
            return None, "completion cannot return to base inside the duty horizon"

    if taxi_used != declared_taxi:
        return None, f"uses_taxi must be {taxi_used} for the submitted route"
    if release_used != declared_release:
        return None, (
            "release_at_relief_point must be " + str(release_used)
            + " for the submitted route"
        )
    return {
        "uses_taxi": taxi_used,
        "release_at_relief_point": release_used,
        "return_station": station,
        "return_time_minutes": clock,
    }, None


def completion_cost(inst, duty, covered_tasks, uses_taxi, release_at_relief_point):
    original = [task for task in duty.get("tasks", []) if task in inst.tasks]
    if covered_tasks == original and not uses_taxi and not release_at_relief_point:
        return 0
    if not covered_tasks and not original and not uses_taxi and not release_at_relief_point:
        return 0

    cost = int(inst.params["cost_duty_changed"])
    original_set = set(original)
    cost += int(inst.params["cost_task_reassigned"]) * sum(
        task not in original_set for task in covered_tasks
    )
    original_pairs = set(zip(original, original[1:]))
    cost += int(inst.params["cost_new_transfer"]) * sum(
        pair not in original_pairs for pair in zip(covered_tasks, covered_tasks[1:])
    )
    if uses_taxi:
        cost += int(inst.params["cost_taxi"])
    return cost


def make_completion(inst, duty, covered_tasks):
    """Construct the canonical taxi/release flags for a proposed route."""
    for uses_taxi in (False, True):
        for released in (False, True):
            derived, error = validate_completion(
                inst, duty, list(covered_tasks), uses_taxi, released
            )
            if error is None:
                return {
                    "covered_tasks": list(covered_tasks),
                    "uses_taxi": derived["uses_taxi"],
                    "release_at_relief_point": derived["release_at_relief_point"],
                    "return_station": derived["return_station"],
                    "return_time_minutes": derived["return_time_minutes"],
                    "cost": completion_cost(
                        inst,
                        duty,
                        list(covered_tasks),
                        derived["uses_taxi"],
                        derived["release_at_relief_point"],
                    ),
                }
    return None


def completion_options(inst, duty):
    """Build a finite, deterministic released completion library."""
    routes = [[]]
    original = [task for task in duty.get("tasks", []) if task in inst.tasks]
    if original:
        routes.append(original)
    # Single-task completions keep the library polynomial and still let the
    # set-covering model rescue many high-penalty tasks.
    routes.extend([[task_id] for task_id in sorted(inst.tasks)])

    options = []
    seen = set()
    for route in routes:
        completion = make_completion(inst, duty, route)
        if completion is None:
            continue
        key = (
            tuple(completion["covered_tasks"]),
            completion["uses_taxi"],
            completion["release_at_relief_point"],
        )
        if key in seen:
            continue
        seen.add(key)
        options.append(completion)
    if not options:
        raise ValueError(f"no released completion exists for duty {duty['duty_id']}")
    return options
