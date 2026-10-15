"""LiSA JSPTWT conversion, limited to integer complete job shops.

Semantics: LiSA handbook, sections 2.3.2 and appendix A.
https://www.math.ovgu.de/math_media/Lisa/Handbook_en-p-6090.pdf
MO[j,m] is the visit rank of machine m, not a machine ID.
LR ranks order operations along both job rows and machine columns.
"""

import re
import xml.etree.ElementTree as ET
from decimal import Decimal


def integers(element, count):
    if element is None:
        raise ValueError("Missing LiSA value")
    text = element.text or ""
    tokens = re.sub(r"[{}]", " ", text).split()
    values = []
    for token in tokens:
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
            raise ValueError("Only integer LiSA values are supported")
        number = Decimal(token)
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("Only integer LiSA values are supported")
        values.append(int(number))
    if len(values) != count:
        raise ValueError("LiSA array size mismatch")
    return values


def matrix(element, n, m):
    values = integers(element, n * m)
    return [values[j * m : (j + 1) * m] for j in range(n)]


def read_instance(root):
    problem = root.find("problem")
    if problem is None or problem.find("alpha").get("env") != "J":
        raise ValueError("Expected job shop")
    if problem.find("gamma").get("objective") != "Sum_wiTi":
        raise ValueError("Expected total weighted tardiness")
    beta = problem.find("beta")
    if beta is not None and any(
        key != "release_times" for key in beta.attrib
    ):
        raise ValueError("Unsupported extra problem constraints")
    values = root.find("values")
    n, m = int(values.get("n")), int(values.get("m"))
    if min(n, m) < 1:
        raise ValueError("Empty instance")
    pt = matrix(values.find("processing_times"), n, m)
    mo = matrix(values.find("machine_order"), n, m)
    ops = integers(values.find("operation_set"), n * m)
    if any(x != 1 for x in ops) or any(x <= 0 for row in pt for x in row):
        raise ValueError("Expected complete shop with positive durations")
    r = integers(values.find("release_times"), n)
    d = integers(values.find("due_dates"), n)
    w = integers(values.find("weights"), n)
    if min(r) < 0 or min(d) < 0 or min(w) <= 0:
        raise ValueError("Invalid release, due date or weight")
    jobs = []
    for j in range(n):
        if sorted(mo[j]) != list(range(1, m + 1)):
            raise ValueError("Machine order is not a rank permutation")
        route = sorted(range(m), key=lambda k: mo[j][k])
        jobs.append(dict(job_id=j, release_date=r[j], due_date=d[j], weight=w[j],
                         operations=[dict(machine=k, processing_time=pt[j][k]) for k in route]))
    return dict(num_jobs=n, num_machines=m, num_operations=n * m, jobs=jobs)


def write_instance(instance, controls=None):
    n, m = instance["num_jobs"], instance["num_machines"]
    jobs = instance["jobs"]
    if [j["job_id"] for j in jobs] != list(range(n)):
        raise ValueError("Contiguous ordered job IDs required")
    root = ET.Element("instance")
    problem = ET.SubElement(root, "problem")
    ET.SubElement(problem, "alpha", env="J")
    ET.SubElement(problem, "beta", release_times="yes")
    ET.SubElement(problem, "gamma", objective="Sum_wiTi")
    values = ET.SubElement(root, "values", n=str(n), m=str(m))
    pt, mo = [], []
    for job in jobs:
        operations = job["operations"]
        if sorted(op["machine"] for op in operations) != list(range(m)):
            raise ValueError("Each job must visit each machine exactly once")
        times, ranks = [0] * m, [0] * m
        for rank, op in enumerate(operations, 1):
            times[op["machine"]] = op["processing_time"]
            ranks[op["machine"]] = rank
        pt.append(times)
        mo.append(ranks)
    for tag, rows in [("processing_times", pt), ("operation_set", [[1] * m for _ in jobs]), ("machine_order", mo)]:
        ET.SubElement(values, tag, model="lisa_native").text = "{\n" + "\n".join("{ " + " ".join(map(str, row)) + " }" for row in rows) + "\n}"
    for tag, key in [("release_times", "release_date"), ("due_dates", "due_date"), ("weights", "weight")]:
        ET.SubElement(values, tag, model="lisa_native").text = "{ " + " ".join(str(j[key]) for j in jobs) + " }"
    if controls is not None:
        root.append(ET.fromstring(ET.tostring(controls)))
    # A semantic round trip catches unsupported or lossy inputs before execution.
    canonical = {k: instance[k] for k in ("num_jobs", "num_machines", "jobs")}
    roundtrip = read_instance(root)
    if any(roundtrip[k] != v for k, v in canonical.items()):
        raise ValueError("Lossy LiSA serialization")
    return '<?xml version="1.0"?>\n<!DOCTYPE instance SYSTEM "LiSA.dtd">\n' + ET.tostring(root, encoding="unicode")


def read_solution(root, instance):
    decoded = read_instance(root)
    for key in ("num_jobs", "num_machines", "jobs"):
        if decoded[key] != instance[key]:
            raise ValueError("Output changed the input problem")
    n, m = instance["num_jobs"], instance["num_machines"]
    schedules = root.findall("schedule")
    if len(schedules) != 1:
        raise ValueError("Expected exactly one schedule")
    schedule = schedules[0]
    if int(schedule.get("n")) != n or int(schedule.get("m")) != m:
        raise ValueError("Schedule dimensions changed")
    lr = matrix(schedule.find("plan"), n, m)
    for row in lr:
        if min(row) <= 0 or len(set(row)) != m:
            raise ValueError("Invalid job ranks")
    for k in range(m):
        if len({lr[j][k] for j in range(n)}) != n:
            raise ValueError("Duplicate machine ranks")
    duration, before = {}, {}
    for job in instance["jobs"]:
        j = job["job_id"]
        route = [op["machine"] for op in job["operations"]]
        if route != sorted(range(m), key=lambda k: lr[j][k]):
            raise ValueError("Output ranks violate technological sequence")
        for pos, op in enumerate(job["operations"]):
            node = (j, op["machine"])
            duration[node] = op["processing_time"]
            before[node] = [] if pos == 0 else [(j, route[pos - 1])]
    for k in range(m):
        order = sorted(range(n), key=lambda j: lr[j][k])
        for first, second in zip(order, order[1:]):
            before[second, k].append((first, k))
    completion_element = schedule.find("completion_times")
    explicit = matrix(completion_element, n, m) if completion_element is not None else None
    starts, ends = {}, {}
    for node in sorted(duration, key=lambda node: lr[node[0]][node[1]]):
        j, k = node
        earliest = max([instance["jobs"][j]["release_date"]] + [ends[p] for p in before[node]])
        start = explicit[j][k] - duration[node] if explicit is not None else earliest
        if start < earliest:
            raise ValueError("Explicit time violates ordering or release")
        starts[node], ends[node] = start, start + duration[node]
    result, objective = [], 0
    for job in instance["jobs"]:
        j = job["job_id"]
        completion = ends[j, job["operations"][-1]["machine"]]
        tardiness = max(0, completion - job["due_date"])
        objective += job["weight"] * tardiness
        result.append(dict(job_id=j, completion_time=completion, tardiness=tardiness,
                           operations=[dict(machine=op["machine"], start_time=starts[j, op["machine"]]) for op in job["operations"]]))
    return dict(objective_value=objective, schedule=result,
                reconstruction="explicit completion times" if explicit is not None else "earliest schedule from LiSA LR ranks")
