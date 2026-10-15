"""Failure-aware instance-level summaries for the proposed comparison.

Definitions implement the locally reviewed protocol. No missing objective is
replaced by a fictional penalty. This module does not authorize final testing.
"""

import random
import statistics


def compare(a, b):
    """+1 favors candidate a; None denotes a documented solver failure."""
    if a is None and b is None:
        return 0
    if a is None:
        return -1
    if b is None:
        return 1
    return (a < b) - (a > b)


def outcomes(rows):
    if len(rows) != 3:
        raise ValueError("Exactly three planned executions required")
    values = []
    for row in rows:
        if row["status"] == "VALID_IN_TIME":
            value = row["objective"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Expected verified nonnegative integer objective")
            values.append(value)
        elif row["status"] == "SOLVER_FAILURE":
            if row.get("objective") is not None:
                raise ValueError("Failures cannot have invented objectives")
            values.append(None)
        else:
            raise ValueError("Unresolved or infrastructure-interrupted execution needs adjudication")
    return values


def summarize_instance(record):
    candidate, comparator = outcomes(record["candidate"]), outcomes(record["comparator"])
    pairs = [compare(a, b) for a in candidate for b in comparator]
    complete = all(x is not None for x in candidate + comparator)
    numeric = None
    if complete:
        a, b = statistics.mean(candidate), statistics.mean(comparator)
        numeric = (b - a) / max(1, a, b)
    valid_a = sum(x is not None for x in candidate)
    valid_b = sum(x is not None for x in comparator)
    return dict(instance=record["instance"], stratum=record["stratum"],
                wins=pairs.count(1), ties=pairs.count(0), losses=pairs.count(-1),
                head_to_head=statistics.mean(pairs), candidate_valid=valid_a,
                comparator_valid=valid_b, reliability_difference=(valid_a - valid_b) / 3,
                complete_numeric=numeric)


def percentile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (index - lower) * (values[upper] - values[lower])


def analyze(records, bootstrap_seed, resamples=10_000, expected_per_stratum=5):
    if len({r["instance"] for r in records}) != len(records):
        raise ValueError("Duplicate instance IDs")
    rows = [summarize_instance(record) for record in records]
    strata = {}
    for row in rows:
        strata.setdefault(row["stratum"], []).append(row)
    if len(strata) != 6 or any(len(group) != expected_per_stratum for group in strata.values()):
        raise ValueError("Expected six balanced strata with all planned instances")
    if resamples < 1:
        raise ValueError("Positive resampling count required")
    numeric_complete = all(row["complete_numeric"] is not None for row in rows)
    fields = ["head_to_head", "reliability_difference"] + (["complete_numeric"] if numeric_complete else [])
    rng = random.Random(bootstrap_seed)
    bootstrap = {field: [] for field in fields}
    # Resample whole instances, preserving all six outcomes and their dependence.
    for _ in range(resamples):
        sample = [rng.choice(strata[key]) for key in sorted(strata) for _ in range(expected_per_stratum)]
        for field in fields:
            bootstrap[field].append(statistics.mean(row[field] for row in sample))
    intervals = {field: [percentile(values, .025), percentile(values, .975)] for field, values in bootstrap.items()}
    reliability_gate = sum(row["candidate_valid"] for row in rows) >= sum(row["comparator_valid"] for row in rows)
    return dict(instances=rows, bootstrap_seed=bootstrap_seed, resamples=resamples,
                unit="instance; stratified percentile bootstrap", intervals=intervals,
                means={field: statistics.mean(row[field] for row in rows) for field in fields},
                candidate_valid=sum(row["candidate_valid"] for row in rows),
                comparator_valid=sum(row["comparator_valid"] for row in rows),
                descriptive_reliability_gate=reliability_gate,
                failure_aware_advantage_supported=reliability_gate and intervals["head_to_head"][0] > 0,
                numerical_quality_supported=numeric_complete and reliability_gate and intervals["head_to_head"][0] > 0 and intervals["complete_numeric"][0] > 0,
                incomplete_numeric_policy="No full-sample numeric advantage if any failures; conditional analyses must be labelled separately")
