"""Confirmatory plan: defines seeds and order without generating any instances."""
import random

PROTOCOL_VERSION = 'final-synthetic-v1'
SOLVER_SEEDS = (104729, 130363, 155921)
ORDER_SEED = 2026100801
BOOTSTRAP_SEED = 2026100802
SOLVERS = ('frozen-v2', 'egrasp')


def instance_plan():
    return [dict(name=f"final-tf{tf.replace('/', '_')}-{weights}-{i}",
                 tightness=tf, weights=weights,
                 seed=f'{PROTOCOL_VERSION}:{tf}:{weights}:{i}')
            for tf in ('3/2', '7/4', '2')
            for weights in ('uniform', 'priority') for i in range(5)]


def execution_plan():
    cases = instance_plan()
    # Each instance/repetition is a two-solver block. Block order is fixed-random;
    # exactly half of the 90 blocks run each solver first.
    blocks = [(case, repeat) for case in cases for repeat in range(3)]
    rng = random.Random(ORDER_SEED)
    rng.shuffle(blocks)
    first = [0, 1] * 45
    rng.shuffle(first)
    schedule = []
    for index, ((case, repeat), lead) in enumerate(zip(blocks, first)):
        order = SOLVERS if lead == 0 else SOLVERS[::-1]
        for solver in order:
            schedule.append(dict(slot=f"{case['name']}-r{repeat}-{solver}",
                instance=case['name'], solver=solver, repeat=repeat, block=index,
                seed=SOLVER_SEEDS[repeat] if solver == 'frozen-v2' else None))
    return schedule


def analysis_records(plan, rows):
    expected = execution_plan()
    if len(rows) != 180 or [(r['slot'], r['solver'], r['repeat'], r['seed']) for r in rows] != [
            (x['slot'], x['solver'], x['repeat'], x['seed']) for x in expected]:
        raise ValueError('All 180 executions must match the fixed schedule')
    records = []
    for case in plan:
        record = dict(instance=case['name'], stratum=f"tf{case['tightness']}-{case['weights']}")
        for solver, key in [('frozen-v2', 'candidate'), ('egrasp', 'comparator')]:
            group = sorted([r for r in rows if r['instance'] == case['name'] and r['solver'] == solver], key=lambda r: r['repeat'])
            if [r['repeat'] for r in group] != [0, 1, 2]:
                raise ValueError('Missing or duplicate repetitions')
            record[key] = group
        records.append(record)
    return records
