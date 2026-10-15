"""Custom scheduling distribution, development version 0.1.

LiSA handbook p70, equations7.1/7.2 motivate due dates and release range:
https://www.math.ovgu.de/math_media/Lisa/Handbook_en-p-6090.pdf
Dimensions, tightness levels, weights and integer rounding are our design choices.
This is not the missing FrontierOR generator and contains no final-test seed.
"""

import hashlib
import random

VERSION = "custom-jsptwt-development-v0.1"
TIGHTNESS = (2, 3, 4)
WEIGHTS = ("uniform", "priority")


def stream(seed, field):
    digest = hashlib.sha256(f"{VERSION}:{seed}:{field}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def generate(seed, tightness, weights, n=30, m=10):
    if tightness not in TIGHTNESS or weights not in WEIGHTS or n < 1 or m < 1:
        raise ValueError("Invalid development configuration")
    durations_rng = stream(seed, "durations")
    routes_rng = stream(seed, "routes")
    releases_rng = stream(seed, "releases")
    weights_rng = stream(seed, "weights")
    times = [[durations_rng.randint(1, 99) for _ in range(m)] for _ in range(n)]
    release_max = sum(map(sum, times)) // (2 * n)
    # Exactly20% high-priority jobs when n is divisible by5. No Bernoulli count drift.
    high_priority = set(weights_rng.sample(range(n), n // 5)) if weights == "priority" else set()
    jobs = []
    for j in range(n):
        route = list(range(m))
        routes_rng.shuffle(route)
        release = releases_rng.randint(0, release_max)
        weight = weights_rng.randint(1, 10) if weights == "uniform" else (10 if j in high_priority else 1)
        jobs.append(dict(job_id=j, release_date=release,
                         due_date=release + tightness * sum(times[j]), weight=weight,
                         operations=[dict(machine=machine, processing_time=times[j][machine]) for machine in route]))
    return dict(num_jobs=n, num_machines=m, num_operations=n * m, jobs=jobs)


def development_plan():
    return [dict(name=f"dev-tf{tf}-{weight}-{repeat}", tightness=tf, weights=weight,
                 seed=f"development:{tf}:{weight}:{repeat}")
            for tf in TIGHTNESS for weight in WEIGHTS for repeat in range(2)]
