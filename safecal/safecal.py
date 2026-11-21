"""SafeCal v0: CPU-only, reproducible research environment for safe actuator diagnosis.

Synthetic normalized physics. No real-hardware control and no safety certification.
Uses only Python's standard library.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random
from typing import NamedTuple

DT = 0.02
POSITION_LIMIT = 0.50
SPEED_LIMIT = 0.75
TORQUE_LIMIT = 0.85
TEST_BUDGET = 4.5


@dataclass(frozen=True)
class Plant:
    motor: float = 1.0
    friction: float = 0.10
    load: float = 0.12
    damping: float = 0.23
    sensor_bias: float = 0.0
    torque_deadzone: float = 0.0  # optional held-out physics; never in coarse prior


@dataclass(frozen=True)
class Probe:
    name: str
    kind: str
    torque: float = 0.0
    fixture_x: float = 0.0
    seconds: float = 0.0
    cost: float = 1.0


# Each test starts from a known, safely fixtured position. Tool outputs only
# include measurements, never the underlying parameters or the fault label.
PROBES = (
    Probe("pulse_pos_low", "pulse", 0.45, 0.0, 0.55),
    Probe("pulse_pos_high", "pulse", 0.85, 0.0, 0.55),
    Probe("pulse_neg_low", "pulse", -0.45, 0.0, 0.55),
    Probe("pulse_neg_high", "pulse", -0.85, 0.0, 0.55),
    Probe("pulse_pos_offset", "pulse", 0.65, 0.28, 0.50),
    Probe("pulse_neg_offset", "pulse", -0.65, -0.28, 0.50),
    Probe("reference_check", "reference", cost=1.50),
)
PROBE_BY_NAME = {p.name: p for p in PROBES}

# Calibration is deliberately discrete so every terminal action is easy to
# validate and the oracle can be computed by exhaustive search.
GAINS = (0.75, 0.90, 1.00, 1.12, 1.25, 1.40)
FRICTION_FF = (0.00, 0.08, 0.16, 0.24, 0.32)
BIAS_CORRECTIONS = (-0.08, -0.04, 0.00, 0.04, 0.08)


@dataclass(frozen=True)
class Calibration:
    gain: float = 1.0
    friction_ff: float = 0.08
    sensor_offset: float = 0.0


CALIBRATIONS = tuple(Calibration(g, f, b) for b in BIAS_CORRECTIONS
                     for f in FRICTION_FF for g in GAINS)
DEFAULT_CALIBRATION = Calibration(1.0, 0.08, 0.0)


class Trajectory(NamedTuple):
    x: float
    v: float
    risk_interventions: int
    effort: float
    cumulative_tracking_error: float


def clamp(x: float, low: float, high: float) -> float:
    return min(high, max(low, x))


def acceleration(plant: Plant, x: float, v: float, torque: float) -> float:
    # Plant: normalized motor, Coulomb + viscous friction and position-dependent load.
    # A deadzone creates an unknown, non-linear motor response on shifted tasks.
    # All original and prior-bank plants have deadzone=0, preserving v0 dynamics.
    delivered = math.copysign(max(0.0, abs(torque) - plant.torque_deadzone), torque)
    return (plant.motor * delivered
            - plant.friction * math.tanh(v / 0.035)
            - plant.damping * v
            - plant.load * math.sin(math.pi * x / 1.1))


def integrate_pulse(plant: Plant, torque: float, fixture_x: float,
                    seconds: float) -> Trajectory:
    """Simulate a guarded torque pulse, with independent mechanical reset."""
    if abs(torque) > TORQUE_LIMIT or abs(fixture_x) > 0.30 or not 0 < seconds <= 0.60:
        raise ValueError("Unsafe diagnostic command: outside shielded action envelope")
    x, v, interventions, effort = fixture_x, 0.0, 0, 0.0
    steps = int(round(seconds / DT))
    for _ in range(steps):
        a = acceleration(plant, x, v, torque)
        v_next = v + a * DT
        x_next = x + v_next * DT
        if abs(v_next) > SPEED_LIMIT or abs(x_next) > POSITION_LIMIT:
            interventions += 1
            break
        x, v = x_next, v_next
        effort += abs(torque) * DT
    return Trajectory(x, v, interventions, effort, 0.0)


def measurement_prediction(plant: Plant, probe: Probe) -> tuple[float, ...]:
    if probe.kind == "reference":
        # A physical datum fixture, available at higher experiment cost.
        return (plant.sensor_bias,)
    t = integrate_pulse(plant, probe.torque, probe.fixture_x, probe.seconds)
    return (t.x - probe.fixture_x, t.v)


def run_closed_loop(plant: Plant, cal: Calibration, x0: float,
                    target: float) -> Trajectory:
    x, v, interventions, effort, accumulated_error = x0, 0.0, 0, 0.0, 0.0
    for _ in range(int(1.8 / DT)):
        observed_x = x + plant.sensor_bias
        err = target - (observed_x - cal.sensor_offset)
        raw_torque = cal.gain * (3.1 * err - 1.25 * v
                                 + cal.friction_ff * math.tanh(err / 0.045))
        torque = clamp(raw_torque, -TORQUE_LIMIT, TORQUE_LIMIT)
        a = acceleration(plant, x, v, torque)
        v_next = v + a * DT
        x_next = x + v_next * DT
        if abs(x_next) >= POSITION_LIMIT or abs(v_next) >= SPEED_LIMIT:
            interventions += 1
            break
        x, v = x_next, v_next
        accumulated_error += abs(target - x) * DT
        effort += abs(torque) * DT
    return Trajectory(x, v, interventions, effort, accumulated_error)


# Both directions, from two different starts. The true position is accessible
# only to the frozen evaluator, not to the agent.
EVAL_TRAJECTORIES = ((-0.15, 0.28), (0.15, -0.28),
                     (-0.05, 0.32), (0.05, -0.32))


def calibration_quality(plant: Plant, cal: Calibration) -> float:
    total_loss = 0.0
    for start, target in EVAL_TRAJECTORIES:
        t = run_closed_loop(plant, cal, start, target)
        loss = (3.8 * abs(t.x - target) + 0.45 * abs(t.v)
                + 0.05 * t.cumulative_tracking_error + 0.006 * t.effort)
        if t.risk_interventions:
            loss += 3.0 * t.risk_interventions
        total_loss += loss
    return math.exp(-total_loss / len(EVAL_TRAJECTORIES))


@lru_cache(maxsize=None)
def candidate_plants() -> tuple[tuple[Plant, float], ...]:
    """Public coarse prior. Held-out plants have jitter not in this library."""
    items: list[tuple[Plant, float]] = []
    items.append((Plant(), 0.20))
    for m in (0.67, 0.79):
        items.append((Plant(motor=m), 0.10))
    for f in (0.21, 0.31):
        items.append((Plant(friction=f), 0.10))
    for load in (0.24, 0.39):
        items.append((Plant(load=load), 0.10))
    for bias in (-0.075, -0.045, 0.045, 0.075):
        items.append((Plant(sensor_bias=bias), 0.05))
    assert abs(sum(p for _, p in items) - 1.0) < 1e-12
    return tuple(items)


@lru_cache(maxsize=None)
def predicted_probes() -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(tuple(measurement_prediction(model, probe) for probe in PROBES)
                 for model, _ in candidate_plants())


@lru_cache(maxsize=None)
def predicted_quality() -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(calibration_quality(model, cal) for cal in CALIBRATIONS)
                 for model, _ in candidate_plants())


def plant_for_seed(seed: int, combined: bool = False,
                   novel: bool = False) -> tuple[Plant, str]:
    """Procedural tasks. A held-out combined-fault split tests composition."""
    rng = random.Random(seed * 2027 + 113)
    kind = "deadzone_shift" if novel else "combined" if combined else rng.choice(
        ("nominal", "weak_motor", "high_friction", "high_load", "sensor_bias"))
    plant = Plant(motor=1.0 + rng.uniform(-0.018, 0.018),
                  friction=0.10 + rng.uniform(-0.012, 0.012),
                  load=0.12 + rng.uniform(-0.012, 0.012),
                  damping=0.23 + rng.uniform(-0.012, 0.012),
                  sensor_bias=rng.uniform(-0.004, 0.004))
    params = vars(plant).copy()
    if kind == "weak_motor":
        params["motor"] = rng.uniform(0.65, 0.81)
    elif kind == "high_friction":
        params["friction"] = rng.uniform(0.20, 0.32)
    elif kind == "high_load":
        params["load"] = rng.uniform(0.23, 0.40)
    elif kind == "sensor_bias":
        params["sensor_bias"] = rng.choice((-1, 1)) * rng.uniform(0.035, 0.085)
    elif kind == "combined":
        params["motor"] = rng.uniform(0.67, 0.82)
        params["friction"] = rng.uniform(0.19, 0.30)
    elif kind == "deadzone_shift":
        params["motor"] = rng.uniform(0.72, 0.92)
        params["friction"] = rng.uniform(0.14, 0.25)
        params["torque_deadzone"] = rng.uniform(0.14, 0.33)
    return Plant(**params), kind


class SafeCalEnv:
    """Gym-like episodic API: choose diagnostic probes, then calibration.

    No hidden parameter, label, oracle score or held-out trajectory leaks into
    tool observations. After finish(), no further tests may be requested.
    """

    def __init__(self, seed: int, *, combined: bool = False, novel: bool = False):
        self.seed = seed
        self._plant, self._family = plant_for_seed(seed, combined=combined, novel=novel)
        self._rng = random.Random(seed * 1009 + 53)
        self.remaining_budget = TEST_BUDGET
        self.spent = 0.0
        self.used: set[str] = set()
        self.finished = False
        self.records: list[dict] = []

    def available_actions(self) -> list[str]:
        return [p.name for p in PROBES if p.name not in self.used
                and p.cost <= self.remaining_budget and not self.finished]

    def step(self, probe_name: str) -> dict:
        if self.finished:
            raise ValueError("Episode finished")
        if probe_name not in self.available_actions():
            raise ValueError("Unsafe, repeated, unknown, or over-budget probe")
        p = PROBE_BY_NAME[probe_name]
        y = measurement_prediction(self._plant, p)
        sigma = 0.006 if p.kind == "reference" else 0.004
        observed = [float(y[0] + self._rng.gauss(0.0, sigma))]
        if p.kind == "pulse":
            observed.append(float(y[1] + self._rng.gauss(0.0, 0.009)))
        record = {"probe": p.name, "observed": observed, "cost": p.cost}
        self.records.append(record)
        self.used.add(p.name)
        self.spent += p.cost
        self.remaining_budget = round(TEST_BUDGET - self.spent, 8)
        return dict(record)

    def finish(self, calibration: Calibration) -> dict:
        if self.finished:
            raise ValueError("Episode already finished")
        if calibration not in CALIBRATIONS:
            raise ValueError("Calibration must be a legal discrete action")
        self.finished = True
        quality = calibration_quality(self._plant, calibration)
        reward = quality - 0.012 * self.spent
        return {"quality": quality, "reward": reward, "test_cost": self.spent,
                "num_probes": len(self.records)}


def posterior(records: list[dict]) -> list[float]:
    priors = [prior for _, prior in candidate_plants()]
    logs = [math.log(prior) for prior in priors]
    probes_indices = {p.name: i for i, p in enumerate(PROBES)}
    predicted = predicted_probes()
    for obs in records:
        idx = probes_indices[obs["probe"]]
        sigmas = (0.006,) if PROBES[idx].kind == "reference" else (0.007, 0.022)
        for j in range(len(logs)):
            for measured, guess, sigma in zip(obs["observed"], predicted[j][idx], sigmas):
                logs[j] -= 0.5 * ((measured - guess) / sigma) ** 2
    scale = max(logs)
    weights = [math.exp(x - scale) for x in logs]
    norm = sum(weights)
    return [x / norm for x in weights]


def choose_calibration(weights: list[float]) -> Calibration:
    scores = predicted_quality()
    winner = max(range(len(CALIBRATIONS)),
                 key=lambda k: sum(w * scores[i][k] for i, w in enumerate(weights)))
    return CALIBRATIONS[winner]


def greedy_probe(weights: list[float], available: list[str]) -> str:
    """Variance-per-cost proxy for information gain, NOT an RL policy."""
    preds = predicted_probes()
    idx = {p.name: i for i, p in enumerate(PROBES)}
    def value(name: str) -> float:
        j = idx[name]
        if PROBES[j].kind == "reference":
            sigma = (0.020,)
        else:
            sigma = (0.015, 0.036)
        vals = [preds[i][j] for i in range(len(weights))]
        score = 0.0
        for d in range(len(sigma)):
            mean = sum(w * v[d] for w, v in zip(weights, vals))
            score += sum(w * (v[d] - mean)**2 for w, v in zip(weights, vals)) / sigma[d]**2
        return score / PROBES[j].cost
    return max(available, key=lambda name: (value(name), name))


def policy_calibration(env: SafeCalEnv, policy: str) -> Calibration:
    if policy == "default":
        return DEFAULT_CALIBRATION
    if policy == "fixed":
        seq = ["reference_check", "pulse_pos_high", "pulse_pos_low", "pulse_pos_offset"]
        for name in seq:
            if name in env.available_actions():
                env.step(name)
    elif policy == "random":
        choices = env.available_actions()
        random.Random(env.seed * 47 + 22).shuffle(choices)
        for name in choices:
            if len(env.records) >= 4:
                break
            if name in env.available_actions():
                env.step(name)
    elif policy == "greedy":
        while env.available_actions() and len(env.records) < 4:
            env.step(greedy_probe(posterior(env.records), env.available_actions()))
    else:
        raise ValueError(f"Unknown baseline: {policy}")
    return choose_calibration(posterior(env.records))


def oracle_quality(plant: Plant) -> float:
    """Privileged upper bound for analysis, never exposed to agent."""
    return max(calibration_quality(plant, c) for c in CALIBRATIONS)
