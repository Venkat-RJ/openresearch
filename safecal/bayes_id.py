"""Bayesian system-identification baseline with budgeted experiment design.

The particle prior and observation model are fixed independently of evaluation
seeds. There is no access to the hidden plant, fault label, oracle, or scoring
result when an action is chosen. This is a classical baseline, not RL.

Requires NumPy. The core SafeCal simulator itself remains stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random

import numpy as np

from safecal import (
    CALIBRATIONS, DEFAULT_CALIBRATION, PROBES, PROBE_BY_NAME,
    Calibration, Plant, SafeCalEnv, calibration_quality, measurement_prediction,
)

N_PARTICLES = 384
# Likelihood accounts for both measurement noise and modest model discrepancy.
SIGMAS = np.array([[0.011, 0.026] for _ in PROBES], dtype=float)
SIGMAS[-1] = (0.009, 1.0)


def _draw_plant(rng: random.Random, family: str) -> Plant:
    params = {
        "motor": 1.0 + rng.uniform(-0.03, 0.03),
        "friction": 0.10 + rng.uniform(-0.025, 0.025),
        "load": 0.12 + rng.uniform(-0.025, 0.025),
        "damping": 0.23 + rng.uniform(-0.022, 0.022),
        "sensor_bias": rng.uniform(-0.009, 0.009),
    }
    if family in ("motor", "compound"):
        params["motor"] = rng.uniform(0.60, 0.85)
    if family in ("friction", "compound"):
        params["friction"] = rng.uniform(0.17, 0.37)
    if family == "load":
        params["load"] = rng.uniform(0.20, 0.46)
    if family == "bias":
        params["sensor_bias"] = rng.choice((-1, 1)) * rng.uniform(0.025, 0.10)
    if family == "broad":
        params.update(motor=rng.uniform(0.60, 1.07),
                      friction=rng.uniform(0.07, 0.36),
                      load=rng.uniform(0.08, 0.46),
                      sensor_bias=rng.uniform(-0.095, 0.095))
    return Plant(**params)


@lru_cache(maxsize=1)
def prior_bank() -> tuple[tuple[Plant, ...], np.ndarray]:
    """Deterministic training prior; explicitly includes compound hypotheses.

    Particles are sampled once with a fixed seed unrelated to task identities.
    Held-out combined tasks therefore test parameter generalization, not
    out-of-support model-family discovery. A separate physics-family split is
    needed before claiming transfer to new model structures.
    """
    rng = random.Random(783617)
    families = (["nominal"] * 48 + ["motor"] * 54 + ["friction"] * 54
                + ["load"] * 54 + ["bias"] * 54 + ["compound"] * 78
                + ["broad"] * 42)
    assert len(families) == N_PARTICLES
    rng.shuffle(families)
    plants = tuple(_draw_plant(rng, family) for family in families)
    # Reference check is one-dimensional; pad its second component as zero.
    p = np.zeros((N_PARTICLES, len(PROBES), 2), dtype=float)
    for i, plant in enumerate(plants):
        for j, probe in enumerate(PROBES):
            y = measurement_prediction(plant, probe)
            p[i, j, :len(y)] = y
    return plants, p


@lru_cache(maxsize=9000)
def _quality_cached(plant: Plant, cal: Calibration) -> float:
    return calibration_quality(plant, cal)


@dataclass
class BayesianID:
    """Particle posterior and information-gain experiment selector."""
    samples: int = 20
    seed: int = 2391

    def __post_init__(self) -> None:
        self.plants, self.predictions = prior_bank()
        self.weights = np.full(N_PARTICLES, 1.0 / N_PARTICLES)
        self.probe_to_idx = {p.name: j for j, p in enumerate(PROBES)}
        self.rng = np.random.default_rng(self.seed)

    def update(self, record: dict) -> None:
        idx = self.probe_to_idx[record["probe"]]
        y = np.asarray(record["observed"], dtype=float)
        residual = (self.predictions[:, idx, :len(y)] - y) / SIGMAS[idx, :len(y)]
        logw = np.log(np.maximum(self.weights, 1e-300)) - 0.5 * np.sum(residual ** 2, axis=1)
        logw -= np.max(logw)
        w = np.exp(logw)
        self.weights = w / np.sum(w)

    def info_gain(self, probe_name: str) -> float:
        """Monte-Carlo expected KL(p(theta|y) || p(theta)) in nats / cost."""
        idx = self.probe_to_idx[probe_name]
        dims = 1 if PROBES[idx].kind == "reference" else 2
        means = self.predictions[:, idx, :dims]
        sigmas = SIGMAS[idx, :dims]
        picks = self.rng.choice(N_PARTICLES, size=self.samples, p=self.weights)
        observations = means[picks] + self.rng.normal(size=(self.samples, dims)) * sigmas
        # For each possible observation, compare hypothetical posterior to prior.
        residual = (observations[:, None, :] - means[None, :, :]) / sigmas
        log_likelihood = -0.5 * np.sum(residual * residual, axis=-1)
        prior_logs = np.log(np.maximum(self.weights, 1e-300))
        posterior_logs = log_likelihood + prior_logs
        posterior_logs -= np.max(posterior_logs, axis=1, keepdims=True)
        posterior = np.exp(posterior_logs)
        posterior /= np.sum(posterior, axis=1, keepdims=True)
        kls = np.sum(posterior * (np.log(np.maximum(posterior, 1e-300)) - prior_logs), axis=1)
        return float(np.mean(kls) / PROBE_BY_NAME[probe_name].cost)

    def choose_probe(self, available: list[str]) -> tuple[str, float]:
        ranked = [(self.info_gain(name), name) for name in available]
        info, name = max(ranked)  # stable tie-break
        return name, info

    def choose_calibration(self) -> Calibration:
        """Approximate Bayes decision maximizing posterior expected quality.

        Decision is based on systematic posterior particles and the posterior
        mean. This is an approximate expectation, not a global Bayes optimum.
        """
        cdf = np.cumsum(self.weights)
        # Deterministic systematic resampling covers posterior modes while
        # bounding simulator calls. Different representative points are not
        # independent draws and do not measure epistemic confidence.
        quantiles = (np.arange(8) + 0.5) / 8.0
        indices = np.unique(np.searchsorted(cdf, quantiles).clip(0, N_PARTICLES - 1))
        representatives = [self.plants[int(i)] for i in indices]
        params = {}
        for name in vars(self.plants[0]):
            params[name] = float(sum(w * getattr(p, name) for w, p in zip(self.weights, self.plants)))
        representatives.append(Plant(**params))
        vals = [
            sum(_quality_cached(p, c) for p in representatives) / len(representatives)
            for c in CALIBRATIONS
        ]
        return CALIBRATIONS[int(np.argmax(vals))]


def policy_bayes(env: SafeCalEnv, *, max_probes: int = 4,
                 min_gain: float = 0.03, samples: int = 20) -> Calibration:
    """Info-gain active identification + posterior-predictive calibration.

    `min_gain` is a fixed heuristic stopping threshold, not an optimized value
    from held-out evaluation. It can be changed on training tasks only.
    """
    model = BayesianID(samples=samples, seed=env.seed * 31 + 379)
    while len(env.records) < max_probes:
        available = env.available_actions()
        if not available:
            break
        name, gain = model.choose_probe(available)
        if len(env.records) > 0 and gain < min_gain:
            break
        model.update(env.step(name))
    return model.choose_calibration()
