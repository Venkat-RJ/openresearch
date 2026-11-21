"""SafeCal multi-turn dialogue contract, independent of Verifiers.

The model chooses JSON actions; the host environment owns the hidden physics.
This module exposes no private plant parameters or evaluation outcomes to
future tool turns. Do not mount this evaluator inside an agent code sandbox.
"""
from __future__ import annotations
import json
import math
from typing import Any

from safecal import (
    BIAS_CORRECTIONS, FRICTION_FF, GAINS, PROBES,
    Calibration, SafeCalEnv,
)

SYSTEM_PROMPT = """You are calibrating a SIMULATED single-axis actuator.
No physical device is connected. You can run diagnostic probes and then
submit one calibration. Hidden actuator parameters are not available.

Reply with exactly one JSON object per turn:
  {"probe": "pulse_pos_low"}
  OR
  {"calibration": {"gain": 1.0, "friction_ff": 0.08, "sensor_offset": 0.0}}

Available probes: pulse_pos_low, pulse_pos_high, pulse_neg_low,
pulse_neg_high, pulse_pos_offset, pulse_neg_offset, reference_check.
Probe costs: each pulse = 1.0; reference_check = 1.5; total budget = 4.5.
Valid gains: 0.75, 0.90, 1.00, 1.12, 1.25, 1.40.
Valid friction_ff: 0.00, 0.08, 0.16, 0.24, 0.32.
Valid sensor_offset: -0.08, -0.04, 0.00, 0.04, 0.08.
Probes may only be used once. The host enforces the safe action envelope.

Goal: maximize hidden actuator tracking quality minus the cost of testing.
The evaluator checks multiple unseen motion trajectories, not your explanation.
"""


class SafeCalDialogue:
    """Per-episode state. Any invalid action consumes an agent turn."""
    MAX_INVALID = 2

    def __init__(self, seed: int, *, combined: bool = False, novel: bool = False):
        self.env = SafeCalEnv(seed, combined=combined, novel=novel)
        self.invalid = 0
        self.completed = False
        self.outcome: dict[str, float | int] | None = None

    def observation(self, record: dict[str, Any] | None = None) -> str:
        body = {
            "status": "ready" if record is None else "measurement",
            "remaining_budget": self.env.remaining_budget,
            "available_probes": self.env.available_actions(),
        }
        if record is not None:
            body["probe_result"] = record
        return json.dumps(body, separators=(",", ":"), sort_keys=True)

    def act(self, response: str) -> tuple[str, bool]:
        """Returns next user message and whether the episode has terminated."""
        if self.completed:
            raise RuntimeError("Dialogue already completed")
        try:
            data = json.loads(response.strip())
            if not isinstance(data, dict) or len(data) != 1:
                raise ValueError("Expected one JSON action")
            if "probe" in data:
                if not isinstance(data["probe"], str):
                    raise ValueError("Probe name must be a string")
                result = self.env.step(data["probe"])
                return self.observation(result), False
            if "calibration" in data:
                c = data["calibration"]
                if not isinstance(c, dict) or set(c) != {"gain", "friction_ff", "sensor_offset"}:
                    raise ValueError("Calibration needs exactly three fields")
                if any(isinstance(v, bool) or not isinstance(v, (int,float)) or
                       not math.isfinite(float(v)) for v in c.values()):
                    raise ValueError("Invalid numeric values")
                cal = Calibration(**{k: float(v) for k, v in c.items()})
                self.outcome = self.env.finish(cal)
                self.completed = True
                # The environment terminates now, so this is not sent back to
                # the policy before its next action. Score is host-only.
                return '{"status":"submitted"}', True
            raise ValueError("Unknown action type")
        except (ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError):
            self.invalid += 1
            if self.invalid >= self.MAX_INVALID:
                self.completed = True
                return '{"error":"invalid_action_limit","status":"terminated"}', True
            return json.dumps({"error": "invalid_action", "remaining_budget": self.env.remaining_budget,
                               "available_probes": self.env.available_actions()}), False

    def reward(self) -> float:
        return float(self.outcome["reward"]) if self.outcome is not None else 0.0
