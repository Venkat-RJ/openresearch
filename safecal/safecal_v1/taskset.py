"""Prime Intellect Verifiers v1 integration for SafeCal.

Architecture follows PrimeIntellect-ai/verifiers >=0.4.0:
  TaskData -> Taskset -> host-side Env.run() -> Trace.record_reward().
The model only sees JSON observations; it never sees private actuator state.
"""
from __future__ import annotations
import itertools
from typing import Iterator, Literal

import verifiers.v1 as vf

from safecal_v1.protocol import SYSTEM_PROMPT, SafeCalDialogue


class SafeCalTaskData(vf.TaskData):
    info: dict


class SafeCalTask(vf.Task[SafeCalTaskData, vf.State, vf.TaskConfig]):
    pass


class SafeCalTasksetConfig(vf.TasksetConfig):
    split: Literal["train", "heldout", "combined", "shifted"] = "train"


class SafeCalTaskset(vf.Taskset[SafeCalTask, SafeCalTasksetConfig]):
    INFINITE = True

    def load(self) -> Iterator[SafeCalTask]:
        config = self.config
        base = {"train": 1000, "heldout": 900000,
                "combined": 1200000, "shifted": 1500000}[config.split]
        for index in itertools.count():
            yield SafeCalTask(
                SafeCalTaskData(
                    name=f"safecal-{config.split}-{index}",
                    prompt="Start the diagnostic episode. Submit one JSON action.",
                    system_prompt=SYSTEM_PROMPT,
                    info={"seed": base + index, "split": config.split},
                ),
                config.task,
            )


class SafeCalVerifierConfig(vf.SingleAgentEnvConfig):
    # The 4-probe task needs five model turns; two malformed responses
    # are tolerated. The host also enforces an explicit seven-turn cap.
    pass


class SafeCalVerifierEnv(vf.Env[SafeCalVerifierConfig]):
    async def run(self, task: SafeCalTask, agents):
        info = task.data.info
        dialogue = SafeCalDialogue(
            info["seed"],
            combined=info["split"] == "combined",
            novel=info["split"] == "shifted",
        )
        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn(dialogue.observation())
            for _ in range(7):
                if segment.terminated or dialogue.completed:
                    break
                observation, completed = dialogue.act(segment.last_reply)
                if completed:
                    break
                segment = await interaction.turn(observation)
        trace = interaction.trace
        trace.record_reward("safecal_net_reward", dialogue.reward(), 1.0)
        trace.record_metric("completed", float(dialogue.outcome is not None))
        trace.record_metric("invalid_actions", float(dialogue.invalid))
        trace.record_metric("diagnostic_cost", float(dialogue.env.spent))
        trace.record_metric("num_probes", float(len(dialogue.env.records)))
        if dialogue.outcome is not None:
            trace.record_metric("calibration_quality", float(dialogue.outcome["quality"]))

