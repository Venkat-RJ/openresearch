# SafeCal

**A trainable, simulated engineering diagnosis task.** The question is whether an agent can learn to choose *safe, informative experiments* and recalibrate an unfamiliar actuator with fewer tests than strong classical system-identification methods.

SafeCal is a CPU-based research simulator with an optional [Prime Intellect Verifiers v1](https://github.com/PrimeIntellect-ai/verifiers) taskset. It is **not** automotive control software, a real hardware controller, or a validated robotics safety benchmark. **No RL policy has been trained.**

The original motivation comes from industrial power-liftgate calibration work, but all plant parameters, faults, trajectories, and rewards here are synthetic and normalized. No employer code, proprietary parameters, or hardware traces are included.

## The task

A hidden single-axis actuator may have motor, friction, load, or encoder-bias faults. The agent can run bounded torque-pulse diagnostics or a more expensive reference check. The simulator returns noisy displacement and velocity measurements. After spending up to 4.5 normalized cost units, the agent chooses one of 150 legal calibrations for feedback gain, friction feed-forward, and encoder offset.

A **host-side** evaluator checks four unseen trajectories for tracking error, residual speed, energy, and safety interventions. Reward is `calibration_quality - 0.012 * test_cost`. The evaluator must never be mounted inside a coding-agent sandbox. Physical constraints are enforced by the host, not merely rewarded by the model.

Three evaluation splits:

- `heldout`: unseen seeds with individual faults.
- `combined`: unseen seeds with simultaneous motor degradation and friction changes. The stronger Bayesian prior already includes this fault class, so this is **parameter generalization**, not unseen model-class discovery.
- `shifted`: a new torque-deadzone term missing from both the simple and Bayesian model priors. This is synthetic **structural model mismatch**, not validated transfer to real hardware.

## Pilot results

Mean **net reward** across 60 seeds per split, higher is better:

| Baseline | Single faults | Combined faults | Shifted dynamics |
|---|---:|---:|---:|
| Factory default, no tests | 0.84053 | 0.79321 | 0.64360 |
| Random tests + coarse inference | 0.83333 | 0.74069 | 0.61665 |
| Fixed tests + coarse inference | 0.85070 | 0.73545 | 0.60238 |
| Coarse greedy info proxy | 0.85546 | 0.74145 | 0.60838 |
| Bayesian identification, up to 4 probes | 0.85311 | 0.85051 | 0.76219 |
| **Bayesian identification, up to 2 probes** | **0.86167** | **0.87811** | **0.78425** |
| Privileged oracle quality, no test cost | 0.91584 | 0.92050 | 0.82010 |

The **two-probe limit was selected using training seeds only**, across single and compound faults. It was not selected on the evaluation seeds. The Bayesian method uses a fixed 384-particle prior, Gaussian likelihood updates, Monte Carlo expected information gain per probe cost, and a posterior-predictive approximation to calibration quality. This is a classical non-neural baseline, **not RL**.

On the single-fault split, the two-probe Bayesian policy is only 0.00621 above the coarse greedy policy, with a paired bootstrap 95% interval of [-0.00460, +0.01622], so that difference is not established. On the combined and shifted splits, classical Bayesian identification closes much of the original gap to the oracle. **The original claim that compound faults require RL is not supported.**

See [RESULTS.md](RESULTS.md), [PUBLIC_RESULTS.json](PUBLIC_RESULTS.json), and [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Reproduce locally

Python 3.11+; core simulator uses only the standard library. Bayesian baseline requires NumPy.

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python benchmark.py --tasks 60 --split heldout --json results_heldout_v1.json
python benchmark.py --tasks 60 --split combined --json results_combined_v1.json
python benchmark.py --tasks 60 --split shifted --json results_shifted_v1.json
python budget_sweep.py --tasks 25 --first-probes 0 --last-probes 2 --json budget_0to2.json
python budget_sweep.py --tasks 25 --first-probes 3 --last-probes 4 --json budget_3to4.json
python benchmark.py --tasks 60 --split shifted --policies default bayes2 --json results_bayes2_shifted.json
python report_analysis.py
```

The budget sweep uses `train` single seeds 1000-1024 and separate training compound seeds 50000-50024. Evaluation splits begin at 900000, 1200000, and 1500000, respectively. Runs use deterministic random generators and recorded seeds.

### Interactive Python episode

```python
from safecal_v1.protocol import SafeCalDialogue

episode = SafeCalDialogue(seed=900001)
print(episode.observation())
print(episode.act('{"probe":"pulse_pos_high"}'))
print(episode.act('{"calibration":{"gain":1.0,"friction_ff":0.08,"sensor_offset":0.0}}'))
print(episode.reward())
```

## Verifiers v1 adapter (experimental)

`safecal_v1/` exports a v1 `Taskset` and a host-side multi-turn `Env` that accepts one JSON probe or calibration action per turn, records final reward on the Verifiers trace, and enforces an invalid-action limit. It targets the **v1 interfaces of Verifiers 0.4.x**, not the removed v0 `ToolEnv` API.

```bash
python -m pip install -e '.[bayes,verifiers]'
# Requires configured model endpoint and an isolated harness/runtime:
vf-eval safecal-v1 -n 10 --model your-provider/your-model \
    --env.taskset.split train --env.agent.runtime.type docker --no-push
```

**Integration status:** The taskset and host-side dialogue are implemented; the dialogue is unit-tested. We have **not** run a model evaluation or RL training in Verifiers because package installation was blocked in the development container. Run an actual `vf-eval` smoke test in a configured environment before publishing the taskset to an environment hub or making integration claims. A coding harness must be isolated from host evaluator code and task seeds.

## Next scientific test

Can a learned policy beat the train-selected, cost-aware Bayesian diagnostic baseline on **unseen dynamics**, across matched diagnostic budgets, while respecting an independently enforced safety shield? Better reward on this simplified simulator alone would not establish real-world safety or scientific generalization. Stronger Bayesian value-of-information and model-mismatch baselines are still needed.

All experiments to date are simulations, with **no RL training and no hardware validation**.
