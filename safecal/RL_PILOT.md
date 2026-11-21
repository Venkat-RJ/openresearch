# SafeCal neural diagnostic policy pilot (21 November 2026)

**Result:** A small model-free policy-gradient experiment ran end-to-end, but did **not** establish adaptive RL capability. The learned policy always chose the same one-probe action (`pulse_neg_low`) and then stopped. A hand-coded one-probe policy produced **exactly identical** held-out rewards. This is an important negative control.

## What was trained

A two-layer MLP (48 hidden units) receives features derived solely from visible probe observations and a fixed 384-particle Bayesian posterior. It chooses among seven shielded diagnostic probes or `STOP`. It is allowed at most two probes. A **fixed classical BayesianID** model makes the final calibration choice. The terminal reward is the original SafeCal net reward (calibration quality minus normalized diagnostic cost). Training uses REINFORCE with a learned value baseline and Adam. This is **not** RL fine-tuning of a language model and not an end-to-end learned calibration controller.

Training was precommitted to 720 CPU-only episodes, RNG seed `20261121`, with equal sampling from synthetic single and compound fault families (seeds 1000 to 300000 and 50000 to 500000). The final checkpoint was fixed before opening held-out results. No shifted deadzone dynamics were used for training. Training occurred in 96-episode sequential chunks using restored optimizer and RNG states. No cloud GPU was used.

## Matched held-out evaluation

60 fixed seeds per split. The Bayesian comparator is the prior two-probe, train-selected baseline. Positive differences indicate higher learned reward.

| Split | Neural RL + Bayes calibration | Bayes-2 | Difference | Paired bootstrap 95% CI |
|---|---:|---:|---:|---|
| Single faults | 0.84763 | 0.86167 | -0.01403 | [-0.03235, +0.00333] |
| Combined faults | 0.89186 | 0.87811 | +0.01375 | [+0.00617, +0.02130] |
| Deadzone dynamics | 0.78962 | 0.78425 | +0.00536 | [-0.00289, +0.01341] |

**However, the RL policy chose `pulse_neg_low` in 60/60 episodes on all three splits (180/180 total), and its reward was numerically identical to a fixed one-probe policy using the same classical posterior and calibration selector.** Thus the difference vs Bayes-2 is caused by a *lower-cost fixed testing choice*, not by intelligent observation-dependent action selection. We make **no claim of RL improvement** over the appropriate fixed-probe comparator.

Net reward includes diagnostic test cost. The learned policy spent 1.00 cost units per task; Bayes-2 averaged about 2.35 units. The positive combined-fault difference has a paired interval excluding zero, but its mechanistic interpretation is an inexpensive, static probe.

## Reproduce

```bash
python -m pip install -e '.[bayes,rl]'
python -m unittest discover -s tests -q
# Recreate the frozen training checkpoint (no held-out evaluation yet):
python train_probe_rl.py --episodes 720 --seed 20261121 --train-only --output rl_pilot_20261121
# Load the checkpoint, run evaluation once:
python train_probe_rl.py --evaluate rl_pilot_20261121/policy.pt --output rl_pilot_20261121 --eval-tasks 60
# Recreate the classical comparison files on the same 180 evaluation seeds:
python benchmark.py --tasks 60 --split heldout --policies default bayes2 --json results_bayes2_heldout.json
python benchmark.py --tasks 60 --split combined --policies default bayes2 --json results_bayes2_combined.json
python benchmark.py --tasks 60 --split shifted --policies default bayes2 --json results_bayes2_shifted.json
# Audit the fixed-probe control and calculate paired uncertainty intervals:
python audit_rl.py
```

The training command above is equivalent to sequential 96-episode resumed blocks, provided the same PyTorch environment. CPU floating-point and random generator details may introduce small cross-platform differences. `analysis.json` records exact paired intervals, bootstrap seed, costs, action histograms and per-split metrics.

## Scientific interpretation and next experiment

The current environment permits a single probe to be sufficient for many tasks. A policy that never adapts can already score well. Before requesting expensive 1B-3B language-model RL training, we should design **predeclared** task families where optimal second measurements depend on first observations and baseline posterior uncertainty; run identifiability checks, robust Bayesian value-of-information stopping baselines, and train/evaluate separate seeds. These should be justified by realistic diagnostic ambiguity rather than adjusting rewards to favor RL.

Existing Verifiers v1 integration passed [real installed package CI smoke](https://github.com/Venkat-RJ/openresearch/actions/runs/44873170003). That tested taskset loading and host-run interactions with scripted actions, not true model inference. A separate small-language-model smoke is in progress.

**Limits:** no real-world hardware data, no sim-to-real testing, no language-model RL, no safety certification, one policy training seed, 60 tasks per held-out split.