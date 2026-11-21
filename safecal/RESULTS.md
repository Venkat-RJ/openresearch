# SafeCal results, November 2026

These are **classical-policy simulation results**, not reinforcement-learning results. All means are over 60 seeded tasks per evaluation split. Higher terminal net reward is better. `net reward = frozen calibration quality - 0.012 * diagnostic cost`.

| Policy | Held-out single | Held-out combined | Held-out shifted |
|---|---:|---:|---:|
| Factory no-test | 0.84053 | 0.79321 | 0.64360 |
| Random, four probes | 0.83333 | 0.74069 | 0.61665 |
| Fixed diagnostic schedule | 0.85070 | 0.73545 | 0.60238 |
| Coarse Bayesian/variance greedy | 0.85546 | 0.74145 | 0.60838 |
| Continuous Bayesian-ID, four-probe cap | 0.85311 | 0.85051 | 0.76219 |
| Continuous Bayesian-ID, **train-selected two-probe cap** | **0.86167** | **0.87811** | **0.78425** |
| Privileged oracle **quality** (not a policy) | 0.91584 | 0.92050 | 0.82010 |

Approximate 95% seed-paired bootstrap intervals (4,000 resamples; not repeated model-training seeds):

| Comparison (net reward) | Single | Combined | Shifted |
|---|---|---|---|
| Bayes-2 minus default | +0.02114 | +0.08490 | +0.14065 |
| Bayes-2 minus coarse greedy | +0.00621, CI [-0.00460, +0.01622] | +0.13666, CI [+0.12434, +0.14888] | +0.17587, CI [+0.16532, +0.18573] |

Other interval values and exact summary are in `PUBLIC_RESULTS.json`; rerun `report_analysis.py` to regenerate. The oracle knows the hidden plant and optimizes over the same calibration grid. It is an upper-bound reference, not a fair deployable baseline.

## Design-selection discipline

We selected the Bayesian probe cap from 0, 1, 2, 3, 4 using 25 training single-fault and 25 training compound-fault seeds, with a macro-average over the two training groups. The **two-probe cap** had the highest mean net reward on these training tasks (0.86132). No evaluation seeds were used to choose that cap. The information-gain selector itself was not tuned on the evaluation splits.

## What changed from v0

The original coarse prior contained only isolated faults and performed poorly on compound faults. A continuous Bayesian particle prior containing compound hypotheses largely eliminated that failure. Reducing the diagnostic budget further improved net reward. This is an important *negative finding for a naive RL pitch*: better priors and stopping rules solve much of the problem without learning a neural policy.

The new shifted split includes a synthetic motor-torque deadzone absent from the particle prior. Despite this mismatch, the Bayesian baseline does well. That limits the remaining potential gain, and a more powerful model-based comparator could reduce it further.

## Limitations

- Synthetic normalized dynamics, no calibrated industrial plant or hardware validation.
- The shifted split is one invented deadzone mechanism, not broad out-of-distribution physics.
- The Bayesian policy approximates information gain with finite Monte Carlo samples and calibration selection with a small posterior representative set. It is **not** a globally optimal Bayes-adaptive controller.
- The reward coefficients, calibration grid, simulator faults, and allowed probes are provisional. No RL agent has been trained. No claim of generalization to real physical systems is warranted.
- Safety is a simulator-side constraint, not a validated hardware safety guarantee.

Reproduce with the README commands. Raw per-seed JSON is included in the full research archive and can be regenerated from seeded scripts.
