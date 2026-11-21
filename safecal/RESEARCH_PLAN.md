# SafeCal research plan

**Question:** When does RL help with safe diagnostic experiment selection and recalibration beyond strong classical system-identification and experimental-design methods?

## Starting point

A working synthetic actuator and frozen evaluator; original coarse model and a stronger Bayesian particle baseline; a train-only selected two-probe cap; 60 held-out tasks for each of three current splits. **Classical Bayesian methods already close much of the original gap.** No neural policy has been trained. See `RESULTS.md`.

## Falsifiable experiments

1. **Validate the task and shields.** Keep hidden plant generation and scoring host-side; verify no response leaks hidden parameters, scoring internals, or seeds into the agent sandbox. Reject unsafe and over-budget actions independently of the model. Record calibration quality, probe count, cost, and rejected actions separately.
2. **Establish competitive classical baselines.** Strengthen the current particle model with a decision-focused expected value-of-information policy, classical system identification (MAP/least-squares), conservative safe probing, and a budget-aware policy. Tune only on training seeds and freeze evaluation.
3. **Measure model misspecification.** Beyond current single, compound and deadzone tasks, create separate held-out *model classes* such as delayed actuation, stick-slip friction and asymmetric load, with identifiability checks. Do not call them realistic without independent physical validation. A blind simulator judge evaluates held-out trajectories.
4. **Train an RL policy if the task remains challenging.** Use `safecal_v1` with Prime Intellect Verifiers v1 and isolated execution. Compare open-weight 1B-3B model zero-shot tool use against RL-trained versions; report actual GPU-hours, rollouts and tokens. Use multiple seeds. Do not claim an RL gain until trained and measured.
5. **Measure causal improvement.** Primary metric: seed-paired net calibration reward at fixed maximum diagnostic cost. Secondary: safety violations/blocked actions, number of experiments, quality at equal excitation, sample efficiency, and transfer to held-out dynamics. Compare with the strongest classical baseline. Report negative findings.

## Success and failure criteria

RL is useful only if it improves over the best tuned **non-privileged** classical method with uncertainty estimates that support the claim, or demonstrates a specific transferable advantage unavailable to a simpler method. Gains must survive held-out models, equal decision budgets, and reward-audit checks.

Stop or publish negative results if classical baselines tie or win, gains depend on seed leakage or evaluator shortcuts, tasks are unidentifiable under available probes, or the learned policy frequently requests invalid actions.

## Safety and ownership

The simulator is a research abstraction. Industrial liftgate calibration inspired the question, but this project contains no confidential data or control procedures from an employer. No policy should be deployed on hardware without separately validated models, safety mechanisms and domain-specific testing.
