# Final synthetic evaluation: independent audit

Verified on 2026-10-09 06:11 IST.

Frozen v2 achieved lower mean total weighted tardiness on 29 of 30 cases. The historical EGRASP executable achieved the lower mean on the remaining case. Each mean uses three repetitions of one implementation on one instance.

All 180 planned executions are preserved: 90 for each implementation. All 180 returned valid schedules captured within the external deadline. There were no failed executions or zero objectives in this final study. The earlier development study's failures and zero ties remain preserved in their original records.

## Design and analysis

Thirty custom synthetic cases cover six balanced difficulty/weight strata. Both implementations requested 290 seconds internally and had a 300-second external deadline in the same emulated Linux x86 image, with one CPU quota and 512 MiB. The CPU quota did not isolate the shared host. Equal deadlines do not establish equal CPU work. V2 seeds were recorded; EGRASP's seed controls are undocumented, so the repetitions are not seed-matched.

The registered primary statistic averages nine cross-repetition comparisons within each instance, then averages across the 30 instances. A valid schedule beats a failure, two failures tie, and a lower valid objective wins. The mean statistic was 0.925926, with a registered 95% stratified instance-bootstrap interval of [0.792593, 1.000000]. The registered advantage criterion passed. Those 270 cross-repetition comparisons are dependent comparisons formed from 180 actual runs, not 270 independent observations.

The registered secondary statistic is R = (B - A) / max(1, A, B), where A is the mean of v2's three objectives and B is the mean of EGRASP's three objectives on one instance. Averaging R across the 30 instances gave a **9.66% mean normalized objective improvement**, with a 95% interval of **8.39% to 10.97%**. This is not an ordinary percentage reduction relative to EGRASP on every case. Bootstrap resampling uses whole instances within the six strata, 10,000 resamples and seed 2026100802.

## Independent checks

The prepared auditor verified all 180 registered slots, source and input hashes, recorded execution arguments, observed deadlines, captured output bytes and input translations. It reran the official checker and independent schedule validator, recomputed the frozen analysis and independently checked the primary statistic and valid counts. No execution was dropped, replaced or retried by this audit.

The completed Trackinizer experiment outcome matched the saved results. The conclusion was linked to the experiment, and its full analysis JSON matched the audited analysis. Per-attempt progress commitments were checked. No database export was used.

- [Audited summary](summary.json)
- [All 180 verified execution records](verified-executions.json)
- [Auditor source snapshot](audit-source.py)
- [Experiment readback](../experiment-readback.json)
- [Conclusion readback](../conclusion-readback.json)
- [Completed Trackinizer experiment]([private-endpoint-removed]
- [Stored final conclusion]([private-endpoint-removed]

To repeat the audit from the campaign root, choose a new output directory:

```sh
python3.13 pilot/audit_final_comparison.py campaign/comparator-setup/final-current.json evidence/NEW-FINAL-AUDIT-DIRECTORY
```

Original evidence: `evidence/final-comparison-1791453882475673000`.
Audit command and exit-zero receipt: `evidence/final-independent-audit-launch-20261009T003734Z-1791506254068933000`.

## Scope and attribution

This supports v2 against the historical EGRASP implementation on the preregistered custom distribution under these execution conditions. It establishes neither optimality nor current state-of-the-art performance, algorithmic novelty or superiority across FrontierOR tasks.

Sagent developed the v2 algorithm. Codex implemented the registered comparison infrastructure and performed this independent audit. Trackinizer stored the experiment, evidence commitments and conclusion. ChatGPT advised methodology; Venkat authorized the research.

The autonomous Sagent research worker remains stopped after the previously reported provider overload. The separate final evaluation has completed; its runner cleaned up its Docker container through its existing teardown. This audit did not restart either worker or modify frozen implementations, the protocol or the environment.
