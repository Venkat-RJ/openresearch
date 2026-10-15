# FrontierOR results and unresolved blockers

Checkpoint: 11 October 2026 at 09:24:59 IST.

The user chose to stop the completion attempt, retain the results and document the blockers for a future return. No benchmark-owned orchestrator, evaluator or solver is running. The 30-minute monitor is paused. Resume only after a new user request.

The benchmark remains incomplete. This checkpoint is a preserved research result with explicit missing work, not a final benchmark completion or final audit.

## Results preserved

| Metric | Result |
| --- | --- |
| Canonical tasks | 180 |
| Planned large cases | 900 |
| Recorded large cases | 883 |
| Official large passes | 679 |
| Recorded large failures | 204 |
| Missing large result slots | 17 |
| Tasks passing all five large cases | 111 |
| Tasks passing all five with matching current sources | 110, excluding the separate pilot |
| Recorded small cases | 180 |
| Official small passes | 179 |
| Large results matching current main and helper hashes | 871, including 674 passes |
| Large results from earlier solver versions | 7 |
| Large results from the preserved pilot | 5 |

The large-case failure breakdown is 98 objective gaps exceeding the official threshold, 80 runtime errors, 23 infeasible solutions and three invalid outputs. Failures remain in the denominator.

The recorded objective comparison has 207 strictly better decimal values, 168 exact ties, 402 worse values and 106 unavailable comparisons. Applying the separate numerical tolerance gives 181 improvements, 231 numerical ties, 365 worsenings and 106 unavailable comparisons. All 181 improvements beyond tolerance match the current solver sources.

There are 29 improvements beyond tolerance over references labelled optimal. These remain investigation flags. An official pass allows up to a 10 percent objective gap and does not itself mean the reference was beaten. Recorded time versus published reference time is not a controlled speed comparison.

## The 17 missing large cases

| Task | Missing instances | Blocker | Work to consider when returning |
| --- | --- | --- | --- |
| castro2021 | large_1 to large_5 | Very large inputs need serial evaluation. Four inputs have not been downloaded. Previous large_2 admission lacked memory headroom. | Measure whole-pipeline memory, verify full pinned inputs and available disk/download budget, then evaluate serially with the existing memory reserve. |
| fragkos2021 | large_1 to large_5 | Memory admission deferred the first case. Its public checker materializes a full 100 million entry cost matrix. Four inputs have not been downloaded. | Use adequate machine memory and a measured job budget. Streaming solver input alone does not reduce the unchanged checker's peak memory. |
| laporte2003 | large_1 to large_5 | Small-case gate fails. The schema, checker and reference have an unresolved cost-indexing inconsistency. | Reconcile the intended contract before changing the solver, checker or reference. Revalidate the small case before any large evaluation. |
| martello1999 | large_5 | An earlier full evaluation exceeded a 4.25 GiB limit, with measured peak RSS about 4.53 GiB. The later 5 GiB request was not admitted. | Measure sufficient available memory for the complete evaluator and checker, then retry with a supported higher budget and the same gate contract. |
| taninmis2022 | large_5 | Fresh small validation now passes. The new large_5 attempt exceeded its 2 GiB job limit and produced no official CSV row. | Inspect the captured attempt and establish an adequate admitted whole-pipeline budget before retrying. |

These are missing result slots, not invented failure rows. Resource and admission records remain separate from official CSV results. The fixed memory reserve is 2 GiB. Do not remove it or terminate unrelated applications to force a run.

## What the brief restart changed

The completion request authorized a restart before the user chose to stop. Two evaluations were launched for `taninmis2022` with unchanged solver and harness sources.

The genuine fresh small case passed at 09:22 IST. Its objective was 15128802.133861, gap 0.000088 and recorded solver time 18.44 seconds. Its saved source hash now matches the current source, so the stale small-validation blocker has been cleared.

The large_5 attempt ended at 09:23:38 IST. The owned process tree reached 2198241280 bytes against a 2147483648 byte cap. The existing supervisor terminated that tree and recorded `memory_exceeded`, return code -15. No official result row was produced. The cause of that measured peak has not been isolated to a solver or checker phase.

There are therefore still 883 recorded large results and 17 missing slots. No new large pass or objective improvement was obtained during this restart. No solver source, public checker, scoring reference, harness or orchestrator file was changed. Ordinary evaluation records and the manifest were updated by the existing orchestrator.

## Laporte contract evidence

The public schema defines purchase-cost keys as `product,market`. The current solver uses that convention. The public checker parses those same keys as `market,product` and treats missing reversed costs as zero in its objective calculation.

For the saved tiny solution, routing cost is 28. Purchase cost under the documented convention is 43, making the reported objective 71 consistent with that convention. The checker instead computes purchase cost 394 and total objective 422, rejecting the solution. Twenty-five positive purchases have no reversed cost key.

Under the documented convention, buying each product at its cheapest available market gives a purchase-only lower bound of 36. The published tiny reference is 18 and labelled optimal, which cannot be reconciled with that bound and nonnegative routing cost. This establishes inconsistency among the benchmark files; it does not establish which component should be corrected.

Preserve this as a benchmark data-quality finding. Do not change the reported objective to exploit the inconsistency or present a checker-compatible score as a scientific improvement before the intended semantics are reconciled.

## Value retained

The strongest independently checked examples remain graph colouring and hyperplane clustering. The checkpoint audit checks the actual saved witnesses and the current source hashes.

For `alfieri2020`, the five colour counts are 13, 12, 20, 27 and 13 against references 16, 14, 20, 31 and 16. Every vertex and edge check passes: four incumbent improvements and one tie.

For `amaldi2013`, the five plane counts are 2, 3, 5, 4 and 5 against references 2, 3, 6, 5 and 7. Complete coverage, assignments, plane normalization and distance checks pass: three incumbent improvements and two ties. The last case is now a matching current-source result, improving the earlier six-plane output in the October 10 report.

These support valid improvements over particular published incumbents. They do not yet establish a new algorithm, current state of the art or performance on untouched confirmation instances. Existing methods include DSATUR, CP-SAT, SVD fitting and set cover. Ablations and specialist comparators would be needed to establish why an approach improves on a baseline.

## How to return later

1. Read this handoff and `paused-decision.json`. Confirm that the user wants to resume.
2. Recheck processes, memory, disk, source hashes and the live manifest. This checkpoint may be stale when revisited.
3. Compare against the immutable checkpoint instead of overwriting it. Keep all failed and earlier-version records.
4. Address the chosen blocker. Keep solver work separate from an investigation of benchmark contract defects. Laporte needs contract reconciliation first.
5. Use the existing orchestrator with explicit tasks, one worker for huge inputs, pinned full inputs, the existing 60-second solver limit, and measured memory admission. Any changed source, helper or harness contract needs a fresh matching small gate before large cases.
6. Inspect native logs, official rows, return codes and measured RSS after each attempt. A completed process or a written solution is not an official pass.
7. Reconcile all 1080 planned slots, current and earlier sources, the preserved pilot, reference access and optimal-reference flags before claiming full benchmark completion. The full final audit is still outstanding.

No automatic retry or future solver run is scheduled by this handoff.

## Evidence locations

Campaign: `research-workspace/Personal_experiments/frontieror-attempt`.

Frozen checkpoint: `research-workspace/Documents/ChatGPT/rekursive_tesitng/frontieror-interim-report-20261010/snapshots/20261011-parked`.

The checkpoint contains the manifest, aggregate CSV, main and helper source copies, current comparison tables, every optimal-reference flag, recent logs, ten independently audited witnesses and a SHA-256 file inventory. It is a checkpoint reconciliation and focused witness audit, not the full campaign final audit.

Evidence archive: `research-workspace/Documents/ChatGPT/rekursive_tesitng/frontieror-interim-report-20261010/snapshots/20261011-parked-evidence.zip`.

This folder contains the before-restart snapshot, `before-sha256.json`, `authorization.json`, `paused-decision.json`, `laporte-contract-evidence.json`, the two new Taninmis run records and driver logs. The full native run directories remain under `orchestration-runs/taninmis2022`.

Report: https://chatgpt.com/space/page_7d842b992b148191916f280ae7bae7fb.
