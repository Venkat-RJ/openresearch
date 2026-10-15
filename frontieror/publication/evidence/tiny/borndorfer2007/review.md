# Saved Borndorfer2007 tiny witness review

Reviewed on 12 October 2026 by a separate Codex agent.

The saved candidate passes a new arithmetic and graph validator against the pinned tiny input. Its weighted objective recomputes to **21,166.220504**, compared with **40,885.807478** for the saved reference. The reduction is **19,719.586974**, or **48.230885%**.

This is an internal review of saved artifacts. It is not outside replication and does not certify a global optimum.

## The concrete excluded passenger route

The saved candidate sends **31 passengers** from station **8** to station **17** on:

**8 → 14 → 19 → 17**

Its directed arcs are **[64, 106, 117]**, corresponding to undirected edges **[32, 53, 58]**. Their travel times are **46.07 + 30.13 + 59.69 = 135.89 seconds**.

The pinned reference source sets `K_PATHS = 10` at line 353. Its preceding comment identifies this as an inferred assumption absent from the paper. Reconstructing only its pure graph functions produces a tenth shortest route of **120.07 seconds** for this OD pair. A separate uniform-cost search over simple paths produces the same ten paths for every one of the instance's 13 OD pairs. The saved candidate route is strictly longer than the tenth, so its exclusion does not depend on a tie-breaking ambiguity.

The saved reference instead routes the same 31 passengers along **8 → 13 → 17**, taking **70.81 seconds**. The excluded route is longer. Its value comes from its place within a cheaper line plan, not faster passenger travel.

The candidate has two active bus lines:

| Saved line label | Ordered stations | Ordered edges | Frequency | Capacity per direction on each line edge |
| --- | --- | --- | ---: | ---: |
| 0 | 0 → 10 → 16 → 17 → 19 | 4, 38, 56, 58 | 1.2375 | 99 |
| 214 | 4 → 3 → 8 → 14 → 19 | 14, 16, 32, 53 | 1.5625 within floating-point rounding | 125 within floating-point rounding |

The highlighted passenger route follows line 214 from 8 to 19, then line 0 in reverse from 19 to 17. The model permits this interchange through arc capacities. It has no separate transfer penalty.

All saved candidate passenger paths have valid consecutive directed arcs, correct OD endpoints, no repeated stations and nonnegative flows. They satisfy every OD demand and the directed capacity constraints.

The highlighted route's arcs have the following **total loads across all passenger paths**, not merely the 31 highlighted passengers:

| Arc | Direction | Total passenger load | Supplied capacity |
| --- | --- | ---: | ---: |
| 64 | 8 → 14 | 89 | 125 within floating-point rounding |
| 106 | 14 → 19 | 89 | 125 within floating-point rounding |
| 117 | 19 → 17 | 89 | 99 |

Seven of the 13 saved candidate passenger paths fall outside their reference ten-path pools:

| OD pair | Saved candidate station sequence | Flow | Travel time, seconds | Tenth shortest time, seconds |
| --- | --- | ---: | ---: | ---: |
| 8 → 17 | 8 → 14 → 19 → 17 | 31 | 135.89 | 120.07 |
| 17 → 8 | 17 → 19 → 14 → 8 | 31 | 135.89 | 120.07 |
| 3 → 17 | 3 → 8 → 14 → 19 → 17 | 20 | 160.27 | 128.82 |
| 17 → 3 | 17 → 19 → 14 → 8 → 3 | 20 | 160.27 | 128.82 |
| 8 → 10 | 8 → 14 → 19 → 17 → 16 → 10 | 20 | 204.74 | 139.66 |
| 10 → 8 | 10 → 16 → 17 → 19 → 14 → 8 | 20 | 204.74 | 139.66 |
| 3 → 10 | 3 → 8 → 14 → 19 → 17 → 16 → 10 | 18 | 229.12 | 133.56 |

## What the objective comparison shows

The released tiny input uses `lambda = 0.9978`:

`objective = 0.9978 × (fixed line cost + frequency-weighted operating cost) + 0.0022 × passenger travel time`

| Recomputed component | Saved reference | Saved candidate |
| --- | ---: | ---: |
| Active lines | 4 | 2 |
| Fixed cost | 40,000 | 20,000 |
| Operating cost | 888.75 | 1,120 |
| Total passenger travel time, passenger-seconds | 39,551.24 | 42,129.32 |
| Weighted line cost | 40,798.79475 | 21,073.536 |
| Weighted passenger travel cost | 87.012728 | 92.684504 |
| Total weighted objective | 40,885.807478 | 21,166.220504 |

The candidate accepts **2,578.08 additional passenger-seconds**, costing **5.671776** in the weighted objective. Its weighted line cost falls by **19,725.25875**. These changes yield the net improvement of **19,719.586974**.

The reference's recorded `optimal` status concerns its restricted path-based model. It does not certify optimality over all legal passenger paths. The saved candidate cannot be represented in that model because seven of its active paths are absent from the reference pools.

This establishes a real restriction and a valid lower-cost witness. It does not isolate how much improvement would result from changing only the passenger-path pool while holding every other implementation choice fixed.

## Review method and provenance

The research directory was read only. A small standard-library validator was written in this article review directory. It checks:

- Simple terminal-to-terminal lines, mode membership, ordered node/edge connectivity and the four-edge maximum length.
- Finite nonnegative frequencies, the global frequency bound and every edge's frequency capacity.
- Complete directed passenger paths, finite nonnegative flow, all 13 OD demands, per-OD station flow conservation and all 120 directed arc capacities.
- The weighted objective directly from line and passenger decisions.

The saved reference passes the same independent checks. Line labels are treated as implementation-specific identifiers; physical station and edge sequences determine validity.

The new check uses Decimal arithmetic with an absolute tolerance of **1e-8**. The candidate objective residual is approximately **1.437e-12**. One saturated arc has a capacity excess of **2.88e-13**, consistent with the saved frequency `1.5624999999999964`. These are floating-point rounding residuals.

To reconstruct the reference shortlist, the validator extracts only `build_directed_arcs`, `dijkstra` and `k_shortest_paths` from the pinned source via its syntax tree. It never imports or executes that module, its optimization model or its `main` function. The second shortest-path computation is newly written uniform-cost enumeration of simple paths.

Dataset revision: **36c0406670b9d33eb35f0fbfdd987b4836b3aaa8**.

Evaluator revision recorded in the saved manifest and observed repository HEAD: **9bd047c9271d3048eff91dd0639941d0bc182d86**.

The benchmark files' contents match their Hugging Face download metadata at the dataset revision, using SHA-256 for LFS objects or Git blob SHA-1 for regular files. The frozen candidate source's SHA-256 matches the saved final300 manifest. The evaluator repository had no reported local changes during this review.

The full absolute artifact paths, SHA-256 hashes, metadata checks and final300 manifest entry are in [review.json](review.json). The validator is [independent_review.py (release note)](../../../../RELEASE-NOTES.md#additional-source-materials).

## Figure-ready evidence

The `figure_data` object in `review.json` includes all **20 station coordinates**, all **60 physical edges**, both saved line plans, the highlighted excluded passenger path, the saved reference path for the same OD pair, and its reconstructed ten-path shortlist.

Use the pinned coordinates, keeping any display inversion of the vertical axis consistent. Highlight station 8 as the origin, station 17 as the destination and station 19 as the interchange. Draw the two candidate bus lines and the route 8 → 14 → 19 → 17 above a subdued physical network. A second route, 8 → 13 → 17, can illustrate a faster path allowed by the reference shortlist.

Suggested caption:

> Actual stations and edges from the pinned tiny instance. The saved candidate carries 31 passengers along 8 → 14 → 19 → 17 in 135.89 seconds. This legal route is absent from the reference's ten shortest paths, whose tenth takes 120.07 seconds. The candidate's two-line plan accepts longer passenger travel while reducing fixed line costs. This is an illustrative reconstruction from saved decisions, not an animation of the optimizer.

## Precise source links

- Original paper: [A Column-Generation Approach to Line Planning in Public Transport](https://doi.org/10.1287/trsc.1060.0161).
- [Pinned formulation](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/borndorfer2007/mathematical_formulation.md).
- [Pinned problem description](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/borndorfer2007/problem_description.txt).
- [Pinned reference source](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/borndorfer2007/gurobi_code.py), especially lines 351 to 353.
- [Pinned tiny input](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/borndorfer2007/instance/tiny_instance.json).
- [Pinned reference witness](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/borndorfer2007/gurobi_solution/tiny_solution.json).
- [Pinned checker](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/borndorfer2007/feasibility_check.py).
- Saved candidate: [solution_tiny.json](candidate.json).
- Frozen candidate source: [code.py](candidate.py).

Public URLs above are constructed from the pinned repository paths. Their contents were verified locally against download metadata; HTTP availability was not tested in this bounded audit.

## Limits to preserve in the article

No solver, official feasibility checker, benchmark evaluator, campaign rerun, large instance or live Trackinizer access was used.

The pinned formulation's displayed objective is unweighted, while its problem description, instance parameter, reference code and checker implement the weighted objective used above. The article should identify this as the released benchmark objective rather than claim the displayed formula and implementation are identical.

The reconstructed shortlist comes from the pinned source. It is not a retained execution trace of the exact reference path pool. The source reconstruction and independent shortest-path check agree, and the seven excluded routes are all strictly longer than their tenth-shortest cutoff.

This establishes saved-certificate feasibility, objective consistency and specific reference restrictions. It does not establish a new algorithm, outside replication or unrestricted global optimality.
