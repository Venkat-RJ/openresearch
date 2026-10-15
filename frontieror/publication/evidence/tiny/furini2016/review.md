# Furini2016 saved witness review

Reviewed on 12 October 2026.

The archived final witness is exactly **400 copies of item 1, each 1 × 4**, worth 4 per piece.

An independently written validator expands its aggregate cut counts into a valid physical guillotine tree with **399 cuts, 400 finished pieces, profit 1600, and zero waste**.

This is a separate Codex review of a saved witness. It is not outside scientific replication. No optimization solver, official feasibility checker, benchmark evaluator, or campaign was executed during this review.

## The actual saved pattern

The witness stores 49 plate types and 48 cut types, with multiplicities. These are types, not 49 physical plates or 48 physical cuts.

The saved items list has one entry:

```json
{"item_id":1,"plate_id":3,"plate_dims":[1,4],"copies":400}
```

The following schedule is reconstructed from the exact saved cut multiset. The witness does not record a chronological cutting log. Several valid execution orders exist.

1. Start with one 40 × 40 panel.
2. Cut vertically at local position 1. This produces a 1 × 40 strip and a 39 × 40 remainder.
3. Repeat on the remainder: 39 × 40 becomes 1 × 40 plus 38 × 40, then 38 × 40 becomes 1 × 40 plus 37 × 40, and so on.
4. The last 2 × 40 remainder is cut at position 1 into two 1 × 40 strips. There are now **40 strips**, obtained by **39 vertical cuts**.
5. Take any 1 × 40 strip and cut horizontally at local position 4. This gives a 1 × 4 piece and a 1 × 36 remainder.
6. Continue cutting the remainder at position 4: widths 36, 32, 28, 24, 20, 16, and 12 each yield one 1 × 4 piece.
7. Cut the final 1 × 8 remainder at position 4 into two 1 × 4 pieces. Each strip yields **ten pieces through nine horizontal cuts**.
8. Repeat steps 5 to 7 for every strip. This adds **360 horizontal cuts** and yields **400 pieces** in total.

The dimensions follow the benchmark's orientation convention: `v` splits length, and `h` splits width.

For a figure, draw the 40 × 40 panel with 40 columns of width 1 and ten rows of height 4. In a coordinate system where x is length and y is width, the finished rectangles are `[x,x+1] × [4k,4k+4]` for x = 0,...,39 and k = 0,...,9.

The global vertical cut coordinates are x = 1,...,39. In each strip, the global horizontal cut coordinates are y = 4,8,...,36. Each horizontal cut runs across its current strip remainder. A line across the entire original panel would describe a different cut schedule.

## Arithmetic and optimality

| Quantity | Recomputed value |
|---|---:|
| Item 1 copies | 400 of 400 allowed |
| Item 2 copies | 0 of 88 allowed |
| Item 3 copies | 0 of 76 allowed |
| Item 4 copies | 0 of 38 allowed |
| Item 5 copies | 0 of 22 allowed |
| Profit | 400 × 4 = 1600 |
| Selected area | 400 × 1 × 4 = 1600 |
| Original panel area | 40 × 40 = 1600 |
| Waste | 0 |
| Physical cuts | 39 + 40 × 9 = 399 |

Every item type in the pinned input has profit equal to area. Consequently, every feasible non-overlapping cutting pattern has profit at most the panel area, 1600. This physically valid saved witness attains that bound, so **1600 is the global optimum of this pinned tiny instance under the stated general guillotine model**.

This conclusion does not establish general optimality of the generated dynamic program, performance on large instances, or a new runtime result.

## Exact midpoint exclusion

The pinned reference source contains these expressions at lines 102 and 103:

```python
h_pos = sorted(q for q in h_all if 2 * q < plate_w)
v_pos = sorted(q for q in v_all if 2 * q < plate_l)
```

For even dimensions, strict `<` excludes an exact midpoint. Removing symmetric complementary cut positions should preserve a midpoint because it is its own complement.

The archived 1600 witness requires two midpoint cut types:

| Saved cut | Multiplicity | Equality excluded by the reference |
|---|---:|---|
| 2 × 40 → 1 × 40 + 1 × 40, vertical q = 1 | 1 | 2 × 1 = 2 |
| 1 × 8 → 1 × 4 + 1 × 4, horizontal q = 4 | 40 | 2 × 4 = 8 |

Both midpoint positions are valid positions in the archived mathematical formulation, which allows q = 1,...,dimension−1. The generated frozen source uses `range(1, dim//2+1)` for this 40 × 40 instance, retaining midpoint positions.

The restriction therefore excludes the displayed valid 1600 pattern from the reference's cut enumeration. The saved reference witness independently expands to a valid physical pattern with profit 1592 and 8 units of unselected area. Its stored `optimal` status refers to the reference's enumerated model, not to a newly verified unrestricted optimum.

The observed difference is 1600 − 1592 = 8. This review did not run a modified reference solver, so it does not experimentally isolate the contribution of changing that one filter to the archived difference.

## What the checker result proves

The archived final checker result says `feasible: true`.

The pinned checker aggregates selected items, checks their bounds and dimensions, computes flow balances from the supplied cut counts and plate IDs, and recomputes the objective.

Its cut-flow block trusts the supplied parent and child IDs. It does not compare the supplied cut coordinates or child dimensions against the parent geometry. Its x-integrality section explicitly performs no cut-count check. The public solution schema lists only `objective_value` and `items_selected`, although this saved witness supplies the extra cut data required for a stronger review.

The independent validator adds explicit checks for positive integer cut counts, valid cut positions, correct child geometry, root dimensions, plate classification, copy bounds, flow balances, and physical reconstruction. It checks that all archived cut copies and selected items can be consumed in a tree rooted in exactly one panel. It checks the expanded finished rectangles for containment and pairwise non-overlap.

Both the final and reference saved witnesses pass these checks. Five deliberately corrupted copies are rejected: a wrong cut coordinate, a missing cut copy, a negative cut count, a false objective, and an exceeded item bound. These checks do not mutate the archived files.

## Provenance

Research root: `FROZEN_CAMPAIGN`.

The final300 manifest identifies evaluator revision `9bd047c9271d3048eff91dd0639941d0bc182d86`, dataset revision `36c0406670b9d33eb35f0fbfdd987b4836b3aaa8`, and frozen generated source SHA-256 `d36dfe460a2b3895fa2215071e0e05295305e016f86c89fa5450283f0f564671`. The computed generated-source hash matches that manifest entry.

The local Hugging Face cache metadata records the same dataset revision for the input and source. The input content SHA-256 matches the cache metadata's content digest.

Revision links are constructed from the local release metadata and the repository's documented dataset location. The web tool could not retrieve the pinned Hugging Face pages during this review, so online retrieval of these links was not independently confirmed.

- [Pinned tiny input](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/furini2016/instance/tiny_instance.json)
- [Pinned formulation](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/furini2016/mathematical_formulation.md)
- [Pinned reference source](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/furini2016/gurobi_code.py)
- [Pinned checker](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/furini2016/feasibility_check.py)
- [Pinned reference witness](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/furini2016/gurobi_solution/tiny_solution.json)
- [Evaluator revision](https://github.com/Minw913/FrontierOR/tree/9bd047c9271d3048eff91dd0639941d0bc182d86)

| File | SHA-256 |
|---|---|
| Saved final witness | `bb59961bd6e0cc8c7ec002b552a9e0ff195553042133c178cc113524126ce14b` |
| Frozen generated source | `d36dfe460a2b3895fa2215071e0e05295305e016f86c89fa5450283f0f564671` |
| Tiny input | `e157c9f43089641e90198adc13cc32f45702c00962f0cb3b581021ab3fcfdbf6` |
| Formulation | `79158ce9399d18468cf98b2dd7cc7dadadac9c21487a9ae3adfdb12c16595f75` |
| Reference source | `61256cbe45272e78058c1c731ebc601d0addad3c91cbc9d99fc9682ab54a921f` |
| Reference witness | `43fb60c04a7ed4fead3c7d1e69307c069ed974f5b5232191a6cd04dd23812ab6` |
| Checker | `e1e5b9327fb48add1d3c20d8d7a8f153a5aa2ab92875122052947707a4e7ed15` |
| Output schema | `79b65a1d0aad313b1694f7ea7b83bc0efd11b7bba7834556baecf2c936092177` |

`review.json` contains the complete input/source/witness hash inventory and the per-plate accounting. `figure_data.json` contains all 400 item rectangles, all 399 physical cuts, an expanded tree, and the two-phase cutting schedule. `independent_validator.py` reproduces the arithmetic and geometric validation using Python's standard library only.

## Suggested article wording

> The saved Furini2016 witness contains 400 pieces of item 1, each 1 × 4. A separate Codex review reconstructs a guillotine sequence with 39 cuts to form 40 strips, then nine cuts per strip to form ten pieces. Its 1600 profit equals the 40 × 40 panel area, an upper bound because every item's profit equals its area. The pinned reference code uses `2 * q < dimension`, which excludes the exact midpoint cuts needed by this saved pattern. The archived reference value is 1592. This is a review of one saved tiny-instance witness, not outside replication or a new benchmark run.
