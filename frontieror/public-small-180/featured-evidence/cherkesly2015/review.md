# Cherkesly saved-witness review

Reviewed on 12 October 2026 by a separate Codex agent.
This is an independent arithmetic implementation within the same authoring workflow.
It is not replication by an outside researcher.
No optimization solver, official evaluator, Trackinizer query or campaign rerun was used.
All research inputs were read only.

The saved candidate legally serves all 30 requests using ten routes under the pinned task's route semantics.
The full objective recomputes to **102,616.4**.
The saved reference recomputes to **303,868.0** across 30 one-request routes.
The cost reduction is **201,251.6**, or **66.22994195%**.

This is primarily a correction of the reference search space.
The reference source chooses 30 vehicles and requires every one to operate.
The task description allows a chosen number of identical vehicles.
The result does not demonstrate an improvement over the original paper's algorithms.

## Checks completed

The validator checks every pickup and delivery appears exactly once, each pair is on the same vehicle, route endpoints and internal depots, unique vehicle IDs, and every used directed arc against the pinned input.
It replays exact cumulative loads and a single linear stack.
It checks each delivery removes the top request, capacity and the node-specific load bounds, and empty load and stack at return.

It reconstructs the earliest feasible service start using each node's service duration and each saved arc's travel time.
It checks all time windows and each pickup's direct-travel minimum precedence to its delivery.
It also checks the saved route details' times and loads rather than accepting the reported totals.
The arc-cost objective is independently summed with decimal arithmetic.

Both witnesses pass these checks with no violations.
The candidate has 60 request-node visits, 80 saved detail rows, maximum load 8 against capacity 22, and maximum stack depth 2.
The reference has 60 request-node visits and 120 saved detail rows.
No optimality proof for the ten-route witness was produced.

## Why the reference requires 30 routes

The pinned `gurobi_code.py` sets `K_size = n` on lines 119 to 122.
Here `n = 30`.
Lines 194 to 200 require one departure from the origin for every modeled vehicle.
Lines 213 to 219 require one return for every modeled vehicle.
The input has no direct origin-to-destination empty-route arc.
Origin arcs lead to pickups, and every pickup is visited exactly once.
Consequently, all 30 vehicles must serve a pickup and all 30 requests are split across 30 vehicles.
The saved reference indeed contains 30 routes, each serving one request.

The pinned formulation calls the vehicle set unrestricted, while its departure and return equations refer to the vehicles that are chosen to operate.
The checker iterates over the submitted routes and does not require 30 of them.
The problem description explicitly allows any number of vehicles.
Ten used routes are therefore legal for the described benchmark task and checked witness semantics, although they cannot be represented by the reference source's forced 30-vehicle model.

| Cost component | Saved reference | Saved candidate | Reduction |
|---|---:|---:|---:|
| Vehicle charge | 300,000 | 100,000 | 200,000 |
| Remaining arc cost | 3,868.0 | 2,616.4 | 1,251.6 |
| Total | 303,868.0 | 102,616.4 | 201,251.6 |

## Stack correction needed in the article

The pinned Cherkesly task defines **one linear LIFO stack per vehicle**.
There is no multi-stack count or stack-assignment variable in this instance, schema, formulation, candidate or checker.
The publisher abstract for [the original paper](https://doi.org/10.1287/trsc.2014.0535) also describes a linear stack loading structure.
Replace the draft's statement about "multiple stacks" with one stack.
The stack states in `review.json` are inferred by replay of the saved route, not an explicit stack-assignment field from the solver.

## What the frozen candidate code actually does

It begins with one legal pickup-delivery route for each request.
It builds new solutions by inserting a request's pickup and delivery into existing routes wherever the total extra cost is lowest and the entire resulting route remains legal.
When no cheaper insertion works, it starts a new route.
The first construction orders requests by pickup deadline; seven more constructions randomize request order.
Later iterations remove a random set of 8 to 15 requests from the best saved routes and reinsert them.
Every route trial is checked for legal arcs, time windows, load bounds and strict LIFO order.
It saves improving full solutions until the time budget expires.
This is a constructive insertion heuristic with randomized removal and reinsertion.
It is not an exact algorithm or an optimality proof.

## Figure-ready saved route

Use vehicle **5**, which serves requests 28, 30, 5 and 4.
Its full saved sequence is:

`0 -> 28 -> 58 -> 30 -> 60 -> 5 -> 4 -> 34 -> 35 -> 61`

Nodes 0 and 61 are the same depot position, represented twice for departure and return.
Coordinates are abstract instance coordinates, not a geographic map.
Draw pickup nodes and delivery nodes differently, connect visits in the saved order, and highlight the four-event nested loading sequence near the end.
Do not invent roads or portray straight segments as real driving paths.

| Node | Event | x | y | Service starts | Time window | Load after | Stack bottom to top |
|---|---|---:|---:|---:|---|---:|---|
| 0 | Depart depot | 50 | 50 | 0.0 | 0 to 600 | 0 | empty |
| 28 | Pick up request 28 | 8 | 78 | 72.0 | 72 to 132 | 4 | 28 |
| 58 | Deliver request 28 | 78 | 59 | 155.0 | 155 to 177 | 0 | empty |
| 30 | Pick up request 30 | 42 | 69 | 202.4 | 171 to 231 | 7 | 30 |
| 60 | Deliver request 30 | 35 | 80 | 225.4 | 216 to 276 | 0 | empty |
| 5 | Pick up request 5 | 30 | 52 | 440.0 | 440 to 500 | 1 | 5 |
| 4 | Pick up request 4 | 12 | 73 | 477.7 | 436 to 496 | 8 | 5, 4 |
| 34 | Deliver request 4 | 5 | 73 | 494.7 | 481 to 539 | 1 | 5 |
| 35 | Deliver request 5 | 4 | 64 | 513.8 | 485 to 541 | 0 | empty |
| 61 | Return depot | 50 | 50 | 571.9 | 0 to 600 | 0 | empty |

The route costs 10,293.7, including one 10,000 vehicle charge.
The highlighted nested example picks up request 5, then request 4 on top, unloads request 4 at node 34, and finally unloads request 5 at node 35.
All ten route sequences, all 62 input nodes, their coordinates and per-visit stack states are in `review.json`.

Suggested figure caption:

"A route from the saved ten-vehicle solution, drawn in benchmark coordinates. Request 4 is loaded after request 5 and delivered first, preserving the single-stack LIFO rule. Combining legal requests into shared routes removes vehicle charges. The saved reference uses 30 vehicles because its source forces every modeled vehicle to operate."

## Evidence and hashes

Dataset revision: `36c0406670b9d33eb35f0fbfdd987b4836b3aaa8`.
Evaluator revision: `9bd047c9271d3048eff91dd0639941d0bc182d86`.
The five pinned public files below were fetched over HTTPS and their complete SHA-256 hashes matched the local files on 12 October 2026.
This readback verifies input and source provenance; it does not rerun either solver.

| Evidence | SHA-256 |
|---|---|
| [Tiny input](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/cherkesly2015/instance/tiny_instance.json) | `a2f90d7f68bdc84e641876537644336549dc866ce434eb6a9a7d48dd4741747d` |
| [Benchmark formulation](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/cherkesly2015/mathematical_formulation.md) | `f73b2061c75113681d8a55495bbd45c48a469c08a55b4c4ebce419ec31e358dc` |
| [Reference source](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/cherkesly2015/gurobi_code.py) | `94051e8c26224f152721c1fd185eb5bc2146323b1c991bd2664631cb2a3171a1` |
| [Pinned checker](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/cherkesly2015/feasibility_check.py) | `a8dbdaa34677a290aab242616e5b9e5f5d4b98fcac9181bdfef601d159078567` |
| [Saved reference](https://huggingface.co/datasets/frontieror/FrontierOR/blob/36c0406670b9d33eb35f0fbfdd987b4836b3aaa8/tasks/cherkesly2015/gurobi_solution/tiny_solution.json) | `99c0a308375334c116bc1aa7ed0d042cb8ca22baf8dbac0ff9facf60776d88e1` |
| Saved final candidate | `363a3a31865cbc337fa2cdec61b68e3736977c819a26205aa458b9a84cb33bd0` |
| Frozen candidate source | `bd4b573e5065b25be0bed3694fe8fc049553b2553890532e7a508aa6fc0d2683` |

The research root is `FROZEN_CAMPAIGN`.
Pinned task files are under `FrontierOR/frontier-or/tasks/cherkesly2015/`.
The final candidate is `results/final300/raw_outputs/cherkesly2015/solution_tiny.json`.
Frozen candidate source is `results/final300/code/cherkesly2015/gpt-5.6-sol/code.py`.
Exact absolute paths and additional file hashes are recorded in `review.json`.

The article can link a read-only copy of the candidate, but no public candidate URL was established by this review.
The original paper's full mathematical model and experimental instances were not reproduced.
The displayed benchmark formulation contains nonstandard or ambiguous activation expressions in its load and time equations.
This review checks the physical route semantics described by the task and implemented by its checker, rather than claiming every displayed algebraic equation is a correct transcription of the paper.

Only files in the assigned article review directory were written.
