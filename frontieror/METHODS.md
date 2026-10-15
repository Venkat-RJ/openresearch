# Methods and comparison limits

These notes retain the methodological qualifications behind the research feature. The recorded results and original evidence remain in the [research archive](README.md).

## Scheduling

The sequence matters because it connects a concrete code revision to measured results and the next experiment. It also preserves the case where V2 still missed the public reference. Several changes went into the revision together, so the records cannot isolate which change contributed most, or establish that Trackinizer itself caused the improvement.

That led back to the original research behind the task. Bierwirth and Kuhpfahl's [2017 paper](https://doi.org/10.1016/j.ejor.2017.03.030) describes EGRASP, a specialized scheduling method. The authors made an [earlier executable](https://prodlog.wiwi.uni-halle.de/forschung/research_data/2623675_2768180/?lang=en) available in 2014. We obtained that historical implementation rather than relying solely on FrontierOR's Gurobi scores. It is an authentic specialist comparison, but we have not established that the old executable reproduces every detail of the final 2017 method.

Getting the two programs to solve equivalent problems required input conversion and independent schedule checks. EGRASP could even write valid output while exiting with an unusual status, so a process exit code was not a reliable measure of success. We also did not retune the historical executable through the same development effort as the candidate.

The first synthetic comparison revealed a problem with the test. On eight of twelve cases, both methods achieved zero tardiness. Those schedules were as good as possible, but the instances were too easy to distinguish the algorithms. I tightened deadlines in the development generator and ran further development checks. This was a deliberate adjustment after seeing the initial results, not an untouched sample of all possible factory problems.

Only then did we freeze the solver, generator settings, evaluation procedure and analysis before generating thirty new synthetic scheduling problems. Each method ran three times per problem, producing 180 scheduling executions. The implementations used the same emulated Linux environment, one CPU quota, 512 MiB, a 290-second internal requested limit and a common 300-second external deadline. The shared host was not perfectly isolated, and EGRASP did not expose documented controls for matching random seeds. I registered the protocol locally in Trackinizer, not in an external public registry.

All final executions produced valid schedules. The Sagent-developed solver achieved a lower three-run mean weighted-tardiness penalty on 29 of the 30 new problems; EGRASP performed better on one. The registered 9.66% mean normalized advantage compares the methods' three-run means on each instance, divides their difference by the larger mean or 1, then averages across the thirty instances. The final problems had positive penalties. This is a defined comparison statistic, not a percentage of factory costs saved.

The result does not establish a new scheduling state of the art. It does show a completed solver performing well on new inputs from the tested distribution, against an independently developed historical method. The C++ program can be run on further compatible inputs without repeating the research session that produced it.

## Campaigns

These were three separate studies: a completed scheduling comparison, an unfinished large-instance campaign, and a completed 180-task small-instance campaign. Their results are not pooled. The next investigation was not a larger run of the factory solver. Routing trucks, cutting material and planning transport services need different algorithms and different checks. The question was whether the research process could produce useful specialized solvers when the mathematical problems kept changing.

In a later, separate campaign, six Sagent workers took thirty public small tasks each, attempting a task-specific solver for each formulation. The evaluator stored submitted solver files under the label `gpt-5.6-sol`, but this label does not establish which model authored them. The retained launch records also do not verify an earlier attribution to `sol-6.1`, so the exact authoring model remains unknown. [Model provenance and its limits](model-provenance.json) are recorded with the evidence. Candidate programs ran under FrontierOR's Docker evaluator with a standard 300-second budget, one CPU and 2 GB of memory. Sagent CLI provided the agent interface, Trackinizer retained records explicitly written to it, and external orchestration and later checks handled other parts of the evidence. Unlike the scheduling study, the CLI workflow did not automatically enforce the full hypothesis-to-conclusion sequence.

That broader result is useful as a measure of what the setup could attempt. It is not evidence that the same solver generalized across 180 problems, nor that the research workflow reliably outperformed specialist methods. The most informative cases were those whose saved decisions let us investigate why they differed from their references.

## Delivery

Additional [checks within this research](README.md#delivery) verified the saved routes and recomputed the objective. We have a better feasible plan for this one public input. We do not have proof that ten is the minimum fleet or that the new routing method is generally superior. The finding is nevertheless instructive: the released implementation could optimize its routes, but could not discover savings that required operating fewer trucks.

## Workflows

The three tool combinations describe how I worked, not a controlled ablation study. “Sagent only” refers to the agent interface; an external harness still ran evaluations. The records do not isolate the causal effect of any tool.

## Research scope

The reported outcomes concern computational benchmarks. Validation was performed within this project with human direction and Codex support. Independent external replication and savings in operating factories or delivery networks have not been demonstrated.
