# FrontierOR research archive

Results, solver code and saved experiment records from my research using Sagent and Trackinizer, with Codex supporting infrastructure and validation.

[Published article](https://ai-scientist-delivery-manufacturing.venkat190859.chatgpt.site/) · [Methods](METHODS.md) · [Source projects and attribution](RELEASE-NOTES.md)

## Start here

These are three separate studies. Each has its own inputs, execution settings and results.

| Study | Results | Code | Saved records and checks |
| --- | --- | --- | --- |
| Factory scheduling | [30-instance comparison](scheduling-study/assets/final-instance-data.csv) and [analysis](scheduling-study/final/analysis.json) | [Frozen C++ solver](scheduling-study/final/source/frozen-v2.cpp) | [180 final executions](scheduling-study/final/results.json), [test protocol](scheduling-study/final/plan-before-generation.json) and [study guide](scheduling-study/README.md) |
| Earlier large-instance campaign | [Checkpoint comparisons](public-large-campaign/checkpoint/case-comparisons.csv) | [Source index](public-large-campaign/CODE.md) | [Saved-solution checks](public-large-campaign/checkpoint/independent-audit.json) and [study guide](public-large-campaign/README.md) |
| 180 small-task campaign | [Selected results](public-small-180/results/campaign-results.csv) | [Task-by-task source and output index](public-small-180/TASKS.md) | [Campaign records](public-small-180/results/campaign-results.json), [primary classification](public-small-180/results/final300/primary-classification.json) and [study guide](public-small-180/README.md) |

For the delivery, cutting, transit, kitchen and bus-duty findings, start with the featured cases below. Each case links its solver, input, saved solution, checker and released reference. The [claim map](CLAIMS.md) connects the article's main findings to their records.

## What the files contain

- **Results** contain objectives, execution status and comparisons. A feasible output may match or trail its reference.
- **Solver code** contains the programs used to construct solutions. The scheduling study has one frozen solver; the broader campaigns have task-specific programs.
- **Saved solutions** contain schedules, routes or other decisions returned by those programs.
- **Checks** recompute feasibility and objectives from saved solutions. The methods explain the comparison limits and unresolved questions.
- **Historical records** preserve earlier development runs, failures and source identities alongside the selected results.

## Download and check the saved results

Clone this repository, or use GitHub’s **Code → Download ZIP** and extract it. Run the command below from the repository root. The three study ZIPs are also available in this folder; extract those into the same directory if downloading them separately.

Use Python 3.10 or newer:

```sh
python3 frontieror/tools/verify_public_release.py
```

The command checks the downloaded file hashes, selected source mappings, featured saved solutions and scheduling statistics. It reads saved files locally and launches no optimization runs or network requests.

To rerun the experiments, follow the individual study guides for runtime settings and dependencies. Obtain the historical EGRASP executable from the [original authors](https://prodlog.wiwi.uni-halle.de/forschung/research_data/2623675_2768180/?lang=en). Checking the saved results and independently rerunning the experiments are separate steps.

## File versions and reuse

[release-manifest.json](release-manifest.json) lists hashes for the current public files. [verification-public.json](verification-public.json) records the saved-result checks. Historical hashes identify the original files; the [technical release records](RELEASE-RECORDS.md) explain their relationship to the public downloads. The [release notes](RELEASE-NOTES.md) list upstream sources and reuse terms.

## Delivery

A feasible 10-truck plan with objective 102,616.4 versus the reference’s 303,868. The minimum fleet remains unknown.

[Solver](public-small-180/featured-evidence/cherkesly2015/candidate.py) · [Saved solution](public-small-180/featured-evidence/cherkesly2015/candidate.json) · [Input](public-small-180/featured-evidence/cherkesly2015/input.json) · [Checker](public-small-180/featured-evidence/cherkesly2015/checker.py) · [Reference code](public-small-180/featured-evidence/cherkesly2015/reference.py) · [Reference solution](public-small-180/featured-evidence/cherkesly2015/reference.json)

[Review and explanation](public-small-180/featured-evidence/cherkesly2015/review.md)

## Cutting

A value of 1,600 reaches the area-based upper bound for this saved instance.

[Solver](public-small-180/featured-evidence/furini2016/candidate.py) · [Saved solution](public-small-180/featured-evidence/furini2016/candidate.json) · [Input](public-small-180/featured-evidence/furini2016/input.json) · [Checker](public-small-180/featured-evidence/furini2016/checker.py) · [Reference code](public-small-180/featured-evidence/furini2016/reference.py) · [Reference solution](public-small-180/featured-evidence/furini2016/reference.json)

[Review and explanation](public-small-180/featured-evidence/furini2016/review.md)

## Transit

A 48.23% lower weighted objective with increased passenger travel time.

[Solver](public-small-180/featured-evidence/borndorfer2007/candidate.py) · [Saved solution](public-small-180/featured-evidence/borndorfer2007/candidate.json) · [Input](public-small-180/featured-evidence/borndorfer2007/input.json) · [Checker](public-small-180/featured-evidence/borndorfer2007/checker.py) · [Reference code](public-small-180/featured-evidence/borndorfer2007/reference.py) · [Reference solution](public-small-180/featured-evidence/borndorfer2007/reference.json)

[Review and explanation](public-small-180/featured-evidence/borndorfer2007/review.md)

## Kitchen

A feasible modeled objective 14.18% lower for the saved instance.

[Solver](public-small-180/featured-evidence/dayarian2019/candidate.py) · [Saved solution](public-small-180/featured-evidence/dayarian2019/candidate.json) · [Input](public-small-180/featured-evidence/dayarian2019/input.json) · [Checker](public-small-180/featured-evidence/dayarian2019/checker.py) · [Reference code](public-small-180/featured-evidence/dayarian2019/reference.py) · [Reference solution](public-small-180/featured-evidence/dayarian2019/reference.json)

## Bus duties

A marginal 0.00232% difference with an unresolved reference-label interpretation.

[Solver](public-small-180/featured-evidence/hadjar2006/candidate.py) · [Saved solution](public-small-180/featured-evidence/hadjar2006/candidate.json) · [Input](public-small-180/featured-evidence/hadjar2006/input.json) · [Checker](public-small-180/featured-evidence/hadjar2006/checker.py) · [Reference code](public-small-180/featured-evidence/hadjar2006/reference.py) · [Reference solution](public-small-180/featured-evidence/hadjar2006/reference.json)

## Recorded scheduling iteration

The [development experiment and decision record](publication/study-evidence/scheduling/research-decision.json) documents the V1-to-V2 revision, two seeds and subsequent research decisions. [Workflow provenance](publication/study-evidence/scheduling/workflow-provenance.json) records the custom tools and source identities. These public development runs are separate from the final fresh-instance comparison.

## Diagrams and figures

The [publication figures](publication/figures/) and [editable research diagrams](publication/diagrams/) accompany the article. The live article retains its animations and original tool recording.
