# Open Research

Research code, results and experiment records from my independent work using Rekursiv’s open-source Sagent and Trackinizer, with Codex supporting infrastructure and validation.

## FrontierOR

[Read the article](https://ai-scientist-delivery-manufacturing.venkat190859.chatgpt.site/) · [Explore the research materials](frontieror/README.md) · [Methods](frontieror/METHODS.md)

| Study | What is included | Start here |
| --- | --- | --- |
| Factory scheduling | Frozen C++ solver, 30 fresh-instance comparisons, all 180 final executions, protocol and checks | [Scheduling study](frontieror/scheduling-study/README.md) |
| Earlier large-instance campaign | Checkpoint results, captured sources, failed evaluations and seven checked improvements | [Large-instance campaign](frontieror/public-large-campaign/README.md) |
| 180 small-task campaign | Selected results, task-specific code, saved solutions, inputs and checkers | [Small-task campaign](frontieror/public-small-180/README.md) |

The studies use different evaluation designs. Their guides explain the results and comparison limits. The [FrontierOR guide](frontieror/README.md) also links the delivery, cutting, transit, kitchen and bus-duty cases.

## Check the saved results

Clone or download this repository, then run from its root with Python 3.10 or newer:

```sh
python3 frontieror/tools/verify_public_release.py
```

This checks public file hashes, source mappings, saved-solution feasibility and scheduling statistics. It launches no optimization runs and makes no network requests. The [study guides](frontieror/README.md) describe the dependencies needed for complete reruns.

## Sources and reuse

The materials combine original research records and separately attributed upstream code and data. [Source projects, dataset revisions and reuse terms](frontieror/RELEASE-NOTES.md) apply to their respective files. No blanket license is assigned to the whole repository.

The [release manifest](frontieror/release-manifest.json) identifies the public files. [Technical release records](frontieror/RELEASE-RECORDS.md) retain their provenance and release history.
