# Factory-scheduling study

A frozen Sagent-developed C++ solver beat a historical EGRASP implementation on 29 of 30 fresh synthetic instances by three-run mean weighted tardiness. All 180 final executions produced valid schedules. The registered mean normalized difference was 9.66%; this is a comparison statistic, not measured factory cost savings.

## Results and validation

- [Thirty instance comparisons](assets/final-instance-data.csv)
- [180 execution rows](assets/final-run-data.csv)
- [Full final results](final/results.json) and [registered analysis](final/analysis.json)
- [Frozen protocol and budget](final/plan-before-generation.json)
- [Saved instances](INSTANCES.md)
- [Original audit](final/independent-audit/)
- [Earlier public and development results](earlier/records.json), including the retained failure

## Code and research records

- [Frozen solver](final/source/frozen-v2.cpp)
- [Final comparison runner](final/source/final_comparison.py)
- [Instance generator](final/source/synthetic_instances_v2.py)
- [Independent schedule checker](final/source/audit_schedule.py)
- [Official checker snapshot](final/source/official_checker.py)
- [Custom research tools](workflow/source/research-worker.py) and [tracking helpers](workflow/source/campaign.py)
- [Workflow provenance](assets/workflow-provenance.json)
- [Registration](final/registration-readback.json), [experiment](final/experiment-readback.json), [conclusion](final/conclusion-readback.json)

## Reproduction

Run `python3 frontieror/scheduling-study/verify_saved.py` from the repository root with Python 3.10 or newer. This portable verifier checks 291 saved execution records: 63 public, 48 development, and 180 final. It preserves one development failure and does not pool the development distributions.

For a complete solver rerun, inspect the [runtime recipes](final/runtime-recipe/) and [original reproduction notes](README.txt). The historical EGRASP executable is not redistributed. Obtain it from the [authors' page](https://prodlog.wiwi.uni-halle.de/forschung/research_data/2623675_2768180/?lang=en) and check its recorded hash. The frozen runner contains original workspace paths and is retained for inspection; adapt paths before executing it.

This is a result on the tested synthetic distribution and historical comparator. It does not establish a new state of the art, matched random seeds, or isolated hardware timing.
