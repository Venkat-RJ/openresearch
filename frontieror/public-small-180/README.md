# All 180 public small FrontierOR tasks

This separate campaign attempted 180 public tiny instances with a 300-second requested budget, one CPU and 2 GB per candidate. The selected results, after eight targeted follow-up evaluations, contain 177 checker-feasible outputs and five reviewed reference improvements, including one marginal gain. Three tasks have no accepted output.

## Results, source and outputs

- [Selected results CSV](results/campaign-results.csv)
- [Complete selected campaign record](results/campaign-results.json)
- [Every task's source and saved output](TASKS.md)
- [Original 180-task results](results/final300/results.csv) and [classification](results/final300/primary-classification.json)
- [Seven follow-ups](results/final300-logging2/results.csv) and [one further diagnostic](results/final300-energy-diagnostic/results.csv)
- [Original primary source manifest](results/final300/manifest.json)
- [Trackinizer outcome snapshots](results/final300/verified-outcomes.json)
- [Evaluation driver (release note)](../RELEASE-NOTES.md#additional-source-materials) and [dependencies (release note)](../RELEASE-NOTES.md#additional-source-materials)

The primary run and the selected campaign summary are deliberately separate. For example, 170 original results met the reported 1% objective threshold; 171 selected results did after follow-ups. The article does not treat a native 10% gate pass as a reference win.

## Five reviewed improvements

| Task | Article result | Saved evidence |
| --- | --- | --- |
| Delivery, cherkesly2015 | 10 trucks versus 30; objective 102,616.4 versus 303,868 | [Review](featured-evidence/cherkesly2015/review.md), [code](featured-evidence/cherkesly2015/candidate.py), [solution](featured-evidence/cherkesly2015/candidate.json) |
| Cutting, furini2016 | Value 1,600, reaching an area-based upper bound | [Proof and review](featured-evidence/furini2016/review.md), [code](featured-evidence/furini2016/candidate.py), [solution](featured-evidence/furini2016/candidate.json) |
| Transit, borndorfer2007 | Weighted objective 48.23% lower, with increased passenger travel time | [Review](featured-evidence/borndorfer2007/review.md), [code](featured-evidence/borndorfer2007/candidate.py), [solution](featured-evidence/borndorfer2007/candidate.json) |
| Kitchen, dayarian2019 | Modeled objective 14.18% lower | [Code](featured-evidence/dayarian2019/candidate.py), [solution](featured-evidence/dayarian2019/candidate.json), [checker](featured-evidence/dayarian2019/checker.py) |
| Bus duties, hadjar2006 | Modeled objective 0.00232% lower, with unresolved reference-label interpretation | [Code](featured-evidence/hadjar2006/candidate.py), [solution](featured-evidence/hadjar2006/candidate.json), [checker](featured-evidence/hadjar2006/checker.py) |

Run `python3 frontieror/tools/verify_archive.py` from the repository root to verify source bindings and replay the five archived checkers without optimization. The saved reviews document additional arithmetic and semantic checks; they are within this research effort, not outside scientific replication. Only the cutting instance has the optimality proof described above.

To plan full reruns, inspect the pinned source and dataset revisions in the campaign record, the [evaluator documentation (release note)](../RELEASE-NOTES.md#additional-source-materials), and [runtime recipe](runtime-build-context/). Some solvers require commercial Gurobi licensing. Original code directories containing model-like labels are historical harness paths and do not independently identify the authoring model.
