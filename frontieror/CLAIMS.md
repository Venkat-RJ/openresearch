# Article claims and supporting evidence

| Claim | Results | Validation or proof | Code |
| --- | --- | --- | --- |
| Scheduling: lower three-run mean on 29 of 30 new instances | [Instance table](scheduling-study/assets/final-instance-data.csv), [180 executions](scheduling-study/final/results.json) | [Frozen plan](scheduling-study/final/plan-before-generation.json), [offline verifier](scheduling-study/verify_final.py) | [Frozen solver](scheduling-study/final/source/frozen-v2.cpp) |
| Earlier large campaign: seven checked incumbent-reference improvements | [Case comparisons](public-large-campaign/checkpoint/case-comparisons.csv) | [Ten-case saved-solution audit](public-large-campaign/checkpoint/independent-audit.json) | [Source index](public-large-campaign/CODE.md) |
| Small campaign: 180 tasks, 177 checker-feasible selected outputs, five reviewed improvements | [Campaign summary](public-small-180/results/campaign-results.json) | [Primary classification](public-small-180/results/final300/primary-classification.json), [source and checker verifier](tools/verify_archive.py) | [180-task index](public-small-180/TASKS.md) |
| Delivery: 30 to 10 trucks; 66.23% lower modeled objective | [Saved solution](public-small-180/featured-evidence/cherkesly2015/candidate.json) | [Route and reference-restriction review](public-small-180/featured-evidence/cherkesly2015/review.md) | [Candidate](public-small-180/featured-evidence/cherkesly2015/candidate.py), [reference](public-small-180/featured-evidence/cherkesly2015/reference.py) |
| Cutting: recovered value 1,600 equals the upper bound | [Saved construction](public-small-180/featured-evidence/furini2016/candidate.json) | [Optimality and witness review](public-small-180/featured-evidence/furini2016/review.md) | [Candidate](public-small-180/featured-evidence/furini2016/candidate.py) |
| Transit, kitchen and bus-duty improvements | [Selected task table](public-small-180/TASKS.md) | [Qualifications and saved checkers](public-small-180/README.md) | [Selected source links](public-small-180/TASKS.md) |

These are computational results. Lower modeled objectives are not measured business savings. Saved-solution review is not outside scientific replication. The cutting upper-bound argument proves optimality only for its specific instance.
